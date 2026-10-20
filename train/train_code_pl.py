import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor,EarlyStopping
from sklearn.metrics import f1_score, classification_report
from transformers import get_linear_schedule_with_warmup

_VALID_METRICS = {'f1_t1', 'f1_t2', 'f1_t3', 'avg_f1'}


# ─────────────────────────── helpers (sin cambios) ───────────────────────────
def resolve_metric_key(cfg) -> str:
    key = getattr(cfg.train, 'BEST_MODEL_METRIC', 'avg_f1')
    if key not in _VALID_METRICS:
        print(f"⚠️  BEST_MODEL_METRIC='{key}' no es válido. Opciones: {_VALID_METRICS}. "
              f"Usando 'avg_f1'.")
        key = 'avg_f1'
    return key


def class_weights_binary(series: pd.Series) -> torch.Tensor:
    n_neg = (series == 0).sum()
    n_pos = (series == 1).sum()
    total = len(series)
    return torch.tensor(
        [total / (2 * max(n_neg, 1)), total / (2 * max(n_pos, 1))],
        dtype=torch.float,
    )


def pos_weights_multilabel(df: pd.DataFrame, label_cols: list[str]) -> torch.Tensor:
    w = []
    for col in label_cols:
        n_pos = (df[col] == 1).sum()
        n_neg = (df[col] == 0).sum()
        w.append(n_neg / max(n_pos, 1))
    return torch.tensor(w, dtype=torch.float)


# ─────────────────────────────── LightningModule ─────────────────────────────
class MultiTaskLitModule(pl.LightningModule):
    """
    model: nn.Module con .backbone, .head_t1, .head_t2, .head_t3
           y forward(ids, mask) -> (lg1, lg2, lg3)
    """

    def __init__(self, model, tokenizer, cfg, train_df, t2_label_cols,
                 verbose_reports: bool = True):
        super().__init__()
        # cfg y flags quedan guardados dentro del .ckpt (hparams)
        #self.save_hyperparameters(ignore=['model', 'tokenizer', 'train_df'])
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.verbose_reports = verbose_reports
        self.metric_key = resolve_metric_key(cfg)

        # Pesos como buffers -> se mueven solos al device y viajan en el ckpt.
        self.register_buffer('cw_t1', class_weights_binary(train_df['label_t1']))
        self.register_buffer('cw_t3', class_weights_binary(train_df['label_t3']))
        self.register_buffer('pw_t2', pos_weights_multilabel(train_df, t2_label_cols))

        self._val_preds = None
        self._val_labels = None

    # ---- setup -------------------------------------------------------------
    def setup(self, stage=None):
        if self.cfg.train.GRADIENT_CHECKPOINTING:
            self.model.backbone.gradient_checkpointing_enable()
            print('[SETUP] Gradient checkpointing: ON')
        else:
            self.model.backbone.gradient_checkpointing_disable()
            print('[SETUP] Gradient checkpointing: OFF')

    def forward(self, ids, mask):
        return self.model(ids, mask)

    # ---- losses ------------------------------------------------------------
    def _task_losses(self, lg1, lg2, lg3, lb1, lb2, lb3):
        l1 = F.cross_entropy(lg1, lb1, weight=self.cw_t1)
        l2 = F.binary_cross_entropy_with_logits(lg2, lb2, pos_weight=self.pw_t2)
        l3 = F.cross_entropy(lg3, lb3, weight=self.cw_t3)
        return l1, l2, l3

    # ---- train -------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        lg1, lg2, lg3 = self(batch['input_ids'], batch['attention_mask'])
        l1, l2, l3 = self._task_losses(
            lg1, lg2, lg3, batch['label_t1'], batch['label_t2'], batch['label_t3']
        )
        t = self.cfg.train
        loss = t.LOSS_W1 * l1 + t.LOSS_W2 * l2 + t.LOSS_W3 * l3

        bs = batch['input_ids'].size(0)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log_dict({'train/l1': l1, 'train/l2': l2, 'train/l3': l3},
                      on_step=False, on_epoch=True, batch_size=bs)
        return loss

    # ---- validation --------------------------------------------------------
    def on_validation_epoch_start(self):
        self._val_preds = {'t1': [], 't2': [], 't3': []}
        self._val_labels = {'t1': [], 't2': [], 't3': []}

    def validation_step(self, batch, batch_idx):
        lb1, lb2, lb3 = batch['label_t1'], batch['label_t2'], batch['label_t3']
        lg1, lg2, lg3 = self(batch['input_ids'], batch['attention_mask'])
        l1, l2, l3 = self._task_losses(lg1, lg2, lg3, lb1, lb2, lb3)

        bs = lb1.size(0)
        self.log_dict({'val_loss_t1': l1, 'val_loss_t2': l2, 'val_loss_t3': l3},
                      on_step=False, on_epoch=True, batch_size=bs)

        thr = self.cfg.train.THRESHOLD_T2
        self._val_preds['t1'].extend(lg1.argmax(-1).cpu().tolist())
        self._val_preds['t2'].extend((torch.sigmoid(lg2) >= thr).int().cpu().tolist())
        self._val_preds['t3'].extend(lg3.argmax(-1).cpu().tolist())
        self._val_labels['t1'].extend(lb1.cpu().tolist())
        self._val_labels['t2'].extend(lb2.int().cpu().tolist())
        self._val_labels['t3'].extend(lb3.cpu().tolist())

    def on_validation_epoch_end(self):
        f1 = {
            k: f1_score(self._val_labels[k], self._val_preds[k],
                        average='macro', zero_division=0)
            for k in ('t1', 't2', 't3')
        }
        avg = sum(f1.values()) / 3

        # sync_dist no hace falta en 1 GPU; con DDP ver nota sobre torchmetrics
        self.log_dict({
            'val_f1_t1': f1['t1'], 'val_f1_t2': f1['t2'],
            'val_f1_t3': f1['t3'], 'val_avg_f1': avg,
        }, prog_bar=True)

        if self.verbose_reports and not self.trainer.sanity_checking:
            for k, title in (('t1', 'T1'), ('t2', 'T2 (Multilabel/Binario)'), ('t3', 'T3')):
                print(f"\n{'=' * 30}\nREPORTE DE CLASIFICACIÓN - {title}\n{'=' * 30}")
                print(classification_report(self._val_labels[k], self._val_preds[k],
                                            zero_division=0))

    # ---- optimizer / scheduler --------------------------------------------
    def configure_optimizers(self):
        t = self.cfg.train
        m = self.model
        optimizer = torch.optim.AdamW([
            {'params': m.backbone.parameters(), 'lr': t.LR_BACKBONE},
            {'params': m.head_t1.parameters(),  'lr': t.LR_HEADS},
            {'params': m.head_t2.parameters(),  'lr': t.LR_HEADS},
            {'params': m.head_t3.parameters(),  'lr': t.LR_HEADS},
        ], weight_decay=t.WEIGHT_DECAY)

        # estimated_stepping_batches ya considera epochs, accumulate_grad_batches, etc.
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(t.WARMUP_RATIO * total_steps),
            num_training_steps=total_steps,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }


# ──────────────────────────────── entry point ────────────────────────────────
def build_trainer(cfg) -> pl.Trainer:
    
    metric_key = resolve_metric_key(cfg)
    monitor_name = f'val_{metric_key}'

    ckpt_cb = ModelCheckpoint(
        dirpath=cfg.paths.SAVE_DIR,
        filename=f'best-{cfg.flags.USE_GLOSS}-gloss-' + '{epoch:02d}-{' + f'val_{metric_key}' + ':.4f}',
        monitor=f'val_{metric_key}',
        mode='max',
        save_top_k=1,
        save_last=True,           # last.ckpt para reanudar entrenamiento
    )

    early_stop_cb = EarlyStopping(
        monitor=monitor_name,
        mode='max',
        patience= cfg.train.PATIENCE,
        min_delta= cfg.train.MIN_DELTA,
        check_on_train_epoch_end=False,   # evalúa tras la validación
        verbose=True,
    )
    
    return pl.Trainer(
        max_epochs=cfg.train.EPOCHS,
        accelerator='auto',
        devices=1,
        gradient_clip_val=1.0,                 # reemplaza clip_grad_norm_
        log_every_n_steps=20,
        callbacks=[ckpt_cb,early_stop_cb, LearningRateMonitor(logging_interval='step')],
         precision='bf16-mixed', #reduce el tiempo por epoch   # opcional
        # accumulate_grad_batches=2,           # opcional
    )


if __name__ == '__main__':
    # model, tokenizer, cfg, train_df, train_loader, val_loader vienen de tu setup
    lit = MultiTaskLitModule(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        train_df=train_df,
        t2_label_cols=['label_t2'],
    )
    trainer = build_trainer(cfg)
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print('Mejor ckpt:', trainer.checkpoint_callback.best_model_path)
    print('Mejor métrica:', trainer.checkpoint_callback.best_model_score)