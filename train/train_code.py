import os

import pickle
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModel,
                          get_linear_schedule_with_warmup)

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# METRIC SELECTOR
_VALID_METRICS = {'f1_t1', 'f1_t2', 'f1_t3', 'avg_f1'}

def save_training_config(save_dir: str, cfg) -> None:
    """
    Guarda el objeto cfg completo usando pickle.
    """
    out_path = os.path.join(save_dir, "training_config.pkl")

    with open(out_path, "wb") as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"📝 Config guardada en {out_path}")


def load_training_config(save_dir: str):
    """
    Recupera el objeto cfg.
    """
    in_path = os.path.join(save_dir, "training_config.pkl")

    with open(in_path, "rb") as f:
        cfg = pickle.load(f)

    return cfg

    


def get_checkpoint_metric(val_metrics: dict, cfg) -> float:
    """
    Extrae el valor de la métrica de guardado definida en cfg.train.BEST_MODEL_METRIC.

    Valores válidos: 'f1_t1' | 'f1_t2' | 'f1_t3' | 'avg_f1'
    Si el parámetro no existe o es inválido, hace fallback a 'avg_f1' con un warning.
    """
    metric_key = getattr(cfg.train, 'BEST_MODEL_METRIC', 'avg_f1')

    if metric_key not in _VALID_METRICS:
        print(
            f"   ⚠️  BEST_MODEL_METRIC='{metric_key}' no es válido. "
            f"Opciones: {_VALID_METRICS}. Usando 'avg_f1' como fallback."
        )
        metric_key = 'avg_f1'

    return val_metrics[metric_key], metric_key


def evaluate(model, loader, criterions, cfg, device):
    """
    criterions: dict with keys 't1', 't2', 't3'
    """
    criterion_t1, criterion_t2, criterion_t3 = (
        criterions['t1'], criterions['t2'], criterions['t3']
    )

    model.eval()
    totals = {'l1': 0., 'l2': 0., 'l3': 0.}
    preds  = {'t1': [], 't2': [], 't3': []}
    labels = {'t1': [], 't2': [], 't3': []}

    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lb1  = batch['label_t1'].to(device)
            lb2  = batch['label_t2'].to(device)
            lb3  = batch['label_t3'].to(device)

            lg1, lg2, lg3 = model(ids, mask)

            totals['l1'] += criterion_t1(lg1, lb1).item()
            totals['l2'] += criterion_t2(lg2, lb2).item()
            totals['l3'] += criterion_t3(lg3, lb3).item()

            preds['t1'].extend(lg1.argmax(-1).cpu().tolist())
            preds['t2'].extend((torch.sigmoid(lg2) >= cfg.train.THRESHOLD_T2).int().cpu().tolist())
            preds['t3'].extend(lg3.argmax(-1).cpu().tolist())

            labels['t1'].extend(lb1.cpu().tolist())
            labels['t2'].extend(lb2.int().cpu().tolist())
            labels['t3'].extend(lb3.cpu().tolist())

    n = len(loader)
    f1_t1 = f1_score(labels['t1'], preds['t1'], average='macro', zero_division=0)
    f1_t2 = f1_score(labels['t2'], preds['t2'], average='macro', zero_division=0)
    f1_t3 = f1_score(labels['t3'], preds['t3'], average='macro', zero_division=0)

    
    print("\n" + "="*30)
    print("REPORTE DE CLASIFICACIÓN - T1")
    print("="*30)
    print(classification_report(labels['t1'], preds['t1'], zero_division=0))

    print("\n" + "="*30)
    print("REPORTE DE CLASIFICACIÓN - T2 (Multilabel/Binario)")
    print("="*30)
    print(classification_report(labels['t2'], preds['t2'], zero_division=0))

    print("\n" + "="*30)
    print("REPORTE DE CLASIFICACIÓN - T3")
    print("="*30)
    print(classification_report(labels['t3'], preds['t3'], zero_division=0))
    # -------------------------------
    
    return {
        'loss_t1': totals['l1'] / n, 'loss_t2': totals['l2'] / n,
        'loss_t3': totals['l3'] / n,
        'f1_t1':   f1_t1, 'f1_t2': f1_t2, 'f1_t3': f1_t3,
        'avg_f1':  (f1_t1 + f1_t2 + f1_t3) / 3,
        'preds':   preds,  'labels': labels,
    }



    
# OPTIMIZER & SCHEDULER
def build_optimizer_and_scheduler(model, train_loader, cfg):
    total_steps  = len(train_loader) * cfg.train.EPOCHS
    warmup_steps = int(cfg.train.WARMUP_RATIO * total_steps)

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': cfg.train.LR_BACKBONE},
        {'params': model.head_t1.parameters(),  'lr': cfg.train.LR_HEADS},
        {'params': model.head_t2.parameters(),  'lr': cfg.train.LR_HEADS},
        {'params': model.head_t3.parameters(),  'lr': cfg.train.LR_HEADS},
    ], weight_decay=cfg.train.WEIGHT_DECAY)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    return optimizer, scheduler


# SINGLE TRAINING STEP
def train_step(model, batch, criterions, cfg, optimizer, scheduler, device):
    """Returns per-task losses as plain floats."""
    ids  = batch['input_ids'].to(device)
    mask = batch['attention_mask'].to(device)
    lb1  = batch['label_t1'].to(device)
    lb2  = batch['label_t2'].to(device)
    lb3  = batch['label_t3'].to(device)

    optimizer.zero_grad()
    lg1, lg2, lg3 = model(ids, mask)

    l1   = criterions['t1'](lg1, lb1)
    l2   = criterions['t2'](lg2, lb2)
    l3   = criterions['t3'](lg3, lb3)
    loss = cfg.train.LOSS_W1 * l1 + cfg.train.LOSS_W2 * l2 + cfg.train.LOSS_W3 * l3

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    return loss.item(), l1.item(), l2.item(), l3.item()

def setup_gradient_checkpointing(model, cfg) -> None:
    """
    Activa o desactiva gradient checkpointing en el backbone.
    Reduce uso de VRAM a costa de ~20-30% más tiempo de entrenamiento.

    cfg.train.GRADIENT_CHECKPOINTING: bool  (default False si no existe)
    """
    enabled = getattr(cfg.train, 'GRADIENT_CHECKPOINTING', False)

    if enabled:
        model.backbone.gradient_checkpointing_enable()
        #model.backbone.config.use_cache = False #only if warning
        print('[SETUP] Gradient checkpointing: ON  (menos VRAM, más lento)')
    else:
        model.backbone.gradient_checkpointing_disable()
        print('[SETUP] Gradient checkpointing: OFF')
        

# SINGLE EPOCH
def train_epoch(model, train_loader, criterions, cfg, optimizer, scheduler, device, epoch):
    model.train()
    running = {'total': 0., 'l1': 0., 'l2': 0., 'l3': 0.}

    for step, batch in enumerate(train_loader):
        total, l1, l2, l3 = train_step(
            model, batch, criterions, cfg, optimizer, scheduler, device
        )
        running['total'] += total
        running['l1']    += l1
        running['l2']    += l2
        running['l3']    += l3

        if (step + 1) % 20 == 0:
            print(
                f'  Epoch {epoch+1} | Step {step+1}/{len(train_loader)} '
                f'| Loss: {total:.4f} '
                f'(T1={l1:.3f} T2={l2:.3f} T3={l3:.3f})'
            )

    n = len(train_loader)
    return {k: v / n for k, v in running.items()}



# CHECKPOINT SAVING
def save_checkpoint(model, tokenizer, cfg):
    os.makedirs(cfg.paths.SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.paths.SAVE_DIR, 'model_weights.pt'))
    tokenizer.save_pretrained(cfg.paths.SAVE_DIR)
    model.backbone.config.save_pretrained(cfg.paths.SAVE_DIR)
    save_training_config(cfg.paths.SAVE_DIR, cfg)


# FULL TRAINING LOOP
def train(model, train_loader, val_loader, criterions, tokenizer, cfg, device):
    """
    Entry point. Returns (history, best_metric_value).

    cfg.train.BEST_MODEL_METRIC controla qué métrica se usa para guardar el
    mejor checkpoint. Valores válidos: 'f1_t1' | 'f1_t2' | 'f1_t3' | 'avg_f1'
    Si se omite o es inválido, hace fallback a 'avg_f1'.
    """
    setup_gradient_checkpointing(model, cfg)
    
    optimizer, scheduler = build_optimizer_and_scheduler(model, train_loader, cfg)

    best_metric_value = 0.0
    history           = []

    # Resolvemos la métrica una sola vez antes del loop para que el nombre
    # aparezca correctamente en los logs desde la primera época.
    _, metric_key = get_checkpoint_metric({'f1_t1': 0, 'f1_t2': 0, 'f1_t3': 0, 'avg_f1': 0}, cfg)

    print('\n[TRAINING] Iniciando entrenamiento multi-task…')
    print(f'[TRAINING] Métrica de guardado: {metric_key}\n')

    for epoch in range(cfg.train.EPOCHS):
        train_losses = train_epoch(
            model, train_loader, criterions, cfg, optimizer, scheduler, device, epoch
        )
        val_metrics = evaluate(model, val_loader, criterions, cfg, device)

        rec = {
            'epoch':      epoch + 1,
            'train_loss': train_losses['total'],
            'train_l1':   train_losses['l1'],
            'train_l2':   train_losses['l2'],
            'train_l3':   train_losses['l3'],
            **{k: v for k, v in val_metrics.items() if k not in ('preds', 'labels')},
        }
        history.append(rec)

        current_metric, metric_key = get_checkpoint_metric(val_metrics, cfg)

        print(f'\n📌 Epoch {epoch+1}/{cfg.train.EPOCHS}')
        print(
            f'   Train loss: {rec["train_loss"]:.4f}  '
            f'(T1={rec["train_l1"]:.3f} T2={rec["train_l2"]:.3f} T3={rec["train_l3"]:.3f})'
        )
        print(
            f'   Val   loss: T1={rec["loss_t1"]:.3f} '
            f'T2={rec["loss_t2"]:.3f} T3={rec["loss_t3"]:.3f}'
        )
        print(
            f'   Val   F1:   T1={rec["f1_t1"]:.4f} T2={rec["f1_t2"]:.4f} '
            f'T3={rec["f1_t3"]:.4f} → Avg={rec["avg_f1"]:.4f}'
        )

        # ── Checkpoint basado en la métrica elegida ──────────────────────────
        if current_metric > best_metric_value:
            best_metric_value = current_metric
            save_checkpoint(model, tokenizer, cfg)
            print(f'   💾 Guardado (mejor {metric_key}: {best_metric_value:.4f})')

    print(
        f'\n[TRAINING] Entrenamiento completo — '
        f'Mejor {metric_key}: {best_metric_value:.4f}'
    )
    return history, best_metric_value


def class_weights_binary(series: pd.Series, device: torch.device) -> torch.Tensor:
    n_neg = (series == 0).sum()
    n_pos = (series == 1).sum()
    total = len(series)
    return torch.tensor(
        [total / (2 * max(n_neg, 1)), total / (2 * max(n_pos, 1))],
        dtype=torch.float,
    ).to(device)


def pos_weights_multilabel(df: pd.DataFrame, label_cols: list[str], device: torch.device) -> torch.Tensor:
    """BCE pos_weight = n_neg / n_pos per label column."""
    weights = []
    for col in label_cols:
        n_pos = (df[col] == 1).sum()
        n_neg = (df[col] == 0).sum()
        weights.append(n_neg / max(n_pos, 1))
    return torch.tensor(weights, dtype=torch.float).to(device)


# CRITERION FACTORY
def build_criterions(
    train_df: pd.DataFrame,
    t2_label_cols: list[str],
    device: torch.device,
) -> dict[str, nn.Module]:
    """
    Returns criterions ready to use.
    Adjust t2_label_cols to whatever columns back task-2.
    """
    cw_t1 = class_weights_binary(train_df['label_t1'], device)
    cw_t3 = class_weights_binary(train_df['label_t3'], device)
    pw_t2 = pos_weights_multilabel(train_df, t2_label_cols, device)

    return {
        't1': nn.CrossEntropyLoss(weight=cw_t1),
        't2': nn.BCEWithLogitsLoss(pos_weight=pw_t2),
        't3': nn.CrossEntropyLoss(weight=cw_t3),
    }


# ENTRY POINT
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterions = build_criterions(
        train_df=train_df,
        t2_label_cols=['label_t2'],
        device=device,
    )

    history, best_metric = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterions=criterions,
        tokenizer=tokenizer,
        cfg=cfg,
        device=device,
    )