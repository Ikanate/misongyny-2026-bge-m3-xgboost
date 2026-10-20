# run_train.py
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl

import config as cfg

import preprocess.data_frame_preprocess as prep
import models.model_bge_m3 as bge_m3
import dataset.Train as multi_task
from train.train_code_pl import MultiTaskLitModule, build_trainer

VALID_METRICS = {"f1_t1", "f1_t2", "f1_t3", "avg_f1"}


def deep_merge(base: dict, extra: dict | None) -> dict:
    """Fusiona por sección: lo que pasas sobreescribe solo esas keys."""
    out = {section: dict(values) for section, values in base.items()}
    for section, values in (extra or {}).items():
        out.setdefault(section, {}).update(values)
    return out


def apply_overrides(cfg, overrides: dict) -> None:
    metric = overrides.get("train", {}).get("BEST_MODEL_METRIC")
    if metric is not None and metric not in VALID_METRICS:
        raise ValueError(f"BEST_MODEL_METRIC inválida: {metric}. Opciones: {VALID_METRICS}")

    # Validar TODO antes de aplicar nada (así no queda cfg a medias si hay un error)
    for section, values in overrides.items():
        if not hasattr(cfg, section):
            raise AttributeError(f"cfg no tiene la sección '{section}'")
        target = getattr(cfg, section)
        unknown = [k for k in values if not hasattr(target, k)]
        if unknown:
            raise AttributeError(f"Keys que no existen en cfg.{section}: {unknown}")

    for section, values in overrides.items():
        target = getattr(cfg, section)
        for key, value in values.items():
            setattr(target, key, value)
    

# ------------------------------------------------------------------ data
def load_dataframe(cfg) -> pd.DataFrame:
    raw_df = pd.read_csv(cfg.paths.CSV_PATH)
    return prep.prepare_dataframe(raw_df, cfg)  # añade lyrics_clean y gloss_clean


def build_strat_key(df: pd.DataFrame, cfg) -> pd.Series:
    t2 = df[cfg.columns.LABEL_COLS_T2].astype(str).agg("-".join, axis=1)
    return df["label_t1"].astype(str) + "_" + df["label_t3"].astype(str) + "_" + t2


def split_dataframe(df: pd.DataFrame, cfg):
    return train_test_split(
        df,
        test_size=cfg.train.TEST_SIZE,
        stratify=build_strat_key(df, cfg),
        random_state=cfg.train.SEED,
    )


def build_dataset(df: pd.DataFrame, cfg, tokenizer):
    return multi_task.MultiTaskDataset(
        df, cfg, tokenizer,
        cfg.tokenization.MAX_LEN,
        head_len=cfg.tokenization.HEAD_LEN,
        model_prefix=cfg.paths.MODEL_PREFIX,
    )


def build_dataloaders(train_df, val_df, cfg, tokenizer, num_workers: int = 6):
    train_ds = build_dataset(train_df, cfg, tokenizer)
    val_ds = build_dataset(val_df, cfg, tokenizer)

    common = dict(batch_size=cfg.train.BATCH_SIZE, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader


# ----------------------------------------------------------------- model
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_lit_module(model, tokenizer, cfg, train_df):
    return MultiTaskLitModule(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        train_df=train_df,
        t2_label_cols=cfg.columns.LABEL_COLS_T2,
    )


# ------------------------------------------------------------------ main
def main_bge(overrides: dict | None = None, num_workers: int = 6):
    
    apply_overrides(cfg, deep_merge({}, overrides))
    
    pl.seed_everything(cfg.train.SEED)
    os.makedirs(cfg.paths.SAVE_DIR, exist_ok=True)

    df = load_dataframe(cfg)
    train_df, val_df = split_dataframe(df, cfg)

    device = get_device()
    tokenizer, model = bge_m3.load_tokenizer_model(cfg, device)

    train_loader, val_loader = build_dataloaders(
        train_df, val_df, cfg, tokenizer, num_workers=num_workers
    )

    lit = build_lit_module(model, tokenizer, cfg, train_df)
    trainer = build_trainer(cfg)
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)

    ckpt = trainer.checkpoint_callback
    print("Mejor ckpt:", ckpt.best_model_path)
    print("Mejor métrica:", ckpt.best_model_score)

    return {
        "trainer": trainer,
        "lit": lit,
        "model": model,
        "tokenizer": tokenizer,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "device": device,
        "best_ckpt_path": ckpt.best_model_path,
    }


if __name__ == "__main__":
    main()