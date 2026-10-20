# xgb_stage.py
import json
import os
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import torch
import xgboost as xgb
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
from torch.utils.tensorboard import SummaryWriter

import models.model_xgboost as xgb_m3
from bge_g4_step import main_bge, deep_merge


optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_best_weights(lit, ckpt_path, device):
    state = torch.load(ckpt_path, map_location="cpu")
    lit.load_state_dict(state["state_dict"])
    lit.to(device).eval()

# ------------------------------------------------------------ logging utils
def _metrics(y_true, y_pred):
    return {
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def _log_confusion_matrix(writer, y_true, y_pred, tag):
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(ax=ax, colorbar=False)
    writer.add_figure(tag, fig)
    plt.close(fig)


def log_top_trials(run_dir, main_writer, top, tag):
    """Escribe en TensorBoard solo los trials de `top` (ya ordenados por F1)."""
    
    for rank, rec in enumerate(top, start=1):
        name = f"{tag}_top{rank}_trial{rec['number']:03d}"
        w = SummaryWriter(os.path.join(run_dir, name))  # un run por trial -> curvas superpuestas

        for split, key in (("train", "validation_0"), ("val", "validation_1")):
            for i, v in enumerate(rec["curves"][key]["logloss"]):
                w.add_scalar(f"{tag}/logloss_{split}", v, i)
        for k, v in rec["metrics"].items():
            w.add_scalar(f"{tag}/val_{k}", v, 0)
        w.add_text(f"{tag}/params", f"```\n{json.dumps(rec['params'], indent=2)}\n```")
        w.close()

        main_writer.add_hparams(rec["params"], {f"hparam/{tag}_f1": rec["metrics"]["f1"]},
                                run_name=name)


# ---------------------------------------------------------------- training
def train_binary_clf(X_train, y_train, X_val, y_val,cfg, tag = "N" ,n_trials=50,
                     device="cuda", top_k=3):
    
    """Devuelve (mejor_clf, mejor_f1, mejores_params, top_trials)."""
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1.0

    top = []                                   # top_k mejores trials (solo datos livianos)
    best = {"f1": -1.0, "clf": None, "params": None, "number": None}

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 0.9),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            colsample_bynode=trial.suggest_float("colsample_bynode", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
        )
        
        clf = xgb.XGBClassifier(
            **params, scale_pos_weight=spw, eval_metric="logloss",
            device=device, random_state=42, tree_method="hist",
        )
        
        # eval_set solo registra curvas; sin early stopping no cambia las predicciones
        clf.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
        
        m = _metrics(y_val, clf.predict(X_val))

        # mantener solo el top_k en memoria
        top.append({"number": trial.number, "params": params,
                    "metrics": m, "curves": clf.evals_result()})
        top.sort(key=lambda r: r["metrics"]["f1"], reverse=True)
        del top[top_k:]

        # mantener solo el clasificador del mejor trial
        if m["f1"] > best["f1"]:
            best.update(f1=m["f1"], clf=clf, params=params, number=trial.number)
        return m["f1"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1)

    # registro completo de todos los trials, por si lo quieres revisar luego
    study.trials_dataframe().to_csv(os.path.join(cfg.paths.SAVE_DIR_XGBOOST, f"{tag}_optuna_trials.csv"), index=False)

    print(f"  → Mejor trial: #{best['number']}")
    return best["clf"], best["f1"], best["params"], top


def log_final_report(writer, clf, X_val, y_val, tag, best_params,cfg):
    pred = clf.predict(X_val)
    proba = clf.predict_proba(X_val)[:, 1]
    m = _metrics(y_val, pred)

    for k, v in m.items():
        writer.add_scalar(f"{tag}/final/val_{k}", v, 0)
    writer.add_pr_curve(f"{tag}/final/pr_curve", y_val, proba)
    _log_confusion_matrix(writer, y_val, pred, f"{tag}/final/confusion_matrix")

    report = classification_report(y_val, pred)
    writer.add_text(f"{tag}/final/classification_report", f"```\n{report}\n```")
    writer.add_text(f"{tag}/final/best_params", f"```\n{json.dumps(best_params, indent=2)}\n```")

    with open(os.path.join(cfg.paths.SAVE_DIR_XGBOOST, f"metrics_{tag}.json"), "w") as f:
        json.dump({"val_metrics": m, "best_params": best_params}, f, indent=2)
    return report


# -------------------------------------------------------------------- run
def run(cfg,bge_overrides=None,seed=30, n_trials=200, top_k=3):

    SAVE_DIR = cfg.paths.SAVE_DIR
    SAVE_DIR_XGBOOST = cfg.paths.SAVE_DIR
    
    TB_ROOT = os.path.join(SAVE_DIR_XGBOOST, "tb_logs")

    #changes the Hparam and objetive of training
    overrides = deep_merge({}, bge_overrides)
    seed = cfg.train.SEED

    #ensure the existance of save dir for BGE_M3
    os.makedirs(SAVE_DIR, exist_ok=True)
    out = main_bge(overrides=overrides)

    #ensure the existance of save dir for XGBOOST
    SAVE_DIR_XGBOOST = cfg.paths.SAVE_DIR_XGBOOST 
    os.makedirs(SAVE_DIR_XGBOOST, exist_ok=True)

    run_dir = os.path.join(SAVE_DIR_XGBOOST, "tb_logs", f"xgb_seed{seed}_{datetime.now():%Y%m%d_%H%M%S}")

    model, device = out["model"], out["device"]
    load_best_weights(out["lit"], out["best_ckpt_path"], device)

    with torch.no_grad():
        X_train, y_t1_train, *_ = xgb_m3.extract_embeddings_and_labels(model, out["train_loader"], device)
        X_val, y_t1_val, *_ = xgb_m3.extract_embeddings_and_labels(model, out["val_loader"], device)

    run_dir = os.path.join(TB_ROOT, f"xgb_seed{seed}_{datetime.now():%Y%m%d_%H%M%S}")
    writer = SummaryWriter(run_dir)
    xgb_device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        
        xgb_t1, best_f1, best_params, top = train_binary_clf(
            X_train, y_t1_train, X_val, y_t1_val, cfg, tag = overrides["train"]["BEST_MODEL_METRIC"],
            n_trials=n_trials, device=xgb_device, top_k=top_k,
        )
        
        print(f"\n  → Mejor F1 ({overrides["train"]["BEST_MODEL_METRIC"]}): {best_f1:.4f}")
        print(f"  → Mejores parámetros: {best_params}")

        log_top_trials(run_dir, writer, top, tag=overrides["train"]["BEST_MODEL_METRIC"])
        print(f"\n── Classification Report {overrides["train"]["BEST_MODEL_METRIC"]} ──")
        print(log_final_report(writer, xgb_t1, X_val, y_t1_val, overrides["train"]["BEST_MODEL_METRIC"] , best_params,cfg))

        
    finally:
        writer.close()

    joblib.dump(xgb_t1, os.path.join(SAVE_DIR_XGBOOST, f"xgb_{overrides["train"]["BEST_MODEL_METRIC"]}_gloss_{overrides["flags"]["USE_GLOSS"]}.pkl"))
    return xgb_t1


if __name__ == "__main__":
    run()