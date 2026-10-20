from pathlib import Path
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import re

import yaml

def get_version_number(folder_name: str) -> int | None:
    """Extrae el número entero de carpetas tipo 'version_X'."""
    match = re.search(r"version_(\d+)", folder_name)
    return int(match.group(1)) if match else None



def parse_filtered_lightning_runs(
    log_dir: str | Path, min_version: int = 71
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_path = Path(log_dir)

    metrics_records = []
    hparams_records = []

    # Iterar sobre las carpetas que coincidan con el patrón version_*
    for version_dir in base_path.glob("**/version_*"):
        if not version_dir.is_dir():
            continue

        ver_num = get_version_number(version_dir.name)
        # Filtro estricto: solo versiones por encima del límite
        if ver_num is None or ver_num < min_version:
            continue

        run_id = f"version_{ver_num}"

        # 1. Leer hparams.yaml si existe
        hparams_file = version_dir / "hparams.yaml"
        if hparams_file.exists():
            with open(hparams_file, "r") as f:
                hparams = yaml.safe_load(f) or {}
            flat_hparams = pd.json_normalize(hparams).to_dict(orient="records")
            hparam_entry = flat_hparams[0] if flat_hparams else {}
            hparams_records.append(
                {"run": run_id, "version": ver_num, **hparam_entry}
            )

        # 2. Leer eventos de TensorBoard
        event_files = list(version_dir.glob("events.out.tfevents.*"))
        for event_file in event_files:
            ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            ea.Reload()

            for tag in ea.Tags().get("scalars", []):
                for e in ea.Scalars(tag):
                    metrics_records.append(
                        {
                            "run": run_id,
                            "version": ver_num,
                            "tag": tag,
                            "step": e.step,
                            "value": e.value,
                        }
                    )

    df_metrics = pd.DataFrame(metrics_records)
    df_hparams = pd.DataFrame(hparams_records)

    return df_metrics, df_hparams


def parse_filtered_lightning_runs_folders(
    log_dir: str | Path,
    min_version: int = 71,
    target_folders: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_path = Path(log_dir)

    metrics_records = []
    hparams_records = []

    # Determinar qué subcarpetas de experimentos se van a recorrer
    if target_folders is not None:
        experiment_dirs = [
            base_path / folder
            for folder in target_folders
            if (base_path / folder).is_dir()
        ]
    else:
        experiment_dirs = [p for p in base_path.iterdir() if p.is_dir()]

    for exp_dir in experiment_dirs:
        experiment_name = exp_dir.name

        # Buscar versiones dentro de este experimento
        for version_dir in exp_dir.glob("**/version_*"):
            if not version_dir.is_dir():
                continue

            ver_num = get_version_number(version_dir.name)
            if ver_num is None or ver_num < min_version:
                continue

            run_id = f"{experiment_name}_version_{ver_num}"

            # 1. Leer hparams.yaml
            hparams_file = version_dir / "hparams.yaml"
            if hparams_file.exists():
                with open(hparams_file, "r") as f:
                    hparams = yaml.safe_load(f) or {}
                flat_hparams = pd.json_normalize(hparams).to_dict(
                    orient="records"
                )
                hparam_entry = flat_hparams[0] if flat_hparams else {}
                hparams_records.append(
                    {
                        "experiment": experiment_name,
                        "run": run_id,
                        "version": ver_num,
                        **hparam_entry,
                    }
                )

            # 2. Leer eventos de TensorBoard
            event_files = list(version_dir.glob("events.out.tfevents.*"))
            for event_file in event_files:
                ea = EventAccumulator(
                    str(event_file), size_guidance={"scalars": 0}
                )
                ea.Reload()

                for tag in ea.Tags().get("scalars", []):
                    for e in ea.Scalars(tag):
                        metrics_records.append(
                            {
                                "experiment": experiment_name,
                                "run": run_id,
                                "version": ver_num,
                                "tag": tag,
                                "step": e.step,
                                "value": e.value,
                            }
                        )

    df_metrics = pd.DataFrame(metrics_records)
    df_hparams = pd.DataFrame(hparams_records)

    return df_metrics, df_hparams