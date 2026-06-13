"""LR-stacker walk-forward evaluation using saved prediction artifacts.

Requires prediction artifacts already saved at:
  predictions/xgb/fold_<id>_val.parquet
  predictions/xgb/fold_<id>_test.parquet
  predictions/lagrangian/fold_<id>_val.parquet
  predictions/lagrangian/fold_<id>_test.parquet

Run with:
    python -m src.training.train_stacked
    python -m src.training.train_stacked +fold_start=20 +fold_end=40
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.evaluation.metrics import evaluate
from src.evaluation.predictions import PredictionArtifact
from src.postprocessing.stacker import LogisticStacker
from src.utils.reproducibility import set_global_seed

log = logging.getLogger(__name__)
MODEL_NAME = "stacked"


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    project_root = Path(hydra.utils.get_original_cwd())
    set_global_seed(cfg.seed)

    xgb_dir = project_root / "predictions" / "xgb"
    lag_dir = project_root / "predictions" / "lagrangian"
    out_dir = Path(".")

    fold_start = getattr(cfg, "fold_start", None)
    fold_end = getattr(cfg, "fold_end", None)
    min_folds = int(getattr(cfg, "stacker_min_folds", 5))

    # Collect available fold IDs from saved XGBoost test artifacts
    fold_ids = sorted(
        int(p.stem.split("_")[1])
        for p in xgb_dir.glob("fold_*_test.parquet")
    )
    if not fold_ids:
        log.error(f"No XGBoost test artifacts found in {xgb_dir}. Run train_baseline.py first.")
        return

    stacker = LogisticStacker(min_folds=min_folds)
    all_metrics: list[dict] = []

    for fold_id in fold_ids:
        if fold_start is not None and fold_id < fold_start:
            continue
        if fold_end is not None and fold_id > fold_end:
            break

        # Fit stacker on history from folds < current fold_id
        stacker.fit()

        xgb_test_path = xgb_dir / f"fold_{fold_id:02d}_test.parquet"
        lag_test_path = lag_dir / f"fold_{fold_id:02d}_test.parquet"
        if not lag_test_path.exists():
            log.warning(f"Missing lagrangian test artifact for fold {fold_id}, skipping")
            continue

        xgb_test = PredictionArtifact.load(xgb_test_path)
        lag_test = PredictionArtifact.load(lag_test_path)

        # Align test sets (lagrangian may be shorter due to window_len)
        n = min(len(xgb_test.probs), len(lag_test.probs))
        xgb_probs = xgb_test.probs[-n:]
        lag_probs = lag_test.probs[-n:]
        y_true = xgb_test.true_labels[-n:]

        stacked_probs = stacker.predict_proba(xgb_probs, lag_probs)
        y_pred = stacked_probs.argmax(axis=1)
        result = evaluate(y_true, y_pred, stacked_probs)

        metrics_dict = {
            "fold_id": fold_id,
            "model": MODEL_NAME,
            "macro_f1": result.macro_f1,
            "balanced_accuracy": result.balanced_accuracy,
            "brier_score": result.brier_score,
            "ece": result.ece,
            "stacker_fitted": stacker._lr is not None,
        }
        all_metrics.append(metrics_dict)
        log.info(
            f"Fold {fold_id}: F1={result.macro_f1:.4f}  "
            f"Brier={result.brier_score:.4f}  ECE={result.ece:.4f}  "
            f"fitted={stacker._lr is not None}"
        )

        # Accumulate val predictions for next fold's stacker
        xgb_val_path = xgb_dir / f"fold_{fold_id:02d}_val.parquet"
        lag_val_path = lag_dir / f"fold_{fold_id:02d}_val.parquet"
        if xgb_val_path.exists() and lag_val_path.exists():
            xgb_val = PredictionArtifact.load(xgb_val_path)
            lag_val = PredictionArtifact.load(lag_val_path)
            nv = min(len(xgb_val.probs), len(lag_val.probs))
            stacker.update(xgb_val.probs[-nv:], lag_val.probs[-nv:], xgb_val.true_labels[-nv:])

    if not all_metrics:
        log.warning("No folds processed.")
        return

    df = pd.DataFrame(all_metrics)
    df.to_csv(out_dir / "walk_forward_summary.csv", index=False)
    stats = df[["macro_f1", "brier_score", "ece"]].agg(["mean", "std"])
    log.info(f"\nStacked summary:\n{stats.to_string()}")


if __name__ == "__main__":
    main()
