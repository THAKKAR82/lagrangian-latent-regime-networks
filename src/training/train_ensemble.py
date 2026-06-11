"""XGBoost + conv1d Lagrangian ensemble walk-forward training entrypoint.

Trains both models per fold, averages softmax probabilities at test time.

Run with:
    python -m src.training.train_ensemble
    python -m src.training.train_ensemble +fold_start=20 +fold_end=40
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.data.download import fetch_all
from src.data.manager import DataManager
from src.evaluation.metrics import evaluate
from src.evaluation.predictions import PredictionArtifact
from src.features.engineer import FeaturesConfig, build_features
from src.labels.quantile_labeler import LabelConfig, QuantileLabeler
from src.models.baseline_xgb import RegimeXGB, XGBConfig
from src.models.lagrangian_regime_net import LagrangianConfig, LagrangianRegimeNet
from src.postprocessing.ensemble import WeightedEnsemble, grid_search_weights
from src.utils.dataset_builder import SplitConfig, build_folds
from src.utils.reproducibility import set_global_seed
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_fold_summary,
    plot_regime_timeline,
)

log = logging.getLogger(__name__)

MODEL_NAME = "ensemble"

_LAG_CFG = dict(
    latent_dim=16, hidden_dim=64, encoder_type="conv1d", encoder_dim=64,
    conv_channels=64, conv_kernel_size=3, n_steps=8, damping=0.1, dt=1.0,
    use_forcing=False, use_vector_damping=True, use_coord_transform=True,
    use_skip_connection=False, eps=1e-4, batch_size=64, lr=5e-4,
    max_epochs=150, patience=30, device="cpu",
)

_XGB_CFG = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50,
    eval_metric="mlogloss", n_jobs=-1,
)


def _train_lag(fold_seq, seed: int, device: torch.device) -> LagrangianRegimeNet:
    n_feat = fold_seq.train_X.shape[2]
    cfg = LagrangianConfig(input_dim=n_feat, window_len=40, seed=seed, **_LAG_CFG)
    model = LagrangianRegimeNet(cfg).to(device)

    Xtr = torch.from_numpy(fold_seq.train_X).float()
    ytr = torch.from_numpy(fold_seq.train_y).long()
    Xva = torch.from_numpy(fold_seq.val_X).float().to(device)
    yva = torch.from_numpy(fold_seq.val_y).long().to(device)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    crit = nn.CrossEntropyLoss()
    best_vl = float("inf")
    best_st = {k: v.clone() for k, v in model.state_dict().items()}
    patience_ctr = 0

    for _ in range(cfg.max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xva), yva).item()
        if vl < best_vl:
            best_vl = vl
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                break

    model.load_state_dict(best_st)
    model.eval()
    return model


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    project_root = Path(hydra.utils.get_original_cwd())

    set_global_seed(cfg.seed)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device("cpu")

    dm = DataManager(
        raw_dir=project_root / cfg.data.raw_dir,
        processed_dir=project_root / cfg.data.processed_dir,
        tickers=list(cfg.data.tickers),
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
    )
    prices = fetch_all(dm)

    feat_cfg = FeaturesConfig(
        roll_windows=list(cfg.features.roll_windows),
        momentum_windows=list(cfg.features.momentum_windows),
        corr_windows=list(cfg.features.corr_windows),
        cross_assets=list(cfg.features.cross_assets),
        primary_asset=cfg.features.primary_asset,
    )
    features = build_features(prices, feat_cfg)
    log.info(f"Features shape: {features.shape}")

    label_cfg = LabelConfig(
        horizon=cfg.labels.horizon,
        vol_window=cfg.labels.vol_window,
        return_quantile=cfg.labels.return_quantile,
        vol_quantile=cfg.labels.vol_quantile,
        smoothing=cfg.labels.smoothing,
        smoothing_min_periods=cfg.labels.smoothing_min_periods,
    )
    labels = QuantileLabeler(label_cfg).fit_transform(
        prices[cfg.labels.label_asset]
    )["label"].reindex(features.index)

    split_cfg = SplitConfig(
        train_start=cfg.splits.train_start,
        val_size=cfg.splits.val_size,
        test_size=cfg.splits.test_size,
        step_size=cfg.splits.step_size,
        min_train_size=cfg.splits.min_train_size,
    )

    folds_flat = list(build_folds(features, labels, split_cfg, window_len=1, flat=True))
    folds_seq  = list(build_folds(features, labels, split_cfg, window_len=40, flat=False))

    output_dir = Path(".")
    figures_dir = project_root / cfg.figures_dir / MODEL_NAME

    artifact_dir = project_root / "predictions" / MODEL_NAME
    artifact_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []
    all_fold_ids: list[int] = []

    fold_start = getattr(cfg, "fold_start", None)
    fold_end   = getattr(cfg, "fold_end", None)

    xgb_cfg = XGBConfig(seed=cfg.seed, **_XGB_CFG)

    for fold_f, fold_s in zip(folds_flat, folds_seq):
        if fold_start is not None and fold_f.fold_id < fold_start:
            continue
        if fold_end is not None and fold_f.fold_id > fold_end:
            break

        log.info(
            f"Fold {fold_f.fold_id}: "
            f"train={len(fold_f.train_y)} val={len(fold_f.val_y)} test={len(fold_f.test_y)}"
        )

        # --- XGBoost ---
        torch.manual_seed(cfg.seed)
        xgb = RegimeXGB(xgb_cfg)
        xgb.fit(fold_f.train_X, fold_f.train_y, fold_f.val_X, fold_f.val_y)
        xgb_prob = xgb.predict_proba(fold_f.test_X)

        # --- Lagrangian conv1d ---
        lag = _train_lag(fold_s, seed=cfg.seed, device=device)
        lag_prob = lag.predict_proba(fold_s.test_X)

        # Align: conv1d test set is 39 shorter (window_len-1) than flat test set
        n = len(lag_prob)
        xgb_prob_aligned = xgb_prob[-n:]
        y_true = fold_f.test_y[-n:]

        # Weighted ensemble with optional grid search
        use_grid_search = getattr(cfg, "ensemble_grid_search", False)
        if use_grid_search:
            xgb_val_prob_ens = xgb.predict_proba(fold_f.val_X)
            lag_val_prob_ens = lag.predict_proba(fold_s.val_X)
            n_val = len(lag_val_prob_ens)
            w_xgb, w_lag = grid_search_weights(
                xgb_val_prob_ens[-n_val:], lag_val_prob_ens,
                fold_f.val_y[-n_val:],
            )
            log.info(f"  Grid search: w_xgb={w_xgb:.1f}, w_lag={w_lag:.1f}")
        else:
            w_xgb, w_lag = 0.5, 0.5

        ens_prob = WeightedEnsemble(w_xgb, w_lag).predict_proba(xgb_prob_aligned, lag_prob)
        y_pred = ens_prob.argmax(axis=1)

        result = evaluate(y_true, y_pred, ens_prob)

        xgb_val_prob = xgb.predict_proba(fold_f.val_X)
        lag_val_prob = lag.predict_proba(fold_s.val_X)
        n_val = len(lag_val_prob)
        xgb_val_aligned = xgb_val_prob[-n_val:]
        ens_val_prob = (xgb_val_aligned + lag_val_prob) / 2.0
        val_dates_aligned = fold_f.val_dates[-n_val:]

        PredictionArtifact(
            fold_id=fold_f.fold_id, split="val", model_name=MODEL_NAME,
            dates=np.array([str(d.date()) for d in val_dates_aligned]),
            true_labels=fold_f.val_y[-n_val:], probs=ens_val_prob,
        ).save(artifact_dir / f"fold_{fold_f.fold_id:02d}_val.parquet")
        PredictionArtifact(
            fold_id=fold_f.fold_id, split="test", model_name=MODEL_NAME,
            dates=np.array([str(d.date()) for d in fold_f.test_dates[-n:]]),
            true_labels=y_true, probs=ens_prob,
        ).save(artifact_dir / f"fold_{fold_f.fold_id:02d}_test.parquet")

        metrics_dict = {
            "fold_id": fold_f.fold_id,
            "model": MODEL_NAME,
            "macro_f1": result.macro_f1,
            "balanced_accuracy": result.balanced_accuracy,
            "brier_score": result.brier_score,
            "ece": result.ece,
            "switch_frequency": result.switch_frequency,
            "mean_entropy": result.mean_entropy,
            "val_start": str(fold_f.val_dates.min().date()),
            "test_start": str(fold_f.test_dates.min().date()),
            "test_end": str(fold_f.test_dates.max().date()),
        }
        all_metrics.append(metrics_dict)
        all_fold_ids.append(fold_f.fold_id)

        fold_dir = output_dir / f"fold_{fold_f.fold_id:02d}"
        fold_dir.mkdir(exist_ok=True)
        (fold_dir / "metrics.json").write_text(json.dumps(metrics_dict, indent=2))

        test_dates = fold_f.test_dates[-n:]
        plot_regime_timeline(
            test_dates, y_true, y_pred,
            figures_dir / f"fold_{fold_f.fold_id:02d}_timeline.png",
        )
        plot_confusion_matrix(
            result.confusion_matrix, f"Fold {fold_f.fold_id} (ensemble)",
            figures_dir / f"fold_{fold_f.fold_id:02d}_cm.png",
        )

        log.info(
            f"  Macro F1={result.macro_f1:.4f}  "
            f"Brier={result.brier_score:.4f}  "
            f"ECE={result.ece:.4f}"
        )

    if not all_metrics:
        log.warning("No folds produced — check split config and data date range.")
        return

    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(output_dir / "walk_forward_summary.csv", index=False)

    numeric_cols = ["macro_f1", "balanced_accuracy", "brier_score", "ece"]
    summary_stats = summary_df[numeric_cols].agg(["mean", "std"])
    log.info(f"\nWalk-Forward Summary ({MODEL_NAME}):\n{summary_stats.to_string()}")

    plot_fold_summary(
        all_fold_ids, summary_df["macro_f1"].tolist(),
        save_path=figures_dir / "fold_summary.png",
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
