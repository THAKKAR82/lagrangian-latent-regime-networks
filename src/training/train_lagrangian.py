"""Lagrangian Regime Network walk-forward training entrypoint.

Run with:
    python -m src.training.train_lagrangian model=lagrangian
    python -m src.training.train_lagrangian model=lagrangian model.latent_dim=16
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
from src.features.econophysics import build_econophysics_features
from src.features.engineer import FeaturesConfig, build_features
from src.labels.quantile_labeler import LabelConfig, QuantileLabeler
from src.models.lagrangian_regime_net import LagrangianConfig, LagrangianRegimeNet
from src.utils.dataset_builder import SplitConfig, build_folds
from src.utils.reproducibility import set_global_seed
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_fold_summary,
    plot_regime_timeline,
)

log = logging.getLogger(__name__)

MODEL_NAME = "lagrangian"


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    weight: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Focal loss: -(1 - p_t)^gamma * log(p_t). Reduces loss for easy samples."""
    log_p = nn.functional.log_softmax(logits, dim=-1)
    p = torch.exp(log_p)
    p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
    loss = -((1 - p_t) ** gamma) * log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
    if weight is not None:
        w = weight[targets]
        loss = loss * w
    return loss.mean()


def make_class_weights(y: np.ndarray) -> np.ndarray:
    """Inverse class frequency weights, normalised to sum to n_classes."""
    counts = np.bincount(y, minlength=4).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / counts
    return (weights / weights.sum() * 4).astype(np.float32)


def _get_labels(
    spy_prices: pd.DataFrame,
    label_cfg: LabelConfig,
    feature_index: pd.DatetimeIndex,
) -> pd.Series:
    labeler = QuantileLabeler(label_cfg)
    label_df = labeler.fit_transform(spy_prices)
    return label_df["label"].reindex(feature_index)


def _train_fold(
    model: nn.Module,
    fold,
    cfg: DictConfig,
    device: torch.device,
    lag_cfg: "LagrangianConfig | None" = None,
    loss_type: str = "ce",
    class_weights: "torch.Tensor | None" = None,
) -> nn.Module:
    """Train one fold: Adam + CrossEntropyLoss + early stopping + gradient clipping."""
    X_tr = torch.from_numpy(fold.train_X).float()
    y_tr = torch.from_numpy(fold.train_y).long()
    X_va = torch.from_numpy(fold.val_X).float().to(device)
    y_va = torch.from_numpy(fold.val_y).long().to(device)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=cfg.model.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights) if loss_type == "weighted_ce" else nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for epoch in range(cfg.model.max_epochs):
        model.train()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            if lag_cfg is not None and lag_cfg.use_transition_head:
                trans_labels = torch.zeros(len(y_batch), 1, dtype=torch.float32, device=device)
                if len(y_batch) > 1:
                    trans_labels[1:, 0] = (y_batch[1:] != y_batch[:-1]).float()
                regime_logits, trans_logits = model.forward_with_transition(X_batch)
                loss = criterion(regime_logits, y_batch) + 0.1 * nn.functional.binary_cross_entropy_with_logits(
                    trans_logits, trans_labels
                )
            else:
                if loss_type == "focal":
                    loss = focal_loss(model(X_batch), y_batch, gamma=2.0, weight=class_weights)
                else:
                    loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_va), y_va).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg.model.patience:
                log.debug(f"  Early stop at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    project_root = Path(hydra.utils.get_original_cwd())

    set_global_seed(cfg.seed)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device(cfg.model.device)

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
    if getattr(cfg.model, 'use_econophysics_features', False):
        eco_features = build_econophysics_features(
            prices,
            primary_asset=cfg.features.primary_asset,
            roll_windows=list(cfg.features.roll_windows),
        )
        eco_features = eco_features.reindex(features.index)
        features = pd.concat([features, eco_features], axis=1)
    n_features = features.shape[1]
    log.info(f"Features shape: {features.shape}")

    label_cfg = LabelConfig(
        horizon=cfg.labels.horizon,
        vol_window=cfg.labels.vol_window,
        return_quantile=cfg.labels.return_quantile,
        vol_quantile=cfg.labels.vol_quantile,
        smoothing=cfg.labels.smoothing,
        smoothing_min_periods=cfg.labels.smoothing_min_periods,
    )
    spy_prices = prices[cfg.labels.label_asset]
    labels = _get_labels(spy_prices, label_cfg, features.index)

    split_cfg = SplitConfig(
        train_start=cfg.splits.train_start,
        val_size=cfg.splits.val_size,
        test_size=cfg.splits.test_size,
        step_size=cfg.splits.step_size,
        min_train_size=cfg.splits.min_train_size,
    )

    lag_cfg = LagrangianConfig(
        input_dim=n_features,
        window_len=cfg.data.window_len,
        latent_dim=cfg.model.latent_dim,
        hidden_dim=cfg.model.hidden_dim,
        n_steps=cfg.model.n_steps,
        damping=cfg.model.damping,
        dt=cfg.model.dt,
        use_forcing=cfg.model.use_forcing,
        use_scalar_damping=getattr(cfg.model, 'use_scalar_damping', True),
        use_vector_damping=getattr(cfg.model, 'use_vector_damping', False),
        use_coord_transform=getattr(cfg.model, 'use_coord_transform', False),
        eps=cfg.model.eps,
        seed=cfg.seed,
        batch_size=cfg.model.batch_size,
        lr=cfg.model.lr,
        max_epochs=cfg.model.max_epochs,
        patience=cfg.model.patience,
        device=str(device),
        encoder_type=getattr(cfg.model, 'encoder_type', 'mlp'),
        encoder_dim=getattr(cfg.model, 'encoder_dim', 64),
        conv_channels=getattr(cfg.model, 'conv_channels', 64),
        conv_kernel_size=getattr(cfg.model, 'conv_kernel_size', 3),
        tcn_channels=getattr(cfg.model, 'tcn_channels', 64),
        tcn_kernel_size=getattr(cfg.model, 'tcn_kernel_size', 3),
        tcn_dilations=list(getattr(cfg.model, 'tcn_dilations', [1, 2, 4, 8])),
        use_skip_connection=getattr(cfg.model, 'use_skip_connection', False),
        use_transition_head=getattr(cfg.model, 'use_transition_head', False),
        use_multi_timescale=getattr(cfg.model, 'use_multi_timescale', False),
        coarse_dt=getattr(cfg.model, 'coarse_dt', 5.0),
    )

    experiment_name = getattr(cfg.model, "name", MODEL_NAME)

    output_dir = project_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = project_root / cfg.figures_dir / experiment_name

    artifact_dir = project_root / "predictions" / experiment_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    import datetime
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    loss_type = getattr(cfg.model, "loss_type", "ce")

    all_metrics: list[dict] = []
    all_fold_ids: list[int] = []

    fold_start = getattr(cfg, 'fold_start', None)
    fold_end = getattr(cfg, 'fold_end', None)

    for fold in build_folds(
        features,
        labels,
        split_cfg,
        window_len=cfg.data.window_len,
        flat=False,
    ):
        if fold_start is not None and fold.fold_id < fold_start:
            continue
        if fold_end is not None and fold.fold_id > fold_end:
            break

        log.info(
            f"Fold {fold.fold_id}: "
            f"train={len(fold.train_y)} val={len(fold.val_y)} test={len(fold.test_y)}"
        )

        torch.manual_seed(cfg.seed)
        model = LagrangianRegimeNet(lag_cfg)

        class_weights_t: torch.Tensor | None = None
        if loss_type in ("weighted_ce", "focal"):
            class_weights_np = make_class_weights(fold.train_y)
            class_weights_t = torch.tensor(class_weights_np, dtype=torch.float32).to(device)

        model = _train_fold(model, fold, cfg, device, lag_cfg=lag_cfg, loss_type=loss_type, class_weights=class_weights_t)

        y_pred = model.predict(fold.test_X)
        y_prob = model.predict_proba(fold.test_X)
        val_prob = model.predict_proba(fold.val_X)
        result = evaluate(fold.test_y, y_pred, y_prob)

        # Save val and test artifacts.
        # fold.val_dates / fold.test_dates are the raw 252-day periods; windowing
        # reduces predictions to (period - window_len + 1) rows, so we align to tail.
        n_val = len(val_prob)
        n_test = len(y_prob)
        PredictionArtifact(
            fold_id=fold.fold_id, split="val", model_name=experiment_name,
            dates=np.array([str(d.date()) for d in fold.val_dates[-n_val:]]),
            true_labels=fold.val_y, probs=val_prob,
        ).save(artifact_dir / f"fold_{fold.fold_id:02d}_val.parquet")
        PredictionArtifact(
            fold_id=fold.fold_id, split="test", model_name=experiment_name,
            dates=np.array([str(d.date()) for d in fold.test_dates[-n_test:]]),
            true_labels=fold.test_y, probs=y_prob,
        ).save(artifact_dir / f"fold_{fold.fold_id:02d}_test.parquet")
        n = n_test  # used below for plot date alignment

        metrics_dict = {
            "fold_id": fold.fold_id,
            "model": experiment_name,
            "fold_start": fold_start,
            "fold_end": fold_end,
            "run_id": run_id,
            "macro_f1": result.macro_f1,
            "balanced_accuracy": result.balanced_accuracy,
            "brier_score": result.brier_score,
            "ece": result.ece,
            "switch_frequency": result.switch_frequency,
            "mean_entropy": result.mean_entropy,
            "val_start": str(fold.val_dates.min().date()),
            "test_start": str(fold.test_dates.min().date()),
            "test_end": str(fold.test_dates.max().date()),
        }
        all_metrics.append(metrics_dict)
        all_fold_ids.append(fold.fold_id)

        fold_dir = output_dir / f"fold_{fold.fold_id:02d}"
        fold_dir.mkdir(exist_ok=True)
        (fold_dir / "metrics.json").write_text(json.dumps(metrics_dict, indent=2))

        plot_regime_timeline(
            fold.test_dates[-n:], fold.test_y, y_pred,
            figures_dir / f"fold_{fold.fold_id:02d}_timeline.png",
        )
        plot_confusion_matrix(
            result.confusion_matrix, f"Fold {fold.fold_id} ({experiment_name})",
            figures_dir / f"fold_{fold.fold_id:02d}_cm.png",
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
    csv_path = output_dir / f"walk_forward_summary_{experiment_name}.csv"
    summary_df.to_csv(csv_path, index=False)
    log.info(f"Saved summary to {csv_path}")

    numeric_cols = ["macro_f1", "balanced_accuracy", "brier_score", "ece"]
    summary_stats = summary_df[numeric_cols].agg(["mean", "std"])
    log.info(f"\nWalk-Forward Summary ({experiment_name}):\n{summary_stats.to_string()}")

    plot_fold_summary(
        all_fold_ids, summary_df["macro_f1"].tolist(),
        save_path=figures_dir / "fold_summary.png",
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
