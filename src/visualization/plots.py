"""Visualization functions for regime forecasting results."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REGIME_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c", 3: "#9b59b6"}
REGIME_NAMES = {0: "Bull/Calm", 1: "Bull/Stress", 2: "Bear/Calm", 3: "Bear/Stress"}


def plot_regime_timeline(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    for ax, labels, title in zip(axes, [y_true, y_pred], ["True Regime", "Predicted Regime"]):
        for regime_id, color in REGIME_COLORS.items():
            mask = labels == regime_id
            ax.fill_between(dates, 0, 1, where=mask, color=color, alpha=0.7,
                            label=REGIME_NAMES[regime_id])
        ax.set_ylabel(title)
        ax.set_yticks([])
    axes[0].legend(loc="upper left", fontsize=8, ncol=4)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = [REGIME_NAMES[i] for i in range(4)]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ece: float,
    save_path: Path | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for cls_id, ax in enumerate(axes):
        prob_cls = y_prob[:, cls_id]
        true_cls = (y_true == cls_id).astype(float)
        accs = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (prob_cls >= lo) & (prob_cls < hi)
            accs.append(true_cls[mask].mean() if mask.sum() > 0 else np.nan)
        ax.bar(bin_centers, np.nan_to_num(accs), width=0.1, alpha=0.7, label="Actual")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
        ax.set_title(f"{REGIME_NAMES[cls_id]}")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=7)

    fig.suptitle(f"Calibration (ECE={ece:.4f})")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_rolling_f1(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window: int = 63,
    save_path: Path | None = None,
) -> plt.Figure:
    from sklearn.metrics import f1_score
    f1s = []
    for i in range(len(dates)):
        start = max(0, i - window + 1)
        yt = y_true[start: i + 1]
        yp = y_pred[start: i + 1]
        if len(np.unique(yt)) < 2:
            f1s.append(np.nan)
        else:
            f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, f1s, color="#3498db", lw=1.5)
    ax.set_ylabel(f"Rolling Macro F1 ({window}d)")
    ax.set_xlabel("Date")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
    save_path: Path | None = None,
) -> plt.Figure:
    top_n = min(top_n, len(importances))
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in idx], importances[idx], color="#3498db")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_fold_summary(
    fold_ids: list[int],
    macro_f1s: list[float],
    save_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(fold_ids, macro_f1s, color="#2ecc71", alpha=0.8)
    mean_f1 = float(np.mean(macro_f1s))
    ax.axhline(mean_f1, color="red", linestyle="--", label=f"Mean={mean_f1:.3f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Macro F1")
    ax.set_title("Walk-Forward Fold Performance")
    ax.legend()
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
