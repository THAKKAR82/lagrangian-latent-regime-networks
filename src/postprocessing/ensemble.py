"""Weighted ensemble and validation-based weight grid search."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


class WeightedEnsemble:
    def __init__(self, w_xgb: float = 0.5, w_lag: float = 0.5) -> None:
        if abs(w_xgb + w_lag - 1.0) > 1e-5:
            raise ValueError(f"Weights must sum to 1.0, got {w_xgb + w_lag:.4f}")
        self.w_xgb = w_xgb
        self.w_lag = w_lag

    def predict_proba(self, xgb_probs: np.ndarray, lag_probs: np.ndarray) -> np.ndarray:
        return (self.w_xgb * xgb_probs + self.w_lag * lag_probs).astype(np.float32)


def grid_search_weights(
    xgb_val_probs: np.ndarray,
    lag_val_probs: np.ndarray,
    val_labels: np.ndarray,
    candidates: list[float] | None = None,
) -> tuple[float, float]:
    """Select w_xgb from candidates that maximises val macro F1. Returns (w_xgb, w_lag).

    Leakage-safe: pass only validation split probs and labels.
    """
    if candidates is None:
        candidates = [round(i / 10, 1) for i in range(1, 10)]  # 0.1 .. 0.9
    best_f1, best_w = -1.0, 0.5
    for w in candidates:
        probs = WeightedEnsemble(w_xgb=w, w_lag=1.0 - w).predict_proba(
            xgb_val_probs, lag_val_probs
        )
        f1 = f1_score(val_labels, probs.argmax(axis=1), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_w = f1, w
    return best_w, 1.0 - best_w
