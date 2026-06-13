"""Per-class threshold tuning on validation probabilities.

Leakage-safe: call fit() on val only. At test time, use predict().
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


class ThresholdTuner:
    def __init__(self, n_thresholds: int = 17) -> None:
        self.thresholds = np.full(4, 0.25, dtype=np.float32)
        self.n_thresholds = n_thresholds

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray) -> "ThresholdTuner":
        """Find per-class threshold in [0.1, 0.9] maximising per-class F1 on val."""
        candidates = np.linspace(0.1, 0.9, self.n_thresholds)
        for c in range(4):
            best_f1, best_t = -1.0, 0.25
            y_true_c = (val_labels == c).astype(int)
            for t in candidates:
                y_pred_c = (val_probs[:, c] >= t).astype(int)
                if y_pred_c.sum() == 0:
                    continue
                f1_c = float(f1_score(y_true_c, y_pred_c, zero_division=0))
                if f1_c > best_f1:
                    best_f1, best_t = f1_c, t
            self.thresholds[c] = best_t
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Assign class by threshold+argmax; fallback to argmax if no class clears threshold."""
        exceeds = probs >= self.thresholds[np.newaxis, :]  # (N, 4)
        masked = np.where(exceeds, probs, -1.0)
        pred = masked.argmax(axis=1)
        no_exceed = ~exceeds.any(axis=1)
        pred[no_exceed] = probs[no_exceed].argmax(axis=1)
        return pred
