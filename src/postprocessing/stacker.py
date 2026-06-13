"""Logistic regression stacker over XGBoost + Lagrangian probabilities.

Leakage-safe: trained on val predictions from folds 0..F-1 only.
Call update() after each fold, fit() before predicting on fold F.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticStacker:
    def __init__(self, min_folds: int = 5, C: float = 1.0) -> None:
        self.min_folds = min_folds
        self.C = C
        self._X_history: list[np.ndarray] = []
        self._y_history: list[np.ndarray] = []
        self._lr: LogisticRegression | None = None

    def update(
        self, xgb_val_probs: np.ndarray, lag_val_probs: np.ndarray, val_labels: np.ndarray
    ) -> None:
        """Accumulate val predictions from a completed fold (call after each fold)."""
        feats = np.concatenate([xgb_val_probs, lag_val_probs], axis=1)
        self._X_history.append(feats)
        self._y_history.append(val_labels)

    def fit(self) -> bool:
        """Fit on accumulated history. Returns True if enough folds available."""
        if len(self._X_history) < self.min_folds:
            return False
        X = np.vstack(self._X_history)
        y = np.concatenate(self._y_history)
        self._lr = LogisticRegression(
            C=self.C, max_iter=1000, solver="lbfgs"
        )
        self._lr.fit(X, y)
        return True

    def predict_proba(self, xgb_probs: np.ndarray, lag_probs: np.ndarray) -> np.ndarray:
        """Predict using stacker if fitted, else return 50/50 average."""
        if self._lr is None:
            return ((xgb_probs + lag_probs) / 2.0).astype(np.float32)
        feats = np.concatenate([xgb_probs, lag_probs], axis=1)
        return self._lr.predict_proba(feats).astype(np.float32)
