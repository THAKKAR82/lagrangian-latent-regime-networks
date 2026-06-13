"""Fold-adaptive ensemble weighting strategies."""
from __future__ import annotations


class FoldAdaptiveEnsemble:
    """Compute per-fold (w_xgb, w_lag) weights without using current fold's test set.

    Modes:
      equal             — fixed 0.5/0.5
      training_size     — w_lag scales with train_size up to max_lag_weight
      previous_folds_val — proportional to mean val F1 over all prior folds
    """

    def __init__(
        self,
        mode: str = "equal",
        size_threshold: int = 2000,
        max_lag_weight: float = 0.7,
    ) -> None:
        valid = {"equal", "training_size", "previous_folds_val"}
        if mode not in valid:
            raise ValueError(f"mode must be one of {valid}, got '{mode}'")
        self.mode = mode
        self.size_threshold = size_threshold
        self.max_lag_weight = max_lag_weight
        self._xgb_f1_history: list[float] = []
        self._lag_f1_history: list[float] = []

    def register_fold_val_f1(self, xgb_f1: float, lag_f1: float) -> None:
        """Record previous fold's val macro F1 for both models."""
        self._xgb_f1_history.append(xgb_f1)
        self._lag_f1_history.append(lag_f1)

    def get_weights(self, fold_id: int, train_size: int) -> tuple[float, float]:
        """Return (w_xgb, w_lag). Never uses current fold's test data."""
        if self.mode == "equal":
            return 0.5, 0.5

        if self.mode == "training_size":
            # XGBoost weight scales with training size (better with larger datasets)
            w_xgb = min(1.0, train_size / self.size_threshold) * self.max_lag_weight
            w_xgb = max(0.1, min(0.9, w_xgb))
            return round(w_xgb, 4), round(1.0 - w_xgb, 4)

        # previous_folds_val
        if not self._xgb_f1_history:
            return 0.5, 0.5
        mean_xgb = sum(self._xgb_f1_history) / len(self._xgb_f1_history)
        mean_lag = sum(self._lag_f1_history) / len(self._lag_f1_history)
        total = mean_xgb + mean_lag
        if total < 1e-8:
            return 0.5, 0.5
        w_xgb = round(mean_xgb / total, 4)
        return w_xgb, round(1.0 - w_xgb, 4)
