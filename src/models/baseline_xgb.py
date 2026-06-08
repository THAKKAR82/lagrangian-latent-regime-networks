"""XGBoost regime classifier wrapper."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from xgboost import XGBClassifier


@dataclass
class XGBConfig:
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 50
    eval_metric: str = "mlogloss"
    seed: int = 42
    n_jobs: int = -1


class RegimeXGB:
    """Thin wrapper around XGBClassifier for 4-class regime forecasting."""

    def __init__(self, cfg: XGBConfig) -> None:
        self.cfg = cfg
        self._model = XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            early_stopping_rounds=cfg.early_stopping_rounds,
            eval_metric=cfg.eval_metric,
            random_state=cfg.seed,
            n_jobs=cfg.n_jobs,
            objective="multi:softprob",
            verbosity=0,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "RegimeXGB":
        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def feature_importances(self) -> np.ndarray:
        return self._model.feature_importances_
