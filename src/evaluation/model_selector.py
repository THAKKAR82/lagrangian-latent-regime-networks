"""Leakage-safe per-fold model selection logic.

For fold t, the selector may only use validation metrics from folds 0..t-1.
Call record_val() AFTER evaluating fold t's test set to ensure causal ordering.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from src.evaluation.metrics import _ece, _LABELS

METHODS = (
    "previous_best_macro_f1",
    "rolling_best_macro_f1",
    "previous_best_calibrated_score",
    "rolling_best_calibrated_score",
    "fallback_static",
)


def compute_val_metrics(true_labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Compute macro_f1 and ece from validation predictions."""
    y_pred = probs.argmax(axis=1)
    macro_f1 = float(
        f1_score(true_labels, y_pred, average="macro", zero_division=0, labels=_LABELS)
    )
    ece = _ece(true_labels, probs)
    return {"macro_f1": macro_f1, "ece": ece}


class ModelSelector:
    """Leakage-safe per-fold model selector.

    Causal usage per fold t:
        selected, scores = selector.select()          # uses history from folds < t only
        # ... load and evaluate test artifact for fold t ...
        selector.record_val("model_a", metrics_a)    # record fold t val metrics
        selector.record_val("model_b", metrics_b)    # (for use in fold t+1 selection)

    Parameters
    ----------
    models:
        Candidate model names (must match prediction artifact directories).
    method:
        One of METHODS.
    k:
        Window size for rolling methods.
    alpha:
        ECE penalty weight for calibrated methods: score = macro_f1 - alpha * ece.
    fallback:
        Model to use when no history is available. Must be in models.
    """

    def __init__(
        self,
        models: list[str],
        method: str,
        k: int = 5,
        alpha: float = 0.5,
        fallback: str | None = None,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}. Valid: {METHODS}")
        self.models = list(models)
        self.method = method
        self.k = k
        self.alpha = alpha
        self.fallback = fallback if fallback is not None else (models[0] if models else "ensemble")
        self._history: dict[str, list[dict[str, float]]] = {m: [] for m in models}

    def record_val(self, model: str, metrics: dict[str, float]) -> None:
        """Record one fold's validation metrics for a model.

        Call AFTER evaluating that fold's test set to maintain causal ordering.
        Silently ignores unknown model names.
        """
        if model in self._history:
            self._history[model].append(metrics)

    def select(self) -> tuple[str, dict[str, float]]:
        """Select a model using only history recorded so far.

        Returns
        -------
        selected : str
            Name of the selected model.
        scores : dict[str, float]
            Selector score per model (NaN if no history for that model).
        """
        if self.method == "fallback_static":
            return self.fallback, {m: float("nan") for m in self.models}

        scores: dict[str, float] = {}
        for model in self.models:
            hist = self._history[model]
            if not hist:
                continue
            window = hist[-self.k :] if self.method.startswith("rolling_") else hist
            if "calibrated" in self.method:
                scores[model] = float(
                    np.mean([h["macro_f1"] - self.alpha * h["ece"] for h in window])
                )
            else:
                scores[model] = float(np.mean([h["macro_f1"] for h in window]))

        all_scores = {m: scores.get(m, float("nan")) for m in self.models}
        if not scores:
            return self.fallback, all_scores
        return max(scores, key=lambda m: scores[m]), all_scores

    def history_lengths(self) -> dict[str, int]:
        """Return number of recorded val folds per model."""
        return {m: len(h) for m, h in self._history.items()}
