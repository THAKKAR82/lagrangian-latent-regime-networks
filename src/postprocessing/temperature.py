"""Temperature scaling calibration for multi-class probabilities.

Fits scalar T on validation NLL. Since we have probs not logits, uses
log(p + eps) / T → re-softmax as the scaling operation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


class TemperatureScaler:
    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(self, probs: np.ndarray, true_labels: np.ndarray) -> "TemperatureScaler":
        """Fit T to minimize NLL on (probs, true_labels). Leakage-safe: call on val only."""
        def nll(T: float) -> float:
            scaled = self.transform(probs, temperature=max(T, 1e-3))
            log_p = np.log(np.clip(scaled, 1e-12, 1.0))
            return -float(log_p[np.arange(len(true_labels)), true_labels].mean())

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(result.x)
        return self

    def transform(self, probs: np.ndarray, temperature: float | None = None) -> np.ndarray:
        T = temperature if temperature is not None else self.temperature
        log_p = np.log(np.clip(probs, 1e-12, 1.0)) / T
        log_p = log_p - np.log(np.exp(log_p).sum(axis=1, keepdims=True))  # log-softmax
        return np.exp(log_p).astype(np.float32)
