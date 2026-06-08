"""QuantileLabeler: 2x2 regime taxonomy via adaptive quantile thresholds."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.labels.base import BaseLabeler

REGIME_NAMES = {
    0: "BullCalm",
    1: "BullStress",
    2: "BearCalm",
    3: "BearStress",
}


@dataclass
class LabelConfig:
    horizon: int = 5
    vol_window: int = 21
    return_quantile: float = 0.5
    vol_quantile: float = 0.5
    smoothing: bool = True
    smoothing_min_periods: int = 3
    label_asset: str = "SPY"


class QuantileLabeler(BaseLabeler):
    """Assigns 4 mutually exclusive regime labels via 2x2 quadrant split.

    Labels:
        0 = Bull/Calm   (positive fwd return, low vol)
        1 = Bull/Stress (positive fwd return, high vol)
        2 = Bear/Calm   (negative fwd return, low vol)
        3 = Bear/Stress (negative fwd return, high vol)

    Thresholds are fit on training data only and applied unchanged to val/test.
    """

    def __init__(self, cfg: LabelConfig) -> None:
        self.cfg = cfg
        self.return_thresh_: float | None = None
        self.vol_thresh_: float | None = None

    def _compute_raw_quantities(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        log_ret = np.log(close / close.shift(1))
        fwd_return = np.log(close.shift(-self.cfg.horizon) / close)
        realized_vol = log_ret.rolling(self.cfg.vol_window, min_periods=self.cfg.vol_window).std() * np.sqrt(252)
        return pd.DataFrame({"fwd_return_h": fwd_return, "roll_vol": realized_vol}, index=data.index)

    def fit(self, data: pd.DataFrame) -> "QuantileLabeler":
        raw = self._compute_raw_quantities(data)
        valid = raw.dropna()
        self.return_thresh_ = float(valid["fwd_return_h"].quantile(self.cfg.return_quantile))
        self.vol_thresh_ = float(valid["roll_vol"].quantile(self.cfg.vol_quantile))
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.return_thresh_ is None or self.vol_thresh_ is None:
            raise RuntimeError("QuantileLabeler must be fit() before transform().")
        raw = self._compute_raw_quantities(data)

        valid_mask = raw["fwd_return_h"].notna() & raw["roll_vol"].notna()

        label_raw_int = pd.Series(np.nan, index=data.index, dtype=float)
        for i in data.index[valid_mask]:
            is_bull = raw.loc[i, "fwd_return_h"] >= self.return_thresh_
            is_stress = raw.loc[i, "roll_vol"] >= self.vol_thresh_
            if is_bull and not is_stress:
                label_raw_int[i] = 0  # BullCalm
            elif is_bull and is_stress:
                label_raw_int[i] = 1  # BullStress
            elif not is_bull and not is_stress:
                label_raw_int[i] = 2  # BearCalm
            else:
                label_raw_int[i] = 3  # BearStress

        label = label_raw_int.copy()
        if self.cfg.smoothing:
            label = self._apply_smoothing(label)

        result = raw.copy()
        result["return_thresh"] = np.where(valid_mask, self.return_thresh_, np.nan)
        result["vol_thresh"] = np.where(valid_mask, self.vol_thresh_, np.nan)
        result["label_raw"] = label_raw_int
        result["label"] = label
        result["regime_name"] = label.map(REGIME_NAMES)
        return result

    def _apply_smoothing(self, label: pd.Series) -> pd.Series:
        """Persistence smoothing: flip accepted only after min_periods consecutive days."""
        smoothed = label.copy()
        valid_idx = label.dropna().index
        if len(valid_idx) == 0:
            return smoothed

        values = label[valid_idx].values.astype(float)
        min_p = self.cfg.smoothing_min_periods
        out = values.copy()
        current = values[0]

        for t in range(1, len(values)):
            if values[t] == current:
                pass  # streak continues
            else:
                candidate = values[t]
                end = min(t + min_p, len(values))
                window = values[t:end]
                if len(window) >= min_p and np.all(window == candidate):
                    current = candidate
            out[t] = current

        smoothed[valid_idx] = out
        return smoothed
