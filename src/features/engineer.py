"""Causal feature engineering. Pure pandas/numpy — no PyTorch imports."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FeaturesConfig:
    roll_windows: list[int] = field(default_factory=lambda: [5, 10, 21, 63])
    momentum_windows: list[int] = field(default_factory=lambda: [5, 21])
    corr_windows: list[int] = field(default_factory=lambda: [21, 63])
    cross_assets: list[str] = field(default_factory=lambda: ["QQQ", "TLT", "GLD", "^VIX"])
    primary_asset: str = "SPY"


def build_features(
    prices: dict[str, pd.DataFrame],
    cfg: FeaturesConfig,
) -> pd.DataFrame:
    """Build causal features. Returns date-indexed DataFrame with NaNs at head."""
    spy = prices[cfg.primary_asset]
    close = spy["close"].copy()
    volume = spy["volume"].copy()

    log_ret = np.log(close / close.shift(1))

    parts: list[pd.Series | pd.DataFrame] = []

    # --- Returns / momentum ---
    parts.append(log_ret.rename("log_return_1d"))
    for w in cfg.momentum_windows:
        parts.append(log_ret.rolling(w, min_periods=w).sum().rename(f"log_return_{w}d"))

    # --- Rolling stats ---
    for w in cfg.roll_windows:
        parts.append(log_ret.rolling(w, min_periods=w).mean().rename(f"roll_mean_return_{w}"))
        realized_vol = log_ret.rolling(w, min_periods=w).std() * np.sqrt(252)
        parts.append(realized_vol.rename(f"roll_realized_vol_{w}"))
        roll_mean = close.rolling(w, min_periods=w).mean()
        roll_std = close.rolling(w, min_periods=w).std()
        parts.append(((close - roll_mean) / roll_std.replace(0, np.nan)).rename(f"roll_zscore_{w}"))

    # --- Drawdown ---
    for w in cfg.roll_windows:
        roll_max = close.rolling(w, min_periods=w).max()
        parts.append(((close - roll_max) / roll_max).rename(f"roll_drawdown_{w}"))

    # --- Moving average distance ---
    for w in cfg.roll_windows:
        sma = close.rolling(w, min_periods=w).mean()
        parts.append(((close - sma) / sma).rename(f"ma_dist_{w}"))

    # --- Volume ---
    vol_shifted = volume.shift(1)
    vol_shifted_safe = vol_shifted.where(vol_shifted != 0, np.nan)
    parts.append(np.log(volume / vol_shifted_safe).rename("vol_change_1d"))
    roll_vol_mean = volume.rolling(21, min_periods=21).mean()
    roll_vol_mean_safe = roll_vol_mean.where(roll_vol_mean != 0, np.nan)
    parts.append((volume / roll_vol_mean_safe).rename("roll_vol_ratio"))

    # --- Volatility ratio ---
    vol5 = log_ret.rolling(5, min_periods=5).std() * np.sqrt(252)
    vol63 = log_ret.rolling(63, min_periods=63).std() * np.sqrt(252)
    vol63_safe = vol63.where(vol63 != 0, np.nan)
    parts.append((vol5 / vol63_safe).rename("vol_ratio_short_long"))

    # --- Cross-asset correlations ---
    spy_ret = log_ret
    for asset in cfg.cross_assets:
        if asset not in prices:
            continue
        asset_close = prices[asset]["close"].copy()
        asset_ret = np.log(asset_close / asset_close.shift(1))
        asset_ret = asset_ret.reindex(spy_ret.index)
        for w in cfg.corr_windows:
            corr = spy_ret.rolling(w, min_periods=w).corr(asset_ret)
            safe_name = asset.replace("^", "")
            parts.append(corr.rename(f"roll_corr_{safe_name}_{w}"))

    # --- VIX-specific ---
    if "^VIX" in prices:
        vix_close = prices["^VIX"]["close"].reindex(spy.index).copy()
        parts.append(vix_close.rename("vix_level"))
        vix_ret = np.log(vix_close / vix_close.shift(1))
        parts.append(vix_ret.rename("vix_change_1d"))
        vix_mean = vix_close.rolling(21, min_periods=21).mean()
        vix_std = vix_close.rolling(21, min_periods=21).std()
        vix_std_safe = vix_std.where(vix_std != 0, np.nan)
        parts.append(((vix_close - vix_mean) / vix_std_safe).rename("vix_roll_zscore_21"))

    result = pd.concat(parts, axis=1)
    result.index.name = "date"
    return result
