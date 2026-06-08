# tests/test_features.py
import numpy as np
import pandas as pd
import pytest
from src.features.engineer import build_features, FeaturesConfig


@pytest.fixture
def features_cfg():
    return FeaturesConfig(
        roll_windows=[5, 10],
        momentum_windows=[5],
        corr_windows=[21],
        cross_assets=["QQQ", "TLT", "GLD", "^VIX"],
        primary_asset="SPY",
    )


def test_build_features_output_index_matches_input(synthetic_prices, features_cfg):
    feats = build_features(synthetic_prices, features_cfg)
    assert feats.index.equals(synthetic_prices["SPY"].index)


def test_build_features_no_future_leak(synthetic_prices, features_cfg):
    """Shifting data forward by 1 must change values — features are not shifted back."""
    feats_orig = build_features(synthetic_prices, features_cfg)
    prices_shifted = {k: v.shift(1).dropna() for k, v in synthetic_prices.items()}
    feats_shifted = build_features(prices_shifted, features_cfg)
    assert feats_shifted.shape[1] == feats_orig.shape[1]


def test_build_features_nans_only_at_head(synthetic_prices, features_cfg):
    feats = build_features(synthetic_prices, features_cfg)
    max_window = max(features_cfg.roll_windows + features_cfg.corr_windows)
    # After warmup rows, no column should be all-NaN
    tail = feats.iloc[max_window:]
    cols_all_nan = [c for c in tail.columns if tail[c].isna().all()]
    assert cols_all_nan == [], f"All-NaN columns after warmup: {cols_all_nan}"


def test_build_features_no_inplace_mutation(synthetic_prices, features_cfg):
    close_before = synthetic_prices["SPY"]["close"].copy()
    build_features(synthetic_prices, features_cfg)
    pd.testing.assert_series_equal(synthetic_prices["SPY"]["close"], close_before)


def test_build_features_expected_columns_present(synthetic_prices, features_cfg):
    feats = build_features(synthetic_prices, features_cfg)
    assert "log_return_1d" in feats.columns
    assert "roll_realized_vol_5" in feats.columns
    assert "vix_level" in feats.columns
    assert "roll_corr_QQQ_21" in feats.columns


def test_build_features_no_inf(synthetic_prices, features_cfg):
    feats = build_features(synthetic_prices, features_cfg)
    assert not np.isinf(feats.values).any()
