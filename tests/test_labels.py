# tests/test_labels.py
import numpy as np
import pandas as pd
import pytest
from src.labels.base import BaseLabeler
from src.labels.quantile_labeler import QuantileLabeler, LabelConfig

REGIME_NAMES = {0: "BullCalm", 1: "BullStress", 2: "BearCalm", 3: "BearStress"}


@pytest.fixture
def label_cfg():
    return LabelConfig(
        horizon=5,
        vol_window=10,
        return_quantile=0.5,
        vol_quantile=0.5,
        smoothing=False,
    )


@pytest.fixture
def label_cfg_smoothed():
    return LabelConfig(
        horizon=5,
        vol_window=10,
        return_quantile=0.5,
        vol_quantile=0.5,
        smoothing=True,
        smoothing_min_periods=3,
    )


def test_base_labeler_is_abstract():
    import inspect
    assert inspect.isabstract(BaseLabeler)


def test_quantile_labeler_produces_4_classes(spy_prices, label_cfg):
    labeler = QuantileLabeler(label_cfg)
    result = labeler.fit_transform(spy_prices)
    labels = result["label"].dropna()
    assert set(labels.unique()).issubset({0, 1, 2, 3})
    assert len(set(labels.unique())) == 4


def test_quantile_labeler_no_nan_after_warmup(spy_prices, label_cfg):
    labeler = QuantileLabeler(label_cfg)
    result = labeler.fit_transform(spy_prices)
    warmup = label_cfg.vol_window + label_cfg.horizon
    valid = result.iloc[warmup:-label_cfg.horizon]["label"]
    assert valid.isna().sum() == 0


def test_quantile_labeler_last_h_rows_nan(spy_prices, label_cfg):
    labeler = QuantileLabeler(label_cfg)
    result = labeler.fit_transform(spy_prices)
    tail_labels = result["label"].iloc[-label_cfg.horizon:]
    assert tail_labels.isna().all()


def test_quantile_labeler_thresholds_fit_on_train_only(spy_prices, label_cfg):
    n = len(spy_prices)
    train = spy_prices.iloc[: n // 2]
    labeler = QuantileLabeler(label_cfg)
    labeler.fit(train)
    thresh_from_train = labeler.return_thresh_
    labeler2 = QuantileLabeler(label_cfg)
    labeler2.fit(spy_prices)
    thresh_from_full = labeler2.return_thresh_
    assert thresh_from_train != pytest.approx(thresh_from_full, rel=1e-6)


def test_quantile_labeler_smoothing_reduces_switches(spy_prices, label_cfg, label_cfg_smoothed):
    raw_labeler = QuantileLabeler(label_cfg)
    smooth_labeler = QuantileLabeler(label_cfg_smoothed)
    raw_result = raw_labeler.fit_transform(spy_prices)
    smooth_result = smooth_labeler.fit_transform(spy_prices)

    def switch_freq(labels):
        l = labels.dropna().values
        return (l[1:] != l[:-1]).mean()

    assert switch_freq(smooth_result["label"]) <= switch_freq(raw_result["label"])


def test_quantile_labeler_output_columns(spy_prices, label_cfg):
    labeler = QuantileLabeler(label_cfg)
    result = labeler.fit_transform(spy_prices)
    for col in ["fwd_return_h", "roll_vol", "return_thresh", "vol_thresh",
                "label_raw", "label", "regime_name"]:
        assert col in result.columns, f"Missing column: {col}"


def test_quantile_labeler_transform_uses_fitted_thresholds(spy_prices, label_cfg):
    n = len(spy_prices)
    train = spy_prices.iloc[: n // 2]
    val = spy_prices.iloc[n // 2 :]
    labeler = QuantileLabeler(label_cfg)
    labeler.fit(train)
    result = labeler.transform(val)
    assert (result["return_thresh"].dropna() == labeler.return_thresh_).all()
    assert (result["vol_thresh"].dropna() == labeler.vol_thresh_).all()
