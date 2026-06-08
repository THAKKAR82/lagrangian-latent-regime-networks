import pandas as pd
import pytest
from pathlib import Path
from src.data.manager import DataManager


def test_data_manager_raw_path(tmp_path):
    dm = DataManager(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        tickers=["SPY", "^VIX"],
        start_date="2020-01-01",
        end_date="2020-12-31",
    )
    path = dm.raw_path("SPY")
    assert path == tmp_path / "raw" / "SPY.parquet"


def test_data_manager_save_and_load_raw(tmp_path, spy_prices):
    dm = DataManager(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        tickers=["SPY"],
        start_date="2010-01-01",
        end_date="2023-12-31",
    )
    dm.save_raw("SPY", spy_prices)
    loaded = dm.load_raw("SPY")
    pd.testing.assert_frame_equal(spy_prices, loaded, check_freq=False)


def test_data_manager_save_creates_dirs(tmp_path, spy_prices):
    dm = DataManager(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        tickers=["SPY"],
        start_date="2010-01-01",
        end_date="2023-12-31",
    )
    dm.save_raw("SPY", spy_prices)
    assert (tmp_path / "raw" / "SPY.parquet").exists()


def test_data_manager_hash_saved(tmp_path, spy_prices):
    dm = DataManager(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        tickers=["SPY"],
        start_date="2010-01-01",
        end_date="2023-12-31",
    )
    dm.save_raw("SPY", spy_prices)
    assert (tmp_path / "raw" / "SPY.sha256").exists()
