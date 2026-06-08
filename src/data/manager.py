"""DataManager: owns all disk I/O and path resolution. No data transforms."""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DataManager:
    raw_dir: Path
    processed_dir: Path
    tickers: list[str]
    start_date: str
    end_date: str

    def __post_init__(self):
        self.raw_dir = Path(self.raw_dir)
        self.processed_dir = Path(self.processed_dir)

    def raw_path(self, ticker: str) -> Path:
        return self.raw_dir / f"{ticker}.parquet"

    def processed_path(self, name: str) -> Path:
        return self.processed_dir / f"{name}.parquet"

    def save_raw(self, ticker: str, df: pd.DataFrame) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_path(ticker)
        df.to_parquet(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(".sha256").write_text(sha)

    def load_raw(self, ticker: str) -> pd.DataFrame:
        return pd.read_parquet(self.raw_path(ticker))

    def save_processed(self, name: str, df: pd.DataFrame) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.processed_path(name))

    def load_processed(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.processed_path(name))

    def raw_exists(self, ticker: str) -> bool:
        return self.raw_path(ticker).exists()

    def log_hashes(self) -> dict[str, str]:
        """Return SHA-256 hashes for all cached raw files."""
        result = {}
        for ticker in self.tickers:
            hash_path = self.raw_path(ticker).with_suffix(".sha256")
            if hash_path.exists():
                result[ticker] = hash_path.read_text().strip()
        return result
