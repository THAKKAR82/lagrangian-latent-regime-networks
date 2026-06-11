"""Prediction artifact: per-fold probability outputs saved as parquet."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PredictionArtifact:
    fold_id: int
    split: str          # "val" or "test"
    model_name: str
    dates: np.ndarray   # shape (N,), string dates
    true_labels: np.ndarray  # shape (N,), int
    probs: np.ndarray   # shape (N, 4), float32

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "date": self.dates,
            "fold_id": self.fold_id,
            "split": self.split,
            "model": self.model_name,
            "true_label": self.true_labels,
        })
        for c in range(4):
            df[f"prob_{c}"] = self.probs[:, c]
        return df

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)

    @classmethod
    def load(cls, path: Path) -> "PredictionArtifact":
        df = pd.read_parquet(path)
        probs = df[[f"prob_{c}" for c in range(4)]].values.astype(np.float32)
        return cls(
            fold_id=int(df["fold_id"].iloc[0]),
            split=str(df["split"].iloc[0]),
            model_name=str(df["model"].iloc[0]),
            dates=df["date"].values,
            true_labels=df["true_label"].values.astype(int),
            probs=probs,
        )
