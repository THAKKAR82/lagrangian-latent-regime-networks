"""BaseLabeler ABC — all regime labelers implement this interface."""
from abc import ABC, abstractmethod

import pandas as pd


class BaseLabeler(ABC):
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "BaseLabeler":
        """Fit labeler parameters on training data. Returns self."""
        ...

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted labeler to data. Must be called after fit()."""
        ...

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(data).transform(data)
