"""Baselines: the Elo system's own probability, raw and recalibrated."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class RawElo:
    """``elo_prob`` from the feature pipeline, untouched. Needs no training."""

    name = "elo"

    def fit(self, df: pd.DataFrame) -> RawElo:
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["elo_prob"].to_numpy(dtype=float)


class CalibratedElo:
    """Rescales Elo's confidence: p = sigmoid(a * logit(elo_prob)).

    No intercept, so it stays symmetric between the two teams; ``a`` < 1 means Elo was
    overconfident, ``a`` > 1 underconfident.
    """

    name = "elo_cal"

    def __init__(self) -> None:
        self.lr = LogisticRegression(fit_intercept=False, C=1e6)

    def fit(self, df: pd.DataFrame) -> CalibratedElo:
        self.lr.fit(_logit(df["elo_prob"].to_numpy(dtype=float)).reshape(-1, 1), df["y"])
        return self

    @property
    def slope(self) -> float:
        return float(self.lr.coef_[0, 0])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = _logit(df["elo_prob"].to_numpy(dtype=float)).reshape(-1, 1)
        return self.lr.predict_proba(x)[:, 1]
