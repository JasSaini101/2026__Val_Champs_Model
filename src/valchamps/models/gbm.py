"""LightGBM on every feature, with early stopping on the most recent maps."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from valchamps.models.dataset import MODEL_FEATURES, time_val_split

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}


class GBMModel:
    name = "gbm"

    def __init__(
        self, params: dict[str, Any] | None = None, early_stopping_rounds: int = 50,
        val_fraction: float = 0.15, calibration: str = "none", seed: int = 0,
    ) -> None:  # fmt: skip
        if calibration not in ("none", "platt"):
            raise ValueError(f"calibration must be 'none' or 'platt', not {calibration!r}")
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        self.calibration = calibration
        self.seed = seed
        self.maps_: list[str] = []
        self.model: lgb.LGBMClassifier | None = None
        self.platt: LogisticRegression | None = None

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[MODEL_FEATURES].copy()
        # Fixed categories: maps unseen in training become missing, which LightGBM handles.
        known = X["map_name"].where(X["map_name"].isin(self.maps_))
        X["map_name"] = pd.Categorical(known, categories=self.maps_)
        return X

    def fit(self, df: pd.DataFrame) -> GBMModel:
        self.maps_ = sorted(df["map_name"].unique())
        train, val = time_val_split(df, self.val_fraction)
        self.model = lgb.LGBMClassifier(**self.params, random_state=self.seed, verbose=-1)
        fit_kwargs: dict[str, Any] = {}
        if len(val):
            fit_kwargs = {
                "eval_X": (self._X(val),),
                "eval_y": (val["y"],),
                "callbacks": [lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
            }
        self.model.fit(self._X(train), train["y"], **fit_kwargs)
        if self.calibration == "platt" and len(val):
            raw = self.model.predict_proba(self._X(val))[:, 1]
            self.platt = LogisticRegression(C=1e6).fit(_logit(raw).reshape(-1, 1), val["y"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit the model first")
        p = self.model.predict_proba(self._X(df))[:, 1]
        if self.platt is not None:
            p = self.platt.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return p

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("fit the model first")
        booster = self.model.booster_
        gain = booster.feature_importance(importance_type="gain")
        return pd.Series(gain, index=booster.feature_name()).sort_values(ascending=False)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
