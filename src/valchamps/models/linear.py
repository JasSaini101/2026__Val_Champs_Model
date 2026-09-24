"""Regularised logistic regression on every feature: the interpretable model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from valchamps.models.dataset import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES

CLIP_Z = 3.0


def clip_z(x: np.ndarray) -> np.ndarray:
    """Cap standardised values at +/-3 so out-of-range inputs can't produce extreme odds.

    Features such as rest days jump across an off-season (~300 days); a linear model or MLP
    would otherwise extrapolate that straight into near-certain predictions.
    """
    return np.clip(x, -CLIP_Z, CLIP_Z)


def numeric_preprocessor() -> Pipeline:
    """Median-impute (plus a was-missing flag per column), standardise, clip to +/-3."""
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        FunctionTransformer(clip_z),
    )


class LinearModel:
    name = "linear"

    def __init__(self, C: float = 0.1, max_iter: int = 2000) -> None:
        self.pipeline = Pipeline([
            ("prep", ColumnTransformer([
                ("num", numeric_preprocessor(), NUMERIC_FEATURES),
                ("map", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ])),
            ("lr", LogisticRegression(C=C, max_iter=max_iter)),
        ])  # fmt: skip

    def fit(self, df: pd.DataFrame) -> LinearModel:
        self.pipeline.fit(df[MODEL_FEATURES], df["y"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(df[MODEL_FEATURES])[:, 1]

    def coefficients(self) -> pd.Series:
        names = self.pipeline.named_steps["prep"].get_feature_names_out()
        coefs = self.pipeline.named_steps["lr"].coef_[0]
        return pd.Series(coefs, index=names).sort_values(key=abs, ascending=False)
