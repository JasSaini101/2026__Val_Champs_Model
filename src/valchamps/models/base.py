"""The interface every map model implements, and perspective symmetry."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class MapModel(Protocol):
    name: str

    def fit(self, df: pd.DataFrame) -> MapModel:
        """Train on rows of the feature table (both perspectives, target column ``y``)."""
        ...

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """P(team_a wins the map) for each row."""
        ...


def partner_index(df: pd.DataFrame) -> np.ndarray:
    """Position of each row's mirrored row (same map, other team's perspective), or -1."""
    position = {
        (g, p): i for i, (g, p) in enumerate(zip(df["game_id"], df["perspective"], strict=True))
    }
    return np.array(
        [
            position.get((g, 1 - p), -1)
            for g, p in zip(df["game_id"], df["perspective"], strict=True)
        ]
    )


def symmetric_predict(model: MapModel, df: pd.DataFrame) -> np.ndarray:
    """Average a model's two views of each map so P(A wins) + P(B wins) = 1 exactly."""
    p = np.asarray(model.predict(df), dtype=float)
    partner = partner_index(df)
    has_partner = partner >= 0
    out = p.copy()
    out[has_partner] = (p[has_partner] + 1.0 - p[partner[has_partner]]) / 2.0
    return out
