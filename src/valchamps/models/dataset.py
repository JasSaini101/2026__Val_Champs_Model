"""Model inputs and time-ordered train/test splits.

Every split is by date: a model is only ever trained on maps played before the maps it is
tested on. Rows come in pairs (one per team's perspective); training uses both, metrics use
``perspective == 0`` so each map is counted once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from valchamps.features.build import FEATURE_COLUMNS

NUMERIC_FEATURES = [*FEATURE_COLUMNS, "is_international", "cross_region"]
CATEGORICAL_FEATURES = ["map_name"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes and sort oldest first (pairs of perspectives kept together)."""
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in ("is_international", "cross_region"):
        df[col] = df[col].astype(float)
    df["map_name"] = df["map_name"].astype(str)
    return df.sort_values(["date", "match_id", "game_id", "perspective"]).reset_index(drop=True)


def load_frame(path: Path) -> pd.DataFrame:
    return prepare(pd.read_parquet(path))


@dataclass(frozen=True)
class Fold:
    name: str
    event_id: int | None
    train_idx: np.ndarray
    test_idx: np.ndarray


def event_starts(df: pd.DataFrame) -> pd.Series:
    """First map date of each event, sorted."""
    return df.groupby("event_id")["date"].min().sort_values()


def walk_forward_folds(
    df: pd.DataFrame, first_test_date: str, holdout_events: list[int]
) -> list[Fold]:
    """One fold per event starting on/after ``first_test_date`` and before the holdout.

    Each fold trains on every map dated before its event starts and tests on that event.
    """
    starts = event_starts(df)
    holdout_start = holdout_start_date(df, holdout_events)
    folds = []
    for event_id, start in starts.items():
        if event_id in holdout_events or start < pd.Timestamp(first_test_date):
            continue
        if holdout_start is not None and start >= holdout_start:
            continue
        train_idx = np.flatnonzero(df["date"] < start)
        test_idx = np.flatnonzero(df["event_id"] == event_id)
        if len(train_idx) and len(test_idx):
            folds.append(Fold(f"event_{event_id}", int(event_id), train_idx, test_idx))
    return folds


def holdout_start_date(df: pd.DataFrame, holdout_events: list[int]) -> pd.Timestamp | None:
    dates = df.loc[df["event_id"].isin(holdout_events), "date"]
    return dates.min() if len(dates) else None


def holdout_fold(df: pd.DataFrame, holdout_events: list[int]) -> Fold:
    """Train on everything before the first holdout event, test on the holdout events."""
    start = holdout_start_date(df, holdout_events)
    if start is None:
        raise ValueError(f"none of the holdout events {holdout_events} are in the data")
    return Fold(
        "holdout",
        None,
        np.flatnonzero(df["date"] < start),
        np.flatnonzero(df["event_id"].isin(holdout_events)),
    )


def time_val_split(df: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off the most recent ``fraction`` of maps (both perspectives stay together)."""
    games = df.drop_duplicates("game_id").sort_values(["date", "game_id"])["game_id"].to_numpy()
    n_val = int(len(games) * fraction)
    if n_val == 0 or n_val == len(games):
        return df, df.iloc[0:0]
    val_games = set(games[-n_val:])
    is_val = df["game_id"].isin(val_games)
    return df[~is_val], df[is_val]
