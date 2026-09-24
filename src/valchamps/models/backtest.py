"""Walk-forward backtests and final training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from valchamps.models.base import MapModel, symmetric_predict
from valchamps.models.dataset import Fold, holdout_fold, walk_forward_folds
from valchamps.models.metrics import evaluate, score

MODEL_NAMES = ("elo", "elo_cal", "linear", "gbm", "nn")

DEFAULT_CONFIG: dict[str, Any] = {
    "backtest": {"first_test_date": "2025-06-01", "holdout_events": []},
    "val_fraction": 0.15,
    "seed": 0,
    "linear": {"C": 0.1},
    "gbm": {"early_stopping_rounds": 50, "calibration": "none", "params": {}},
    "nn": {"params": {}},
}


def load_config(path: Path | None) -> dict[str, Any]:
    raw = {}
    if path is not None and path.exists():
        raw = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("models", {})
    return _merge(DEFAULT_CONFIG, raw)


def make_model(name: str, config: dict[str, Any]) -> MapModel:
    from valchamps.models.elo import CalibratedElo, RawElo
    from valchamps.models.gbm import GBMModel
    from valchamps.models.linear import LinearModel

    val, seed = config["val_fraction"], config["seed"]
    builders: dict[str, Callable[[], MapModel]] = {
        "elo": RawElo,
        "elo_cal": CalibratedElo,
        "linear": lambda: LinearModel(**config["linear"]),
        "gbm": lambda: GBMModel(
            params=config["gbm"].get("params"),
            early_stopping_rounds=config["gbm"]["early_stopping_rounds"],
            calibration=config["gbm"]["calibration"],
            val_fraction=val,
            seed=seed,
        ),
        "nn": lambda: _nn(config["nn"].get("params"), val, seed),
    }
    if name not in builders:
        raise ValueError(f"unknown model {name!r}; choose from {', '.join(MODEL_NAMES)}")
    return builders[name]()


def _nn(params: dict[str, Any] | None, val: float, seed: int) -> MapModel:
    from valchamps.models.nn import NNModel

    return NNModel(params=params, val_fraction=val, seed=seed)


@dataclass
class Result:
    model: str
    predictions: pd.DataFrame  # one row per test map (perspective 0)
    metrics: dict[str, dict[str, float]]
    fold_metrics: list[dict[str, Any]] = field(default_factory=list)


def predict_fold(model_name: str, config: dict[str, Any], df: pd.DataFrame, fold: Fold):
    """Train on the fold's past, predict its test maps (symmetrised), keep one row per map."""
    model = make_model(model_name, config).fit(df.iloc[fold.train_idx])
    test = df.iloc[fold.test_idx]
    p = symmetric_predict(model, test)
    out = test.assign(p=p, fold=fold.name)[
        ["fold", "event_id", "match_id", "game_id", "date", "map_name", "team_a", "team_b",
         "is_international", "cross_region", "perspective", "y", "p"]
    ]  # fmt: skip
    return model, out[out["perspective"] == 0].drop(columns="perspective")


def run_backtest(df: pd.DataFrame, model_name: str, config: dict[str, Any]) -> Result:
    folds = walk_forward_folds(
        df, config["backtest"]["first_test_date"], config["backtest"]["holdout_events"]
    )
    if not folds:
        raise ValueError("no backtest folds: check first_test_date and holdout_events")
    parts, fold_metrics = [], []
    for fold in folds:
        _, preds = predict_fold(model_name, config, df, fold)
        parts.append(preds)
        fold_metrics.append({
            "fold": fold.name, "train_maps": int(len(fold.train_idx) // 2),
            **score(preds["y"].to_numpy(), preds["p"].to_numpy()),
        })  # fmt: skip
    predictions = pd.concat(parts, ignore_index=True)
    return Result(model_name, predictions, evaluate(predictions), fold_metrics)


def run_holdout(df: pd.DataFrame, model_name: str, config: dict[str, Any]) -> Result:
    fold = holdout_fold(df, config["backtest"]["holdout_events"])
    _, preds = predict_fold(model_name, config, df, fold)
    return Result(model_name, preds, evaluate(preds))


def format_table(results: list[Result]) -> str:
    """Model x segment table of log loss / Brier / accuracy / ECE (lower is better, except acc)."""
    header = (
        f"{'model':<9}{'segment':<15}{'maps':>6}{'log_loss':>10}{'brier':>8}{'acc':>7}{'ece':>7}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        for segment, m in r.metrics.items():
            lines.append(
                f"{r.model:<9}{segment:<15}{m['n']:>6}{m['log_loss']:>10.4f}{m['brier']:>8.4f}"
                f"{m['accuracy']:>7.3f}{m['ece']:>7.3f}"
            )
    return "\n".join(lines)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out
