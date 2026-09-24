"""MLflow tracking. Points at ``MLFLOW_TRACKING_URI`` (e.g. DagsHub) or a local SQLite file.

DagsHub credentials come from ``MLFLOW_TRACKING_USERNAME`` / ``MLFLOW_TRACKING_PASSWORD``,
which MLflow reads itself; nothing secret passes through this code.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pandas as pd

from valchamps.config import PROJECT_ROOT
from valchamps.models.base import MapModel, symmetric_predict
from valchamps.models.dataset import MODEL_FEATURES

DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
REGISTERED_MODEL = "map-model"


def _mlflow():
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    return mlflow


def configure() -> str:
    """Set the tracking URI and return it (credentials are never part of it)."""
    uri = os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    _mlflow().set_tracking_uri(uri)
    return uri


@contextmanager
def run(experiment: str, run_name: str, params: dict[str, Any]) -> Iterator[Any]:
    mlflow = _mlflow()
    configure()
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as active:
        mlflow.log_params({k: str(v)[:500] for k, v in _flatten(params).items()})
        yield active


def log_segment_metrics(metrics: dict[str, dict[str, float]], step: int | None = None) -> None:
    """Log {segment: {metric: value}} as ``segment/metric`` (e.g. ``cross_region/log_loss``)."""
    flat = {
        f"{segment}/{name}": float(value)
        for segment, values in metrics.items()
        for name, value in values.items()
        if value == value  # skip NaN (empty segment)
    }
    _mlflow().log_metrics(flat, step=step)


def log_artifact(path: Any) -> None:
    _mlflow().log_artifact(str(path))


def log_and_register_model(model: MapModel, example: pd.DataFrame) -> str | None:
    """Log the model as an MLflow pyfunc and register it as ``map-model``.

    Returns the registered version, or None when the server has no model registry.
    """
    mlflow = _mlflow()

    from mlflow.models import ModelSignature
    from mlflow.types import ColSpec, Schema

    class MapModelPyfunc(mlflow.pyfunc.PythonModel):
        def __init__(self, inner: MapModel) -> None:
            self.inner = inner

        def predict(self, context, model_input, params=None):
            return symmetric_predict(self.inner, model_input)

    # Explicit schema: features are doubles (they can be missing), map_name a string, and
    # game_id/perspective integers that pair up the two views of each map.
    signature = ModelSignature(
        inputs=Schema(
            [ColSpec("double", c) for c in MODEL_FEATURES if c != "map_name"]
            + [
                ColSpec("string", "map_name"),
                ColSpec("long", "game_id"),
                ColSpec("long", "perspective"),
            ]
        ),
        outputs=Schema([ColSpec("double")]),
    )
    example = example[[*MODEL_FEATURES, "game_id", "perspective"]].head(4).copy()
    for col in example.columns:
        if col not in ("map_name", "game_id", "perspective"):
            example[col] = example[col].astype(float)
    info = mlflow.pyfunc.log_model(
        name="model", python_model=MapModelPyfunc(model), signature=signature,
        input_example=example,
    )  # fmt: skip
    try:
        version = mlflow.register_model(info.model_uri, REGISTERED_MODEL)
    except Exception as exc:  # registry unsupported by the tracking server
        print(f"model logged but not registered: {exc}")
        return None
    return str(version.version)


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out
