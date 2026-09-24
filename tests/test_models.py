from __future__ import annotations

import json
import math
import pickle

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from tests.synthetic_history import make_history
from valchamps.features import build_feature_frame, load_events, load_records, team_regions
from valchamps.models import (
    MODEL_NAMES,
    load_config,
    make_model,
    prepare,
    run_backtest,
    run_holdout,
    symmetric_predict,
)
from valchamps.models.base import partner_index
from valchamps.models.dataset import (
    NUMERIC_FEATURES,
    holdout_fold,
    time_val_split,
    walk_forward_folds,
)
from valchamps.models.metrics import ece, evaluate, score

HOLDOUT = [6]  # synthetic Masters 2026
TEST_MODELS = {
    "backtest": {"first_test_date": "2025-01-01", "holdout_events": HOLDOUT},
    "gbm": {"params": {"min_child_samples": 10, "n_estimators": 300}},
    "nn": {"params": {"epochs": 60, "patience": 10}},
}


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def needs_model(name: str):
    return pytest.mark.skipif(
        name == "nn" and not _torch_available(), reason="PyTorch not installed (extra `nn`)"
    )


@pytest.fixture(scope="module")
def frame(tmp_path_factory):
    from valchamps.data import db

    engine = db.get_engine(f"sqlite:///{tmp_path_factory.mktemp('models') / 'h.db'}")
    db.init_db(engine)
    make_history(engine, teams_per_region=8, spread=300)
    raw, _ = build_feature_frame(
        load_records(engine), team_regions(engine), events=load_events(engine)
    )
    return prepare(raw)


@pytest.fixture(scope="module")
def config():
    from valchamps.models.backtest import _merge

    return _merge(load_config(None), TEST_MODELS)


@pytest.fixture(scope="module")
def backtests(frame, config):
    return {
        name: run_backtest(frame, name, config)
        for name in MODEL_NAMES
        if name != "nn" or _torch_available()
    }


# --- splits -----------------------------------------------------------------------------------


def test_folds_only_train_on_the_past(frame):
    folds = walk_forward_folds(frame, "2025-01-01", HOLDOUT)
    assert [f.event_id for f in folds] == [2, 3, 4, 5]  # event 1 has no past; 6 is held out
    for fold in folds:
        assert frame["date"].iloc[fold.train_idx].max() < frame["date"].iloc[fold.test_idx].min()
        assert not frame["event_id"].iloc[fold.train_idx].isin(HOLDOUT).any()


def test_holdout_is_never_trained_on(frame):
    fold = holdout_fold(frame, HOLDOUT)
    assert set(frame["event_id"].iloc[fold.test_idx]) == set(HOLDOUT)
    assert frame["date"].iloc[fold.train_idx].max() < frame["date"].iloc[fold.test_idx].min()
    with pytest.raises(ValueError, match="holdout"):
        holdout_fold(frame, [999])


def test_validation_split_is_most_recent_and_keeps_pairs(frame):
    train, val = time_val_split(frame, 0.2)
    assert train["date"].max() <= val["date"].min()
    assert (val.groupby("game_id").size() == 2).all()
    assert abs(val["game_id"].nunique() / frame["game_id"].nunique() - 0.2) < 0.01


# --- models -----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", [pytest.param(n, marks=needs_model(n)) for n in MODEL_NAMES])
def test_every_model_beats_a_coin_flip(backtests, name):
    overall = backtests[name].metrics["overall"]
    assert overall["n"] > 150
    assert overall["log_loss"] < math.log(2) - 0.02


def test_trained_models_are_not_worse_than_raw_elo(backtests):
    elo = backtests["elo"].metrics["overall"]["log_loss"]
    for name in ("elo_cal", "linear", "gbm"):
        assert backtests[name].metrics["overall"]["log_loss"] < elo + 0.04, name


@pytest.mark.parametrize("name", [pytest.param(n, marks=needs_model(n)) for n in MODEL_NAMES])
def test_predictions_are_symmetric(frame, config, name):
    fold = holdout_fold(frame, HOLDOUT)
    model = make_model(name, config).fit(frame.iloc[fold.train_idx])
    test = frame.iloc[fold.test_idx]
    p = symmetric_predict(model, test)
    partner = partner_index(test)
    assert (partner >= 0).all()
    np.testing.assert_allclose(p + p[partner], 1.0, atol=1e-9)


@needs_model("nn")
def test_nn_is_antisymmetric_without_averaging(frame, config):
    model = make_model("nn", config).fit(frame)
    p = model.predict(frame)
    np.testing.assert_allclose(p + p[partner_index(frame)], 1.0, atol=1e-5)


@pytest.mark.parametrize("name", [pytest.param(n, marks=needs_model(n)) for n in MODEL_NAMES])
def test_unseen_map_still_predicts(frame, config, name):
    fold = holdout_fold(frame, HOLDOUT)
    model = make_model(name, config).fit(frame.iloc[fold.train_idx])
    test = frame.iloc[fold.test_idx].assign(map_name="Corrode")  # a map added mid-season
    p = symmetric_predict(model, test)
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()


def test_linear_inputs_are_clipped(frame, config):
    model = make_model("linear", config).fit(frame)
    num = model.pipeline.named_steps["prep"].named_transformers_["num"]
    extreme = frame[NUMERIC_FEATURES].head(4).copy()
    extreme["rest_days_a"] = 10_000.0  # e.g. a very long off-season
    assert np.abs(num.transform(extreme)).max() <= 3.0


def test_calibrated_elo_learns_a_positive_slope(frame, config):
    model = make_model("elo_cal", config).fit(frame)
    assert model.slope > 0


def test_gbm_platt_calibration_option(frame, config):
    from valchamps.models.gbm import GBMModel

    model = GBMModel(params={"min_child_samples": 10}, calibration="platt").fit(frame)
    assert model.platt is not None
    assert ((model.predict(frame) > 0) & (model.predict(frame) < 1)).all()
    with pytest.raises(ValueError, match="calibration"):
        GBMModel(calibration="isotonic")


def test_holdout_run(frame, config):
    result = run_holdout(frame, "gbm", config)
    assert set(result.predictions["event_id"]) == set(HOLDOUT)
    assert result.metrics["overall"]["log_loss"] < math.log(2)


# --- metrics ----------------------------------------------------------------------------------


def test_metrics_by_hand():
    y, p = np.array([1, 0]), np.array([0.8, 0.3])
    m = score(y, p)
    assert m["log_loss"] == pytest.approx(-(math.log(0.8) + math.log(0.7)) / 2)
    assert m["brier"] == pytest.approx((0.2**2 + 0.3**2) / 2)
    assert m["accuracy"] == 1.0


def test_ece_is_zero_when_calibrated_and_large_when_not():
    y = np.array([1, 0] * 50)
    assert ece(y, np.full(100, 0.5)) == pytest.approx(0.0)
    assert ece(y, np.full(100, 0.9)) == pytest.approx(0.4)


def test_evaluate_reports_every_segment():
    preds = pd.DataFrame({
        "y": [1, 0, 1], "p": [0.7, 0.4, 0.6],
        "cross_region": [1.0, 0.0, 0.0], "is_international": [1.0, 1.0, 0.0],
    })  # fmt: skip
    out = evaluate(preds)
    assert {k: v["n"] for k, v in out.items()} == {
        "overall": 3, "cross_region": 1, "international": 2,
    }  # fmt: skip


# --- CLI + MLflow -----------------------------------------------------------------------------


@pytest.fixture
def cli_env(frame, tmp_path, monkeypatch):
    import valchamps.cli as cli

    features = tmp_path / "maps.parquet"
    frame.to_parquet(features, index=False)
    params = tmp_path / "params.yaml"
    params.write_text(yaml.safe_dump({"models": TEST_MODELS}))
    monkeypatch.chdir(tmp_path)  # MLflow's local artifact store
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(cli, "REPORTS_DIR", tmp_path / "reports")
    return features, params, tmp_path


def test_cli_backtest_logs_runs_to_mlflow(cli_env):
    import mlflow

    from valchamps.cli import app

    features, params, tmp = cli_env
    args = ["backtest", "--models", "elo,elo_cal", "--features", str(features),
            "--params-file", str(params)]  # fmt: skip
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "elo_cal  cross_region" in result.output
    mlflow.set_tracking_uri(f"sqlite:///{tmp / 'mlflow.db'}")
    runs = mlflow.search_runs(experiment_names=["map-model-backtest"])
    assert set(runs["tags.mlflow.runName"]) == {"elo", "elo_cal"}
    assert runs["metrics.overall/log_loss"].between(0, math.log(2)).all()
    assert (tmp / "reports" / "backtest" / "elo" / "reliability.png").exists()


def test_cli_train_saves_and_registers_model(cli_env, frame):
    import mlflow

    from valchamps.cli import app

    features, params, tmp = cli_env
    out, metrics = tmp / "map_model.pkl", tmp / "metrics.json"
    result = CliRunner().invoke(app, [
        "train", "--model", "gbm", "--features", str(features), "--params-file", str(params),
        "--out", str(out), "--metrics-file", str(metrics),
    ])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "registered map-model v1" in result.output

    bundle = pickle.loads(out.read_bytes())
    assert bundle["name"] == "gbm"
    p = symmetric_predict(bundle["model"], frame)
    assert ((p > 0) & (p < 1)).all()
    assert json.loads(metrics.read_text())["holdout"]["overall"]["n"] > 0

    mlflow.set_tracking_uri(f"sqlite:///{tmp / 'mlflow.db'}")
    loaded = mlflow.pyfunc.load_model("models:/map-model/1")
    np.testing.assert_allclose(loaded.predict(frame.head(10)), p[:10])


def test_cli_rejects_unknown_model(cli_env):
    from valchamps.cli import app

    features, _, _ = cli_env
    result = CliRunner().invoke(app, ["backtest", "--models", "xgb", "--features", str(features)])
    assert result.exit_code != 0
    assert "unknown model" in result.output
