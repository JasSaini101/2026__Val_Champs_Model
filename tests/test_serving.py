from __future__ import annotations

import pickle
from pathlib import Path

import httpx
import pytest
import respx

pytest.importorskip("fastapi")
pytest.importorskip("streamlit")

from fastapi.testclient import TestClient

from tests.synthetic_history import CHAMPIONS_EVENT, add_champions_event, make_history
from valchamps.api.app import ApiSettings, create_app
from valchamps.bracket import forecast_event, publish
from valchamps.cli import DEFAULT_BRACKET
from valchamps.dashboard import charts
from valchamps.dashboard import client as client_module
from valchamps.dashboard.client import ApiClient, ApiError
from valchamps.data import db
from valchamps.features import (
    build_feature_frame,
    load_events,
    load_records,
    team_regions,
)
from valchamps.models import load_config, make_model, prepare

DASHBOARD = Path(__file__).parents[1] / "src" / "valchamps" / "dashboard" / "app.py"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Synthetic history + Champions event, a trained linear map model, and published odds."""
    tmp = tmp_path_factory.mktemp("serving")
    engine = db.get_engine(f"sqlite:///{tmp / 'hist.db'}")
    db.init_db(engine)
    truth = make_history(engine, teams_per_region=8, spread=200)
    teams = sorted(truth.strength)
    add_champions_event(engine, teams, played=1)
    records = load_records(engine)
    frame, _ = build_feature_frame(records, team_regions(engine), events=load_events(engine))
    model = make_model("linear", load_config(None)).fit(prepare(frame))
    model_path = tmp / "map_model.pkl"
    model_path.write_bytes(pickle.dumps(
        {"model": model, "name": "linear", "trained_through": "2027-03-01", "maps": 1}
    ))  # fmt: skip
    odds_dir = tmp / "odds"
    forecast = forecast_event(engine, CHAMPIONS_EVENT, bracket_file=DEFAULT_BRACKET, odds="elo",
                              runs=2_000)  # fmt: skip
    publish(forecast, odds_dir / str(CHAMPIONS_EVENT))
    settings = ApiSettings(db_url=str(engine.url), odds_dir=odds_dir, model_path=model_path,
                           params_file=tmp / "missing.yaml")  # fmt: skip
    return engine, teams, settings


@pytest.fixture(scope="module")
def api(world):
    _, _, settings = world
    return TestClient(create_app(settings))


# --- API --------------------------------------------------------------------------------------


def test_health(api):
    assert api.get("/health").json() == {"status": "ok"}


def test_published_odds(api):
    latest = api.get(f"/events/{CHAMPIONS_EVENT}/odds").json()
    assert len(latest["teams"]) == 16 and latest["fixed_results"] == 1
    assert sum(t["title"] for t in latest["teams"]) == pytest.approx(1, abs=1e-4)
    history = api.get(f"/events/{CHAMPIONS_EVENT}/odds/history").json()
    assert len(history) == 16 and {"updated_at", "team", "title"} <= set(history[0])
    assert api.get("/events/1/odds").status_code == 404
    assert api.get("/events/1/odds/history").status_code == 404


def test_event_teams(api, world):
    _, teams, _ = world
    got = api.get(f"/events/{CHAMPIONS_EVENT}/teams").json()
    assert sorted(t["team_id"] for t in got) == sorted(teams)
    assert {t["group"] for t in got} == {"A", "B", "C", "D"}
    assert api.get("/events/1/teams").status_code == 404


def test_predict_is_symmetric(api, world):
    _, teams, _ = world
    a, b = teams[0], teams[9]
    for best_of in (3, 5):
        ab = api.get("/predict", params={"team_a": a, "team_b": b, "best_of": best_of}).json()
        ba = api.get("/predict", params={"team_a": b, "team_b": a, "best_of": best_of}).json()
        assert ab["p_a"] + ba["p_a"] == pytest.approx(1)
        assert ab["p_a"] + ab["p_b"] == pytest.approx(1)
        assert sum(s["p"] for s in ab["scores"]) == pytest.approx(1)
        assert len(ab["maps"]) == 7 and len(ab["vetoes"]) == 5
        assert all(len(v["maps"]) == best_of for v in ab["vetoes"])
        assert ab["model"]["name"] == "linear"


def test_predict_rejects_bad_requests(api, world):
    _, teams, _ = world
    a = teams[0]
    assert api.get("/predict", params={"team_a": a, "team_b": a}).status_code == 422
    r = api.get("/predict", params={"team_a": a, "team_b": teams[1], "best_of": 7})
    assert r.status_code == 422 and "best_of" in r.json()["detail"]
    r = api.get("/predict", params={"team_a": a, "team_b": 999_999})
    assert r.status_code == 404 and "no completed matches" in r.json()["detail"]


def test_state_reloads_when_the_database_changes(world, tmp_path):
    engine, teams, settings = world
    import shutil

    db_copy = tmp_path / "copy.db"
    shutil.copy(Path(engine.url.database), db_copy)
    from dataclasses import replace

    local = replace(settings, db_url=f"sqlite:///{db_copy}")
    api = TestClient(create_app(local))
    params = {"team_a": teams[0], "team_b": teams[9]}
    before = api.get("/predict", params=params).json()["data_through"]
    import os

    add_champions_event(db.get_engine(f"sqlite:///{db_copy}"), teams, played=4)
    st = os.stat(db_copy)
    os.utime(db_copy, (st.st_atime, st.st_mtime + 5))  # make sure the mtime moves
    after = api.get("/predict", params=params).json()["data_through"]
    assert after > before


# --- dashboard --------------------------------------------------------------------------------


def test_charts_build(api, world):
    import pandas as pd

    _, teams, _ = world
    latest = api.get(f"/events/{CHAMPIONS_EVENT}/odds").json()
    table = pd.DataFrame(latest["teams"])
    history = pd.DataFrame(api.get(f"/events/{CHAMPIONS_EVENT}/odds/history").json())
    pred = api.get("/predict", params={"team_a": teams[0], "team_b": teams[9]}).json()
    for mode, colors in charts.PALETTE.items():
        specs = [
            charts.title_bars(table, colors).to_dict(),
            charts.history_lines(history, table["team"].iloc[0], colors).to_dict(),
            charts.score_bars(pred["scores"], "A", "B", colors).to_dict(),
        ]
        assert all(
            s["$schema"].startswith("https://vega.github.io/schema/vega-lite") for s in specs
        )
        assert colors["accent"] in str(specs[0]), mode
        assert colors["team_b"] in str(specs[2]), mode


@respx.mock
def test_api_client_reports_errors():
    respx.get("http://api.test/events/1/odds").mock(
        return_value=httpx.Response(404, json={"detail": "no odds published for event 1"})
    )
    respx.get("http://api.test/events/2/odds").mock(side_effect=httpx.ConnectError("refused"))
    client = ApiClient("http://api.test/")
    with pytest.raises(ApiError, match="404: no odds published"):
        client.odds(1)
    with pytest.raises(ApiError, match="cannot reach the API"):
        client.odds(2)


class _ClientOverTestApp(ApiClient):
    """The dashboard's client, but sending requests to an in-process TestClient."""

    app: TestClient

    def _get(self, path, **params):
        r = self.app.get(path, params=params)
        if r.status_code != 200:
            raise ApiError(f"{r.status_code}: {r.json().get('detail')}")
        return r.json()


def test_dashboard_renders_against_the_api(api, monkeypatch):
    from streamlit.testing.v1 import AppTest

    _ClientOverTestApp.app = api
    monkeypatch.setattr(client_module, "ApiClient", _ClientOverTestApp)
    at = AppTest.from_file(str(DASHBOARD), default_timeout=120)
    at.run()
    at.sidebar.number_input[0].set_value(CHAMPIONS_EVENT).run()
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    assert "Favourite" in labels and "Finished series" in labels
    # Match predictor: two different teams are preselected, so a prediction is shown.
    assert any(m.label.endswith(" wins") for m in at.metric)
    assert not at.error


def test_dashboard_without_published_odds(monkeypatch, api):
    from streamlit.testing.v1 import AppTest

    _ClientOverTestApp.app = api
    monkeypatch.setattr(client_module, "ApiClient", _ClientOverTestApp)
    at = AppTest.from_file(str(DASHBOARD), default_timeout=120)
    at.run()
    at.sidebar.number_input[0].set_value(1).run()  # nothing published, no groups
    assert not at.exception, at.exception
    assert any("No published odds yet" in i.value for i in at.info)
