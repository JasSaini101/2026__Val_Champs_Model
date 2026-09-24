from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

import valchamps.cli as cli
from tests.synthetic_history import CHAMPIONS_EVENT, add_champions_event, make_history
from valchamps.bracket import FixedResult, forecast_event, publish, results_fingerprint
from valchamps.cli import DEFAULT_BRACKET, app
from valchamps.data import db
from valchamps.data.ingest import IngestReport

WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "update-odds.yml"


@pytest.fixture
def champions(tmp_path):
    engine = db.get_engine(f"sqlite:///{tmp_path / 'hist.db'}")
    db.init_db(engine)
    truth = make_history(engine, teams_per_region=8, spread=200)
    teams = sorted(truth.strength)
    add_champions_event(engine, teams, played=1)
    return engine, teams


def _forecast(engine):
    return forecast_event(engine, CHAMPIONS_EVENT, bracket_file=DEFAULT_BRACKET, odds="elo",
                          runs=2_000)  # fmt: skip


def test_fingerprint_ignores_order():
    a = FixedResult("Group Stage: Opening (A)", 1, 2, 1)
    b = FixedResult("Group Stage: Opening (B)", 3, 4, 4)
    swapped = FixedResult("Group Stage: Opening (A)", 2, 1, 1)
    assert results_fingerprint([a, b]) == results_fingerprint([b, swapped])
    assert results_fingerprint([a]) != results_fingerprint([a, b])
    assert results_fingerprint([a]) != results_fingerprint([FixedResult(a.stage, 1, 2, 2)])


def test_publish_only_when_results_change(champions, tmp_path):
    engine, teams = champions
    out = tmp_path / "odds"
    first = _forecast(engine)
    assert publish(first, out)
    assert not publish(_forecast(engine), out)  # same results: nothing written
    history = pd.read_csv(out / "history.csv")
    assert len(history) == 16 and history["results_fingerprint"].nunique() == 1

    add_champions_event(engine, teams, played=2)  # a second opening match finishes
    second = _forecast(engine)
    assert second.fingerprint != first.fingerprint
    assert publish(second, out)
    history = pd.read_csv(out / "history.csv")
    assert len(history) == 32
    assert history.groupby("updated_at")["title"].sum().round(6).eq(1).all()
    latest = json.loads((out / "latest.json").read_text())
    assert latest["results_fingerprint"] == second.fingerprint
    assert latest["fixed_results"] == 2
    assert len(latest["teams"]) == 16

    assert publish(second, out, force=True)
    assert len(pd.read_csv(out / "history.csv")) == 48


def test_cli_update_without_scraping(champions, tmp_path, monkeypatch):
    engine, _ = champions
    monkeypatch.setenv("VALCHAMPS_DB_URL", str(engine.url))
    out = tmp_path / "odds"
    args = ["update", "--event", str(CHAMPIONS_EVENT), "--no-ingest", "--odds", "elo",
            "--runs", "2000", "--out-dir", str(out)]  # fmt: skip
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "published to" in first.output
    again = CliRunner().invoke(app, args)
    assert again.exit_code == 0, again.output
    assert "no new results since the last update" in again.output
    assert len(pd.read_csv(out / "history.csv")) == 16


class _FakeClient:
    def __init__(self, settings):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_cli_update_scrapes_then_fails_on_scrape_errors(champions, tmp_path, monkeypatch):
    """New results are ingested before simulating; failed scrapes still publish, then exit 1."""
    engine, teams = champions
    monkeypatch.setenv("VALCHAMPS_DB_URL", str(engine.url))
    calls = []

    def fake_ingest(client, eng, spec, **kwargs):
        calls.append(spec.event_id)
        add_champions_event(engine, teams, played=3)  # "scrapes" two more finished matches
        return IngestReport(event_id=spec.event_id, listed=8, fetched=2, failed=[123])

    monkeypatch.setattr(cli, "VlrClient", _FakeClient)
    monkeypatch.setattr(cli, "ingest_event", fake_ingest)
    out = tmp_path / "odds"
    result = CliRunner().invoke(app, [
        "update", "--event", str(CHAMPIONS_EVENT), "--odds", "elo", "--runs", "2000",
        "--out-dir", str(out),
    ])  # fmt: skip
    assert result.exit_code == 1, result.output
    assert calls == [CHAMPIONS_EVENT]
    assert "2 fetched" in result.output and "1 failed" in result.output
    assert json.loads((out / "latest.json").read_text())["fixed_results"] == 3


def test_workflow_runs_update_and_commits_odds():
    wf = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = wf["jobs"]["update"]["steps"]
    commands = "\n".join(s.get("run", "") for s in steps)
    assert "dvc pull data/valchamps.db models/map_model.pkl" in commands
    assert "valchamps update" in commands
    assert "[skip ci]" in commands and "git push origin HEAD:main" in commands
    assert wf["permissions"] == {"contents": "write"}
    triggers = wf[True]  # YAML 1.1 reads the bare key `on` as True
    assert "workflow_dispatch" in triggers and triggers["schedule"]
