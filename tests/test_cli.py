from __future__ import annotations

from sqlalchemy import inspect
from typer.testing import CliRunner

from valchamps.cli import app
from valchamps.data import db


def test_init_db_creates_tables(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    monkeypatch.setenv("VALCHAMPS_DB_URL", url)
    result = CliRunner().invoke(app, ["init-db"])
    assert result.exit_code == 0, result.output
    tables = set(inspect(db.get_engine(url)).get_table_names())
    assert {"events", "teams", "matches", "maps", "vetoes", "player_map_stats"} <= tables


def test_inspect_event_reads_saved_page(tmp_path, monkeypatch):
    from tests.conftest import load_fixture
    from valchamps.data.scraper import cache_key

    cache = tmp_path / "cache"
    cache.mkdir()
    page = load_fixture("real/event_2501_americas_stage2_2025.html")
    (cache / cache_key("/event/2501")).write_text(page, encoding="utf-8")
    monkeypatch.setenv("VALCHAMPS_CACHE_DIR", str(cache))
    result = CliRunner().invoke(app, ["inspect-event", "2501"])
    assert result.exit_code == 0, result.output
    assert "'dates': 'Jul 18 \u2013 Sep 1, 2025'" in result.output
    assert "standings (8 rows)" in result.output
    assert "G2 Esports (team 11058)" in result.output


def test_inspect_event_without_saved_page(tmp_path, monkeypatch):
    monkeypatch.setenv("VALCHAMPS_CACHE_DIR", str(tmp_path))
    result = CliRunner().invoke(app, ["inspect-event", "1"])
    assert result.exit_code == 1
    assert "no saved page" in result.output
