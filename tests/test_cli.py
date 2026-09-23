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
