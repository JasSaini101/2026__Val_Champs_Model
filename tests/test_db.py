from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from tests.conftest import load_fixture
from valchamps.data import db
from valchamps.data.parser import parse_match


def count(conn, table):
    return conn.execute(select(func.count()).select_from(table)).scalar_one()


@pytest.fixture
def completed():
    return parse_match(load_fixture("match_378829_completed.html"), 378829)


def test_save_match_populates_all_tables(engine, completed):
    with engine.begin() as conn:
        db.save_match(conn, completed)
    with engine.connect() as conn:
        assert count(conn, db.events) == 1
        assert count(conn, db.teams) == 2
        assert count(conn, db.matches) == 1
        assert count(conn, db.maps) == 3
        assert count(conn, db.vetoes) == 7
        assert count(conn, db.players) == 10
        assert count(conn, db.player_map_stats) == 30
        row = conn.execute(select(db.matches)).one()
        assert (row.winner_id, row.status, row.best_of) == (2593, "completed", 3)
        winners = conn.execute(select(db.maps.c.winner_id).order_by(db.maps.c.map_order)).scalars()
        assert list(winners) == [2593, 1001, 2593]


def test_save_match_is_idempotent(engine, completed):
    for _ in range(2):
        with engine.begin() as conn:
            db.save_match(conn, completed)
    with engine.connect() as conn:
        assert count(conn, db.matches) == 1
        assert count(conn, db.player_map_stats) == 30


def test_upcoming_match_keeps_known_team_tag(engine, completed):
    upcoming = parse_match(load_fixture("match_378830_upcoming.html"), 378830)
    with engine.begin() as conn:
        db.save_match(conn, completed)
        db.save_match(conn, upcoming)
    with engine.connect() as conn:
        tag = conn.execute(select(db.teams.c.tag).where(db.teams.c.team_id == 2593)).scalar_one()
        assert tag == "FNC"
        assert db.completed_match_ids(conn) == {378829}


def test_foreign_keys_enforced(engine):
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(db.maps.insert().values(
            game_id=1, match_id=999, map_order=1, map_name="Bind", team1_rounds=13, team2_rounds=5,
        ))  # fmt: skip


def test_match_does_not_clobber_existing_event_name(engine, completed):
    from valchamps.data.models import Event

    with engine.begin() as conn:
        db.upsert_event(conn, Event(2097, "Champions 2024 (curated)"), tier="champions")
        db.save_match(conn, completed)
    with engine.connect() as conn:
        row = conn.execute(select(db.events)).one()
        assert (row.name, row.tier) == ("Champions 2024 (curated)", "champions")
