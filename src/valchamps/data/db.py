"""Relational schema and idempotent writes.

SQLite is the default (a single file DVC can version); the same Core schema runs on Postgres by
pointing ``VALCHAMPS_DB_URL`` at it. All writes are upserts so re-ingesting a match is harmless.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    select,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.engine import Connection

from valchamps.data.models import Event, Match

metadata = MetaData()

events = Table(
    "events", metadata,
    Column("event_id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("tier", String),
    Column("region", String),
    Column("is_lan", Boolean),
    Column("start_date", Date),
    Column("end_date", Date),
    Column("location", String),
)  # fmt: skip

teams = Table(
    "teams", metadata,
    Column("team_id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("tag", String),
)  # fmt: skip

players = Table(
    "players", metadata,
    Column("player_id", Integer, primary_key=True),
    Column("handle", String, nullable=False),
)  # fmt: skip

matches = Table(
    "matches", metadata,
    Column("match_id", Integer, primary_key=True),
    Column("event_id", Integer, ForeignKey("events.event_id")),
    Column("stage", String),
    Column("date_utc", DateTime),
    Column("status", String, nullable=False),
    Column("best_of", Integer),
    Column("team1_id", Integer, ForeignKey("teams.team_id"), nullable=False),
    Column("team2_id", Integer, ForeignKey("teams.team_id"), nullable=False),
    Column("team1_score", Integer),
    Column("team2_score", Integer),
    Column("winner_id", Integer, ForeignKey("teams.team_id")),
    Column("scraped_at", DateTime, nullable=False),
)  # fmt: skip

maps = Table(
    "maps", metadata,
    Column("game_id", Integer, primary_key=True),
    Column("match_id", Integer, ForeignKey("matches.match_id"), nullable=False, index=True),
    Column("map_order", Integer, nullable=False),
    Column("map_name", String, nullable=False),
    Column("picked_by", Integer, ForeignKey("teams.team_id")),
    Column("team1_rounds", Integer, nullable=False),
    Column("team2_rounds", Integer, nullable=False),
    Column("team1_ct", Integer),
    Column("team1_t", Integer),
    Column("team2_ct", Integer),
    Column("team2_t", Integer),
    Column("winner_id", Integer, ForeignKey("teams.team_id")),
    Column("duration", String),
)  # fmt: skip

vetoes = Table(
    "vetoes", metadata,
    Column("match_id", Integer, ForeignKey("matches.match_id"), primary_key=True),
    Column("step", Integer, primary_key=True),
    Column("action", String, nullable=False),
    Column("map_name", String, nullable=False),
    Column("team_id", Integer, ForeignKey("teams.team_id")),
)  # fmt: skip

player_map_stats = Table(
    "player_map_stats", metadata,
    Column("game_id", Integer, ForeignKey("maps.game_id"), primary_key=True),
    Column("player_id", Integer, ForeignKey("players.player_id"), primary_key=True),
    Column("team_id", Integer, ForeignKey("teams.team_id"), nullable=False),
    Column("agent", String),
    Column("rating", Float),
    Column("acs", Float),
    Column("kills", Integer),
    Column("deaths", Integer),
    Column("assists", Integer),
    Column("kast", Float),
    Column("adr", Float),
    Column("hs_pct", Float),
    Column("first_kills", Integer),
    Column("first_deaths", Integer),
)  # fmt: skip

scrape_log = Table(
    "scrape_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("url_path", String, nullable=False),
    Column("ok", Boolean, nullable=False),
    Column("message", Text),
    Column("at", DateTime, nullable=False),
)  # fmt: skip


def get_engine(url: str) -> Engine:
    engine = create_engine(url, future=True)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn: Any, _record: Any) -> None:
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

    return engine


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)


def upsert(
    conn: Connection, table: Table, rows: Sequence[dict[str, Any]], *, update: bool = True
) -> None:
    """Insert rows; on a primary-key clash update non-key columns (or skip if not ``update``)."""
    if not rows:
        return
    dialect = {"sqlite": sqlite, "postgresql": postgresql}.get(conn.dialect.name)
    if dialect is None:
        raise NotImplementedError(f"upsert not supported for {conn.dialect.name}")
    keys = [c.name for c in table.primary_key.columns]
    stmt = dialect.insert(table).values(list(rows))
    updates = {c: stmt.excluded[c] for c in rows[0] if c not in keys} if update else {}
    stmt = (
        stmt.on_conflict_do_update(index_elements=keys, set_=updates)
        if updates
        else (stmt.on_conflict_do_nothing(index_elements=keys))
    )
    conn.execute(stmt)


def upsert_event(
    conn: Connection, ev: Event, *, tier: str | None = None, region: str | None = None,
    is_lan: bool | None = None,
) -> None:  # fmt: skip
    upsert(conn, events, [{
        "event_id": ev.event_id, "name": ev.name, "tier": tier, "region": region,
        "is_lan": is_lan, "start_date": ev.start_date, "end_date": ev.end_date,
        "location": ev.location,
    }])  # fmt: skip


def save_match(conn: Connection, match: Match, *, event_id: int | None = None) -> None:
    """Persist a parsed match and all its children. Safe to call repeatedly."""
    event_id = event_id if event_id is not None else match.event_id
    if event_id is not None:
        # Placeholder row if the event wasn't ingested first; never clobbers a real one.
        name = match.event_name or f"event-{event_id}"
        upsert(conn, events, [{"event_id": event_id, "name": name}], update=False)

    team_rows = [
        {"team_id": t.team_id, "name": t.name, "tag": t.tag} for t in (match.team1, match.team2)
    ]
    for row in team_rows:
        # Don't overwrite a known tag with None (e.g. when an upcoming match has no stats yet).
        if row["tag"] is None:
            existing = conn.execute(
                select(teams.c.tag).where(teams.c.team_id == row["team_id"])
            ).first()
            row["tag"] = existing.tag if existing else None
    upsert(conn, teams, team_rows)

    upsert(conn, matches, [{
        "match_id": match.match_id, "event_id": event_id, "stage": match.stage,
        "date_utc": match.date_utc, "status": match.status, "best_of": match.best_of,
        "team1_id": match.team1.team_id, "team2_id": match.team2.team_id,
        "team1_score": match.team1_score, "team2_score": match.team2_score,
        "winner_id": match.winner_id, "scraped_at": datetime.now(UTC).replace(tzinfo=None),
    }])  # fmt: skip

    upsert(conn, vetoes, [{
        "match_id": match.match_id, "step": v.step, "action": v.action,
        "map_name": v.map_name, "team_id": v.team_id,
    } for v in match.veto])  # fmt: skip

    for m in match.maps:
        winner = None
        if m.completed:
            winner = match.team1.team_id if m.team1_rounds > m.team2_rounds else match.team2.team_id
        upsert(conn, maps, [{
            "game_id": m.game_id, "match_id": match.match_id, "map_order": m.map_order,
            "map_name": m.map_name, "picked_by": m.picked_by, "team1_rounds": m.team1_rounds,
            "team2_rounds": m.team2_rounds, "team1_ct": m.team1_ct, "team1_t": m.team1_t,
            "team2_ct": m.team2_ct, "team2_t": m.team2_t, "winner_id": winner,
            "duration": m.duration,
        }])  # fmt: skip
        unique_players = {
            p.player_id: {"player_id": p.player_id, "handle": p.handle} for p in m.players
        }
        upsert(conn, players, list(unique_players.values()))
        upsert(conn, player_map_stats, [{
            "game_id": m.game_id, "player_id": p.player_id, "team_id": p.team_id,
            "agent": p.agent, "rating": p.rating, "acs": p.acs, "kills": p.kills,
            "deaths": p.deaths, "assists": p.assists, "kast": p.kast, "adr": p.adr,
            "hs_pct": p.hs_pct, "first_kills": p.first_kills, "first_deaths": p.first_deaths,
        } for p in m.players])  # fmt: skip


def log_scrape(conn: Connection, url_path: str, ok: bool, message: str | None = None) -> None:
    conn.execute(scrape_log.insert().values(
        url_path=url_path, ok=ok, message=message, at=datetime.now(UTC).replace(tzinfo=None),
    ))  # fmt: skip


def completed_match_ids(conn: Connection) -> set[int]:
    rows = conn.execute(select(matches.c.match_id).where(matches.c.status == "completed"))
    return {r.match_id for r in rows}
