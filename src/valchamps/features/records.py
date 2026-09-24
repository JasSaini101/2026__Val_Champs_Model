"""Chronological match history, the input to every feature tracker.

The trackers never touch the database: they consume :class:`MatchRecord` objects in time order.
That keeps them easy to unit-test with hand-built histories and makes the point-in-time
guarantee (features only see earlier matches) a property of the loop in :mod:`.build`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd
from sqlalchemy import Engine, case, func, select

from valchamps.data import db

INTERNATIONAL_TIERS = frozenset({"international", "champions"})


@dataclass(frozen=True)
class EventInfo:
    event_id: int
    tier: str | None
    end_date: date | None
    # team_id -> (place, circuit points); only placed teams, usually the top 8.
    standings: dict[int, tuple[int, int | None]] = field(default_factory=dict)

    @property
    def is_international(self) -> bool:
        return self.tier in INTERNATIONAL_TIERS


@dataclass(frozen=True)
class MapRecord:
    game_id: int
    map_order: int
    map_name: str
    team1_rounds: int
    team2_rounds: int
    picked_by: int | None

    @property
    def team1_won(self) -> bool:
        return self.team1_rounds > self.team2_rounds


@dataclass(frozen=True)
class MatchRecord:
    match_id: int
    date: datetime
    event_id: int | None
    tier: str | None
    team1_id: int
    team2_id: int
    best_of: int | None = None
    stage: str | None = None
    maps: tuple[MapRecord, ...] = ()
    # (team_id, action, map_name) in veto order; action is "ban" or "pick".
    vetoes: tuple[tuple[int, str, str], ...] = ()
    # team_id -> {player_id: mean rating over the maps they played in this match}
    lineups: dict[int, dict[int, float | None]] = field(default_factory=dict)

    @property
    def is_international(self) -> bool:
        return self.tier in INTERNATIONAL_TIERS

    @property
    def teams(self) -> tuple[int, int]:
        return self.team1_id, self.team2_id

    def maps_won(self, team_id: int) -> int:
        won_as_1 = sum(m.team1_won for m in self.maps)
        return won_as_1 if team_id == self.team1_id else len(self.maps) - won_as_1

    def round_diff(self, team_id: int) -> int:
        diff = sum(m.team1_rounds - m.team2_rounds for m in self.maps)
        return diff if team_id == self.team1_id else -diff


def load_records(engine: Engine) -> list[MatchRecord]:
    """Completed matches with at least one finished map, oldest first."""
    with engine.connect() as conn:
        matches = pd.read_sql(
            select(
                db.matches.c.match_id, db.matches.c.date_utc, db.matches.c.event_id,
                db.matches.c.team1_id, db.matches.c.team2_id, db.matches.c.best_of,
                db.matches.c.stage, db.events.c.tier, db.events.c.start_date,
            )
            .select_from(db.matches.outerjoin(db.events))
            .where(db.matches.c.status == "completed"),
            conn,
        )  # fmt: skip
        maps = pd.read_sql(select(db.maps), conn)
        vetoes = pd.read_sql(select(db.vetoes).order_by(db.vetoes.c.step), conn)
        stats = pd.read_sql(
            select(
                db.player_map_stats.c.game_id, db.player_map_stats.c.player_id,
                db.player_map_stats.c.team_id, db.player_map_stats.c.rating, db.maps.c.match_id,
            ).select_from(db.player_map_stats.join(db.maps)),
            conn,
        )  # fmt: skip

    # Exhibition matches (e.g. "Showmatch: Override") are not competitive results.
    matches = matches[~matches.stage.fillna("").str.lower().str.startswith("showmatch")]
    maps = maps[maps.team1_rounds != maps.team2_rounds]
    maps_by_match = {mid: g.sort_values("map_order") for mid, g in maps.groupby("match_id")}
    vetoes_by_match = {mid: g for mid, g in vetoes.groupby("match_id")}
    lineups: dict[int, dict[int, dict[int, float | None]]] = defaultdict(dict)
    for (mid, tid, pid), g in stats.groupby(["match_id", "team_id", "player_id"]):
        rating = g.rating.mean()
        lineups[mid].setdefault(tid, {})[pid] = None if pd.isna(rating) else float(rating)

    records: list[MatchRecord] = []
    for row in matches.itertuples(index=False):
        game_rows = maps_by_match.get(row.match_id)
        if game_rows is None or game_rows.empty:
            continue
        date = row.date_utc if pd.notna(row.date_utc) else row.start_date
        if date is None or pd.isna(date):
            continue
        veto_rows = vetoes_by_match.get(row.match_id)
        records.append(
            MatchRecord(
                match_id=int(row.match_id),
                date=pd.Timestamp(date).to_pydatetime(),
                event_id=None if pd.isna(row.event_id) else int(row.event_id),
                tier=row.tier,
                team1_id=int(row.team1_id),
                team2_id=int(row.team2_id),
                best_of=None if pd.isna(row.best_of) else int(row.best_of),
                stage=row.stage,
                maps=tuple(
                    MapRecord(
                        game_id=int(m.game_id),
                        map_order=int(m.map_order),
                        map_name=m.map_name,
                        team1_rounds=int(m.team1_rounds),
                        team2_rounds=int(m.team2_rounds),
                        picked_by=None if pd.isna(m.picked_by) else int(m.picked_by),
                    )
                    for m in game_rows.itertuples(index=False)
                ),
                vetoes=tuple(
                    (int(v.team_id), v.action, v.map_name)
                    for v in veto_rows.itertuples(index=False)
                    if v.action in ("ban", "pick") and pd.notna(v.team_id)
                )
                if veto_rows is not None
                else (),
                lineups=lineups.get(int(row.match_id), {}),
            )
        )
    records.sort(key=lambda r: (r.date, r.match_id))
    return records


def load_events(engine: Engine) -> dict[int, EventInfo]:
    """Tier, end date and final placements of every event.

    If the event page gave no end date but every match of the event is finished, the date of
    its last match stands in (still schedule information, never a result).
    """
    with engine.connect() as conn:
        events = conn.execute(select(db.events.c.event_id, db.events.c.tier, db.events.c.end_date))
        placed = conn.execute(select(db.placements)).all()
        last_match = {
            r.event_id: (r.last, r.unfinished)
            for r in conn.execute(
                select(
                    db.matches.c.event_id,
                    func.max(db.matches.c.date_utc).label("last"),
                    func.sum(case((db.matches.c.status != "completed", 1), else_=0)).label(
                        "unfinished"
                    ),
                ).group_by(db.matches.c.event_id)
            )
        }
    standings: dict[int, dict[int, tuple[int, int | None]]] = defaultdict(dict)
    for p in placed:
        standings[p.event_id][p.team_id] = (p.place, p.circuit_points)
    infos = {}
    for e in events:
        end = e.end_date
        last, unfinished = last_match.get(e.event_id, (None, 1))
        if end is None and last is not None and not unfinished:
            end = pd.Timestamp(last).date()
        infos[e.event_id] = EventInfo(e.event_id, e.tier, end, standings.get(e.event_id, {}))
    return infos


def team_regions(engine: Engine) -> dict[int, str]:
    """Each team's home region: the region of the regional-league events it played most."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(db.matches.c.team1_id, db.matches.c.team2_id, db.events.c.region)
            .select_from(db.matches.join(db.events))
            .where(db.events.c.tier == "regional", db.events.c.region.is_not(None))
        ).all()
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    for t1, t2, region in rows:
        counts[t1][region] += 1
        counts[t2][region] += 1
    return {team: c.most_common(1)[0][0] for team, c in counts.items()}
