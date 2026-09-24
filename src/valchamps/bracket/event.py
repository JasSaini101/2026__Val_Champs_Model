"""The state of an event from the database: its groups and the results so far."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence

from sqlalchemy import Engine, select

from valchamps.bracket.simulate import FixedResult
from valchamps.data import db
from valchamps.features.records import MatchRecord

_OPENING = re.compile(r"^Group Stage: Opening \((\w+)\)$")


def group_openings(engine: Engine, event_id: int) -> dict[str, tuple[tuple[int, int], ...]]:
    """Each group's opening pairs (team ids), from the event's scheduled or played matches."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(db.matches.c.match_id, db.matches.c.stage, db.matches.c.team1_id,
                   db.matches.c.team2_id)
            .where(db.matches.c.event_id == event_id)
            .order_by(db.matches.c.match_id)
        ).all()  # fmt: skip
    out: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in rows:
        m = _OPENING.match(r.stage or "")
        if m:
            out[m.group(1)].append((r.team1_id, r.team2_id))
    bad = {g: len(p) for g, p in out.items() if len(p) != 2}
    if bad:
        raise ValueError(f"expected 2 opening matches per group, got {bad}")
    return {g: tuple(p) for g, p in sorted(out.items())}


def fixed_results(records: Sequence[MatchRecord], event_id: int) -> list[FixedResult]:
    """Finished series of the event, as (stage, teams, winner)."""
    out = []
    for r in records:
        if r.event_id != event_id or not r.stage:
            continue
        won = r.maps_won(r.team1_id)
        lost = len(r.maps) - won
        if won != lost:
            out.append(FixedResult(r.stage, r.team1_id, r.team2_id,
                                   r.team1_id if won > lost else r.team2_id))  # fmt: skip
    return out
