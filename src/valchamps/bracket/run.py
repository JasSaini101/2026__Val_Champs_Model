"""Forecast an event end to end: history, bracket, pairwise odds, simulation."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, select

from valchamps.bracket.event import fixed_results, group_openings
from valchamps.bracket.format import build_bracket, load_format
from valchamps.bracket.odds import elo_odds, model_odds
from valchamps.bracket.simulate import FixedResult, SimulationResult, simulate
from valchamps.data import db
from valchamps.features import (
    FeatureParams,
    MatchRecord,
    build_feature_frame,
    load_events,
    load_records,
    team_regions,
)

ODDS_SOURCES = ("model", "elo")


def current_map_pool(records: Sequence[MatchRecord]) -> tuple[str, ...]:
    """The seven maps of the most recent vetoed match: the pool currently in play."""
    return next((r.map_pool for r in reversed(records) if len(r.map_pool) == 7), ())


def prediction_date(records: Sequence[MatchRecord]) -> datetime:
    """Now, or just after the last known match if the clock is behind the data."""
    return max(datetime.now(UTC).replace(tzinfo=None), records[-1].date)


def results_fingerprint(results: Sequence[FixedResult]) -> str:
    """Identifies the set of finished results a forecast was made from."""
    key = sorted((r.stage, min(r.team_a, r.team_b), max(r.team_a, r.team_b), r.winner)
                 for r in results)  # fmt: skip
    return hashlib.sha256(repr(key).encode()).hexdigest()[:16]


@dataclass
class EventForecast:
    event: int
    table: pd.DataFrame  # one row per team: every final standing plus playoffs/top4/final/title
    sim: SimulationResult
    results: list[FixedResult]
    names: dict[int, str]
    source: str  # where the series odds came from
    runs: int
    seed: int
    as_of: datetime  # date of the last finished match in the data

    @property
    def fingerprint(self) -> str:
        return results_fingerprint(self.results)

    def meta(self) -> dict:
        return {
            "event": self.event, "odds": self.source, "runs": self.runs, "seed": self.seed,
            "as_of": self.as_of.isoformat(), "fixed_results": len(self.sim.used_results),
            "results_fingerprint": self.fingerprint,
        }  # fmt: skip


def forecast_event(
    engine: Engine, event: int, *, bracket_file: Path, odds: str = "model",
    model_path: Path | None = None, feature_params: FeatureParams | None = None,
    veto_prior: float = 5.0, map_pool: Sequence[str] | None = None, runs: int = 100_000,
    seed: int = 0,
) -> EventForecast:  # fmt: skip
    """Simulate ``event`` from everything in the database. Raises ValueError if it can't."""
    if odds not in ODDS_SOURCES:
        raise ValueError(f"odds must be one of {ODDS_SOURCES}, got {odds!r}")
    records = load_records(engine)
    if not records:
        raise ValueError("no completed matches in the database")
    events = load_events(engine)
    openings = group_openings(engine, event)
    if not openings:
        raise ValueError(f"no group-stage opening matches found for event {event}")
    bracket = build_bracket(load_format(bracket_file), openings)
    teams = sorted({t for pairs in openings.values() for pair in pairs for t in pair})
    formats = sorted({m.best_of for m in bracket})

    _, builder = build_feature_frame(records, team_regions(engine), feature_params, events)
    if odds == "model":
        if model_path is None:
            raise ValueError("odds='model' needs a trained map model")
        bundle = pickle.loads(model_path.read_bytes())
        pool = tuple(map_pool) if map_pool else current_map_pool(records)
        if len(pool) != 7:
            raise ValueError(f"need 7 maps in the pool, got {list(pool)}")
        info = events.get(event)
        matrices = model_odds(
            builder, bundle["model"], teams, prediction_date(records), pool, formats=formats,
            veto_prior=veto_prior, event_id=event, tier=info.tier if info else None,
        )  # fmt: skip
        source = f"map model {bundle['name']} (trained through {bundle['trained_through']})"
    else:
        matrices = elo_odds(builder, teams, formats)
        source = "raw Elo"

    results = fixed_results(records, event)
    sim = simulate(bracket, matrices, teams, runs, results, seed=seed)
    with engine.connect() as conn:
        names = dict(conn.execute(select(db.teams.c.team_id, db.teams.c.name)).all())
    table = sim.to_frame(names)
    group_of = {t: g for g, pairs in openings.items() for pair in pairs for t in pair}
    table.insert(2, "group", table["team_id"].map(group_of))
    return EventForecast(event, table, sim, results, names, source, runs, seed, records[-1].date)
