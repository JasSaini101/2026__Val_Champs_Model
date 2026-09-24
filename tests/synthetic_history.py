"""Synthetic VCT seasons with known team strengths, written through the real DB layer.

Two regions play round-robin leagues; each season ends with an international event between
the top teams of each region. Map results are drawn from an Elo-style model of the true
strengths, so tests can check the ratings recover them.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import Engine

from valchamps.data import db
from valchamps.data.models import (
    Event,
    MapResult,
    Match,
    PlayerMapStats,
    Team,
    VetoStep,
)

MAP_POOL = ("Ascent", "Bind", "Haven", "Lotus", "Split", "Sunset", "Icebox")
REGIONS = {"americas": 0.0, "emea": 120.0}  # true region strength offsets


@dataclass
class Truth:
    strength: dict[int, float]  # team_id -> true strength incl. region offset
    region: dict[int, str]


def make_history(
    engine: Engine,
    seed: int = 7,
    teams_per_region: int = 6,
    seasons: tuple[int, ...] = (2025, 2026),
    spread: float = 120.0,
) -> Truth:
    """``spread`` is the standard deviation of true team strength (Elo points) within a region."""
    rng = random.Random(seed)
    strength: dict[int, float] = {}
    region_of: dict[int, str] = {}
    teams: dict[str, list[Team]] = {}
    roster: dict[int, list[int]] = {}
    next_player = itertools.count(10_000)
    for r_idx, (region, offset) in enumerate(REGIONS.items()):
        teams[region] = []
        for i in range(teams_per_region):
            tid = 100 * (r_idx + 1) + i
            teams[region].append(Team(tid, f"{region}-{i}", f"{region[:2].upper()}{i}"))
            strength[tid] = offset + rng.gauss(0, spread)
            region_of[tid] = region
            roster[tid] = [next(next_player) for _ in range(5)]

    match_ids = itertools.count(1)
    game_ids = itertools.count(1)
    event_ids = itertools.count(1)
    day = datetime(seasons[0], 1, 10, 12)

    def play(a: Team, b: Team, event_id: int, tier: str) -> None:
        nonlocal day
        day += timedelta(hours=20)
        p_a = 1 / (1 + 10 ** (-(strength[a.team_id] - strength[b.team_id]) / 400))
        pool = list(MAP_POOL)
        rng.shuffle(pool)
        veto = [
            VetoStep(1, "ban", pool[0], a.team_id),
            VetoStep(2, "ban", pool[1], b.team_id),
            VetoStep(3, "pick", pool[2], a.team_id),
            VetoStep(4, "pick", pool[3], b.team_id),
            VetoStep(5, "ban", pool[4], a.team_id),
            VetoStep(6, "ban", pool[5], b.team_id),
            VetoStep(7, "remains", pool[6], None),
        ]
        maps, won_a, won_b = [], 0, 0
        for order, (map_name, picker) in enumerate(
            ((pool[2], a.team_id), (pool[3], b.team_id), (pool[6], None)), start=1
        ):
            if max(won_a, won_b) == 2:
                break
            a_wins = rng.random() < p_a
            loser_rounds = rng.randint(0, 11)
            r_a, r_b = (13, loser_rounds) if a_wins else (loser_rounds, 13)
            won_a, won_b = won_a + a_wins, won_b + (not a_wins)
            gid = next(game_ids)
            players = [
                PlayerMapStats(pid, f"p{pid}", t.team_id, "Omen",
                               round(1.0 + strength[t.team_id] / 1000 + rng.gauss(0, 0.1), 2),
                               200, 15, 14, 5, 70, 140, 25, 2, 2)
                for t in (a, b) for pid in roster[t.team_id]
            ]  # fmt: skip
            maps.append(MapResult(gid, order, map_name, r_a, r_b, picker, players=players))
        match = Match(
            match_id=next(match_ids), event_id=event_id, event_name=None, stage=None,
            date_utc=day, status="completed", best_of=3, team1=a, team2=b,
            team1_score=won_a, team2_score=won_b, veto=veto, maps=maps,
        )  # fmt: skip
        with engine.begin() as conn:
            db.save_match(conn, match)

    for year in seasons:
        day = max(day, datetime(year, 1, 10, 12))
        if year != seasons[0]:  # one roster change per region between seasons
            for region in REGIONS:
                roster[teams[region][0].team_id][0] = next(next_player)
        for region, league in teams.items():
            eid = next(event_ids)
            with engine.begin() as conn:
                db.upsert_event(conn, Event(eid, f"{region} league {year}"), tier="regional",
                                region=region, is_lan=True)  # fmt: skip
            for a, b in itertools.combinations(league, 2):
                play(a, b, eid, "regional")
        eid = next(event_ids)
        with engine.begin() as conn:
            db.upsert_event(conn, Event(eid, f"Masters {year}"), tier="international",
                            region="international", is_lan=True)  # fmt: skip
        top = [t for r in REGIONS for t in sorted(teams[r], key=lambda t: -strength[t.team_id])[:3]]
        for a, b in itertools.combinations(top, 2):
            play(a, b, eid, "international")
    return Truth(strength=strength, region=region_of)


CHAMPIONS_EVENT = 99


def add_champions_event(engine: Engine, teams: list[int], played: int = 1) -> dict[str, list[int]]:
    """A Champions-style event: 16 ``teams`` in 4 groups, the first ``played`` openings finished.

    Upserts, so calling it again with a larger ``played`` finishes more opening matches.
    """
    groups = {g: teams[i::4] for i, g in enumerate("ABCD")}
    day = datetime(2027, 3, 1, 12)
    actions = ["ban", "ban", "pick", "pick", "ban", "ban", "remains"]
    with engine.begin() as conn:
        db.upsert_event(conn, Event(CHAMPIONS_EVENT, "Champions"), tier="champions",
                        region="international")  # fmt: skip
        n = 0
        for g, (a, b, c, d) in groups.items():
            for t1, t2 in ((a, b), (c, d)):
                n += 1
                done = n <= played
                veto = [VetoStep(i + 1, act, MAP_POOL[i],
                                 None if act == "remains" else (t1 if i % 2 == 0 else t2))
                        for i, act in enumerate(actions)]  # fmt: skip
                maps = [MapResult(900_000 + 10 * n + k, k + 1, MAP_POOL[2 + k], 13, 7, t1)
                        for k in (0, 1)]  # fmt: skip
                db.save_match(conn, Match(
                    match_id=90_000 + n, event_id=CHAMPIONS_EVENT, event_name="Champions",
                    stage=f"Group Stage: Opening ({g})", date_utc=day + timedelta(hours=3 * n),
                    status="completed" if done else "upcoming", best_of=3,
                    team1=Team(t1, f"t{t1}"), team2=Team(t2, f"t{t2}"),
                    team1_score=2 if done else None, team2_score=0 if done else None,
                    veto=veto if done else [], maps=maps if done else [],
                ))  # fmt: skip
    return groups
