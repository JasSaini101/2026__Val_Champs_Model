"""Monte Carlo simulation of a bracket from pairwise series odds.

All runs are simulated at once: each match slot holds an array with one team per run, and a
match is one vectorised draw. A match already played is fixed to its real result in every run
where the same two teams meet at that stage.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from valchamps.bracket.format import MatchSlot, place_order


@dataclass(frozen=True)
class FixedResult:
    stage: str
    team_a: int
    team_b: int
    winner: int


@dataclass
class SimulationResult:
    teams: list[int]
    runs: int
    places: dict[str, np.ndarray]  # standing -> P(each team finishes there), aligned with teams
    used_results: set[FixedResult] = field(default_factory=set)
    unused_results: list[FixedResult] = field(default_factory=list)
    # match id -> (winners, losers) as team ids, one per run
    outcomes: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def to_frame(self, names: dict[int, str] | None = None) -> pd.DataFrame:
        """One row per team: P(each final standing) and summary columns, best title odds first."""
        labels = sorted(self.places, key=place_order)
        df = pd.DataFrame({label: self.places[label] for label in labels})
        df.insert(0, "team_id", self.teams)
        df.insert(1, "team", [(names or {}).get(t, str(t)) for t in self.teams])
        top = lambda n: df[[c for c in labels if place_order(c) <= n]].sum(axis=1)  # noqa: E731
        df["title"] = df["1"] if "1" in labels else 0.0
        df["final"] = top(2)
        df["top4"] = top(4)
        df["top8"] = top(8)
        return df.sort_values(["title", "final", "top8"], ascending=False).reset_index(drop=True)


def simulate(
    bracket: list[MatchSlot], odds: dict[int, np.ndarray], teams: list[int], runs: int,
    results: list[FixedResult] | tuple[FixedResult, ...] = (), seed: int = 0,
) -> SimulationResult:  # fmt: skip
    """Play ``bracket`` ``runs`` times.

    ``odds[best_of][i, j]`` is P(teams[i] beats teams[j]) in a best-of series.
    """
    index = {t: i for i, t in enumerate(teams)}
    rng = np.random.default_rng(seed)
    by_stage: defaultdict[str, list[FixedResult]] = defaultdict(list)
    for r in results:
        by_stage[r.stage].append(r)

    slots: dict[str, np.ndarray] = {}  # "W:<id>" / "L:<id>" -> team index per run
    counts: defaultdict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(teams)))
    used: set[FixedResult] = set()
    outcomes = {}

    def resolve(slot: str) -> np.ndarray:
        if slot.startswith("T:"):
            return np.full(runs, index[int(slot[2:])])
        return slots[slot]

    for m in bracket:
        a, b = resolve(m.a), resolve(m.b)
        if m.best_of not in odds:
            raise ValueError(f"no best-of-{m.best_of} odds for {m.id}")
        a_wins = rng.random(runs) < odds[m.best_of][a, b]
        for r in by_stage.get(m.stage, []):
            ra, rb, rw = index.get(r.team_a), index.get(r.team_b), index.get(r.winner)
            if ra is None or rb is None:
                continue
            same = (a == ra) & (b == rb)
            swapped = (a == rb) & (b == ra)
            if same.any() or swapped.any():
                used.add(r)
            a_wins[same] = rw == ra
            a_wins[swapped] = rw == rb
        winners, losers = np.where(a_wins, a, b), np.where(a_wins, b, a)
        slots[f"W:{m.id}"], slots[f"L:{m.id}"] = winners, losers
        outcomes[m.id] = (np.asarray(teams)[winners], np.asarray(teams)[losers])
        for place, who in ((m.loser_place, losers), (m.winner_place, winners)):
            if place is not None:
                counts[place] += np.bincount(who, minlength=len(teams))
    return SimulationResult(
        teams=list(teams), runs=runs, places={k: v / runs for k, v in counts.items()},
        used_results=used, unused_results=[r for r in results if r not in used],
        outcomes=outcomes,
    )  # fmt: skip
