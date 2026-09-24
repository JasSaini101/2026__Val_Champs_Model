"""Series odds between every pair of teams, computed once before a simulation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd

from valchamps.features.build import FeatureBuilder
from valchamps.models.base import MapModel
from valchamps.series.predict import map_probabilities
from valchamps.series.series import series_win_prob, veto_win_prob
from valchamps.series.veto import simulate_veto, tendencies


def _matrices(teams: Sequence[int], formats: Iterable[int], p_first_wins) -> dict[int, np.ndarray]:
    """Fill odds[bo][i, j] from ``p_first_wins(i, j, bo)`` for i < j; the rest by symmetry."""
    n = len(teams)
    out = {}
    for bo in formats:
        m = np.full((n, n), 0.5)
        for i, j in combinations(range(n), 2):
            m[i, j] = p_first_wins(i, j, bo)
            m[j, i] = 1.0 - m[i, j]
        out[bo] = m
    return out


def model_odds(
    builder: FeatureBuilder, model: MapModel, teams: Sequence[int], date: datetime,
    map_pool: Sequence[str], *, formats: Iterable[int] = (3, 5), veto_prior: float = 5.0,
    event_id: int | None = None, tier: str | None = None,
) -> dict[int, np.ndarray]:  # fmt: skip
    """Simulated veto + map model, as in ``predict-match``, for every pair (one model call)."""
    pairs = list(combinations(range(len(teams)), 2))
    rows = pd.concat([
        builder.hypothetical_rows(
            teams[i], teams[j], date, map_pool, event_id=event_id, tier=tier, match_id=k + 1
        )
        for k, (i, j) in enumerate(pairs)
    ])  # fmt: skip
    probs = map_probabilities(model, rows)
    key = {pair: k + 1 for k, pair in enumerate(pairs)}
    veto = {t: tendencies(builder.map_pool, t) for t in teams}

    def p(i: int, j: int, bo: int) -> float:
        outcomes = simulate_veto(map_pool, bo, veto[teams[i]], veto[teams[j]], veto_prior)
        return veto_win_prob(outcomes, probs[key[(i, j)]])

    return _matrices(teams, formats, p)


def elo_odds(
    builder: FeatureBuilder, teams: Sequence[int], formats: Iterable[int] = (3, 5)
) -> dict[int, np.ndarray]:
    """Raw Elo: the map-agnostic Elo probability on every map of the series."""
    return _matrices(
        teams, formats,
        lambda i, j, bo: series_win_prob([builder.elo.expected(teams[i], teams[j])] * bo),
    )  # fmt: skip
