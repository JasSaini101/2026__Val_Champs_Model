"""Series odds for a match that hasn't been played, from the current feature state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from valchamps.features.build import FeatureBuilder
from valchamps.models.base import MapModel, symmetric_predict
from valchamps.models.dataset import prepare
from valchamps.series.series import series_win_prob, veto_score_distribution, veto_win_prob
from valchamps.series.veto import VetoOutcome, simulate_veto, tendencies


def map_probabilities(
    model: MapModel, rows: pd.DataFrame
) -> dict[int, dict[tuple[str, str], float]]:
    """P(team_a wins) for every (map, context) in ``hypothetical_rows`` output, per match_id."""
    df = prepare(rows)
    df["p"] = symmetric_predict(model, df)
    first = df[df["perspective"] == 0]
    out: dict[int, dict[tuple[str, str], float]] = {}
    for match_id, m, context, p in zip(
        first["match_id"], first["map_name"], first["context"], first["p"], strict=True
    ):
        out.setdefault(int(match_id), {})[(m, context)] = float(p)
    return out


@dataclass(frozen=True)
class SeriesPrediction:
    team_a: int
    team_b: int
    best_of: int
    p_a: float  # P(team_a wins the series)
    scores: dict[tuple[int, int], float]  # (maps won by a, maps won by b) -> probability
    map_probs: dict[tuple[str, str], float]  # (map, context) -> P(team_a wins the map)
    vetoes: list[VetoOutcome]  # likeliest first
    elo_p: float  # raw Elo series probability (no map or veto information)


def predict_series(
    builder: FeatureBuilder, model: MapModel, team_a: int, team_b: int, date: datetime,
    map_pool: Sequence[str], *, best_of: int = 3, veto_prior: float = 5.0,
    event_id: int | None = None, tier: str | None = None,
) -> SeriesPrediction:  # fmt: skip
    rows = builder.hypothetical_rows(
        team_a, team_b, date, map_pool, best_of=best_of, event_id=event_id, tier=tier
    )
    map_probs = map_probabilities(model, rows)[0]
    vetoes = simulate_veto(
        map_pool, best_of, tendencies(builder.map_pool, team_a),
        tendencies(builder.map_pool, team_b), veto_prior,
    )  # fmt: skip
    return SeriesPrediction(
        team_a=team_a, team_b=team_b, best_of=best_of,
        p_a=veto_win_prob(vetoes, map_probs),
        scores=veto_score_distribution(vetoes, map_probs),
        map_probs=map_probs, vetoes=vetoes,
        elo_p=series_win_prob([builder.elo.expected(team_a, team_b)] * best_of),
    )  # fmt: skip
