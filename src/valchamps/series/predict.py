"""Series odds for a match that hasn't been played, from the current feature state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import Any

import pandas as pd

from valchamps.features.build import FeatureBuilder
from valchamps.models.base import MapModel, symmetric_predict
from valchamps.models.dataset import prepare
from valchamps.series.series import series_win_prob, veto_score_distribution, veto_win_prob
from valchamps.series.veto import (
    VetoOutcome,
    map_play_probabilities,
    simulate_veto,
    tendencies,
)


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
    return _prediction(builder, team_a, team_b, best_of, map_probs, vetoes)


def predict_matchups(
    builder: FeatureBuilder, model: MapModel, teams: Sequence[int], date: datetime,
    map_pool: Sequence[str], *, formats: Sequence[int] = (3, 5), veto_prior: float = 5.0,
    event_id: int | None = None, tier: str | None = None,
) -> dict[tuple[int, int, int], SeriesPrediction]:  # fmt: skip
    """``predict_series`` for every pair of ``teams`` (in the order given) and every format.

    Keyed by (team_a, team_b, best_of). All maps of all pairs are scored in one model call.
    """
    pairs = list(combinations(teams, 2))
    rows = pd.concat([
        builder.hypothetical_rows(a, b, date, map_pool, event_id=event_id, tier=tier,
                                  match_id=k + 1)
        for k, (a, b) in enumerate(pairs)
    ])  # fmt: skip
    probs = map_probabilities(model, rows)
    veto = {t: tendencies(builder.map_pool, t) for t in teams}
    return {
        (a, b, bo): _prediction(
            builder,
            a,
            b,
            bo,
            probs[k + 1],
            simulate_veto(map_pool, bo, veto[a], veto[b], veto_prior),
        )
        for k, (a, b) in enumerate(pairs)
        for bo in formats
    }


def _prediction(
    builder: FeatureBuilder, team_a: int, team_b: int, best_of: int,
    map_probs: dict[tuple[str, str], float], vetoes: list[VetoOutcome],
) -> SeriesPrediction:  # fmt: skip
    return SeriesPrediction(
        team_a=team_a, team_b=team_b, best_of=best_of,
        p_a=veto_win_prob(vetoes, map_probs),
        scores=veto_score_distribution(vetoes, map_probs),
        map_probs=map_probs, vetoes=vetoes,
        elo_p=series_win_prob([builder.elo.expected(team_a, team_b)] * best_of),
    )  # fmt: skip


def prediction_payload(
    pred: SeriesPrediction, pool: Sequence[str], names: dict[int, str], vetoes: int = 5
) -> dict[str, Any]:
    """A prediction as JSON: the body of the API's ``/predict`` (without model metadata)."""
    played = map_play_probabilities(pred.vetoes)
    return {
        "team_a": {"team_id": pred.team_a, "name": names.get(pred.team_a, str(pred.team_a))},
        "team_b": {"team_id": pred.team_b, "name": names.get(pred.team_b, str(pred.team_b))},
        "best_of": pred.best_of,
        "p_a": pred.p_a,
        "p_b": 1 - pred.p_a,
        "elo_p_a": pred.elo_p,
        "scores": [
            {"a": a, "b": b, "p": p}
            for (a, b), p in sorted(pred.scores.items(), key=lambda kv: kv[0][1] - kv[0][0])
        ],
        "maps": [
            {
                "map": m,
                "pick_a": pred.map_probs[(m, "pick_a")],
                "pick_b": pred.map_probs[(m, "pick_b")],
                "decider": pred.map_probs[(m, "decider")],
                "in_series": played.get(m, 0.0),
            }
            for m in sorted(pool, key=lambda m: -played.get(m, 0.0))
        ],
        "vetoes": [
            {"p": o.prob, "maps": [{"map": m, "picked_by": c} for m, c in o.maps]}
            for o in pred.vetoes[:vetoes]
        ],
        "map_pool": list(pool),
    }
