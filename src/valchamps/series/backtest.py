"""Walk-forward series backtest, on the same folds as the map model.

History is replayed once. Before each Bo3/Bo5 in a test event the pre-match state gives the
match's hypothetical rows (every map of its pool in every pick context), its simulated veto and
the raw Elo probability. Then, per fold, the map model is trained on maps before the event and
every series in it is scored three ways:

* ``veto``: averaged over the simulated veto (what a prediction of an upcoming match uses)
* ``actual_maps``: on the maps the veto actually produced (unplayed decider included), in order
* ``elo``: the Elo system's map-agnostic probability on every map, no model or veto
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from valchamps.features.build import FeatureBuilder, build_feature_frame
from valchamps.features.params import FeatureParams
from valchamps.features.records import EventInfo, MatchRecord
from valchamps.models.backtest import make_model
from valchamps.models.dataset import prepare, walk_forward_folds
from valchamps.models.metrics import evaluate
from valchamps.series.predict import map_probabilities
from valchamps.series.series import series_win_prob, veto_win_prob
from valchamps.series.veto import FORMATS, PICK, MapSequence, VetoOutcome, simulate_veto, tendencies

METHODS = {"veto": "p_veto", "actual_maps": "p_actual", "elo": "p_elo"}


def actual_veto(match: MatchRecord) -> MapSequence | None:
    """The maps the veto produced, in play order, from team1's view; None if it's incomplete."""
    steps = FORMATS.get(match.best_of or 0)
    if steps is None or len(match.map_pool) != len(steps) + 1:
        return None
    if tuple(action for _, action, _ in match.vetoes) != steps:
        return None
    first = match.vetoes[0][0]
    if first not in match.teams or any(
        (team == first) != (i % 2 == 0) for i, (team, _, _) in enumerate(match.vetoes)
    ):
        return None
    chosen = {m for _, _, m in match.vetoes}
    deciders = [m for m in match.map_pool if m not in chosen]
    if len(deciders) != 1 or len(chosen) != len(steps):
        return None
    picks = tuple(
        (m, "pick_a" if team == match.team1_id else "pick_b")
        for team, action, m in match.vetoes
        if action == PICK
    )
    return (*picks, (deciders[0], "decider"))


def series_winner_is_team1(match: MatchRecord) -> bool | None:
    need = (match.best_of or 0) // 2 + 1
    won = match.maps_won(match.team1_id)
    lost = len(match.maps) - won
    if won == need and lost < need:
        return True
    if lost == need and won < need:
        return False
    return None  # unfinished or forfeited


@dataclass
class Case:
    """A test series and everything known about it before it was played."""

    match: MatchRecord
    y: int  # 1 if team1 won
    cross_region: bool
    elo_map_p: float
    simulated: list[VetoOutcome]
    actual: MapSequence
    rows: pd.DataFrame


def collect_cases(
    records: Sequence[MatchRecord], regions: dict[int, str], params: FeatureParams,
    events: dict[int, EventInfo] | None, first_test_date: str, veto_prior: float,
) -> tuple[pd.DataFrame, list[Case]]:  # fmt: skip
    """Replay history; return the map training table and a Case for every scorable series."""
    start = pd.Timestamp(first_test_date).to_pydatetime()
    cases: list[Case] = []

    def on_match(builder: FeatureBuilder, match: MatchRecord) -> None:
        if match.date < start or match.best_of not in FORMATS:
            return
        actual = actual_veto(match)
        y = series_winner_is_team1(match)
        if actual is None or y is None:
            return
        a, b = match.teams
        cases.append(Case(
            match=match, y=int(y),
            cross_region=bool(regions.get(a) and regions.get(b)
                              and regions.get(a) != regions.get(b)),
            elo_map_p=builder.elo.expected(a, b),
            simulated=simulate_veto(
                match.map_pool, match.best_of, tendencies(builder.map_pool, a),
                tendencies(builder.map_pool, b), veto_prior,
            ),
            actual=actual,
            rows=builder.hypothetical_rows(
                a, b, match.date, match.map_pool, best_of=match.best_of,
                event_id=match.event_id, tier=match.tier, match_id=match.match_id,
            ),
        ))  # fmt: skip

    frame, _ = build_feature_frame(records, regions, params, events, on_match=on_match)
    return frame, cases


@dataclass
class SeriesResult:
    model: str
    predictions: pd.DataFrame  # one row per series
    metrics: dict[str, dict[str, dict[str, float]]]  # method -> segment -> metric
    skipped: int = 0  # series in test events without a usable veto or result
    fold_counts: list[dict[str, Any]] = field(default_factory=list)


def run_series_backtest(
    records: Sequence[MatchRecord], regions: dict[int, str], feature_params: FeatureParams,
    events: dict[int, EventInfo] | None, model_name: str, config: dict[str, Any],
    veto_prior: float,
) -> SeriesResult:  # fmt: skip
    bt = config["backtest"]
    frame, cases = collect_cases(
        records, regions, feature_params, events, bt["first_test_date"], veto_prior
    )
    df = prepare(frame)
    folds = walk_forward_folds(df, bt["first_test_date"], bt["holdout_events"])
    if not folds:
        raise ValueError("no backtest folds: check first_test_date and holdout_events")
    fold_events = {f.event_id for f in folds}
    skipped = sum(
        1 for r in records
        if r.event_id in fold_events and r.best_of in FORMATS
        and (actual_veto(r) is None or series_winner_is_team1(r) is None)
    )  # fmt: skip

    out, fold_counts = [], []
    for fold in folds:
        fold_cases = [c for c in cases if c.match.event_id == fold.event_id]
        if not fold_cases:
            continue
        model = make_model(model_name, config).fit(df.iloc[fold.train_idx])
        probs = map_probabilities(model, pd.concat([c.rows for c in fold_cases]))
        for c in fold_cases:
            m, mp = c.match, probs[c.match.match_id]
            veto_lookup = {o.maps: o.prob for o in c.simulated}
            out.append({
                "fold": fold.name, "event_id": m.event_id, "match_id": m.match_id,
                "date": m.date, "team_a": m.team1_id, "team_b": m.team2_id,
                "best_of": m.best_of, "is_international": m.is_international,
                "cross_region": c.cross_region, "y": c.y,
                "p_veto": veto_win_prob(c.simulated, mp),
                "p_actual": series_win_prob([mp[x] for x in c.actual]),
                "p_elo": series_win_prob([c.elo_map_p] * m.best_of),
                "actual_veto_prob": veto_lookup.get(c.actual, 0.0),
                "actual_maps": " / ".join(f"{name} ({ctx})" for name, ctx in c.actual),
            })  # fmt: skip
        fold_counts.append({"fold": fold.name, "series": len(fold_cases)})
    predictions = pd.DataFrame(out)
    metrics = {
        method: evaluate(predictions.assign(p=predictions[col])) for method, col in METHODS.items()
    }
    return SeriesResult(model_name, predictions, metrics, skipped, fold_counts)


def veto_summary(predictions: pd.DataFrame) -> dict[str, float]:
    """How well the simulated veto anticipated the real one.

    ``log_lik`` is the mean log probability given to the exact real map sequence (maps, order
    and pickers). ``uniform_log_lik`` is the same for a veto where every choice is a coin flip:
    then all orderings of distinct maps are equally likely, and half of them have the wrong
    team picking first.
    """
    p = predictions["actual_veto_prob"].clip(lower=1e-12)
    uniform = predictions["best_of"].map(lambda b: math.log(0.5 / math.perm(7, b)))
    return {"log_lik": float(p.map(math.log).mean()), "uniform_log_lik": float(uniform.mean())}


def format_series_table(result: SeriesResult) -> str:
    header = (
        f"{'method':<13}{'segment':<15}{'series':>7}{'log_loss':>10}{'brier':>8}"
        f"{'acc':>7}{'ece':>7}"
    )
    lines = [header, "-" * len(header)]
    for method, segments in result.metrics.items():
        for segment, m in segments.items():
            lines.append(
                f"{method:<13}{segment:<15}{m['n']:>7}{m['log_loss']:>10.4f}{m['brier']:>8.4f}"
                f"{m['accuracy']:>7.3f}{m['ece']:>7.3f}"
            )
    return "\n".join(lines)
