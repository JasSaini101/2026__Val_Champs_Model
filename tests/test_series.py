from __future__ import annotations

import itertools
import math
import pickle
import random
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from tests.synthetic_history import MAP_POOL, make_history
from valchamps.cli import app
from valchamps.features import (
    FEATURE_COLUMNS,
    MapRecord,
    MatchRecord,
    build_feature_frame,
    load_events,
    load_records,
    team_regions,
)
from valchamps.features.params import FeatureParams
from valchamps.models import load_config, make_model, prepare
from valchamps.models.backtest import _merge
from valchamps.series import (
    FORMATS,
    SeriesParams,
    VetoTendencies,
    actual_veto,
    predict_series,
    run_series_backtest,
    score_distribution,
    series_win_prob,
    simulate_veto,
    veto_win_prob,
)
from valchamps.series.veto import PICK

POOL = MAP_POOL  # 7 maps
FLIP = {"pick_a": "pick_b", "pick_b": "pick_a", "decider": "decider"}


def random_tendencies(rng: random.Random) -> VetoTendencies:
    return VetoTendencies(
        picks={m: rng.choice([0, 0, 1, 3, 8]) for m in POOL},
        bans={m: rng.choice([0, 0, 2, 5, 12]) for m in POOL},
    )


# --- series odds ------------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.0, 0.1, 0.37, 0.5, 0.8, 1.0])
def test_bo3_with_constant_p(p):
    assert series_win_prob([p] * 3) == pytest.approx(p**2 * (3 - 2 * p))
    assert series_win_prob([p] * 5) == pytest.approx(p**3 * (10 - 15 * p + 6 * p**2))


def test_score_distribution():
    ps = [0.7, 0.4, 0.55]
    dist = score_distribution(ps)
    assert set(dist) == {(2, 0), (2, 1), (1, 2), (0, 2)}
    assert sum(dist.values()) == pytest.approx(1)
    assert dist[(2, 0)] == pytest.approx(0.7 * 0.4)
    assert dist[(0, 2)] == pytest.approx(0.3 * 0.6)
    # Brute force over every full 3-map outcome.
    brute = sum(
        math.prod(p if w else 1 - p for p, w in zip(ps, wins, strict=True))
        for wins in itertools.product([0, 1], repeat=3)
        if sum(wins) >= 2
    )
    assert series_win_prob(ps) == pytest.approx(brute)
    with pytest.raises(ValueError, match="odd"):
        score_distribution([0.5, 0.5])


# --- veto -------------------------------------------------------------------------------------


@pytest.mark.parametrize("best_of", [3, 5])
@pytest.mark.parametrize("prior", [0.0, 1.0])
def test_veto_outcome_probabilities_sum_to_one(best_of, prior):
    rng = random.Random(best_of)
    outcomes = simulate_veto(POOL, best_of, random_tendencies(rng), random_tendencies(rng), prior)
    assert sum(o.prob for o in outcomes) == pytest.approx(1)
    assert all(o.prob > 0 for o in outcomes)
    for o in outcomes:
        assert len(o.maps) == best_of
        assert len({m for m, _ in o.maps}) == best_of
        assert o.maps[-1][1] == "decider"
        contexts = [c for _, c in o.maps[:-1]]
        assert sorted(contexts) == sorted(["pick_a", "pick_b"] * ((best_of - 1) // 2))
    assert [o.prob for o in outcomes] == sorted((o.prob for o in outcomes), reverse=True)


@pytest.mark.parametrize("best_of", [3, 5])
def test_veto_matches_brute_force_enumeration(best_of):
    """The memoised enumeration equals walking all 7! choice orders by hand."""
    rng = random.Random(11)
    a, b, prior = random_tendencies(rng), random_tendencies(rng), 0.5
    steps = FORMATS[best_of]
    brute: defaultdict[tuple, float] = defaultdict(float)
    for first, second, labels in ((a, b, ("pick_a", "pick_b")), (b, a, ("pick_b", "pick_a"))):
        for order in itertools.permutations(POOL):
            prob, remaining, maps = 0.5, sorted(POOL), []
            for step, (action, m) in enumerate(zip(steps, order, strict=False)):
                team = (first, second)[step % 2]
                w = team.weights(action, remaining, prior)
                prob *= w[remaining.index(m)] / sum(w)
                remaining.remove(m)
                if action == PICK:
                    maps.append((m, labels[step % 2]))
            brute[(*maps, (remaining[0], "decider"))] += prob
    got = {o.maps: o.prob for o in simulate_veto(POOL, best_of, a, b, prior)}
    assert set(got) == {k for k, v in brute.items() if v > 0}
    for k, v in got.items():
        assert v == pytest.approx(brute[k])


def test_uniform_veto_without_history():
    outcomes = simulate_veto(POOL, 3, VetoTendencies(), VetoTendencies(), prior=1.0)
    # Any ordered (pick, pick, decider) of distinct maps, with either team picking first.
    assert len(outcomes) == 2 * 7 * 6 * 5
    assert all(o.prob == pytest.approx(1 / 420) for o in outcomes)


def test_veto_follows_history_without_prior():
    a = VetoTendencies(picks={"Bind": 1}, bans={"Lotus": 4})
    b = VetoTendencies(bans={"Split": 2})
    outcomes = simulate_veto(POOL, 3, a, b, prior=0.0)
    # Whoever goes first, A's first ban is Lotus and B's is Split. A always picks Bind unless
    # B (with no pick history) took it first.
    assert all({"Lotus", "Split"}.isdisjoint(m for m, _ in o.maps) for o in outcomes)
    assert all({("Bind", "pick_a"), ("Bind", "pick_b")} & set(o.maps) for o in outcomes)
    p_a_picks_bind = sum(o.prob for o in outcomes if ("Bind", "pick_a") in o.maps)
    assert p_a_picks_bind == pytest.approx(0.5 + 0.5 * 4 / 5)  # B picks among 5 maps


def test_veto_rejects_wrong_pool_or_format():
    with pytest.raises(ValueError, match="needs 7 maps"):
        simulate_veto(POOL[:6], 3, VetoTendencies(), VetoTendencies(), 1.0)
    with pytest.raises(ValueError, match="best-of-1"):
        simulate_veto(POOL, 1, VetoTendencies(), VetoTendencies(), 1.0)


@pytest.mark.parametrize("best_of", [3, 5])
def test_series_odds_are_symmetric(best_of):
    rng = random.Random(3)
    ta, tb = random_tendencies(rng), random_tendencies(rng)
    probs_a = {(m, c): rng.random() for m in POOL for c in ("pick_a", "pick_b", "decider")}
    probs_b = {(m, c): 1 - probs_a[(m, FLIP[c])] for m, c in probs_a}  # same maps, B's view
    p_a = veto_win_prob(simulate_veto(POOL, best_of, ta, tb, 1.0), probs_a)
    p_b = veto_win_prob(simulate_veto(POOL, best_of, tb, ta, 1.0), probs_b)
    assert p_a + p_b == pytest.approx(1)


# --- hypothetical rows ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def history(tmp_path_factory):
    from valchamps.data import db

    path = tmp_path_factory.mktemp("series") / "hist.db"
    engine = db.get_engine(f"sqlite:///{path}")
    db.init_db(engine)
    make_history(engine, teams_per_region=6, spread=250)
    return engine, load_records(engine), team_regions(engine), load_events(engine)


def _row_key(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["map_name", "perspective", *FEATURE_COLUMNS]].reset_index(drop=True)


def test_records_carry_the_veto_map_pool(history):
    _, records, _, _ = history
    assert all(sorted(r.map_pool) == sorted(POOL) for r in records)
    seq = actual_veto(records[0])
    assert [c for _, c in seq] == ["pick_a", "pick_b", "decider"]  # team1 vetoes first
    assert [m for m, _ in seq][: len(records[0].maps)] == [m.map_name for m in records[0].maps]


def test_hypothetical_rows_match_real_rows_and_leave_state_alone(history):
    """For a played match, the hypothetical row for (its map, its context) is its real row, and
    building hypothetical rows first changes nothing about the real rows."""
    _, records, regions, events = history
    real, _ = build_feature_frame(records, regions, events=events)
    captured: dict[int, pd.DataFrame] = {}

    def on_match(builder, match):
        if match.match_id % 7 == 0:
            captured[match.match_id] = builder.hypothetical_rows(
                *match.teams, match.date, match.map_pool, event_id=match.event_id,
                tier=match.tier, match_id=match.match_id,
            )  # fmt: skip

    with_hyp, _ = build_feature_frame(records, regions, events=events, on_match=on_match)
    pd.testing.assert_frame_equal(real, with_hyp)
    assert captured
    by_id = {r.match_id: r for r in records}
    for match_id, hyp in captured.items():
        assert len(hyp) == 7 * 3 * 2
        assert hyp["y"].isna().all()
        match = by_id[match_id]
        for m in match.maps:
            context = {match.team1_id: "pick_a", match.team2_id: "pick_b"}.get(
                m.picked_by, "decider"
            )
            want = real[real.game_id == m.game_id]
            got = hyp[(hyp.map_name == m.map_name) & (hyp.context == context)]
            pd.testing.assert_frame_equal(_row_key(got), _row_key(want), check_dtype=False)


def test_hypothetical_rows_only_use_earlier_matches(history):
    """Flipping every later result leaves a match's hypothetical rows unchanged."""
    _, records, regions, events = history
    cutoff = len(records) // 2
    target = records[cutoff]

    def capture(recs):
        out = {}

        def on_match(builder, match):
            if match.match_id == target.match_id:
                out["rows"] = builder.hypothetical_rows(*match.teams, match.date, POOL)

        build_feature_frame(recs, regions, events=events, on_match=on_match)
        return out["rows"]

    flipped = records[: cutoff + 1] + [
        MatchRecord(**{**r.__dict__, "maps": tuple(
            MapRecord(m.game_id, m.map_order, m.map_name, m.team2_rounds, m.team1_rounds,
                      m.picked_by) for m in r.maps)})
        for r in records[cutoff + 1 :]
    ]  # fmt: skip
    pd.testing.assert_frame_equal(capture(records), capture(flipped))
    pd.testing.assert_frame_equal(capture(records), capture(records[: cutoff + 1]))


def test_hypothetical_rows_across_a_new_season_do_not_touch_ratings(history):
    _, records, regions, events = history
    _, builder = build_feature_frame(records, regions, events=events)
    before = dict(builder.elo.rating)
    a, b = records[-1].teams
    rows = builder.hypothetical_rows(a, b, datetime(2031, 1, 1), POOL)
    assert dict(builder.elo.rating) == before
    assert builder.elo.season == records[-1].date.year
    # Ratings were regressed toward the mean for the new season, on a copy.
    assert abs(rows.elo_a.iloc[0] - 1500) < abs(before[a] - 1500)


# --- prediction and backtest ------------------------------------------------------------------

TEST_MODELS = {"backtest": {"first_test_date": "2025-01-01", "holdout_events": [6]}}


@pytest.fixture(scope="module")
def fitted(history):
    _, records, regions, events = history
    frame, builder = build_feature_frame(records, regions, events=events)
    model = make_model("linear", load_config(None)).fit(prepare(frame))
    return builder, model


def test_predicted_series_odds_are_symmetric(history, fitted):
    _, records, _, _ = history
    builder, model = fitted
    a, b = records[-1].teams
    date = records[-1].date + timedelta(days=1)
    for best_of in (3, 5):
        ab = predict_series(builder, model, a, b, date, POOL, best_of=best_of, tier="champions")
        ba = predict_series(builder, model, b, a, date, POOL, best_of=best_of, tier="champions")
        assert ab.p_a + ba.p_a == pytest.approx(1)
        assert ab.elo_p + ba.elo_p == pytest.approx(1)
        assert sum(ab.scores.values()) == pytest.approx(1)
        for (m, c), p in ab.map_probs.items():
            assert p + ba.map_probs[(m, FLIP[c])] == pytest.approx(1)


def test_series_backtest(history):
    _, records, regions, events = history
    config = _merge(load_config(None), TEST_MODELS)
    result = run_series_backtest(
        records, regions, FeatureParams(), events, "linear", config, veto_prior=1.0
    )
    preds = result.predictions
    assert set(preds.event_id) == {2, 3, 4, 5}  # same folds as the map model
    assert result.skipped == 0
    for col in ("p_veto", "p_actual", "p_elo"):
        assert preds[col].between(0, 1).all()
    assert set(result.metrics) == {"veto", "actual_maps", "elo"}
    assert result.metrics["actual_maps"]["overall"]["log_loss"] < math.log(2)
    assert (preds.actual_veto_prob > 0).all()


# --- CLI --------------------------------------------------------------------------------------


@pytest.fixture
def cli_env(history, fitted, tmp_path, monkeypatch):
    import valchamps.cli as cli

    engine, *_ = history
    _, model = fitted
    monkeypatch.setenv("VALCHAMPS_DB_URL", str(engine.url))
    monkeypatch.setattr(cli, "REPORTS_DIR", tmp_path / "reports")
    params = tmp_path / "params.yaml"
    params.write_text(yaml.safe_dump({"models": TEST_MODELS, "series": {"veto_prior": 2.0}}))
    model_path = tmp_path / "map_model.pkl"
    model_path.write_bytes(pickle.dumps(
        {"model": model, "name": "linear", "trained_through": "2026-12-31", "maps": 1}
    ))  # fmt: skip
    return params, model_path, tmp_path


def test_cli_series_backtest(cli_env):
    params, _, tmp = cli_env
    result = CliRunner().invoke(app, [
        "series-backtest", "--model", "linear", "--params-file", str(params), "--no-track",
    ])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "actual_maps  cross_region" in result.output
    assert "veto_prior 2.0" in result.output
    assert (tmp / "reports" / "series_backtest" / "linear" / "predictions.csv").exists()


def test_cli_predict_match(cli_env):
    params, model_path, _ = cli_env
    args = ["predict-match", "emea-0", "AM1", "--best-of", "5", "--model", str(model_path),
            "--params-file", str(params)]  # fmt: skip
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "emea-0 vs americas-1: Bo5" in result.output
    line = next(ln for ln in result.output.splitlines() if ln.startswith("P(emea-0 wins) ="))
    p_a, p_b = (float(x.split("=")[1]) for x in line.split("    "))
    assert p_a + p_b == pytest.approx(1, abs=1e-3)
    assert "likeliest vetoes" in result.output


def test_cli_predict_match_rejects_unknown_team(cli_env):
    _, model_path, _ = cli_env
    result = CliRunner().invoke(app, ["predict-match", "nobody", "AM1", "--model", str(model_path)])
    assert result.exit_code != 0
    assert "no team matches" in result.output


def test_repo_params_have_a_series_section():
    from valchamps.cli import DEFAULT_PARAMS

    assert SeriesParams.from_yaml(DEFAULT_PARAMS).veto_prior >= 0
    with pytest.raises(ValueError, match="unknown series params"):
        SeriesParams.from_dict({"veto_prir": 1})
