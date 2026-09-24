from __future__ import annotations

import itertools
import math
import pickle

import numpy as np
import pytest
from typer.testing import CliRunner

from tests.synthetic_history import MAP_POOL, add_champions_event, make_history
from valchamps.bracket import (
    FixedResult,
    MatchSlot,
    build_bracket,
    elo_odds,
    fixed_results,
    group_openings,
    load_format,
    model_odds,
    simulate,
    validate,
)
from valchamps.cli import DEFAULT_BRACKET, app
from valchamps.data import db
from valchamps.features import build_feature_frame, load_events, load_records, team_regions
from valchamps.models import load_config, make_model, prepare

CONFIG = load_format(DEFAULT_BRACKET)
TEAMS = list(range(1, 17))
OPENINGS = {g: ((1 + 4 * i, 2 + 4 * i), (3 + 4 * i, 4 + 4 * i)) for i, g in enumerate("ABCD")}
PLACE_SIZES = {"1": 1, "2": 1, "3": 1, "4": 1, "5-6": 2, "7-8": 2, "9-12": 4, "13-16": 4}


def random_odds(n: int, seed: int = 0) -> dict[int, np.ndarray]:
    """Consistent random series odds: P[i, j] + P[j, i] = 1."""
    rng = np.random.default_rng(seed)
    out = {}
    for bo in (3, 5):
        m = rng.uniform(0.1, 0.9, size=(n, n))
        out[bo] = np.triu(m, 1) + np.tril(1 - m.T, -1) + np.eye(n) * 0.5
    return out


# --- format -----------------------------------------------------------------------------------


def test_champions_format_builds():
    bracket = build_bracket(CONFIG, OPENINGS)
    assert len(bracket) == 34  # 20 group matches + 14 playoff matches, as on vlr.gg
    assert [m.id for m in bracket if m.best_of == 5] == ["LF", "GF"]
    assert {m.stage for m in bracket} >= {"Group Stage: Opening (A)", "Playoffs: Grand Final"}
    uqf1 = next(m for m in bracket if m.id == "UQF1")
    assert (uqf1.a, uqf1.b) == ("W:A-W", "W:D-D")  # A1 vs D2


def test_format_validation():
    with pytest.raises(ValueError, match="before that match"):
        validate([MatchSlot("X", "s", "W:Y", "T:1"), MatchSlot("Y", "s", "T:2", "T:3")])
    with pytest.raises(ValueError, match="loser of F goes to 0 places"):
        validate([MatchSlot("F", "s", "T:1", "T:2", winner_place="1")])
    with pytest.raises(ValueError, match="no opening matches"):
        build_bracket(CONFIG, {"A": OPENINGS["A"]})


# --- simulation -------------------------------------------------------------------------------


def test_every_run_has_one_team_in_each_place():
    sim = simulate(build_bracket(CONFIG, OPENINGS), random_odds(16), TEAMS, runs=5_000)
    assert {k: round(v.sum(), 9) for k, v in sim.places.items()} == PLACE_SIZES
    per_team = sum(sim.places.values())
    assert np.allclose(per_team, 1.0)  # every team finishes somewhere exactly once
    table = sim.to_frame()
    assert table["title"].sum() == pytest.approx(1)
    assert table["top8"].sum() == pytest.approx(8)
    assert (table["title"] <= table["final"]).all() and (table["top4"] <= table["top8"]).all()


def _group_and_final() -> list[MatchSlot]:
    config = {
        "groups": {"names": ["A"], "elimination_place": "3-4", "decider_place": "3-4"},
        "playoffs": [{"id": "F", "stage": "final", "a": "A1", "b": "A2",
                      "loser_place": "2", "winner_place": "1"}],
    }  # fmt: skip
    return build_bracket(config, {"A": ((1, 2), (3, 4))})


def test_toy_bracket_matches_exact_odds():
    """A GSL group plus a final between its top two, against brute force over all outcomes."""
    teams = [1, 2, 3, 4]
    odds = random_odds(4, seed=5)
    p = odds[3]
    exact = np.zeros(4)
    bracket = _group_and_final()
    for wins in itertools.product([True, False], repeat=len(bracket)):
        slot, prob = {}, 1.0
        for m, a_wins in zip(bracket, wins, strict=True):
            a = int(m.a[2:]) - 1 if m.a.startswith("T:") else slot[m.a]
            b = int(m.b[2:]) - 1 if m.b.startswith("T:") else slot[m.b]
            prob *= p[a, b] if a_wins else p[b, a]
            slot[f"W:{m.id}"], slot[f"L:{m.id}"] = (a, b) if a_wins else (b, a)
        exact[slot["W:F"]] += prob
    assert exact.sum() == pytest.approx(1)
    sim = simulate(bracket, odds, teams, runs=400_000, seed=1)
    np.testing.assert_allclose(sim.places["1"], exact, atol=0.004)


def test_fixed_results_are_respected():
    bracket = build_bracket(CONFIG, OPENINGS)
    odds = random_odds(16, seed=2)
    # Play one tournament, then fix every result of it: all runs must replay it exactly.
    once = simulate(bracket, odds, TEAMS, runs=1, seed=9)
    results = []
    for m in bracket:
        (w,), (loser,) = once.outcomes[m.id]
        results.append(FixedResult(m.stage, int(loser), int(w), int(w)))
    replay = simulate(bracket, random_odds(16, seed=3), TEAMS, runs=2_000, results=results)
    assert not replay.unused_results
    for place, probs in replay.places.items():
        np.testing.assert_array_equal(probs, once.places[place])

    # A single fixed opening result: its loser can never top the group.
    loser, winner = OPENINGS["A"][0]
    fixed = FixedResult("Group Stage: Opening (A)", loser, winner, winner)
    sim = simulate(bracket, odds, TEAMS, runs=5_000, results=[fixed])
    a_winner = sim.outcomes["A-W"][0]
    assert (a_winner != loser).all()
    assert (sim.outcomes["A-O1"][0] == winner).all()


def test_results_outside_the_bracket_are_reported():
    bogus = FixedResult("Playoffs: Grand Final", 1, 2, 1)  # 1 and 2 share a group: never a GF
    sim = simulate(build_bracket(CONFIG, OPENINGS), random_odds(16), TEAMS, 1_000, [bogus])
    assert sim.unused_results == [bogus]


# --- odds and the event from the database -----------------------------------------------------


@pytest.fixture(scope="module")
def champions(tmp_path_factory):
    """Synthetic history plus a Champions-style event: 4 groups, one opening already played."""
    path = tmp_path_factory.mktemp("bracket") / "hist.db"
    engine = db.get_engine(f"sqlite:///{path}")
    db.init_db(engine)
    truth = make_history(engine, teams_per_region=8, spread=200)
    return engine, add_champions_event(engine, sorted(truth.strength))


def test_event_openings_and_results(champions):
    engine, groups = champions
    openings = group_openings(engine, 99)
    assert {g: sorted(t for pair in p for t in pair) for g, p in openings.items()} == {
        g: sorted(t) for g, t in groups.items()
    }
    (result,) = fixed_results(load_records(engine), 99)
    a, b = groups["A"][:2]
    assert (result.stage, result.team_a, result.team_b, result.winner) == (
        "Group Stage: Opening (A)", a, b, a,
    )  # fmt: skip


def test_pairwise_odds_are_symmetric(champions):
    engine, groups = champions
    records = load_records(engine)
    frame, builder = build_feature_frame(records, team_regions(engine), events=load_events(engine))
    model = make_model("linear", load_config(None)).fit(prepare(frame))
    teams = sorted(t for g in groups.values() for t in g)[:6]
    for odds in (
        elo_odds(builder, teams),
        model_odds(builder, model, teams, records[-1].date, MAP_POOL, tier="champions"),
    ):
        for m in odds.values():
            np.testing.assert_allclose(m + m.T, 1.0)
        assert not np.allclose(odds[3], 0.5)
        # Bo5 exaggerates whatever edge Bo3 shows.
        edge3, edge5 = odds[3] - 0.5, odds[5] - 0.5
        assert (np.abs(edge5) >= np.abs(edge3) - 1e-9).all()


@pytest.mark.parametrize("source", ["elo", "model"])
def test_cli_simulate(champions, tmp_path, monkeypatch, source):
    engine, _ = champions
    records = load_records(engine)
    frame, _ = build_feature_frame(records, team_regions(engine), events=load_events(engine))
    model = make_model("linear", load_config(None)).fit(prepare(frame))
    model_path = tmp_path / "map_model.pkl"
    model_path.write_bytes(pickle.dumps(
        {"model": model, "name": "linear", "trained_through": "2027-03-01", "maps": 1}
    ))  # fmt: skip
    monkeypatch.setenv("VALCHAMPS_DB_URL", str(engine.url))
    out = tmp_path / "odds.csv"
    result = CliRunner().invoke(app, [
        "simulate", "--event", "99", "--runs", "2000", "--odds", source,
        "--model", str(model_path), "--out", str(out),
    ])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "16 teams, 2,000 runs" in result.output
    assert "1 finished series fixed" in result.output
    assert "warning" not in result.output
    import pandas as pd

    table = pd.read_csv(out)
    assert len(table) == 16
    assert table["title"].sum() == pytest.approx(1)
    assert math.isclose(table["top8"].sum(), 8)
    assert out.with_suffix(".json").exists()


def test_cli_simulate_without_groups(champions, monkeypatch):
    engine, _ = champions
    monkeypatch.setenv("VALCHAMPS_DB_URL", str(engine.url))
    result = CliRunner().invoke(app, ["simulate", "--event", "1", "--odds", "elo"])
    assert result.exit_code == 1
    assert "no group-stage opening matches" in result.output
