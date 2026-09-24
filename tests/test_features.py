from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from tests.synthetic_history import make_history
from valchamps.cli import DEFAULT_PARAMS, app
from valchamps.features import (
    FEATURE_COLUMNS,
    FeatureParams,
    MapRecord,
    MatchRecord,
    build_feature_frame,
    load_records,
    team_regions,
)
from valchamps.features.params import EloParams
from valchamps.features.ratings import EloTracker

START = datetime(2025, 3, 1, 12)
_game_ids = iter(range(1, 10_000))


def rec(
    match_id: int, day: int, a: int, b: int, maps: list[tuple[str, int, int, int | None]],
    tier: str = "regional", lineups: dict | None = None, vetoes: tuple = (),
) -> MatchRecord:  # fmt: skip
    return MatchRecord(
        match_id=match_id, date=START + timedelta(days=day), event_id=1, tier=tier,
        team1_id=a, team2_id=b, best_of=3,
        maps=tuple(MapRecord(next(_game_ids), i + 1, name, r1, r2, pick)
                   for i, (name, r1, r2, pick) in enumerate(maps)),
        vetoes=vetoes, lineups=lineups or {},
    )  # fmt: skip


@pytest.fixture(scope="module")
def history(tmp_path_factory):
    from valchamps.data import db

    path = tmp_path_factory.mktemp("hist") / "hist.db"
    engine = db.get_engine(f"sqlite:///{path}")
    db.init_db(engine)
    truth = make_history(engine)
    return engine, truth, load_records(engine), team_regions(engine)


# --- loading ----------------------------------------------------------------------------------


def test_load_records(history):
    _, truth, records, regions = history
    assert len(records) == 90
    assert [(r.date, r.match_id) for r in records] == sorted((r.date, r.match_id) for r in records)
    assert all(2 <= len(r.maps) <= 3 for r in records)
    assert all(len(r.vetoes) == 6 for r in records)  # "remains" steps are dropped
    assert all(len(r.lineups[t]) == 5 for r in records for t in r.teams)
    assert sum(r.is_international for r in records) == 30
    assert regions == truth.region


# --- Elo --------------------------------------------------------------------------------------


def test_elo_recovers_true_strength(history):
    _, truth, records, regions = history
    _, builder = build_feature_frame(records, regions)
    teams = sorted(truth.strength)
    learned = [builder.elo.strength(t) for t in teams]
    actual = [truth.strength[t] for t in teams]
    assert np.corrcoef(learned, actual)[0, 1] > 0.75
    offsets = builder.elo.region_offset
    assert offsets["emea"] > offsets["americas"]  # EMEA is 120 points stronger in truth


def test_elo_is_zero_sum_within_a_region():
    elo = EloTracker(EloParams(), {1: "emea", 2: "emea", 3: "emea"})
    for i, (a, b) in enumerate([(1, 2), (2, 3), (1, 3), (3, 1)]):
        elo.update(rec(i, i, a, b, [("Bind", 13, 7, None), ("Lotus", 9, 13, None)]))
    assert sum(elo.rating[t] for t in (1, 2, 3)) == pytest.approx(3 * 1500)
    assert all(v == 0 for v in elo.region_offset.values())


def test_region_offset_moves_only_on_cross_region_maps():
    elo = EloTracker(EloParams(), {1: "emea", 2: "pacific"})
    elo.update(rec(1, 0, 1, 2, [("Bind", 13, 5, None)], tier="international"))
    assert elo.region_offset["emea"] > 0 > elo.region_offset["pacific"]
    assert elo.region_offset["emea"] == pytest.approx(-elo.region_offset["pacific"])


def test_bigger_margin_moves_rating_more():
    close, blowout = (EloTracker(EloParams(), {}) for _ in range(2))
    close.update(rec(1, 0, 1, 2, [("Bind", 14, 12, None)]))
    blowout.update(rec(1, 0, 1, 2, [("Bind", 13, 0, None)]))
    assert blowout.rating[1] - 1500 > close.rating[1] - 1500 > 0


def test_new_season_regresses_ratings_toward_mean():
    elo = EloTracker(EloParams(season_carryover=0.75), {})
    elo.start_season(datetime(2025, 5, 1))
    elo.rating[1] = 1600.0
    elo.map_dev[(1, "Bind")] = 40.0
    elo.start_season(datetime(2025, 12, 1))  # same season: unchanged
    assert elo.rating[1] == 1600.0
    elo.start_season(datetime(2026, 1, 15))
    assert elo.rating[1] == pytest.approx(1575.0)
    assert elo.map_dev[(1, "Bind")] == pytest.approx(30.0)


# --- point-in-time and symmetry ---------------------------------------------------------------


def test_features_never_see_the_future(history):
    """Rows for a match must not change when later matches are added or altered."""
    _, _, records, regions = history
    cutoff = len(records) // 2
    full, _ = build_feature_frame(records, regions)
    past, _ = build_feature_frame(records[:cutoff], regions)
    head = full[full.match_id.isin({r.match_id for r in records[:cutoff]})]
    pd.testing.assert_frame_equal(head.reset_index(drop=True), past.reset_index(drop=True))

    # Flip every later result: earlier rows must be identical.
    flipped = records[:cutoff] + [
        MatchRecord(**{**r.__dict__, "maps": tuple(
            MapRecord(m.game_id, m.map_order, m.map_name, m.team2_rounds, m.team1_rounds,
                      m.picked_by) for m in r.maps)})
        for r in records[cutoff:]
    ]  # fmt: skip
    altered, _ = build_feature_frame(flipped, regions)
    pd.testing.assert_frame_equal(
        altered.iloc[: len(past)].reset_index(drop=True), past.reset_index(drop=True)
    )


def test_each_map_has_mirrored_rows(history):
    _, _, records, regions = history
    frame, _ = build_feature_frame(records, regions)
    a = frame[frame.perspective == 0].set_index("game_id")
    b = frame[frame.perspective == 1].set_index("game_id").loc[a.index]
    assert (a.y + b.y == 1).all()
    assert np.allclose(a.elo_prob + b.elo_prob, 1)
    assert (a.team_a == b.team_b).all() and (a.pick_a == b.pick_b).all()
    for col in [c for c in FEATURE_COLUMNS if c.endswith("_diff")]:
        pd.testing.assert_series_equal(a[col], -b[col], check_names=False)
    for f in ("elo", "map_winrate", "form_winrate", "lineup_rating"):
        pd.testing.assert_series_equal(a[f"{f}_a"], b[f"{f}_b"], check_names=False)


def test_records_must_be_sorted():
    records = [rec(2, 5, 1, 2, [("Bind", 13, 3, None)]), rec(1, 1, 1, 2, [("Bind", 13, 3, None)])]
    with pytest.raises(ValueError, match="sorted"):
        build_feature_frame(records, {})


# --- individual features ----------------------------------------------------------------------


def _rows(records, **params):
    frame, _ = build_feature_frame(records, {}, FeatureParams.from_dict(params))
    return frame[frame.perspective == 0]


def test_form_and_head_to_head():
    records = [
        rec(1, 0, 1, 2, [("Bind", 13, 5, None), ("Lotus", 13, 7, None)]),
        rec(2, 3, 2, 3, [("Bind", 13, 11, None), ("Split", 13, 9, None)]),
        rec(3, 4, 2, 1, [("Haven", 13, 10, None)]),
    ]
    rows = _rows(records, h2h_prior=3)
    first, second, third = (rows[rows.match_id == i].iloc[0] for i in (1, 2, 3))
    assert math.isnan(first.form_winrate_a) and math.isnan(first.rest_days_a)
    # Team 2 before match 2: lost 0-2, round diff -(8 + 6) over 2 maps, 3 days' rest.
    assert second.form_winrate_a == 0.0
    assert second.form_round_diff_a == -7.0
    assert second.rest_days_a == 3.0
    assert second.matches_30d_a == 1.0
    # Match 3 (team 2 vs team 1): team 2 has lost both maps to team 1 before.
    assert third.h2h_maps == 2.0
    assert third.h2h_winrate == pytest.approx((0 + 0.5 * 3) / (2 + 3))
    assert third.form_winrate_a == pytest.approx(2 / 4)


def test_map_winrate_is_shrunk_toward_overall_rate():
    records = [
        rec(1, 0, 1, 2, [("Bind", 13, 5, None), ("Lotus", 13, 7, None)]),
        rec(2, 1, 1, 2, [("Bind", 13, 5, None)]),
    ]
    row = _rows(records, map_prior=5).query("match_id == 2").iloc[0]
    overall = (2 + 0.5 * 5) / (2 + 5)
    assert row.map_played_a == 1.0
    assert row.map_winrate_a == pytest.approx((1 + overall * 5) / (1 + 5))
    assert row.map_winrate_b < 0.5 < row.map_winrate_a


def test_pick_and_ban_rates_and_pick_context():
    vetoes = ((1, "ban", "Split"), (2, "ban", "Bind"), (1, "pick", "Lotus"), (2, "pick", "Haven"))
    records = [
        rec(1, 0, 1, 2, [("Lotus", 13, 5, 1), ("Haven", 13, 7, 2)], vetoes=vetoes),
        rec(2, 1, 1, 2, [("Lotus", 13, 5, 1), ("Ascent", 13, 7, None)], vetoes=vetoes),
    ]
    rows = _rows(records)
    lotus = rows.query("match_id == 2 and map_name == 'Lotus'").iloc[0]
    assert (lotus.pick_a, lotus.pick_b, lotus.decider) == (1.0, 0.0, 0.0)
    assert (lotus.pick_rate_a, lotus.ban_rate_a, lotus.pick_rate_b) == (1.0, 0.0, 0.0)
    ascent = rows.query("match_id == 2 and map_name == 'Ascent'").iloc[0]
    assert ascent.decider == 1.0


def test_roster_continuity_and_lineup_rating():
    old = {p: 1.0 + p / 100 for p in range(1, 6)}
    new = {**{p: 1.0 for p in range(2, 6)}, 99: 1.2}  # player 1 replaced by 99
    opp = {p: 1.0 for p in range(50, 55)}
    records = [
        rec(1, 0, 1, 2, [("Bind", 13, 5, None)], lineups={1: old, 2: opp}),
        rec(2, 1, 1, 2, [("Bind", 13, 5, None)], lineups={1: new, 2: opp}),
    ]
    row = _rows(records).query("match_id == 2").iloc[0]
    assert row.roster_continuity_a == pytest.approx(4 / 5)
    assert row.roster_continuity_b == 1.0
    # Player 99 has no history yet; the other four averaged 1.02..1.05.
    assert row.lineup_rating_a == pytest.approx(np.mean([1.02, 1.03, 1.04, 1.05]))


# --- params and CLI ---------------------------------------------------------------------------


def test_repo_params_load():
    params = FeatureParams.from_yaml(DEFAULT_PARAMS)
    assert params.elo.k > 0 and params.form_window > 0


def test_unknown_params_are_rejected():
    with pytest.raises(ValueError, match="unknown"):
        FeatureParams.from_dict({"form_windw": 5})


def test_cli_build_features(history, tmp_path, monkeypatch):
    engine, *_ = history
    monkeypatch.setenv("VALCHAMPS_DB_URL", str(engine.url))
    out = tmp_path / "maps.parquet"
    result = CliRunner().invoke(app, ["build-features", "--out", str(out)])
    assert result.exit_code == 0, result.output
    frame = pd.read_parquet(out)
    assert len(frame) == 2 * frame.game_id.nunique()
    assert frame.elo_prob.between(0, 1).all()
    assert "cross-region maps" in result.output
