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
    EventInfo,
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
    tier: str = "regional", lineups: dict | None = None, vetoes: tuple = (), event_id: int = 1,
) -> MatchRecord:  # fmt: skip
    return MatchRecord(
        match_id=match_id, date=START + timedelta(days=day), event_id=event_id, tier=tier,
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


def test_league_and_international_placements_are_point_in_time():
    end = lambda day: (START + timedelta(days=day)).date()  # noqa: E731
    events = {
        # League: team 1 won it, team 2 placed 4th, team 3 played but is not listed.
        10: EventInfo(10, "regional", end(10), {1: (1, 11), 2: (4, 6)}),
        20: EventInfo(20, "international", end(30), {2: (1, None)}),
    }
    league = [("Bind", 13, 5, None)]
    records = [
        rec(1, 0, 1, 2, league, event_id=10),
        rec(2, 5, 1, 3, league, event_id=10),
        rec(3, 10, 2, 3, league, event_id=10),  # final day of the league
        rec(4, 25, 1, 2, league, tier="international", event_id=20),
        rec(5, 40, 1, 2, league, tier="international", event_id=99),
    ]
    frame, _ = build_feature_frame(records, {}, events=events)
    rows = frame[frame.perspective == 0].set_index("match_id")

    # During the league (up to and including its last day) nothing is known yet.
    assert rows.loc[[1, 2, 3], "league_place_a"].isna().all()
    # After it: team 1 won (1st, 11 points), team 2 finished 4th.
    m4 = rows.loc[4]
    assert (m4.league_place_a, m4.league_won_a, m4.season_points_a) == (1.0, 1.0, 11.0)
    assert (m4.league_place_b, m4.league_won_b, m4.league_place_diff) == (4.0, 0.0, -3.0)
    assert math.isnan(m4.intl_place_a)  # the international has not finished
    # Once the international is over, team 2's win there is visible.
    m5 = rows.loc[5]
    # Team 1 played the international but is not listed: just below the last listed place.
    assert (m5.intl_place_b, m5.intl_place_a) == (1.0, 2.0)


def test_unlisted_participant_finishes_below_last_listed_place():
    events = {10: EventInfo(10, "regional", START.date(), {1: (1, 11), 2: (4, 6)})}
    records = [
        rec(1, -3, 3, 1, [("Bind", 13, 5, None)], event_id=10),
        rec(2, 5, 3, 1, [("Bind", 13, 5, None)], event_id=11),
    ]
    frame, _ = build_feature_frame(records, {}, events=events)
    row = frame[(frame.perspective == 0) & (frame.match_id == 2)].iloc[0]
    assert (row.league_place_a, row.season_points_a) == (5.0, 0.0)


def test_showmatches_are_excluded(tmp_path):
    from valchamps.data import db
    from valchamps.data.models import MapResult, Match, Team

    engine = db.get_engine(f"sqlite:///{tmp_path / 'show.db'}")
    db.init_db(engine)
    make_history(engine, teams_per_region=3, seasons=(2025,))
    real = load_records(engine)
    with engine.begin() as conn:
        base = real[0]
        db.save_match(conn, Match(
            match_id=999_999, event_id=base.event_id, event_name=None, stage="Showmatch: Override",
            date_utc=base.date, status="completed", best_of=1,
            team1=Team(base.team1_id, "a"), team2=Team(base.team2_id, "b"),
            team1_score=1, team2_score=0, maps=[MapResult(999_999, 1, "Bind", 13, 9, None)],
        ))  # fmt: skip
    assert 999_999 not in {r.match_id for r in load_records(engine)}


def test_missing_event_end_date_falls_back_to_last_match(history):
    from sqlalchemy import update

    from valchamps.data import db
    from valchamps.features import load_events

    engine, _, records, _ = history
    event_id = records[0].event_id
    last = max(r.date for r in records if r.event_id == event_id).date()
    # Synthetic events have no dates, and every match is finished: last match date is used.
    assert load_events(engine)[event_id].end_date == last

    # With a match still to play, the event is unfinished and gets no end date.
    with engine.begin() as conn:
        conn.execute(update(db.matches).where(db.matches.c.match_id == records[0].match_id)
                     .values(status="upcoming"))  # fmt: skip
    try:
        assert load_events(engine)[event_id].end_date is None
    finally:
        with engine.begin() as conn:
            conn.execute(update(db.matches).where(db.matches.c.match_id == records[0].match_id)
                         .values(status="completed"))  # fmt: skip


def test_season_points_include_masters_and_reset_each_year():
    end = lambda day: (START + timedelta(days=day)).date()  # noqa: E731
    events = {
        10: EventInfo(10, "regional", end(10), {1: (1, 3)}),  # league win, 3 points
        20: EventInfo(20, "international", end(20), {1: (1, 7)}),  # Masters win, 7 points
    }
    league = [("Bind", 13, 5, None)]
    records = [
        rec(1, 0, 1, 2, league, event_id=10),
        rec(2, 15, 1, 2, league, tier="international", event_id=20),
        rec(3, 30, 1, 2, league, event_id=30),
        rec(4, 400, 1, 2, league, event_id=40),  # next calendar year
    ]
    frame, _ = build_feature_frame(records, {}, events=events)
    rows = frame[frame.perspective == 0].set_index("match_id")
    assert rows.loc[2, "season_points_a"] == 3.0
    assert rows.loc[3, "season_points_a"] == 10.0
    assert rows.loc[4, "season_points_a"] == 0.0
