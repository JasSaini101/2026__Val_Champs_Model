from __future__ import annotations

from datetime import date, datetime

import pytest

from tests.conftest import load_fixture
from valchamps.data.parser import (
    ParseError,
    _parse_date_range,
    match_id_from_path,
    parse_event,
    parse_event_matches,
    parse_match,
    parse_veto,
)

FNC, TH = 2593, 1001


@pytest.fixture(scope="module")
def completed():
    return parse_match(load_fixture("match_378829_completed.html"), 378829)


def test_match_header(completed):
    assert (completed.team1.team_id, completed.team1.name, completed.team1.tag) == (
        FNC,
        "FNATIC",
        "FNC",
    )
    assert (completed.team2.team_id, completed.team2.name, completed.team2.tag) == (
        TH,
        "Team Heretics",
        "TH",
    )
    assert completed.status == "completed"
    assert completed.best_of == 3
    assert (completed.team1_score, completed.team2_score) == (2, 1)
    assert completed.winner_id == FNC
    assert completed.date_utc == datetime(2024, 8, 22, 12, 0)
    assert completed.event_id == 2097
    assert completed.event_name == "Valorant Champions 2024"
    assert completed.stage == "Playoffs: Upper Final"


def test_veto_maps_tags_to_team_ids(completed):
    steps = [(v.action, v.map_name, v.team_id) for v in completed.veto]
    assert steps == [
        ("ban", "Icebox", FNC),
        ("ban", "Sunset", TH),
        ("pick", "Lotus", FNC),
        ("pick", "Split", TH),
        ("ban", "Bind", FNC),
        ("ban", "Haven", TH),
        ("remains", "Abyss", None),
    ]


def test_maps_skip_all_maps_tab_and_keep_order(completed):
    summary = [(m.game_id, m.map_order, m.map_name, m.picked_by, m.team1_rounds, m.team2_rounds)
               for m in completed.maps]  # fmt: skip
    assert summary == [
        (180001, 1, "Lotus", FNC, 13, 10),
        (180002, 2, "Split", TH, 7, 13),
        (180003, 3, "Abyss", None, 14, 12),
    ]


def test_map_halves_and_duration(completed):
    lotus = completed.maps[0]
    assert (lotus.team1_ct, lotus.team1_t, lotus.team2_ct, lotus.team2_t) == (7, 6, 4, 6)
    assert lotus.duration == "49:31"
    # Overtime map: regulation halves don't sum to the final score.
    abyss = completed.maps[2]
    assert abyss.team1_ct + abyss.team1_t == 12 and abyss.team1_rounds == 14


def test_player_stats(completed):
    lotus = completed.maps[0]
    assert len(lotus.players) == 10
    assert {p.team_id for p in lotus.players[:5]} == {FNC}
    assert {p.team_id for p in lotus.players[5:]} == {TH}
    boaster = lotus.players[0]
    assert (boaster.player_id, boaster.handle, boaster.agent) == (9001, "Boaster", "Astra")
    assert (boaster.rating, boaster.acs, boaster.kills, boaster.deaths, boaster.assists) == (
        0.83, 189.0, 15, 13, 6,
    )  # fmt: skip
    assert (boaster.kast, boaster.adr, boaster.hs_pct) == (68.0, 121.0, 21.0)
    assert (boaster.first_kills, boaster.first_deaths) == (3, 3)


def test_upcoming_match():
    m = parse_match(load_fixture("match_378830_upcoming.html"), 378830)
    assert m.status == "upcoming"
    assert m.best_of == 5
    assert (m.team1_score, m.team2_score) == (None, None)
    assert m.winner_id is None
    assert m.maps == [] and m.veto == []
    assert m.team1.tag is None


def test_missing_teams_raises():
    with pytest.raises(ParseError):
        parse_match("<html><body>nothing here</body></html>", 1)


def test_event_matches_listing():
    listings = parse_event_matches(load_fixture("event_matches_2097.html"))
    assert [(x.match_id, x.status) for x in listings] == [
        (378829, "completed"),
        (378831, "live"),
        (378830, "upcoming"),
    ]  # TBD placeholder card (one team) is dropped
    assert listings[0].team1_name == "FNATIC"
    assert listings[0].stage == "Playoffs\u2013Upper Final"  # vlr uses an en dash


def test_event_page():
    ev = parse_event(load_fixture("event_2097.html"), 2097)
    assert ev.name == "Valorant Champions 2024"
    assert (ev.start_date, ev.end_date) == (date(2024, 8, 1), date(2024, 8, 25))
    assert ev.location == "Seoul & Incheon, South Korea"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Aug 1, 2024 - Aug 25, 2024", (date(2024, 8, 1), date(2024, 8, 25))),
        ("Sep 12 - Oct 5, 2026", (date(2026, 9, 12), date(2026, 10, 5))),
        ("TBD", (None, None)),
    ],
)
def test_date_range(raw, expected):
    assert _parse_date_range(raw) == expected


def test_veto_unknown_team_is_none():
    steps = parse_veto("XYZ ban Bind; Pearl remains", {"fnc": FNC})
    assert steps[0].team_id is None and steps[0].action == "ban"
    assert steps[1].action == "remains"


def test_match_id_from_path():
    assert match_id_from_path("/378829/fnatic-vs-th") == 378829
    with pytest.raises(ParseError):
        match_id_from_path("/team/2593/fnatic")
