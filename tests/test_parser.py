from __future__ import annotations

from datetime import date, datetime

import pytest

from tests.conftest import load_fixture
from valchamps.data.models import Team
from valchamps.data.parser import (
    ParseError,
    TeamsNotDecided,
    _parse_date_range,
    _parse_players,
    match_id_from_path,
    parse_event,
    parse_event_matches,
    parse_match,
    parse_standings,
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
    assert (lotus.team1_ct, lotus.team1_t, lotus.team2_ct, lotus.team2_t) == (7, 6, 5, 5)
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


def test_tbd_match_page_raises_teams_not_decided():
    """A bracket slot like /754737/tbd-valorant-champions-2026-gf: header links have no team."""
    html = load_fixture("match_378830_upcoming.html")
    for team_href in ('href="/team/1120/edward-gaming"', 'href="/team/2593/fnatic"'):
        html = html.replace(team_href, 'href="#"')
    with pytest.raises(TeamsNotDecided):
        parse_match(html, 754737)


def test_event_matches_listing():
    listings = parse_event_matches(load_fixture("event_matches_2097.html"))
    assert [(x.match_id, x.status) for x in listings] == [
        (378829, "completed"),
        (378831, "live"),
        (378830, "upcoming"),
        (378999, "upcoming"),  # bracket slot with teams still TBD
    ]
    assert [x.teams_decided for x in listings] == [True, True, True, False]
    assert (listings[3].team1_name, listings[3].team2_name) == ("TBD", "TBD")
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
        ("Jul 18 \u2013 Sep 1, 2025", (date(2025, 7, 18), date(2025, 9, 1))),  # vlr's en dash
        ("Sep 12 - 30, 2025", (date(2025, 9, 12), date(2025, 9, 30))),
        ("Jan 10\u201325, 2025", (date(2025, 1, 10), date(2025, 1, 25))),  # unspaced en dash
        ("Jun 5\u201321, 2026", (date(2026, 6, 5), date(2026, 6, 21))),
        ("Dec 28, 2025 \u2013 Jan 5, 2026", (date(2025, 12, 28), date(2026, 1, 5))),
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


def test_rounds_match_map_scores(completed):
    for m in completed.maps:
        assert [r.round_num for r in m.rounds] == list(
            range(1, m.team1_rounds + m.team2_rounds + 1)
        )
        assert sum(r.winner_team_id == FNC for r in m.rounds) == m.team1_rounds
        assert sum(r.winner_team_id == TH for r in m.rounds) == m.team2_rounds
        assert all(r.outcome in {"elim", "defuse", "boom", "time"} for r in m.rounds)


def test_round_sides_match_half_scores(completed):
    lotus = completed.maps[0]
    fnc_rounds = [r for r in lotus.rounds if r.winner_team_id == FNC]
    assert sum(r.winner_side == "ct" for r in fnc_rounds) == lotus.team1_ct
    assert sum(r.winner_side == "t" for r in fnc_rounds) == lotus.team1_t
    # Pistol rounds are round 1 and round 13.
    assert {r.round_num for r in lotus.rounds} >= {1, 13}


@pytest.fixture(scope="module")
def no_stats():
    """A page with the round strip but no scoreboard (e.g. stats not posted yet)."""
    return parse_match(load_fixture("match_378829_no_stats.html"), 378829)


def test_no_stats_page_still_yields_tags_vetoes_and_rounds(no_stats, completed):
    assert (no_stats.team1.tag, no_stats.team2.tag) == ("FNC", "TH")
    assert [v.team_id for v in no_stats.veto] == [v.team_id for v in completed.veto]
    assert all(v.team_id is not None for v in no_stats.veto if v.action != "remains")
    assert [len(m.rounds) for m in no_stats.maps] == [23, 20, 26]
    assert all(m.players == [] for m in no_stats.maps)


def test_player_stats_parse_from_div_scoreboard(completed):
    """vlr.gg now renders the scoreboard without <table>; the parser must not depend on it."""
    html = load_fixture("match_378829_completed.html")
    for tag in ("table", "thead", "tbody", "tr", "th", "td"):
        html = html.replace(f"<{tag}", "<div").replace(f"</{tag}>", "</div>")
    assert "<table" not in html
    parsed = parse_match(html, 378829)
    assert (parsed.team1.tag, parsed.team2.tag) == ("FNC", "TH")
    for got, want in zip(parsed.maps, completed.maps, strict=True):
        assert got.players == want.players


def test_current_scoreboard_layout_matches_legacy(completed):
    ovw = parse_match(load_fixture("match_378829_ovw.html"), 378829)
    for got, want in zip(ovw.maps, completed.maps, strict=True):
        assert got.players == want.players


def _real_row_game(html: str | None = None):
    from bs4 import BeautifulSoup

    html = html or load_fixture("real/ovw_scoreboard_row.html")
    return BeautifulSoup(html, "lxml").select_one(".vm-stats-game")


FNC_TEAM = Team(2593, "FNATIC", "FNC")
BLG_TEAM = Team(12010, "Bilibili Gaming", "BLG")


def test_real_scoreboard_row():
    """Real vlr.gg markup: K/D/A share one cell, every cell is labelled by data-col."""
    players = _parse_players(_real_row_game(), FNC_TEAM, BLG_TEAM)
    assert len(players) == 1
    p = players[0]
    assert (p.player_id, p.handle, p.team_id, p.agent) == (458, "Chronicle", 2593, "Viper")
    assert (p.rating, p.acs, p.kills, p.deaths, p.assists) == (1.39, 251.0, 19, 14, 8)
    assert (p.kast, p.adr, p.hs_pct, p.first_kills, p.first_deaths) == (73.0, 174.0, 35.0, 4, 2)


def test_misaligned_columns_raise():
    """The Sep 2026 bug: reading the new layout by position shifted every column."""
    html = load_fixture("real/ovw_scoreboard_row.html").replace('data-col="', 'data-x="')
    with pytest.raises(ParseError, match="misaligned"):
        _parse_players(_real_row_game(html), FNC_TEAM, BLG_TEAM)


@pytest.mark.parametrize(
    ("fixture", "event_id", "name", "dates", "location"),
    [
        ("real/event_2501_americas_stage2_2025.html", 2501, "VCT 2025: Americas Stage 2",
         (date(2025, 7, 18), date(2025, 9, 1)), "Riot Games Arena, Los Angeles"),
        ("real/event_2283_champions_2025.html", 2283, "Valorant Champions 2025",
         (date(2025, 9, 12), date(2025, 10, 5)), "Accor Arena, Paris"),
    ],
)  # fmt: skip
def test_real_event_header(fixture, event_id, name, dates, location):
    ev = parse_event(load_fixture(fixture), event_id)
    assert (ev.name, (ev.start_date, ev.end_date), ev.location) == (name, dates, location)


def test_real_league_standings():
    rows = parse_standings(load_fixture("real/event_2501_americas_stage2_2025.html"))
    assert [(r.place, r.place_max) for r in rows] == [
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (5, 6), (7, 8), (7, 8),
    ]  # fmt: skip
    g2 = rows[0]
    assert (g2.team_id, g2.team_name, g2.circuit_points, g2.note) == (
        11058, "G2 Esports", 11, "Champions",
    )  # fmt: skip
    assert rows[2].note is None  # 3rd place did not qualify through this event
    assert [r.circuit_points for r in rows] == [11, 9, 9, 6, 3, 3, 2, 2]
    assert rows[4].team_name == "LEVIAT\u00c1N"


def test_real_international_standings_have_no_points_column():
    rows = parse_standings(load_fixture("real/event_2283_champions_2025.html"))
    assert [(r.place, r.team_name) for r in rows[:2]] == [(1, "NRG"), (2, "FNATIC")]
    assert len(rows) == 8
    assert all(r.circuit_points is None and r.note is None for r in rows)


def test_event_without_standings():
    assert parse_standings(load_fixture("event_2097.html")) == []
