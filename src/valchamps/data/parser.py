"""Parse vlr.gg HTML into :mod:`valchamps.data.models` records.

The selectors follow vlr.gg's markup as of 2024-2026. Every lookup is defensive: a missing
optional element yields ``None`` rather than an exception, while a page that lacks the core
structure (team links, match id) raises :class:`ParseError` so bad pages are noticed.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from bs4 import BeautifulSoup, Tag

from valchamps.data.models import (
    Event,
    MapResult,
    Match,
    MatchListing,
    PlayerMapStats,
    RoundResult,
    Standing,
    Team,
    VetoStep,
)


class ParseError(ValueError):
    """The page does not have the structure the parser expects."""


class TeamsNotDecided(ParseError):
    """A bracket match whose teams are not known yet (shown as "TBD"); retry later."""


_WS = re.compile(r"\s+")
_MATCH_PATH = re.compile(r"^/(\d+)/")
_ID_IN_HREF = re.compile(r"/(?:team|player|event)/(?:matches/)?(\d+)")
_VETO_STEP = re.compile(r"^(?P<who>.+?)\s+(?P<action>ban|pick)\s+(?P<map>.+)$", re.IGNORECASE)
_VETO_REMAINS = re.compile(r"^(?P<map>.+?)\s+remains$", re.IGNORECASE)

# vlr.gg's ``data-col`` labels on scoreboard cells, mapped to our field names.
_DATA_COLS = {
    "rating2": "rating", "rating": "rating", "acs": "acs", "kills": "kills",
    "deaths": "deaths", "assists": "assists", "kd-diff": "plus_minus", "kast": "kast",
    "adr": "adr", "hsp": "hs_pct", "fb": "first_kills", "fd": "first_deaths",
    "fk-diff": "fk_plus_minus",
}  # fmt: skip

# Column order of the legacy <table> scoreboard (after the player and agent cells), used when
# cells carry no data-col labels.
_STAT_COLUMNS = (
    "rating", "acs", "kills", "deaths", "assists", "plus_minus",
    "kast", "adr", "hs_pct", "first_kills", "first_deaths", "fk_plus_minus",
)  # fmt: skip


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _text(node: Tag | None) -> str:
    return _WS.sub(" ", node.get_text(" ", strip=True)).strip() if node else ""


def _own_text(node: Tag | None) -> str:
    """Text of ``node`` excluding its child elements (e.g. drop the 'PICK' badge)."""
    if node is None:
        return ""
    return _WS.sub(" ", "".join(s for s in node.find_all(string=True, recursive=False))).strip()


def _num(raw: str | None, cast: type = float) -> Any:
    if raw is None:
        return None
    cleaned = raw.replace("%", "").replace("+", "").strip(" /")
    if cleaned in {"", "-", "\xa0"}:
        return None
    try:
        return cast(cleaned)
    except ValueError:
        return None


def _id_from_href(href: str | None) -> int | None:
    if not href:
        return None
    m = _ID_IN_HREF.search(href)
    return int(m.group(1)) if m else None


def match_id_from_path(path: str) -> int:
    m = _MATCH_PATH.match(path)
    if not m:
        raise ParseError(f"not a match path: {path!r}")
    return int(m.group(1))


def _normalise_status(raw: str) -> str:
    raw = raw.lower()
    if "live" in raw:
        return "live"
    if "final" in raw or "completed" in raw:
        return "completed"
    return "upcoming"


# ---------------------------------------------------------------------------------------------
# Event pages
# ---------------------------------------------------------------------------------------------


def _parse_date(raw: str) -> date | None:
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


_DATE_RANGE_SEP = re.compile(r"\s*[-\u2013\u2014]\s*")  # hyphen/en/em dash, spaced or not
_DAY_YEAR = re.compile(r"^(\d{1,2}), (\d{4})$")


def _parse_date_range(raw: str) -> tuple[date | None, date | None]:
    """Parse 'Aug 1, 2024 - Aug 25, 2024', 'Jul 18 \u2013 Sep 1, 2025' or 'Jan 10\u201325, 2025'."""
    parts = _DATE_RANGE_SEP.split(raw.strip())
    if len(parts) != 2:
        return _parse_date(raw), None
    start_raw, end_raw = parts
    end = _parse_date(end_raw)
    if end is None and (m := _DAY_YEAR.match(end_raw)):
        month = start_raw.split()[0] if start_raw else ""
        end = _parse_date(f"{month} {m.group(1)}, {m.group(2)}")
    start = _parse_date(start_raw)
    if start is None and end is not None:
        start = _parse_date(f"{start_raw}, {end.year}")
    return start, end


def event_details(soup: BeautifulSoup) -> dict[str, str]:
    """Label -> value pairs from the event header ('Dates', 'Prize', 'Location')."""
    details: dict[str, str] = {}
    for item in soup.select(".event-header-main-meta > div"):  # current layout
        label = _text(item.select_one(".label")).lower()
        if label:
            details[label] = _text(item.select_one(".value"))
    for item in soup.select(".event-desc-item"):  # older layout
        label = _text(item.select_one(".event-desc-item-label")).lower()
        details[label] = _text(item.select_one(".event-desc-item-value"))
    return details


def parse_event(html: str, event_id: int) -> Event:
    soup = _soup(html)
    title = soup.select_one("h1.event-header-main-title") or soup.select_one("h1.wf-title")
    details = event_details(soup)
    start, end = _parse_date_range(details.get("dates", ""))
    return Event(
        event_id=event_id,
        name=_text(title) or f"event-{event_id}",
        start_date=start,
        end_date=end,
        location=details.get("location") or None,
    )


_PLACE = re.compile(r"(\d+)(?:st|nd|rd|th)?(?:\s*[-\u2013]\s*(\d+))?")


def parse_standings(html: str) -> list[Standing]:
    """Final placements from the event page's "Prize Distribution" table.

    Only placed teams are listed (usually the top 8); columns are found by their header label
    because league events carry Points/Note columns that international events lack.
    """
    table = _soup(html).select_one(".wf-ptable--standings")
    if table is None:
        return []
    rows = table.select(":scope > .row")
    if not rows:
        return []
    header = [_text(c).lower() for c in rows[0].select(":scope > .cell")]
    col = {name: i for i, name in enumerate(header)}
    standings: list[Standing] = []
    for row in rows[1:]:
        cells = row.select(":scope > .cell")
        place_m = _PLACE.match(_text(cells[col.get("place", 0)]).replace(" ", ""))
        link = row.select_one("a[href*='/team/']")
        team_id = _id_from_href(link.get("href") if link else None)
        if place_m is None or team_id is None:
            continue  # TBD slot of an unfinished event
        name_node = link.select_one(".text-of")
        name = (
            _WS.sub(" ", name_node.find(string=True, recursive=False) or "").strip()
            if name_node
            else ""
        )
        place = int(place_m.group(1))
        points = note = None
        if "points" in col and col["points"] < len(cells):
            points = _num(_text(cells[col["points"]]).replace(" ", ""), int)
        if "note" in col and col["note"] < len(cells):
            note = _text(cells[col["note"]]) or None
        standings.append(
            Standing(
                team_id=team_id,
                team_name=name or _text(link),
                place=place,
                place_max=int(place_m.group(2)) if place_m.group(2) else place,
                circuit_points=points,
                note=note,
            )
        )
    return standings


def parse_event_matches(html: str) -> list[MatchListing]:
    """Parse ``/event/matches/<id>/?series_id=all`` into match listings."""
    listings: list[MatchListing] = []
    for card in _soup(html).select("a.match-item"):
        href = card.get("href", "")
        teams = [_text(t) for t in card.select(".match-item-vs-team-name .text-of")]
        if not href or len(teams) > 2:
            continue
        # Undecided bracket slots may show one or no team name; they are kept as "TBD".
        teams += ["TBD"] * (2 - len(teams))
        listings.append(
            MatchListing(
                match_id=match_id_from_path(href),
                url_path=href,
                team1_name=teams[0],
                team2_name=teams[1],
                status=_normalise_status(_text(card.select_one(".ml-status"))),
                stage=_text(card.select_one(".match-item-event-series")) or None,
            )
        )
    return listings


# ---------------------------------------------------------------------------------------------
# Match pages
# ---------------------------------------------------------------------------------------------


def _parse_header_team(soup: BeautifulSoup, side: int) -> Team:
    link = soup.select_one(f"a.match-header-link.mod-{side}")
    team_id = _id_from_href(link.get("href") if link else None)
    if link is None or team_id is None:
        raise TeamsNotDecided(f"team {side} not decided yet (no team link in match header)")
    name = _text(link.select_one(".wf-title-med")) or _text(link)
    return Team(team_id=team_id, name=name)


def _team_tags(soup: BeautifulSoup) -> list[str | None]:
    """Team tags (e.g. 'FNC') for team 1 and team 2; vetoes refer to teams by tag.

    Read from the first map's round-by-round header, falling back to the player tables.
    """
    first_game = next(
        (g for g in soup.select(".vm-stats-game") if g.get("data-game-id") != "all"), None
    )
    tags: list[str | None] = [None, None]
    if first_game is None:
        return tags
    label_col = first_game.select_one(".vlr-rounds-row-col:not([title]):has(.team)")
    if label_col is not None:
        found = [_text(t) for t in label_col.select(".team")]
        if len(found) == 2 and all(found):
            return [found[0], found[1]]
    # Fallback: first and last scoreboard rows belong to team 1 and team 2.
    rows = _player_rows(first_game)
    if rows:
        tags[0] = _text(rows[0].select_one(".mod-player .ge-text-light")) or None
        tags[1] = _text(rows[-1].select_one(".mod-player .ge-text-light")) or None
    return tags


_ROUND_OUTCOME = re.compile(r"/round/([a-z]+)\.")


def _parse_rounds(game: Tag, team1: Team, team2: Team) -> list[RoundResult]:
    """Round-by-round winners from the ``vlr-rounds`` strip under the map header."""
    rounds: list[RoundResult] = []
    for col in game.select(".vlr-rounds-row-col[title]"):
        num = _num(_text(col.select_one(".rnd-num")), int)
        squares = col.select(".rnd-sq")
        if num is None or len(squares) != 2:
            continue
        winner_idx = next(
            (i for i, sq in enumerate(squares) if "mod-win" in sq.get("class", [])), None
        )
        if winner_idx is None:
            continue
        classes = squares[winner_idx].get("class", [])
        side = "ct" if "mod-ct" in classes else "t" if "mod-t" in classes else None
        img = squares[winner_idx].select_one("img")
        outcome_match = _ROUND_OUTCOME.search(img.get("src", "")) if img else None
        rounds.append(
            RoundResult(
                round_num=num,
                winner_team_id=(team1 if winner_idx == 0 else team2).team_id,
                winner_side=side,
                outcome=outcome_match.group(1) if outcome_match else None,
            )
        )
    return rounds


def parse_veto(raw: str, teams: dict[str, int]) -> list[VetoStep]:
    """Parse 'FNC ban Icebox; TH pick Lotus; ...; Abyss remains'.

    ``teams`` maps lower-cased tags/names to team ids. Unknown teams become ``None``.
    """
    steps: list[VetoStep] = []
    for i, part in enumerate(p.strip() for p in raw.split(";") if p.strip()):
        if m := _VETO_REMAINS.match(part):
            steps.append(VetoStep(i + 1, "remains", m.group("map").strip(), None))
        elif m := _VETO_STEP.match(part):
            steps.append(
                VetoStep(
                    i + 1,
                    m.group("action").lower(),
                    m.group("map").strip(),
                    teams.get(m.group("who").strip().lower()),
                )
            )
    return steps


def _half(node: Tag | None, cls: str) -> int | None:
    return _num(_text(node.select_one(f".{cls}")), int) if node else None


def _player_rows(game: Tag) -> list[Tag]:
    """Rows of the per-map scoreboard, whatever element vlr.gg uses for them.

    vlr.gg has served the scoreboard both as ``<table>`` rows and as nested ``<div>``s, so a
    row is found from its player cell: the nearest ancestor that also holds stat cells.
    """
    rows: list[Tag] = []
    for cell in game.select(".mod-player"):
        row = cell.parent
        while row is not None and row is not game and row.select_one(".stats-sq") is None:
            row = row.parent
        if row is None or row is game or any(row is r for r in rows):
            continue
        rows.append(row)
    return rows


def _stat_cells(row: Tag) -> list[Tag]:
    cells = row.select(".mod-stat")
    if cells:
        return cells
    # No .mod-stat cells: use the outermost stat squares, skipping the agent icons.
    return [
        sq
        for sq in row.select(".stats-sq")
        if "mod-agent" not in sq.get("class", []) and sq.find_parent(class_="stats-sq") is None
    ]


def _cell_value(cell: Tag) -> str:
    both = cell.select_one(".mod-both")  # vlr shows both-sides, attack and defence values
    return _text(both) if both else _text(cell)


def _row_values(row: Tag) -> dict[str, str]:
    """Stat values of one scoreboard row, keyed by field name."""
    labelled = {}
    for cell in row.select("[data-col]"):
        field = _DATA_COLS.get(cell["data-col"])
        if field and field not in labelled:
            labelled[field] = _cell_value(cell)
    if labelled:
        return labelled
    return {
        col: _cell_value(cell) for col, cell in zip(_STAT_COLUMNS, _stat_cells(row), strict=False)
    }


def _check_row(values: dict[str, str], player_id: int) -> None:
    """Catch misaligned columns: the +/- columns must equal the differences they summarise."""
    for a, b, diff in (
        ("kills", "deaths", "plus_minus"),
        ("first_kills", "first_deaths", "fk_plus_minus"),
    ):
        x, y, d = (_num(values.get(k), int) for k in (a, b, diff))
        if None not in (x, y, d) and x - y != d:
            raise ParseError(
                f"scoreboard columns look misaligned for player {player_id}: "
                f"{a}={x}, {b}={y}, {diff}={d}"
            )


def _parse_players(game: Tag, team1: Team, team2: Team) -> list[PlayerMapStats]:
    rows = _player_rows(game)
    by_tag = {t.tag.lower(): t.team_id for t in (team1, team2) if t.tag}
    players: list[PlayerMapStats] = []
    for i, row in enumerate(rows):
        cell = row.select_one(".mod-player")
        link = cell.select_one("a[href*='/player/']") if cell else None
        player_id = _id_from_href(link.get("href") if link else None)
        if player_id is None:
            continue
        tag = _text(cell.select_one(".ge-text-light")).lower()
        # Scoreboards list team 1's players first, so position is the fallback.
        default_team = team1.team_id if i < len(rows) / 2 else team2.team_id
        agent_img = row.select_one(".mod-agents img") or row.select_one(".mod-agent img")
        values = _row_values(row)
        _check_row(values, player_id)
        players.append(
            PlayerMapStats(
                player_id=player_id,
                handle=_text(cell.select_one(".text-of")),
                team_id=by_tag.get(tag, default_team),
                agent=(agent_img.get("title") or agent_img.get("alt")) if agent_img else None,
                rating=_num(values.get("rating")),
                acs=_num(values.get("acs")),
                kills=_num(values.get("kills"), int),
                deaths=_num(values.get("deaths"), int),
                assists=_num(values.get("assists"), int),
                kast=_num(values.get("kast")),
                adr=_num(values.get("adr")),
                hs_pct=_num(values.get("hs_pct")),
                first_kills=_num(values.get("first_kills"), int),
                first_deaths=_num(values.get("first_deaths"), int),
            )
        )
    return players


def _parse_map(game: Tag, order: int, team1: Team, team2: Team) -> MapResult | None:
    header = game.select_one(".vm-stats-game-header")
    if header is None:
        return None
    left = header.select_one(".team:not(.mod-right)")
    right = header.select_one(".team.mod-right")
    map_label = header.select_one(".map span")
    map_name = _own_text(map_label) or _text(map_label)
    if not map_name:
        return None

    picked_by = None
    pick_badge = header.select_one(".map .picked")
    if pick_badge is not None:
        classes = pick_badge.get("class", [])
        picked_by = (
            team1.team_id if "mod-1" in classes else team2.team_id if "mod-2" in classes else None
        )

    t1 = _num(_text(left.select_one(".score")) if left else None, int)
    t2 = _num(_text(right.select_one(".score")) if right else None, int)
    if t1 is None or t2 is None:
        return None

    players = _parse_players(game, team1, team2)

    return MapResult(
        game_id=int(game["data-game-id"]),
        map_order=order,
        map_name=map_name,
        team1_rounds=t1,
        team2_rounds=t2,
        picked_by=picked_by,
        team1_ct=_half(left, "mod-ct"),
        team1_t=_half(left, "mod-t"),
        team2_ct=_half(right, "mod-ct"),
        team2_t=_half(right, "mod-t"),
        duration=_text(header.select_one(".map-duration")) or None,
        players=players,
        rounds=_parse_rounds(game, team1, team2),
    )


def parse_match(html: str, match_id: int) -> Match:
    soup = _soup(html)
    team1 = _parse_header_team(soup, 1)
    team2 = _parse_header_team(soup, 2)
    tag1, tag2 = _team_tags(soup)
    if tag1:
        team1 = Team(team1.team_id, team1.name, tag1)
    if tag2:
        team2 = Team(team2.team_id, team2.name, tag2)

    event_link = soup.select_one("a.match-header-event")
    event_id = _id_from_href(event_link.get("href") if event_link else None)
    event_name = _text(event_link.select_one("div > div")) if event_link else None
    stage = _text(event_link.select_one(".match-header-event-series")) if event_link else None

    date_utc = None
    ts = soup.select_one(".match-header-date .moment-tz-convert[data-utc-ts]")
    if ts is not None:
        try:
            date_utc = datetime.strptime(ts["data-utc-ts"], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            date_utc = None

    notes = [_text(n) for n in soup.select(".match-header-vs-note")]
    status = _normalise_status(" ".join(notes))
    best_of = next((int(m.group(1)) for n in notes if (m := re.search(r"Bo(\d)", n))), None)

    score1 = score2 = None
    spoiler = soup.select_one(".match-header-vs-score .js-spoiler")
    if spoiler is not None and (m := re.search(r"(\d+)\s*:\s*(\d+)", _text(spoiler))):
        score1, score2 = int(m.group(1)), int(m.group(2))

    lookup = {team1.name.lower(): team1.team_id, team2.name.lower(): team2.team_id}
    for team in (team1, team2):
        if team.tag:
            lookup[team.tag.lower()] = team.team_id
    veto = parse_veto(_text(soup.select_one(".match-header-note")), lookup)

    maps: list[MapResult] = []
    for game in soup.select(".vm-stats-game"):
        if game.get("data-game-id") in (None, "all"):
            continue
        parsed = _parse_map(game, len(maps) + 1, team1, team2)
        if parsed is not None:
            maps.append(parsed)

    return Match(
        match_id=match_id,
        event_id=event_id,
        event_name=event_name or None,
        stage=stage or None,
        date_utc=date_utc,
        status=status,
        best_of=best_of,
        team1=team1,
        team2=team2,
        team1_score=score1,
        team2_score=score2,
        veto=veto,
        maps=maps,
    )
