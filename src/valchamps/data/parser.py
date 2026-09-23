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
    Team,
    VetoStep,
)


class ParseError(ValueError):
    """The page does not have the structure the parser expects."""


_WS = re.compile(r"\s+")
_MATCH_PATH = re.compile(r"^/(\d+)/")
_ID_IN_HREF = re.compile(r"/(?:team|player|event)/(?:matches/)?(\d+)")
_VETO_STEP = re.compile(r"^(?P<who>.+?)\s+(?P<action>ban|pick)\s+(?P<map>.+)$", re.IGNORECASE)
_VETO_REMAINS = re.compile(r"^(?P<map>.+?)\s+remains$", re.IGNORECASE)

# Column order of the per-map overview tables (after the player and agent cells).
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


def _parse_date_range(raw: str) -> tuple[date | None, date | None]:
    """Parse strings like 'Aug 1, 2024 - Aug 25, 2024' or 'Sep 12 - Oct 5, 2026'."""
    parts = [p.strip() for p in raw.split(" - ")]
    if len(parts) != 2:
        return _parse_date(raw), None
    start_raw, end_raw = parts
    end = _parse_date(end_raw)
    start = _parse_date(start_raw)
    if start is None and end is not None:
        start = _parse_date(f"{start_raw}, {end.year}")
    return start, end


def parse_event(html: str, event_id: int) -> Event:
    soup = _soup(html)
    name = _text(soup.select_one("h1.wf-title")) or f"event-{event_id}"
    details: dict[str, str] = {}
    for item in soup.select(".event-desc-item"):
        label = _text(item.select_one(".event-desc-item-label")).lower()
        details[label] = _text(item.select_one(".event-desc-item-value"))
    start, end = _parse_date_range(details.get("dates", ""))
    return Event(
        event_id=event_id,
        name=name,
        start_date=start,
        end_date=end,
        location=details.get("location") or None,
    )


def parse_event_matches(html: str) -> list[MatchListing]:
    """Parse ``/event/matches/<id>/?series_id=all`` into match listings."""
    listings: list[MatchListing] = []
    for card in _soup(html).select("a.match-item"):
        href = card.get("href", "")
        teams = [_text(t) for t in card.select(".match-item-vs-team-name .text-of")]
        if len(teams) != 2:
            continue
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
        raise ParseError(f"team {side} link missing from match header")
    name = _text(link.select_one(".wf-title-med")) or _text(link)
    return Team(team_id=team_id, name=name)


def _team_tags(soup: BeautifulSoup) -> list[str | None]:
    """Team tags (e.g. 'FNC') from the first map's player tables; vetoes refer to teams by tag."""
    first_game = next(
        (g for g in soup.select(".vm-stats-game") if g.get("data-game-id") != "all"), None
    )
    tags: list[str | None] = [None, None]
    if first_game is None:
        return tags
    for i, table in enumerate(first_game.select("table.wf-table-inset.mod-overview")[:2]):
        tag = _text(table.select_one("td.mod-player .ge-text-light"))
        tags[i] = tag or None
    return tags


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


def _parse_players(table: Tag, team_id: int) -> list[PlayerMapStats]:
    players: list[PlayerMapStats] = []
    for row in table.select("tbody tr"):
        link = row.select_one("td.mod-player a")
        player_id = _id_from_href(link.get("href") if link else None)
        if player_id is None:
            continue
        agent_img = row.select_one("td.mod-agents img")
        values: dict[str, str | None] = {}
        for col, cell in zip(_STAT_COLUMNS, row.select("td.mod-stat"), strict=False):
            both = cell.select_one(".mod-both")
            values[col] = _text(both) if both else _text(cell)
        players.append(
            PlayerMapStats(
                player_id=player_id,
                handle=_text(row.select_one("td.mod-player .text-of")),
                team_id=team_id,
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

    tables = game.select("table.wf-table-inset.mod-overview")
    players: list[PlayerMapStats] = []
    if len(tables) >= 2:
        players = _parse_players(tables[0], team1.team_id) + _parse_players(
            tables[1], team2.team_id
        )

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
