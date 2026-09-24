"""Typed records produced by the parser and persisted by :mod:`valchamps.data.db`."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass(frozen=True)
class Team:
    team_id: int
    name: str
    tag: str | None = None


@dataclass(frozen=True)
class Event:
    event_id: int
    name: str
    start_date: date | None = None
    end_date: date | None = None
    location: str | None = None


@dataclass(frozen=True)
class Standing:
    """A team's final placement in an event (vlr.gg lists the prize-winning places)."""

    team_id: int
    team_name: str
    place: int  # best place of a shared range: "5th-6th" -> 5
    place_max: int  # worst place of the range: "5th-6th" -> 6
    circuit_points: int | None = None  # VCT circuit points, listed for league events
    note: str | None = None  # e.g. the event the team qualified for


@dataclass(frozen=True)
class MatchListing:
    """A row from an event's match list: enough to decide whether to fetch the match page."""

    match_id: int
    url_path: str
    team1_name: str
    team2_name: str
    status: str  # "completed" | "live" | "upcoming"
    stage: str | None = None

    @property
    def teams_decided(self) -> bool:
        """False for bracket slots whose teams are not known yet ("TBD")."""
        return all(n and n.strip().upper() != "TBD" for n in (self.team1_name, self.team2_name))


@dataclass(frozen=True)
class VetoStep:
    step: int
    action: str  # "ban" | "pick" | "remains"
    map_name: str
    team_id: int | None  # None for the decider map


@dataclass(frozen=True)
class PlayerMapStats:
    player_id: int
    handle: str
    team_id: int
    agent: str | None
    rating: float | None
    acs: float | None
    kills: int | None
    deaths: int | None
    assists: int | None
    kast: float | None
    adr: float | None
    hs_pct: float | None
    first_kills: int | None
    first_deaths: int | None


@dataclass(frozen=True)
class RoundResult:
    round_num: int
    winner_team_id: int
    winner_side: str | None  # "ct" (defence) | "t" (attack)
    outcome: str | None  # "elim" | "defuse" | "boom" (spike detonated) | "time"


@dataclass(frozen=True)
class MapResult:
    game_id: int
    map_order: int
    map_name: str
    team1_rounds: int
    team2_rounds: int
    picked_by: int | None
    team1_ct: int | None = None
    team1_t: int | None = None
    team2_ct: int | None = None
    team2_t: int | None = None
    duration: str | None = None
    players: list[PlayerMapStats] = field(default_factory=list)
    rounds: list[RoundResult] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return self.team1_rounds != self.team2_rounds


@dataclass(frozen=True)
class Match:
    match_id: int
    event_id: int | None
    event_name: str | None
    stage: str | None
    date_utc: datetime | None
    status: str
    best_of: int | None
    team1: Team
    team2: Team
    team1_score: int | None
    team2_score: int | None
    veto: list[VetoStep] = field(default_factory=list)
    maps: list[MapResult] = field(default_factory=list)

    @property
    def winner_id(self) -> int | None:
        if self.status != "completed" or self.team1_score is None or self.team2_score is None:
            return None
        if self.team1_score == self.team2_score:
            return None
        return self.team1.team_id if self.team1_score > self.team2_score else self.team2.team_id
