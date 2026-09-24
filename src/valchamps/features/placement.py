"""How teams finished their most recent completed events.

An event's final standings only become visible to matches played after the event's end date,
so a team's Stage 2 result can inform its Champions matches but never its own Stage 2 matches.
vlr.gg lists the placed teams (usually the top 8); a participant missing from the list is
counted as finishing just below the last listed place.

Season circuit points (which decide Champions qualification) sum every finished event of the
calendar year, leagues and Masters.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime

from valchamps.features.records import EventInfo, MatchRecord

NAN = math.nan


class PlacementTracker:
    def __init__(self, events: dict[int, EventInfo]) -> None:
        self.events = events
        self.played_in: defaultdict[int, set[int]] = defaultdict(set)  # team -> event ids

    def _finish(self, team: int, event: EventInfo) -> tuple[float, float]:
        """(place, circuit points) of ``team`` in a finished ``event``."""
        if not event.standings:
            return NAN, NAN
        if team in event.standings:
            place, points = event.standings[team]
            return float(place), float(points) if points is not None else NAN
        return float(max(p for p, _ in event.standings.values()) + 1), 0.0

    def snapshot(self, team: int, date: datetime) -> dict[str, float]:
        finished = sorted(
            (e for e in (self.events.get(i) for i in self.played_in[team]) if e is not None
             and e.end_date is not None and e.end_date < date.date()),
            key=lambda e: e.end_date,
        )  # fmt: skip
        league = [e for e in finished if not e.is_international]
        intl = [e for e in finished if e.is_international]
        league_place = self._finish(team, league[-1])[0] if league else NAN
        # Circuit points come from leagues and Masters alike (Champions awards none).
        season_points = [self._finish(team, e)[1] for e in finished if e.end_date.year == date.year]
        return {
            "league_place": league_place,
            "league_won": float(league_place == 1) if not math.isnan(league_place) else NAN,
            "season_points": math.fsum(p for p in season_points if not math.isnan(p)),
            "intl_place": self._finish(team, intl[-1])[0] if intl else NAN,
        }

    def update(self, match: MatchRecord) -> None:
        if match.event_id is not None:
            for team in match.teams:
                self.played_in[team].add(match.event_id)
