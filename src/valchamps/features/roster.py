"""Roster stability and lineup quality from per-player ratings."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from statistics import fmean

from valchamps.features.records import MatchRecord

NAN = math.nan


class RosterTracker:
    def __init__(self, player_window: int) -> None:
        self.last_lineup: dict[int, frozenset[int]] = {}
        self.history: defaultdict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=player_window)
        )

    def lineup(self, team: int, match: MatchRecord) -> frozenset[int]:
        """Players fielded in ``match``; falls back to the last known lineup (for predictions)."""
        players = match.lineups.get(team)
        return frozenset(players) if players else self.last_lineup.get(team, frozenset())

    def snapshot(self, team: int, match: MatchRecord) -> dict[str, float]:
        lineup = self.lineup(team, match)
        previous = self.last_lineup.get(team)
        known = [fmean(self.history[p]) for p in lineup if self.history[p]]
        return {
            "roster_continuity": len(lineup & previous) / len(lineup)
            if lineup and previous is not None
            else NAN,
            "lineup_rating": fmean(known) if known else NAN,
        }

    def update(self, match: MatchRecord) -> None:
        for team, players in match.lineups.items():
            if not players:
                continue
            self.last_lineup[team] = frozenset(players)
            for player, rating in players.items():
                if rating is not None:
                    self.history[player].append(rating)
