"""Recent form, schedule load, international record and head-to-head."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import datetime

from valchamps.features.records import MatchRecord

NAN = math.nan


def shrunk_rate(won: float, played: float, prior_rate: float, prior_strength: float) -> float:
    """Win rate pulled toward ``prior_rate`` as if ``prior_strength`` extra games were played."""
    return (won + prior_rate * prior_strength) / (played + prior_strength)


class FormTracker:
    def __init__(self, window: int, intl_prior: float, h2h_prior: float) -> None:
        self.intl_prior = intl_prior
        self.h2h_prior = h2h_prior
        # Per match: (maps won, maps played, round difference)
        self.recent: defaultdict[int, deque[tuple[int, int, int]]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self.match_dates: defaultdict[int, list[datetime]] = defaultdict(list)
        self.maps_played: defaultdict[int, int] = defaultdict(int)
        self.intl: defaultdict[int, list[int]] = defaultdict(lambda: [0, 0])  # [won, played]
        self.h2h: defaultdict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])

    def snapshot(self, team: int, opponent: int, date: datetime) -> dict[str, float]:
        recent = self.recent[team]
        won = sum(r[0] for r in recent)
        played = sum(r[1] for r in recent)
        dates = self.match_dates[team]
        intl_won, intl_played = self.intl[team]
        h2h_won, h2h_played = self.h2h[(team, opponent)]
        return {
            "form_winrate": won / played if played else NAN,
            "form_round_diff": sum(r[2] for r in recent) / played if played else NAN,
            "rest_days": (date - dates[-1]).total_seconds() / 86400 if dates else NAN,
            "matches_30d": float(sum((date - d).days < 30 for d in dates[-30:])),
            "maps_played": float(self.maps_played[team]),
            "intl_maps": float(intl_played),
            "intl_winrate": shrunk_rate(intl_won, intl_played, 0.5, self.intl_prior),
            "h2h_maps": float(h2h_played),
            "h2h_winrate": shrunk_rate(h2h_won, h2h_played, 0.5, self.h2h_prior),
        }

    def update(self, match: MatchRecord) -> None:
        played = len(match.maps)
        for team, opponent in (match.teams, match.teams[::-1]):
            won = match.maps_won(team)
            self.recent[team].append((won, played, match.round_diff(team)))
            self.match_dates[team].append(match.date)
            self.maps_played[team] += played
            if match.is_international:
                self.intl[team][0] += won
                self.intl[team][1] += played
            self.h2h[(team, opponent)][0] += won
            self.h2h[(team, opponent)][1] += played
