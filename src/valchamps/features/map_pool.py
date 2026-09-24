"""How good a team is on a given map, and how it treats that map in the veto."""

from __future__ import annotations

import math
from collections import defaultdict

from valchamps.features.form import shrunk_rate
from valchamps.features.records import MatchRecord

NAN = math.nan


class MapPoolTracker:
    def __init__(self, prior_strength: float) -> None:
        self.prior = prior_strength
        self.map_record: defaultdict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])
        self.overall: defaultdict[int, list[int]] = defaultdict(lambda: [0, 0])
        self.veto_matches: defaultdict[int, int] = defaultdict(int)
        self.picks: defaultdict[tuple[int, str], int] = defaultdict(int)
        self.bans: defaultdict[tuple[int, str], int] = defaultdict(int)

    def snapshot(self, team: int, map_name: str) -> dict[str, float]:
        won, played = self.map_record[(team, map_name)]
        overall_won, overall_played = self.overall[team]
        # Map win rate is shrunk toward the team's overall rate, itself shrunk toward 50%.
        overall_rate = shrunk_rate(overall_won, overall_played, 0.5, self.prior)
        vetoes = self.veto_matches[team]
        return {
            "map_played": float(played),
            "map_winrate": shrunk_rate(won, played, overall_rate, self.prior),
            "pick_rate": self.picks[(team, map_name)] / vetoes if vetoes else NAN,
            "ban_rate": self.bans[(team, map_name)] / vetoes if vetoes else NAN,
        }

    def update(self, match: MatchRecord) -> None:
        a, b = match.teams
        for m in match.maps:
            for team, won in ((a, m.team1_won), (b, not m.team1_won)):
                self.map_record[(team, m.map_name)][0] += won
                self.map_record[(team, m.map_name)][1] += 1
                self.overall[team][0] += won
                self.overall[team][1] += 1
        for team in {t for t, _, _ in match.vetoes}:
            self.veto_matches[team] += 1
        for team, action, map_name in match.vetoes:
            (self.picks if action == "pick" else self.bans)[(team, map_name)] += 1
