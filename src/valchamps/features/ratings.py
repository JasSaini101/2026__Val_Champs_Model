"""Elo ratings: per team, per team-and-map, and per region.

A team's effective strength on a map is::

    rating[team] + region_offset[region(team)] + map_dev[team, map]

* ``rating`` moves on every map the team plays; within a region it is zero-sum.
* ``region_offset`` moves only on maps between teams from different regions (Masters,
  Champions), so it learns how the regions compare from the few matches that show it.
* ``map_dev`` is a small per-map adjustment on top of the overall rating.

Updates scale with the round margin (a 13-2 moves ratings more than 14-12), damped when the
favourite wins, as in FiveThirtyEight's Elo.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime

from valchamps.features.params import EloParams
from valchamps.features.records import MatchRecord

_MOV_NORMALISER = math.log(8)  # a 13-6 map gets a multiplier of about 1


class EloTracker:
    def __init__(self, params: EloParams, regions: dict[int, str]) -> None:
        self.params = params
        self.regions = regions
        self.rating: defaultdict[int, float] = defaultdict(lambda: params.init)
        self.map_dev: defaultdict[tuple[int, str], float] = defaultdict(float)
        self.region_offset: defaultdict[str, float] = defaultdict(float)
        self.season: int | None = None

    def region(self, team: int) -> str | None:
        return self.regions.get(team)

    def offset(self, team: int) -> float:
        region = self.region(team)
        return self.region_offset[region] if region else 0.0

    def strength(self, team: int, map_name: str | None = None) -> float:
        dev = self.map_dev[(team, map_name)] if map_name else 0.0
        return self.rating[team] + self.offset(team) + dev

    def expected(self, a: int, b: int, map_name: str | None = None) -> float:
        """P(a beats b) on ``map_name`` (or on an unknown map)."""
        diff = self.strength(a, map_name) - self.strength(b, map_name)
        return 1.0 / (1.0 + 10 ** (-diff / self.params.scale))

    def start_season(self, date: datetime) -> None:
        """At a new calendar year, pull team and map ratings part way back to the mean.

        Region offsets are kept: the gap between regions is structural, not seasonal.
        """
        if self.season is not None and date.year != self.season:
            keep = self.params.season_carryover ** (date.year - self.season)
            for team, r in self.rating.items():
                self.rating[team] = self.params.init + keep * (r - self.params.init)
            for key, dev in self.map_dev.items():
                self.map_dev[key] = keep * dev
        self.season = date.year

    def update(self, match: MatchRecord) -> None:
        a, b = match.teams
        cross_region = (
            self.region(a) is not None
            and self.region(b) is not None
            and self.region(a) != self.region(b)
        )
        for m in match.maps:
            expected = self.expected(a, b, m.map_name)
            won = 1.0 if m.team1_won else 0.0
            margin = abs(m.team1_rounds - m.team2_rounds)
            winner_edge = self.strength(a, m.map_name) - self.strength(b, m.map_name)
            if not m.team1_won:
                winner_edge = -winner_edge
            mov = math.log(margin + 1) / _MOV_NORMALISER * 2.2 / (winner_edge * 0.001 + 2.2)
            delta = mov * (won - expected)

            self.rating[a] += self.params.k * delta
            self.rating[b] -= self.params.k * delta
            self.map_dev[(a, m.map_name)] += self.params.map_k * delta
            self.map_dev[(b, m.map_name)] -= self.params.map_k * delta
            if cross_region:
                self.region_offset[self.region(a)] += self.params.region_k * delta
                self.region_offset[self.region(b)] -= self.params.region_k * delta
