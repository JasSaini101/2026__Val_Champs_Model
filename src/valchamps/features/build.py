"""Assemble the per-map training table.

Matches are replayed oldest first. For each match every tracker is read *before* the match is
applied (those values are the match's features) and only then updated with its result, so a
row can never see its own match or anything later. Each map produces two rows, one from each
team's point of view, so the model cannot learn anything from which team vlr.gg lists first.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd

from valchamps.features.form import FormTracker
from valchamps.features.map_pool import MapPoolTracker
from valchamps.features.params import FeatureParams
from valchamps.features.placement import PlacementTracker
from valchamps.features.ratings import EloTracker
from valchamps.features.records import EventInfo, MapRecord, MatchRecord
from valchamps.features.roster import RosterTracker

META_COLUMNS = [
    "game_id", "match_id", "date", "event_id", "tier", "is_international", "cross_region",
    "map_name", "map_order", "best_of", "team_a", "team_b", "region_a", "region_b",
    "perspective", "y",
]  # fmt: skip

# Per-team values: each appears as <name>_a and <name>_b.
_TEAM_FEATURES = [
    "elo", "map_dev", "form_winrate", "form_round_diff", "rest_days", "matches_30d",
    "maps_played", "intl_maps", "intl_winrate", "map_played", "map_winrate", "pick_rate",
    "ban_rate", "roster_continuity", "lineup_rating", "league_place", "league_won",
    "season_points", "intl_place",
]  # fmt: skip
# Per-team values that also get an a-minus-b column.
_DIFF_FEATURES = [
    "elo", "region_offset", "map_dev", "form_winrate", "form_round_diff", "intl_winrate",
    "map_winrate", "lineup_rating", "league_place", "season_points",
]  # fmt: skip
_PAIR_FEATURES = ["pick_a", "pick_b", "decider", "elo_prob", "h2h_maps", "h2h_winrate"]

FEATURE_COLUMNS = (
    [f"{f}_{s}" for f in _TEAM_FEATURES for s in ("a", "b")]
    + [f"{f}_diff" for f in _DIFF_FEATURES]
    + _PAIR_FEATURES
)


def _diff(a: float, b: float) -> float:
    return a - b if not (math.isnan(a) or math.isnan(b)) else math.nan


class FeatureBuilder:
    """Holds every tracker; also the state used to score future (unplayed) matches."""

    def __init__(
        self, params: FeatureParams, regions: dict[int, str],
        events: dict[int, EventInfo] | None = None,
    ) -> None:  # fmt: skip
        self.params = params
        self.elo = EloTracker(params.elo, regions)
        self.form = FormTracker(params.form_window, params.intl_prior, params.h2h_prior)
        self.map_pool = MapPoolTracker(params.map_prior)
        self.roster = RosterTracker(params.player_window)
        self.placement = PlacementTracker(events or {})

    def rows_for(self, match: MatchRecord) -> list[dict]:
        """Feature rows for every map of ``match`` from the current (pre-match) state."""
        self.elo.start_season(match.date)
        a, b = match.teams
        team_state = {
            t: {
                "elo": self.elo.rating[t],
                "region_offset": self.elo.offset(t),
                **self.form.snapshot(t, o, match.date),
                **self.roster.snapshot(t, match),
                **self.placement.snapshot(t, match.date),
            }
            for t, o in ((a, b), (b, a))
        }
        rows = []
        for m in match.maps:
            for perspective, (x, y) in enumerate(((a, b), (b, a))):
                rows.append(self._row(match, m, x, y, perspective, team_state))
        return rows

    def _row(
        self, match: MatchRecord, m: MapRecord, a: int, b: int, perspective: int,
        team_state: dict[int, dict[str, float]],
    ) -> dict:  # fmt: skip
        region_a, region_b = self.elo.region(a), self.elo.region(b)
        side = {
            t: {
                **team_state[t],
                "map_dev": self.elo.map_dev[(t, m.map_name)],
                **self.map_pool.snapshot(t, m.map_name),
            }
            for t in (a, b)
        }
        h2h = self.form.snapshot(a, b, match.date)
        row: dict = {
            "game_id": m.game_id,
            "match_id": match.match_id,
            "date": match.date,
            "event_id": match.event_id,
            "tier": match.tier,
            "is_international": match.is_international,
            "cross_region": bool(region_a and region_b and region_a != region_b),
            "map_name": m.map_name,
            "map_order": m.map_order,
            "best_of": match.best_of,
            "team_a": a,
            "team_b": b,
            "region_a": region_a,
            "region_b": region_b,
            "perspective": perspective,
            "y": int(m.team1_won == (a == match.team1_id)),
            "pick_a": float(m.picked_by == a),
            "pick_b": float(m.picked_by == b),
            "decider": float(m.picked_by is None),
            "elo_prob": self.elo.expected(a, b, m.map_name),
            "h2h_maps": h2h["h2h_maps"],
            "h2h_winrate": h2h["h2h_winrate"],
        }
        for f in _TEAM_FEATURES:
            row[f"{f}_a"], row[f"{f}_b"] = side[a][f], side[b][f]
        for f in _DIFF_FEATURES:
            row[f"{f}_diff"] = _diff(side[a][f], side[b][f])
        return row

    def update(self, match: MatchRecord) -> None:
        self.elo.update(match)
        self.form.update(match)
        self.map_pool.update(match)
        self.roster.update(match)
        self.placement.update(match)


def build_feature_frame(
    records: Iterable[MatchRecord], regions: dict[int, str], params: FeatureParams | None = None,
    events: dict[int, EventInfo] | None = None,
) -> tuple[pd.DataFrame, FeatureBuilder]:  # fmt: skip
    """Replay ``records`` (oldest first) and return the training table and final state."""
    builder = FeatureBuilder(params or FeatureParams(), regions, events)
    rows: list[dict] = []
    previous = None
    for match in records:
        key = (match.date, match.match_id)
        if previous is not None and key < previous:
            raise ValueError("records must be sorted oldest first")
        previous = key
        rows.extend(builder.rows_for(match))
        builder.update(match)
    frame = pd.DataFrame(rows, columns=META_COLUMNS + FEATURE_COLUMNS)
    return frame, builder
