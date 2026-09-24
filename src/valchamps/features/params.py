"""Feature hyper-parameters, loaded from the ``features`` section of ``params.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EloParams:
    init: float = 1500.0
    scale: float = 400.0
    k: float = 32.0  # overall team rating
    map_k: float = 16.0  # team-on-this-map deviation
    region_k: float = 16.0  # region offset, cross-region maps only
    # Share of a rating's distance from the mean kept across a new calendar year.
    season_carryover: float = 0.75


@dataclass(frozen=True)
class FeatureParams:
    elo: EloParams = field(default_factory=EloParams)
    form_window: int = 5  # matches
    map_prior: float = 5.0  # pseudo-maps pulling map win rate toward overall win rate
    h2h_prior: float = 3.0  # pseudo-maps pulling head-to-head win rate toward 0.5
    intl_prior: float = 5.0  # pseudo-maps pulling international win rate toward 0.5
    player_window: int = 10  # matches of player history for lineup rating

    @classmethod
    def from_yaml(cls, path: Path) -> FeatureParams:
        raw = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("features", {})
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FeatureParams:
        known = {f.name for f in fields(cls)} - {"elo"}
        unknown = set(raw) - known - {"elo"}
        if unknown:
            raise ValueError(f"unknown feature params: {sorted(unknown)}")
        return cls(elo=EloParams(**raw.get("elo", {})), **{k: raw[k] for k in known & set(raw)})
