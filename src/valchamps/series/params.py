"""Series-model hyper-parameters, loaded from the ``series`` section of ``params.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SeriesParams:
    # Pseudo-vetoes per map added to each team's pick and ban counts: 0 follows a team's
    # history exactly, large values make every veto choice uniform.
    veto_prior: float = 5.0

    @classmethod
    def from_yaml(cls, path: Path) -> SeriesParams:
        if not path.exists():
            return cls()
        raw = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("series", {})
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SeriesParams:
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown series params: {sorted(unknown)}")
        return cls(**raw)
