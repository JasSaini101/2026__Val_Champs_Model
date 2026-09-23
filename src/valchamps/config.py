"""Runtime settings, overridable through ``VALCHAMPS_*`` environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(f"VALCHAMPS_{name}", default)


@dataclass(frozen=True)
class Settings:
    db_url: str = field(
        default_factory=lambda: _env(
            "DB_URL", f"sqlite:///{PROJECT_ROOT / 'data' / 'valchamps.db'}"
        )
    )
    cache_dir: Path = field(
        default_factory=lambda: Path(_env("CACHE_DIR", str(PROJECT_ROOT / "data" / "raw" / "vlr")))
    )
    base_url: str = field(default_factory=lambda: _env("BASE_URL", "https://www.vlr.gg"))
    # Seconds between requests. vlr.gg is a community site, so stay polite.
    request_interval: float = field(default_factory=lambda: float(_env("REQUEST_INTERVAL", "2.0")))
    user_agent: str = field(
        default_factory=lambda: _env(
            "USER_AGENT",
            "valchamps/0.1 (research project; +https://github.com/JasSaini101/2026__Val_Champs_Model)",
        )
    )
