"""Published odds: ``latest.json`` and an append-only ``history.csv`` per event.

A forecast is published only when the set of finished results differs from the last one
published, so a scheduled job that runs every hour writes (and commits) only when a match has
actually finished. Between results the odds would drift only through the passing days.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from valchamps.bracket.run import EventForecast

HISTORY_COLUMNS = ["updated_at", "as_of", "fixed_results", "results_fingerprint", "odds",
                   "team_id", "team", "group", "top8", "top4", "final", "title"]  # fmt: skip


def published_fingerprint(out_dir: Path) -> str | None:
    latest = out_dir / "latest.json"
    if not latest.exists():
        return None
    return json.loads(latest.read_text(encoding="utf-8")).get("results_fingerprint")


def publish(forecast: EventForecast, out_dir: Path, *, force: bool = False) -> bool:
    """Write the forecast unless the same results were already published. True if written."""
    if not force and published_fingerprint(out_dir) == forecast.fingerprint:
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    updated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    meta = {"updated_at": updated_at, **forecast.meta()}
    teams = forecast.table.round(5).to_dict(orient="records")
    (out_dir / "latest.json").write_text(
        json.dumps({**meta, "teams": teams}, indent=2) + "\n", encoding="utf-8"
    )
    rows = forecast.table.assign(**{k: meta[k] for k in HISTORY_COLUMNS if k in meta})
    rows = rows[HISTORY_COLUMNS].round(5)
    history = out_dir / "history.csv"
    rows.to_csv(history, mode="a", header=not history.exists(), index=False, lineterminator="\n")
    return True


def load_history(out_dir: Path) -> pd.DataFrame:
    return pd.read_csv(out_dir / "history.csv")
