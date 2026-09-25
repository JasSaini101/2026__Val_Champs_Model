"""Published odds per event: ``latest.json``, an append-only ``history.csv``, and
``matchups.json`` (every pairing's series prediction, for a dashboard without the API).

The update job runs once a day (around noon US Eastern). A forecast is published when the set
of finished results differs from the last one published, or when nothing has been published yet
that Eastern-time day, so ``history.csv`` gets one snapshot per day of the event. A second run
on the same day with the same results writes nothing (and so commits nothing).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from valchamps.bracket.run import EventForecast

HISTORY_COLUMNS = ["updated_at", "as_of", "fixed_results", "results_fingerprint", "odds",
                   "team_id", "team", "group", "top8", "top4", "final", "title"]  # fmt: skip


# The day a snapshot belongs to, for the daily schedule and the day-by-day chart.
EASTERN = ZoneInfo("America/New_York")


def eastern_day(timestamp: datetime | str) -> date:
    """The US Eastern calendar day of a timezone-aware timestamp (or its ISO string)."""
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)
    return timestamp.astimezone(EASTERN).date()


def _published(out_dir: Path) -> dict | None:
    latest = out_dir / "latest.json"
    if not latest.exists():
        return None
    return json.loads(latest.read_text(encoding="utf-8"))


def published_fingerprint(out_dir: Path) -> str | None:
    last = _published(out_dir)
    return last.get("results_fingerprint") if last else None


def publish(
    forecast: EventForecast, out_dir: Path, *, force: bool = False, now: datetime | None = None
) -> bool:
    """Write the forecast if results changed or nothing was published yet today (US Eastern).

    True if written. ``now`` (timezone-aware, default the current time) is for tests.
    """
    now = (now or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    last = _published(out_dir)
    if not force and last is not None:
        same_results = last.get("results_fingerprint") == forecast.fingerprint
        same_day = "updated_at" in last and eastern_day(last["updated_at"]) == eastern_day(now)
        if same_results and same_day:
            return False
    out_dir.mkdir(parents=True, exist_ok=True)
    updated_at = now.isoformat()
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


def write_matchups(forecast: EventForecast, out_dir: Path) -> bool:
    """Write every pairing's prediction (``matchups.json``), if the forecast has them."""
    if forecast.matchups is None:
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(forecast.matchups, separators=(",", ":"))  # compact: rewritten each update
    (out_dir / "matchups.json").write_text(text + "\n", encoding="utf-8")
    return True


def load_history(out_dir: Path) -> pd.DataFrame:
    return pd.read_csv(out_dir / "history.csv")
