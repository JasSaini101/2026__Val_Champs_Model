"""Scrape → parse → store, for one event or every event in ``configs/events.yaml``."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml
from sqlalchemy import Engine

from valchamps.data import db
from valchamps.data.parser import (
    ParseError,
    parse_event,
    parse_event_matches,
    parse_match,
    parse_standings,
)
from valchamps.data.scraper import ScrapeError, VlrClient

log = logging.getLogger(__name__)

# Match lists change while an event is live; refetch them if older than this (seconds).
LISTING_MAX_AGE = 15 * 60


@dataclass(frozen=True)
class EventSpec:
    event_id: int
    name: str
    tier: str | None = None
    region: str | None = None
    is_lan: bool | None = None


@dataclass
class IngestReport:
    event_id: int
    listed: int = 0
    fetched: int = 0
    skipped: int = 0
    failed: list[int] = field(default_factory=list)


def load_event_specs(path: Path) -> list[EventSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [EventSpec(**entry) for entry in raw.get("events", [])]


def ingest_event(
    client: VlrClient, engine: Engine, spec: EventSpec, *, refresh: bool = False
) -> IngestReport:
    """Fetch every match of an event. Completed matches already stored are skipped unless
    ``refresh``; live and upcoming matches are always refetched so the bracket stays current."""
    report = IngestReport(event_id=spec.event_id)

    event_html = client.get(f"/event/{spec.event_id}", max_age=LISTING_MAX_AGE)
    event = parse_event(event_html, spec.event_id)
    if event.name == f"event-{spec.event_id}":  # title not found on the page
        event = replace(event, name=spec.name)
    listing_path = f"/event/matches/{spec.event_id}/?series_id=all"
    listings = parse_event_matches(client.get(listing_path, max_age=LISTING_MAX_AGE))
    report.listed = len(listings)

    with engine.begin() as conn:
        db.upsert_event(conn, event, tier=spec.tier, region=spec.region, is_lan=spec.is_lan)
        done = set() if refresh else db.completed_match_ids(conn)

    for listing in listings:
        if listing.match_id in done:
            report.skipped += 1
            continue
        # Finished matches never change, so their cached page is always valid.
        max_age = None if listing.status == "completed" else 0
        try:
            match = parse_match(client.get(listing.url_path, max_age=max_age), listing.match_id)
        except (ScrapeError, ParseError) as exc:
            log.error("match %s failed: %s", listing.match_id, exc)
            report.failed.append(listing.match_id)
            with engine.begin() as conn:
                db.log_scrape(conn, listing.url_path, ok=False, message=str(exc))
            continue
        with engine.begin() as conn:
            db.save_match(conn, match, event_id=spec.event_id)
            db.log_scrape(conn, listing.url_path, ok=True)
        report.fetched += 1

    # After the matches, so every placed team already exists.
    with engine.begin() as conn:
        db.save_standings(conn, spec.event_id, parse_standings(event_html))

    log.info(
        "event %s: %d listed, %d fetched, %d skipped, %d failed",
        spec.event_id, report.listed, report.fetched, report.skipped, len(report.failed),
    )  # fmt: skip
    return report
