"""Scrape → parse → store, for one event or every event in ``configs/events.yaml``."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml
from sqlalchemy import Engine

from valchamps.data import db
from valchamps.data.models import Match, MatchListing
from valchamps.data.parser import (
    ParseError,
    TeamsNotDecided,
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
    pending: int = 0  # bracket matches whose teams are not decided yet
    failed: list[int] = field(default_factory=list)
    # Listed as completed, but the match page doesn't show a final result yet (vlr.gg lag).
    # Stored as they are and fetched again on the next run.
    not_final: list[int] = field(default_factory=list)


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
        if not listing.teams_decided:
            report.pending += 1  # fetched on a later run, once the bracket fills in
            continue
        try:
            match = _fetch_match(client, listing)
        except TeamsNotDecided:
            report.pending += 1
            continue
        except (ScrapeError, ParseError) as exc:
            log.error("match %s failed: %s", listing.match_id, exc)
            report.failed.append(listing.match_id)
            with engine.begin() as conn:
                db.log_scrape(conn, listing.url_path, ok=False, message=str(exc))
            continue
        if listing.status == "completed" and match.status != "completed":
            log.warning("match %s is listed as completed but its page is not final yet",
                        listing.match_id)  # fmt: skip
            report.not_final.append(listing.match_id)
        with engine.begin() as conn:
            db.save_match(conn, match, event_id=spec.event_id)
            db.log_scrape(conn, listing.url_path, ok=True)
        report.fetched += 1

    # After the matches, so every placed team already exists.
    with engine.begin() as conn:
        db.save_standings(conn, spec.event_id, parse_standings(event_html))

    log.info(
        "event %s: %d listed, %d fetched, %d skipped, %d pending, %d failed, %d not final",
        spec.event_id, report.listed, report.fetched, report.skipped, report.pending,
        len(report.failed), len(report.not_final),
    )  # fmt: skip
    return report


def _fetch_match(client: VlrClient, listing: MatchListing) -> Match:
    """Fetch and parse a match page, from the cache when that copy can be trusted.

    A finished match's page never changes, so a cached copy of it is reused forever. But the
    copy may have been saved before the match finished (a run while it was upcoming or live);
    then the listing says completed while the cached page doesn't, and the page is refetched.
    Matches not yet finished are always refetched.
    """
    if listing.status != "completed":
        return parse_match(client.get(listing.url_path, max_age=0), listing.match_id)
    match = parse_match(client.get(listing.url_path, max_age=None), listing.match_id)
    if match.status != "completed":
        log.info("cached page of match %s predates its result; refetching", listing.match_id)
        match = parse_match(client.get(listing.url_path, max_age=0), listing.match_id)
    return match
