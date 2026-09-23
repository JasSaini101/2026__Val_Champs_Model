from __future__ import annotations

import re

import pytest
import respx
from sqlalchemy import select

from tests.conftest import load_fixture
from valchamps.data import db
from valchamps.data.ingest import EventSpec, ingest_event, load_event_specs
from valchamps.data.scraper import VlrClient

SPEC = EventSpec(event_id=2097, name="Champions 2024", tier="champions", is_lan=True)


@pytest.fixture
def vlr():
    with respx.mock(base_url="https://vlr.test", assert_all_called=False) as mock:
        mock.get("/event/2097").respond(200, text=load_fixture("event_2097.html"))
        mock.get(url__regex=r"/event/matches/2097/").respond(
            200, text=load_fixture("event_matches_2097.html")
        )
        mock.get(url__regex=r"^https://vlr.test/378829/").respond(
            200, text=load_fixture("match_378829_completed.html")
        )
        mock.get(url__regex=r"^https://vlr.test/378830/").respond(
            200, text=load_fixture("match_378830_upcoming.html")
        )
        mock.get(url__regex=r"^https://vlr.test/378831/").respond(503)
        yield mock


def client(settings):
    return VlrClient(settings, sleep=lambda _s: None, max_retries=1)


def test_ingest_event_end_to_end(settings, engine, vlr):
    with client(settings) as c:
        report = ingest_event(c, engine, SPEC)

    assert (report.listed, report.fetched, report.skipped) == (3, 2, 0)
    assert report.failed == [378831]
    with engine.connect() as conn:
        ev = conn.execute(select(db.events)).one()
        assert (ev.name, ev.tier, ev.is_lan) == ("Valorant Champions 2024", "champions", True)
        statuses = dict(conn.execute(select(db.matches.c.match_id, db.matches.c.status)).all())
        assert statuses == {378829: "completed", 378830: "upcoming"}
        failures = conn.execute(
            select(db.scrape_log.c.url_path).where(~db.scrape_log.c.ok)
        ).scalars()
        assert [re.match(r"/(\d+)/", p).group(1) for p in failures] == ["378831"]


def test_second_run_skips_completed_but_refreshes_upcoming(settings, engine, vlr):
    with client(settings) as c:
        ingest_event(c, engine, SPEC)
        report = ingest_event(c, engine, SPEC)
    assert report.skipped == 1  # the completed match
    assert report.fetched == 1  # the upcoming match is fetched again
    upcoming_route = next(r for r in vlr.routes if "378830" in str(r.pattern))
    assert upcoming_route.call_count == 2


def test_refresh_refetches_everything(settings, engine, vlr):
    with client(settings) as c:
        ingest_event(c, engine, SPEC)
        report = ingest_event(c, engine, SPEC, refresh=True)
    assert report.skipped == 0 and report.fetched == 2


def test_load_event_specs(tmp_path):
    path = tmp_path / "events.yaml"
    path.write_text("events:\n  - {event_id: 1, name: A, tier: champions, is_lan: true}\n")
    assert load_event_specs(path) == [EventSpec(1, "A", "champions", None, True)]


def test_repo_events_config_is_valid():
    from valchamps.cli import DEFAULT_EVENTS

    specs = load_event_specs(DEFAULT_EVENTS)
    assert specs and len({s.event_id for s in specs}) == len(specs)
