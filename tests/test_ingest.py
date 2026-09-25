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

    assert (report.listed, report.fetched, report.skipped, report.pending) == (4, 2, 0, 1)
    assert report.failed == [378831]
    # The TBD bracket slot is never requested.
    assert not any("378999" in str(call.request.url) for call in vlr.calls)
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


def _listing_with_378830(status: str) -> str:
    """The fixture's match list, with the upcoming grand final (378830) given ``status``."""
    html = load_fixture("event_matches_2097.html")
    upcoming = '<div class="ml-status">Upcoming</div>'
    assert html.count(upcoming) == 1
    return html.replace(upcoming, f'<div class="ml-status">{status}</div>')


def _calls(vlr, match_id: int) -> int:
    return sum(f"/{match_id}/" in str(call.request.url) for call in vlr.calls)


def _statuses(engine) -> dict[int, str]:
    with engine.connect() as conn:
        return dict(conn.execute(select(db.matches.c.match_id, db.matches.c.status)).all())


def test_finished_match_with_a_stale_cached_page_is_refetched(settings, engine, vlr):
    """A page cached while the match was upcoming must not hide its result once it finishes."""
    with client(settings) as c:
        ingest_event(c, engine, SPEC)  # 378830 is upcoming: its page is cached as upcoming
        c.cache_path("/event/matches/2097/?series_id=all").unlink()  # as if 15 minutes passed
    assert _statuses(engine)[378830] == "upcoming"

    vlr.get(url__regex=r"/event/matches/2097/").respond(200, text=_listing_with_378830("Completed"))
    vlr.get(url__regex=r"^https://vlr.test/378830/").respond(
        200, text=load_fixture("match_378829_completed.html")
    )
    before = _calls(vlr, 378830)
    with client(settings) as c:
        report = ingest_event(c, engine, SPEC)
    assert _calls(vlr, 378830) == before + 1  # the stale copy was replaced
    assert report.fetched == 1 and report.not_final == []
    assert _statuses(engine)[378830] == "completed"


def test_finished_match_with_a_final_cached_page_is_not_refetched(settings, engine, vlr):
    with client(settings) as c:
        ingest_event(c, engine, SPEC)
        ingest_event(c, engine, SPEC, refresh=True)  # re-reads 378829, which is final
    assert _calls(vlr, 378829) == 1


def test_listed_as_finished_but_page_not_final(settings, engine, vlr):
    """vlr.gg can mark a match completed before its page shows the result: retry next run."""
    vlr.get(url__regex=r"/event/matches/2097/").respond(200, text=_listing_with_378830("Completed"))
    with client(settings) as c:
        report = ingest_event(c, engine, SPEC)
    assert report.not_final == [378830]
    assert _calls(vlr, 378830) == 2  # no cached copy, then one refetch to make sure
    assert _statuses(engine)[378830] == "upcoming"
    with engine.connect() as conn:
        assert 378830 not in db.completed_match_ids(conn)  # so the next run tries again


def test_load_event_specs(tmp_path):
    path = tmp_path / "events.yaml"
    path.write_text("events:\n  - {event_id: 1, name: A, tier: champions, is_lan: true}\n")
    assert load_event_specs(path) == [EventSpec(1, "A", "champions", None, True)]


def test_repo_events_config_is_valid():
    from valchamps.cli import DEFAULT_EVENTS

    specs = load_event_specs(DEFAULT_EVENTS)
    assert specs and len({s.event_id for s in specs}) == len(specs)


def test_match_page_still_tbd_counts_as_pending(settings, engine, vlr):
    """The listing names both teams, but the match page has no team links yet."""
    tbd = load_fixture("match_378830_upcoming.html")
    for team_href in ('href="/team/1120/edward-gaming"', 'href="/team/2593/fnatic"'):
        tbd = tbd.replace(team_href, 'href="#"')
    vlr.get(url__regex=r"^https://vlr.test/378830/").respond(200, text=tbd)
    with client(settings) as c:
        report = ingest_event(c, engine, SPEC)
    assert report.pending == 2  # the TBD listing card plus this page
    assert report.failed == [378831]  # only the real HTTP failure
    with engine.connect() as conn:
        assert not conn.execute(
            select(db.scrape_log).where(db.scrape_log.c.url_path.contains("378830"))
        ).first()


def test_cli_exit_code_ignores_pending_matches(settings, monkeypatch, tmp_path):
    """Pending (TBD) matches must not make `valchamps ingest` fail, so `dvc repro` succeeds."""
    from typer.testing import CliRunner

    from valchamps.cli import app

    events = tmp_path / "events.yaml"
    events.write_text("events:\n  - {event_id: 2097, name: Champions, tier: champions}\n")
    monkeypatch.setenv("VALCHAMPS_DB_URL", settings.db_url)
    monkeypatch.setenv("VALCHAMPS_CACHE_DIR", str(settings.cache_dir))
    monkeypatch.setenv("VALCHAMPS_BASE_URL", settings.base_url)
    monkeypatch.setenv("VALCHAMPS_REQUEST_INTERVAL", "0")
    with respx.mock(base_url="https://vlr.test", assert_all_called=False) as mock:
        mock.get("/event/2097").respond(200, text=load_fixture("event_2097.html"))
        mock.get(url__regex=r"/event/matches/2097/").respond(
            200, text=load_fixture("event_matches_2097.html")
        )
        for mid, fixture in ((378829, "match_378829_completed.html"),
                             (378830, "match_378830_upcoming.html"),
                             (378831, "match_378830_upcoming.html")):  # fmt: skip
            mock.get(url__regex=rf"^https://vlr.test/{mid}/").respond(
                200, text=load_fixture(fixture)
            )
        result = CliRunner().invoke(app, ["ingest", "--events-file", str(events)])
    assert result.exit_code == 0, result.output
    assert "1 pending (teams TBD), 0 failed" in result.output
