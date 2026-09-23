from __future__ import annotations

import os
import time

import httpx
import pytest
import respx

from valchamps.data.scraper import ScrapeError, VlrClient, cache_key


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


def make_client(settings, clock, **kw):
    return VlrClient(settings, sleep=clock.sleep, clock=clock.time, **kw)


@respx.mock
def test_fetch_writes_cache_and_reuses_it(settings, clock):
    route = respx.get("https://vlr.test/123/a-vs-b").mock(
        return_value=httpx.Response(200, text="<p>hi</p>")
    )
    with make_client(settings, clock) as client:
        assert client.get("/123/a-vs-b") == "<p>hi</p>"
        assert client.get("/123/a-vs-b") == "<p>hi</p>"
        assert client.cache_path("/123/a-vs-b").exists()
    assert route.call_count == 1


@respx.mock
def test_max_age_refetches_stale_pages(settings, clock):
    route = respx.get("https://vlr.test/event/matches/1/").mock(
        side_effect=[httpx.Response(200, text="old"), httpx.Response(200, text="new")]
    )
    with make_client(settings, clock) as client:
        assert client.get("/event/matches/1/", max_age=60) == "old"
        path = client.cache_path("/event/matches/1/")
        stale = time.time() - 120
        os.utime(path, (stale, stale))
        assert client.get("/event/matches/1/", max_age=60) == "new"
    assert route.call_count == 2


@respx.mock
def test_retries_on_server_errors_with_backoff(settings, clock):
    route = respx.get("https://vlr.test/1/x").mock(
        side_effect=[httpx.Response(503), httpx.Response(429), httpx.Response(200, text="ok")]
    )
    with make_client(settings, clock, backoff=1.0) as client:
        assert client.get("/1/x") == "ok"
    assert route.call_count == 3
    assert clock.sleeps == [1.0, 2.0]


@respx.mock
def test_gives_up_after_max_retries(settings, clock):
    respx.get("https://vlr.test/1/x").mock(return_value=httpx.Response(500))
    with make_client(settings, clock, max_retries=2) as client, pytest.raises(ScrapeError):
        client.get("/1/x")
    assert not client.cache_path("/1/x").exists()


@respx.mock
def test_transport_errors_are_retried(settings, clock):
    respx.get("https://vlr.test/1/x").mock(
        side_effect=[httpx.ConnectError("boom"), httpx.Response(200, text="ok")]
    )
    with make_client(settings, clock) as client:
        assert client.get("/1/x") == "ok"


@respx.mock
def test_client_errors_are_not_retried(settings, clock):
    route = respx.get("https://vlr.test/404").mock(return_value=httpx.Response(404))
    with make_client(settings, clock) as client, pytest.raises(ScrapeError, match="404"):
        client.get("/404")
    assert route.call_count == 1


@respx.mock
def test_requests_are_throttled(settings, clock):
    from dataclasses import replace

    respx.get(url__regex=r"https://vlr.test/\d+/p").mock(return_value=httpx.Response(200, text="x"))
    with make_client(replace(settings, request_interval=2.0), clock) as client:
        client.get("/1/p")
        clock.now += 0.5
        client.get("/2/p")
    assert clock.sleeps == [1.5]


def test_cache_key_is_stable_and_distinct():
    assert cache_key("/event/matches/2097/?series_id=all") == cache_key(
        "/event/matches/2097/?series_id=all"
    )
    assert cache_key("/a?x=1") != cache_key("/a?x=2")
    assert "/" not in cache_key("/event/matches/2097/")
