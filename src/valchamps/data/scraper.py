"""Polite, cached HTTP client for vlr.gg.

Every page fetched is written to ``cache_dir`` so the parsing stage can be rerun (and versioned
with DVC) without hitting the site again. Pages that change while an event is running (event
match lists, live matches) can bypass the cache with ``max_age``.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from valchamps.config import Settings

log = logging.getLogger(__name__)

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class ScrapeError(RuntimeError):
    """Raised when a page cannot be fetched after all retries."""


def cache_key(path: str) -> str:
    """Map a URL path (plus query) to a readable, filesystem-safe file name."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", path.strip("/")).strip("_")[:80] or "index"
    digest = hashlib.sha1(path.encode()).hexdigest()[:10]
    return f"{slug}__{digest}.html"


class VlrClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 4,
        backoff: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings or Settings()
        self.cache_dir = Path(self.settings.cache_dir)
        self.max_retries = max_retries
        self.backoff = backoff
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None
        self._http = httpx.Client(
            base_url=self.settings.base_url,
            headers={"User-Agent": self.settings.user_agent},
            timeout=30.0,
            follow_redirects=True,
            transport=transport,
        )

    def __enter__(self) -> VlrClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def cache_path(self, path: str) -> Path:
        return self.cache_dir / cache_key(path)

    def get(self, path: str, *, max_age: float | None = None) -> str:
        """Return the HTML at ``path``, using the disk cache when it is fresh enough.

        ``max_age=None`` means cached pages never expire (right for finished matches);
        ``max_age=0`` always refetches.
        """
        cached = self.cache_path(path)
        if cached.exists() and (max_age is None or time.time() - cached.stat().st_mtime < max_age):
            log.debug("cache hit %s", path)
            return cached.read_text(encoding="utf-8")

        html = self._fetch(path)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(html, encoding="utf-8")
        return html

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self.settings.request_interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def _fetch(self, path: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                self._sleep(self.backoff * 2 ** (attempt - 1))
            self._throttle()
            try:
                response = self._http.get(path)
            except httpx.TransportError as exc:
                last_error = exc
                log.warning("transport error on %s (attempt %d): %s", path, attempt + 1, exc)
                continue
            if response.status_code in RETRY_STATUSES:
                last_error = ScrapeError(f"HTTP {response.status_code} for {path}")
                log.warning("HTTP %d on %s (attempt %d)", response.status_code, path, attempt + 1)
                continue
            if response.status_code != 200:
                raise ScrapeError(f"HTTP {response.status_code} for {path}")
            log.info("fetched %s", path)
            return response.text
        raise ScrapeError(
            f"giving up on {path} after {self.max_retries + 1} attempts"
        ) from last_error
