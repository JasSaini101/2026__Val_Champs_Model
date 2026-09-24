"""The dashboard's view of the API."""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_API_URL = os.environ.get("VALCHAMPS_API_URL", "http://localhost:8000")


class ApiError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, **params: Any) -> Any:
        try:
            r = httpx.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise ApiError(f"cannot reach the API at {self.base_url} ({exc})") from exc
        if r.status_code != 200:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ApiError(f"{r.status_code}: {detail}")
        return r.json()

    def odds(self, event: int) -> dict:
        return self._get(f"/events/{event}/odds")

    def history(self, event: int) -> list[dict]:
        return self._get(f"/events/{event}/odds/history")

    def teams(self, event: int) -> list[dict]:
        return self._get(f"/events/{event}/teams")

    def predict(self, team_a: int, team_b: int, best_of: int, event: int) -> dict:
        return self._get("/predict", team_a=team_a, team_b=team_b, best_of=best_of, event_id=event)
