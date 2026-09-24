"""The dashboard's view of the data: the API, or the files ``valchamps update`` publishes."""

from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

DEFAULT_API_URL = os.environ.get("VALCHAMPS_API_URL", "http://localhost:8000")
# The published odds folder (a URL or a path). When set, the dashboard reads it instead of the API.
DEFAULT_DATA_URL = os.environ.get("VALCHAMPS_DATA_URL")


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


class StaticClient:
    """The same calls as :class:`ApiClient`, answered from the published ``odds/`` folder.

    ``base`` is that folder, as a local path or a URL (e.g. the repository's raw GitHub view),
    holding ``<event>/latest.json``, ``history.csv`` and ``matchups.json``. Predictions come from
    ``matchups.json``, so only pairings of the event's teams are available.
    """

    def __init__(self, base: str, timeout: float = 30.0) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout

    def _read(self, event: int, name: str) -> str:
        if self.base.startswith(("http://", "https://")):
            url = f"{self.base}/{event}/{name}"
            try:
                r = httpx.get(url, timeout=self.timeout, follow_redirects=True)
            except httpx.HTTPError as exc:
                raise ApiError(f"cannot fetch {url} ({exc})") from exc
            if r.status_code != 200:
                raise ApiError(f"{r.status_code}: nothing published at {url}")
            return r.text
        path = Path(self.base) / str(event) / name
        if not path.exists():
            raise ApiError(f"404: nothing published at {path}")
        return path.read_text(encoding="utf-8")

    def odds(self, event: int) -> dict:
        return json.loads(self._read(event, "latest.json"))

    def history(self, event: int) -> list[dict]:
        rows = pd.read_csv(io.StringIO(self._read(event, "history.csv"))).to_dict("records")
        return [
            {k: None if isinstance(v, float) and math.isnan(v) else v for k, v in row.items()}
            for row in rows
        ]

    def teams(self, event: int) -> list[dict]:
        return [{**t, "tag": None} for t in self._matchups(event)["teams"]]

    def predict(self, team_a: int, team_b: int, best_of: int, event: int) -> dict:
        doc = self._matchups(event)
        lo, hi = sorted((team_a, team_b))
        body = doc["matchups"].get(str(best_of), {}).get(f"{lo}-{hi}")
        if body is None:
            raise ApiError(f"404: no Bo{best_of} prediction published for teams {lo} and {hi}")
        if team_a > team_b:
            body = flip_prediction(body)
        names = {t["team_id"]: t["name"] for t in doc["teams"]}
        return {
            "team_a": {"team_id": team_a, "name": names.get(team_a, str(team_a))},
            "team_b": {"team_id": team_b, "name": names.get(team_b, str(team_b))},
            "best_of": best_of,
            **body,
            "map_pool": doc["map_pool"],
            "model": doc["model"],
            "data_through": doc["data_through"],
        }

    def _matchups(self, event: int) -> dict:
        return json.loads(self._read(event, "matchups.json"))


_SWAP = {"pick_a": "pick_b", "pick_b": "pick_a", "decider": "decider"}


def flip_prediction(body: dict) -> dict:
    """A published prediction from the other team's point of view."""
    return {
        "p_a": body["p_b"],
        "p_b": body["p_a"],
        "elo_p_a": 1 - body["elo_p_a"],
        "scores": sorted(
            ({"a": s["b"], "b": s["a"], "p": s["p"]} for s in body["scores"]),
            key=lambda s: s["b"] - s["a"],
        ),
        "maps": [
            {
                "map": m["map"],
                "pick_a": 1 - m["pick_b"],
                "pick_b": 1 - m["pick_a"],
                "decider": 1 - m["decider"],
                "in_series": m["in_series"],
            }
            for m in body["maps"]
        ],
        "vetoes": [
            {
                "p": v["p"],
                "maps": [{"map": m["map"], "picked_by": _SWAP[m["picked_by"]]} for m in v["maps"]],
            }
            for v in body["vetoes"]
        ],
    }
