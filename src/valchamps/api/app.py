"""HTTP API: the published odds and live match predictions.

Run with ``valchamps serve`` (or ``uvicorn valchamps.api.app:app``). Published odds come from
``odds/<event>/`` (written by ``valchamps update``). Match predictions replay the database and
use the trained map model. That state is loaded on first use and reloaded when the database
or model file changes, so the scheduled update is picked up without a restart.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.engine import make_url

from valchamps.bracket.event import group_openings
from valchamps.bracket.run import current_map_pool, prediction_date
from valchamps.config import PROJECT_ROOT, Settings
from valchamps.data import db
from valchamps.features import (
    FeatureParams,
    build_feature_frame,
    load_events,
    load_records,
    team_regions,
)
from valchamps.series import FORMATS, SeriesParams, map_play_probabilities, predict_series

CHAMPIONS_2026 = 2766


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(f"VALCHAMPS_{name}", PROJECT_ROOT / default))


@dataclass(frozen=True)
class ApiSettings:
    db_url: str = field(default_factory=lambda: Settings().db_url)
    odds_dir: Path = field(default_factory=lambda: _env_path("ODDS_DIR", "odds"))
    model_path: Path = field(
        default_factory=lambda: _env_path("MODEL_PATH", "models/map_model.pkl")
    )
    params_file: Path = PROJECT_ROOT / "params.yaml"


class _State:
    """History replayed into a FeatureBuilder, plus the map model; reloaded when files change."""

    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._key: tuple | None = None
        self._value: dict[str, Any] | None = None

    def _files_key(self) -> tuple:
        url = make_url(self.settings.db_url)
        paths = [self.settings.model_path, self.settings.params_file]
        if url.get_backend_name() == "sqlite" and url.database:
            paths.append(Path(url.database))
        return tuple(p.stat().st_mtime if p.exists() else None for p in paths)

    def get(self) -> dict[str, Any]:
        with self._lock:
            key = self._files_key()
            if self._value is None or key != self._key:
                self._value, self._key = self._load(), key
            return self._value

    def _load(self) -> dict[str, Any]:
        s = self.settings
        engine = db.get_engine(s.db_url)
        records = load_records(engine)
        if not records:
            raise HTTPException(503, "no completed matches in the database yet")
        events = load_events(engine)
        params = FeatureParams.from_yaml(s.params_file) if s.params_file.exists() else None
        _, builder = build_feature_frame(records, team_regions(engine), params, events)
        with engine.connect() as conn:
            teams = conn.execute(select(db.teams.c.team_id, db.teams.c.name, db.teams.c.tag))
            info = {t.team_id: {"name": t.name, "tag": t.tag} for t in teams}
        played = {t for r in records for t in r.teams}
        bundle = pickle.loads(s.model_path.read_bytes()) if s.model_path.exists() else None
        return {
            "engine": engine,
            "records": records,
            "events": events,
            "builder": builder,
            "teams": info,
            "played": played,
            "bundle": bundle,
            "veto_prior": SeriesParams.from_yaml(s.params_file).veto_prior,
        }


def _read_published(odds_dir: Path, event: int) -> tuple[dict | None, pd.DataFrame | None]:
    d = odds_dir / str(event)
    latest = (
        json.loads((d / "latest.json").read_text("utf-8")) if (d / "latest.json").exists() else None
    )
    history = pd.read_csv(d / "history.csv") if (d / "history.csv").exists() else None
    return latest, history


def _clean(value: Any) -> Any:
    """JSON-safe: NaN becomes null."""
    return None if isinstance(value, float) and math.isnan(value) else value


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings()
    state = _State(settings)
    app = FastAPI(
        title="VALORANT Champions 2026 odds",
        description="Title odds from a Monte Carlo bracket simulation, and series predictions.",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/events/{event_id}/odds")
    def odds(event_id: int) -> dict[str, Any]:
        """The latest published forecast: per-team odds of every final standing."""
        latest, _ = _read_published(settings.odds_dir, event_id)
        if latest is None:
            raise HTTPException(
                404, f"no odds published for event {event_id}; run `valchamps update`"
            )
        return latest

    @app.get("/events/{event_id}/odds/history")
    def odds_history(event_id: int) -> list[dict[str, Any]]:
        """Every published forecast, one row per team per update."""
        _, history = _read_published(settings.odds_dir, event_id)
        if history is None:
            raise HTTPException(404, f"no odds published for event {event_id}")
        return [{k: _clean(v) for k, v in row.items()} for row in history.to_dict(orient="records")]

    @app.get("/events/{event_id}/teams")
    def event_teams(event_id: int) -> list[dict[str, Any]]:
        """The event's teams and groups (from its opening matches)."""
        st = state.get()
        openings = group_openings(st["engine"], event_id)
        if not openings:
            raise HTTPException(404, f"no group-stage matches found for event {event_id}")
        return [
            {"team_id": t, "group": g, **st["teams"].get(t, {"name": str(t), "tag": None})}
            for g, pairs in openings.items()
            for pair in pairs
            for t in pair
        ]

    @app.get("/predict")
    def predict(
        team_a: int = Query(..., description="vlr.gg team id"),
        team_b: int = Query(..., description="vlr.gg team id"),
        best_of: int = Query(3, description="3 or 5"),
        event_id: int = Query(CHAMPIONS_2026, description="Event the match belongs to"),
        vetoes: int = Query(5, ge=0, le=50, description="How many likeliest vetoes to return"),
    ) -> dict[str, Any]:
        """Series odds from the simulated veto and the map model, as of the latest data."""
        if best_of not in FORMATS:
            raise HTTPException(422, f"best_of must be one of {sorted(FORMATS)}")
        if team_a == team_b:
            raise HTTPException(422, "pick two different teams")
        st = state.get()
        for t in (team_a, team_b):
            if t not in st["played"]:
                raise HTTPException(404, f"team {t} has no completed matches to rate it on")
        if st["bundle"] is None:
            raise HTTPException(503, "no trained map model; run `valchamps train`")
        pool = current_map_pool(st["records"])
        info = st["events"].get(event_id)
        pred = predict_series(
            st["builder"],
            st["bundle"]["model"],
            team_a,
            team_b,
            prediction_date(st["records"]),
            pool,
            best_of=best_of,
            veto_prior=st["veto_prior"],
            event_id=event_id,
            tier=info.tier if info else None,
        )
        played = map_play_probabilities(pred.vetoes)
        name = lambda t: st["teams"].get(t, {}).get("name", str(t))  # noqa: E731
        return {
            "team_a": {"team_id": team_a, "name": name(team_a)},
            "team_b": {"team_id": team_b, "name": name(team_b)},
            "best_of": best_of,
            "p_a": pred.p_a,
            "p_b": 1 - pred.p_a,
            "elo_p_a": pred.elo_p,
            "scores": [
                {"a": a, "b": b, "p": p}
                for (a, b), p in sorted(pred.scores.items(), key=lambda kv: kv[0][1] - kv[0][0])
            ],
            "maps": [
                {
                    "map": m,
                    "pick_a": pred.map_probs[(m, "pick_a")],
                    "pick_b": pred.map_probs[(m, "pick_b")],
                    "decider": pred.map_probs[(m, "decider")],
                    "in_series": played.get(m, 0.0),
                }
                for m in sorted(pool, key=lambda m: -played.get(m, 0.0))
            ],
            "vetoes": [
                {"p": o.prob, "maps": [{"map": m, "picked_by": c} for m, c in o.maps]}
                for o in pred.vetoes[:vetoes]
            ],
            "map_pool": list(pool),
            "model": {
                "name": st["bundle"]["name"],
                "trained_through": st["bundle"]["trained_through"],
            },
            "data_through": str(st["records"][-1].date),
        }

    return app


app = create_app()
