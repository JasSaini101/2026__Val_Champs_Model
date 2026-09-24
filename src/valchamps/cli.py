"""Command-line entry point: ``valchamps --help``."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from valchamps.config import PROJECT_ROOT, Settings
from valchamps.data import db
from valchamps.data.ingest import EventSpec, ingest_event, load_event_specs
from valchamps.data.scraper import VlrClient

app = typer.Typer(help="VALORANT Champions 2026 model tooling.", no_args_is_help=True)

DEFAULT_EVENTS = PROJECT_ROOT / "configs" / "events.yaml"
DEFAULT_PARAMS = PROJECT_ROOT / "params.yaml"
DEFAULT_FEATURES = PROJECT_ROOT / "data" / "features" / "maps.parquet"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "map_model.pkl"
DEFAULT_METRICS = PROJECT_ROOT / "metrics.json"
DEFAULT_BRACKET = PROJECT_ROOT / "configs" / "bracket.yaml"
CHAMPIONS_2026 = 2766  # vlr.gg event id of the tournament being simulated
REPORTS_DIR = PROJECT_ROOT / "reports"


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command("init-db")
def init_db_cmd() -> None:
    """Create the database tables (no-op if they exist)."""
    settings = Settings()
    db.init_db(db.get_engine(settings.db_url))
    typer.echo(f"initialised {settings.db_url}")


@app.command()
def ingest(
    events_file: Path = typer.Option(DEFAULT_EVENTS, help="YAML list of events to scrape."),
    event_id: list[int] = typer.Option(None, "--event", help="Only these event ids."),
    refresh: bool = typer.Option(False, help="Refetch matches already stored as completed."),
) -> None:
    """Scrape events from vlr.gg into the database."""
    settings = Settings()
    engine = db.get_engine(settings.db_url)
    db.init_db(engine)
    specs = load_event_specs(events_file)
    if event_id:
        known = {s.event_id: s for s in specs}
        specs = [known.get(e, EventSpec(event_id=e, name=f"event-{e}")) for e in event_id]

    failed = 0
    with VlrClient(settings) as client:
        for spec in specs:
            report = ingest_event(client, engine, spec, refresh=refresh)
            failed += len(report.failed)
            typer.echo(
                f"{spec.name}: {report.fetched} fetched, {report.skipped} skipped, "
                f"{report.pending} pending (teams TBD), {len(report.failed)} failed"
            )
    if failed:
        raise typer.Exit(code=1)


@app.command("inspect-event")
def inspect_event(
    event_id: int,
    fetch: bool = typer.Option(False, help="Download the page instead of using the saved copy."),
) -> None:
    """Show what the parser reads from an event page (header details and standings)."""
    from valchamps.data.parser import _soup, event_details, parse_event, parse_standings

    settings = Settings()
    path = f"/event/{event_id}"
    with VlrClient(settings) as client:
        cached = client.cache_path(path)
        if not fetch and not cached.exists():
            typer.echo(f"no saved page for {path} (looked for {cached}); use --fetch")
            raise typer.Exit(code=1)
        html = client.get(path, max_age=0 if fetch else None)
    typer.echo(f"page: {cached.name}")
    typer.echo(f"raw header details: {event_details(_soup(html))}")
    typer.echo(f"parsed: {parse_event(html, event_id)}")
    standings = parse_standings(html)
    typer.echo(f"standings ({len(standings)} rows):")
    for s in standings:
        typer.echo(f"  {s.place}-{s.place_max}  {s.team_name} (team {s.team_id})  "
                   f"points={s.circuit_points} note={s.note}")  # fmt: skip


@app.command("build-features")
def build_features(
    out: Path = typer.Option(DEFAULT_FEATURES, help="Parquet file to write."),
    params_file: Path = typer.Option(DEFAULT_PARAMS, help="YAML with a `features` section."),
) -> None:
    """Build the per-map training table from the database."""
    from valchamps.features import (
        FeatureParams,
        build_feature_frame,
        load_events,
        load_records,
        team_regions,
    )

    engine = db.get_engine(Settings().db_url)
    records = load_records(engine)
    if not records:
        typer.echo("no completed matches in the database; run `valchamps ingest` first")
        raise typer.Exit(code=1)
    params = FeatureParams.from_yaml(params_file) if params_file.exists() else FeatureParams()
    frame, _ = build_feature_frame(records, team_regions(engine), params, load_events(engine))
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    maps = frame[frame.perspective == 0]
    typer.echo(
        f"wrote {out}: {len(frame)} rows ({len(maps)} maps, {maps.match_id.nunique()} matches), "
        f"{maps.date.min():%Y-%m-%d} to {maps.date.max():%Y-%m-%d}, "
        f"{int(maps.cross_region.sum())} cross-region maps"
    )


def _model_list(models: str) -> list[str]:
    from valchamps.models import MODEL_NAMES

    names = [m.strip() for m in models.split(",") if m.strip()]
    unknown = [m for m in names if m not in MODEL_NAMES]
    if unknown:
        raise typer.BadParameter(f"unknown model(s) {unknown}; choose from {list(MODEL_NAMES)}")
    return names


@app.command()
def backtest(
    models: str = typer.Option("elo,elo_cal,linear,gbm,nn", help="Comma-separated models."),
    features: Path = typer.Option(DEFAULT_FEATURES, help="Feature table from build-features."),
    params_file: Path = typer.Option(DEFAULT_PARAMS, help="YAML with a `models` section."),
    track: bool = typer.Option(True, help="Log each model's run to MLflow."),
) -> None:
    """Walk-forward backtest: train on the past, predict each later event."""
    import json

    from valchamps.models import format_table, load_config, load_frame, run_backtest
    from valchamps.models.metrics import reliability_plot

    config = load_config(params_file)
    df = load_frame(features)
    results = []
    for name in _model_list(models):
        try:
            result = run_backtest(df, name, config)
        except ImportError as exc:  # e.g. nn without torch installed
            typer.echo(f"skipping {name}: {exc}")
            continue
        results.append(result)
        if track:
            from valchamps.models import tracking

            out_dir = REPORTS_DIR / "backtest" / name
            out_dir.mkdir(parents=True, exist_ok=True)
            with tracking.run("map-model-backtest", name, {"model": name, **config}):
                tracking.log_segment_metrics(result.metrics)
                for step, fold in enumerate(result.fold_metrics):
                    tracking.log_segment_metrics(
                        {"fold": {k: v for k, v in fold.items() if k not in ("fold", "n")}},
                        step=step,
                    )
                preds = out_dir / "predictions.csv"
                result.predictions.to_csv(preds, index=False)
                (out_dir / "folds.json").write_text(json.dumps(result.fold_metrics, indent=2))
                plot = reliability_plot(
                    result.predictions["y"], result.predictions["p"],
                    out_dir / "reliability.png", f"{name}: backtest",
                )  # fmt: skip
                for artifact in (preds, out_dir / "folds.json", plot):
                    tracking.log_artifact(artifact)
    typer.echo(format_table(results))
    if track:
        from valchamps.models import tracking

        typer.echo(f"runs logged to {_display_uri(tracking.configure())}")


@app.command()
def train(
    model: str = typer.Option("gbm", help="Model to train (see `backtest`)."),
    features: Path = typer.Option(DEFAULT_FEATURES, help="Feature table from build-features."),
    params_file: Path = typer.Option(DEFAULT_PARAMS, help="YAML with a `models` section."),
    out: Path = typer.Option(DEFAULT_MODEL, help="Where to save the trained model."),
    metrics_file: Path = typer.Option(DEFAULT_METRICS, help="Holdout metrics (JSON) for DVC."),
    track: bool = typer.Option(True, help="Log the run to MLflow and register the model."),
) -> None:
    """Score the model on the holdout events, then refit on all data and save it."""
    import json
    import pickle

    from valchamps.models import format_table, load_config, load_frame, make_model, run_holdout
    from valchamps.models.metrics import reliability_plot

    (name,) = _model_list(model)
    config = load_config(params_file)
    df = load_frame(features)
    holdout = run_holdout(df, name, config)
    typer.echo("holdout (trained only on earlier maps):")
    typer.echo(format_table([holdout]))

    final = make_model(name, config).fit(df)
    out.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": final, "name": name, "trained_through": str(df["date"].max().date()),
              "maps": int(df["game_id"].nunique())}  # fmt: skip
    out.write_bytes(pickle.dumps(bundle))
    metrics_file.write_text(json.dumps({"model": name, "holdout": holdout.metrics}, indent=2))
    typer.echo(
        f"saved {out} (trained on {bundle['maps']} maps through {bundle['trained_through']})"
    )

    if track:
        from valchamps.models import tracking

        out_dir = REPORTS_DIR / "train" / name
        plot = reliability_plot(
            holdout.predictions["y"], holdout.predictions["p"],
            out_dir / "holdout_reliability.png", f"{name}: holdout",
        )  # fmt: skip
        with tracking.run("map-model", name, {"model": name, **config}):
            tracking.log_segment_metrics(holdout.metrics)
            tracking.log_artifact(plot)
            tracking.log_artifact(metrics_file)
            version = tracking.log_and_register_model(final, df)
        where = _display_uri(tracking.configure())
        typer.echo(f"logged to {where}" + (f"; registered map-model v{version}" if version else ""))


def _load_history(engine):
    from valchamps.features import load_events, load_records, team_regions

    records = load_records(engine)
    if not records:
        typer.echo("no completed matches in the database; run `valchamps ingest` first")
        raise typer.Exit(code=1)
    return records, team_regions(engine), load_events(engine)


def _feature_params(params_file: Path):
    from valchamps.features import FeatureParams

    return FeatureParams.from_yaml(params_file) if params_file.exists() else FeatureParams()


@app.command("series-backtest")
def series_backtest(
    model: str = typer.Option("nn", help="Map model to use (see `backtest`)."),
    params_file: Path = typer.Option(DEFAULT_PARAMS, help="YAML with features/models/series."),
    track: bool = typer.Option(True, help="Log the run to MLflow."),
) -> None:
    """Walk-forward backtest of Bo3/Bo5 odds: simulated veto vs actual maps vs raw Elo."""
    from valchamps.models import load_config
    from valchamps.series import (
        SeriesParams,
        format_series_table,
        run_series_backtest,
        veto_summary,
    )

    (name,) = _model_list(model)
    records, regions, events = _load_history(db.get_engine(Settings().db_url))
    config = load_config(params_file)
    series_params = SeriesParams.from_yaml(params_file)
    result = run_series_backtest(
        records, regions, _feature_params(params_file), events, name, config,
        series_params.veto_prior,
    )  # fmt: skip
    preds = result.predictions
    veto = veto_summary(preds)
    typer.echo(
        f"{len(preds)} series ({(preds.best_of == 3).sum()} Bo3, {(preds.best_of == 5).sum()} "
        f"Bo5) in {preds.fold.nunique()} events, {result.skipped} skipped (no usable veto or "
        f"result); map model {name}, veto_prior {series_params.veto_prior}"
    )
    typer.echo(format_series_table(result))
    typer.echo(
        f"veto simulation: mean log P(actual map sequence) {veto['log_lik']:.3f} "
        f"vs {veto['uniform_log_lik']:.3f} for a uniform veto"
    )
    out_dir = REPORTS_DIR / "series_backtest" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_dir / "predictions.csv", index=False)
    typer.echo(f"per-series predictions: {out_dir / 'predictions.csv'}")
    if track:
        from valchamps.models import tracking

        run_params = {"model": name, "veto_prior": series_params.veto_prior, **config}
        with tracking.run("series-model-backtest", name, run_params):
            for method, metrics in result.metrics.items():
                tracking.log_segment_metrics({f"{method}/{seg}": m for seg, m in metrics.items()})
            tracking.log_segment_metrics({"veto": veto})
            tracking.log_artifact(out_dir / "predictions.csv")
        typer.echo(f"run logged to {_display_uri(tracking.configure())}")


def _resolve_team(engine, query: str, records) -> tuple[int, str]:
    """Team id and name from an id, tag or name (case-insensitive); most recently active wins."""
    from sqlalchemy import select

    with engine.connect() as conn:
        teams = conn.execute(select(db.teams.c.team_id, db.teams.c.name, db.teams.c.tag)).all()
    q = query.strip().lower()
    hits = [t for t in teams if str(t.team_id) == q]
    hits = hits or [t for t in teams if q in ((t.tag or "").lower(), t.name.lower())]
    hits = hits or [t for t in teams if q in t.name.lower()]
    if not hits:
        raise typer.BadParameter(f"no team matches {query!r}")
    last_played = {t: r.date for r in records for t in r.teams}  # records are oldest first
    active = [t for t in hits if t.team_id in last_played]
    if not active:
        raise typer.BadParameter(f"{hits[0].name} has no completed matches to rate it on")
    best = max(active, key=lambda t: last_played[t.team_id])
    return best.team_id, best.name


def _map_pool(records, maps: str | None) -> tuple[str, ...]:
    """The pool given as ``--maps``, else the seven maps of the most recent vetoed match."""
    if maps:
        pool = tuple(m.strip().title() for m in maps.split(",") if m.strip())
    else:
        pool = next((r.map_pool for r in reversed(records) if len(r.map_pool) == 7), ())
    if len(pool) != 7:
        raise typer.BadParameter(f"need 7 maps, got {list(pool)}")
    return pool


def _prediction_date(records):
    """Now, or just after the last known match if the clock is behind the data."""
    from datetime import UTC, datetime

    return max(datetime.now(UTC).replace(tzinfo=None), records[-1].date)


@app.command("predict-match")
def predict_match(
    team_a: str = typer.Argument(..., help="Team name, tag or vlr.gg id."),
    team_b: str = typer.Argument(..., help="Team name, tag or vlr.gg id."),
    best_of: int = typer.Option(3, "--best-of", help="3 or 5."),
    maps: str = typer.Option(
        None, help="Comma-separated map pool (default: the pool of the latest vetoed match)."
    ),
    event: int = typer.Option(CHAMPIONS_2026, help="Event the match belongs to."),
    model_path: Path = typer.Option(DEFAULT_MODEL, "--model", help="Trained map model."),
    params_file: Path = typer.Option(DEFAULT_PARAMS, help="YAML with features/series."),
    show_vetoes: int = typer.Option(5, help="How many of the likeliest vetoes to list."),
) -> None:
    """Series odds for an upcoming match, from the simulated veto and the map model."""
    import pickle

    from valchamps.features import build_feature_frame
    from valchamps.series import FORMATS, SeriesParams, map_play_probabilities, predict_series

    if best_of not in FORMATS:
        raise typer.BadParameter(f"--best-of must be one of {sorted(FORMATS)}")
    engine = db.get_engine(Settings().db_url)
    records, regions, events = _load_history(engine)
    (a, a_name), (b, b_name) = (_resolve_team(engine, t, records) for t in (team_a, team_b))
    if a == b:
        raise typer.BadParameter("pick two different teams")
    pool = _map_pool(records, maps)

    _, builder = build_feature_frame(records, regions, _feature_params(params_file), events)
    bundle = pickle.loads(model_path.read_bytes())
    info = events.get(event)
    pred = predict_series(
        builder, bundle["model"], a, b, _prediction_date(records), pool, best_of=best_of,
        veto_prior=SeriesParams.from_yaml(params_file).veto_prior,
        event_id=event, tier=info.tier if info else None,
    )  # fmt: skip

    typer.echo(
        f"{a_name} vs {b_name}: Bo{best_of}, event {event}; map model {bundle['name']} "
        f"trained through {bundle['trained_through']}"
    )
    typer.echo(f"map pool: {', '.join(pool)}")
    typer.echo(f"P({a_name} wins) = {pred.p_a:.3f}    P({b_name} wins) = {1 - pred.p_a:.3f}")
    typer.echo(f"raw Elo (no maps, no veto): {pred.elo_p:.3f}")
    scores = sorted(pred.scores.items(), key=lambda kv: kv[0][1] - kv[0][0])
    typer.echo("final score: " + "  ".join(f"{w}-{lost} {q:.3f}" for (w, lost), q in scores))

    played = map_play_probabilities(pred.vetoes)
    mp = pred.map_probs
    typer.echo(f"\nP({a_name} wins the map), by who picks it:")
    typer.echo(f"{'map':<10}{a_name[:12] + ' pick':>18}{b_name[:12] + ' pick':>18}"
               f"{'decider':>9}{'in series':>11}")  # fmt: skip
    for m in sorted(pool, key=lambda m: -played.get(m, 0.0)):
        typer.echo(f"{m:<10}{mp[(m, 'pick_a')]:>18.3f}{mp[(m, 'pick_b')]:>18.3f}"
                   f"{mp[(m, 'decider')]:>9.3f}{played.get(m, 0.0):>11.3f}")  # fmt: skip
    label = {"pick_a": a_name, "pick_b": b_name, "decider": "decider"}
    typer.echo(f"\nlikeliest vetoes (of {len(pred.vetoes)}):")
    for o in pred.vetoes[:show_vetoes]:
        typer.echo(f"  {o.prob:.3f}  " + " -> ".join(f"{m} ({label[c]})" for m, c in o.maps))


@app.command()
def simulate(
    event: int = typer.Option(CHAMPIONS_2026, help="Event to simulate."),
    runs: int = typer.Option(100_000, help="Number of simulated tournaments."),
    odds: str = typer.Option("model", help="Series odds: `model` (veto + map model) or `elo`."),
    bracket_file: Path = typer.Option(DEFAULT_BRACKET, help="Tournament format (YAML)."),
    model_path: Path = typer.Option(DEFAULT_MODEL, "--model", help="Trained map model."),
    params_file: Path = typer.Option(DEFAULT_PARAMS, help="YAML with features/series."),
    maps: str = typer.Option(None, help="Comma-separated map pool (default: latest vetoed)."),
    seed: int = typer.Option(0, help="Random seed."),
    out: Path = typer.Option(None, help="CSV of per-team odds (default: reports/bracket/)."),
) -> None:
    """Monte Carlo the event's bracket: each team's chance of every final standing."""
    import json
    import pickle

    from sqlalchemy import select

    from valchamps.bracket import (
        build_bracket,
        elo_odds,
        fixed_results,
        group_openings,
        load_format,
        model_odds,
    )
    from valchamps.bracket import simulate as run_simulation
    from valchamps.features import build_feature_frame
    from valchamps.series import SeriesParams

    if odds not in ("model", "elo"):
        raise typer.BadParameter("--odds must be `model` or `elo`")
    engine = db.get_engine(Settings().db_url)
    records, regions, events = _load_history(engine)
    try:
        openings = group_openings(engine, event)
        bracket = build_bracket(load_format(bracket_file), openings)
    except ValueError as exc:
        typer.echo(f"cannot build the bracket for event {event}: {exc}")
        raise typer.Exit(code=1) from exc
    teams = sorted({t for pairs in openings.values() for pair in pairs for t in pair})
    group_of = {t: g for g, pairs in openings.items() for pair in pairs for t in pair}
    formats = sorted({m.best_of for m in bracket})

    _, builder = build_feature_frame(records, regions, _feature_params(params_file), events)
    if odds == "model":
        bundle = pickle.loads(model_path.read_bytes())
        info = events.get(event)
        matrices = model_odds(
            builder, bundle["model"], teams, _prediction_date(records), _map_pool(records, maps),
            formats=formats, veto_prior=SeriesParams.from_yaml(params_file).veto_prior,
            event_id=event, tier=info.tier if info else None,
        )  # fmt: skip
        source = f"map model {bundle['name']} (trained through {bundle['trained_through']})"
    else:
        matrices = elo_odds(builder, teams, formats)
        source = "raw Elo"

    results = fixed_results(records, event)
    sim = run_simulation(bracket, matrices, teams, runs, results, seed=seed)
    with engine.connect() as conn:
        names = dict(conn.execute(select(db.teams.c.team_id, db.teams.c.name)).all())
    table = sim.to_frame(names)
    table.insert(2, "group", table["team_id"].map(group_of))

    typer.echo(
        f"event {event}: {len(teams)} teams, {runs:,} runs, odds from {source}; "
        f"{len(sim.used_results)} finished series fixed"
    )
    for r in sim.unused_results:
        typer.echo(
            f"warning: result not in the bracket ({r.stage}: {names.get(r.team_a)} vs "
            f"{names.get(r.team_b)}); check {bracket_file.name}"
        )
    typer.echo(f"{'team':<22}{'grp':>4}{'playoffs':>10}{'top4':>8}{'final':>8}{'title':>8}")
    for row in table.itertuples(index=False):
        typer.echo(f"{row.team[:21]:<22}{row.group:>4}{row.top8:>10.3f}{row.top4:>8.3f}"
                   f"{row.final:>8.3f}{row.title:>8.3f}")  # fmt: skip

    out = out or REPORTS_DIR / "bracket" / f"odds_{event}_{odds}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    meta = {"event": event, "runs": runs, "odds": source, "seed": seed,
            "fixed_results": len(sim.used_results), "as_of": str(records[-1].date)}  # fmt: skip
    out.with_suffix(".json").write_text(
        json.dumps({**meta, "teams": table.to_dict(orient="records")}, indent=2)
    )
    typer.echo(f"wrote {out} and {out.with_suffix('.json').name}")


def _display_uri(uri: str) -> str:
    """Tracking URI without any credentials that might be embedded in it."""
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(uri)
    if parts.password or parts.username:
        netloc = parts.hostname + (f":{parts.port}" if parts.port else "")
        return urlunsplit(parts._replace(netloc=netloc))
    return uri


if __name__ == "__main__":
    app()
