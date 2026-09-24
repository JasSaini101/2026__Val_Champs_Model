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
