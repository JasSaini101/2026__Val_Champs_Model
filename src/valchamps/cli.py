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
                f"{len(report.failed)} failed"
            )
    if failed:
        raise typer.Exit(code=1)


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


if __name__ == "__main__":
    app()
