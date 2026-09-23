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


if __name__ == "__main__":
    app()
