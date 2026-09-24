# VALORANT Champions 2026 — Title Odds

[![CI](https://github.com/JasSaini101/2026__Val_Champs_Model/actions/workflows/ci.yml/badge.svg)](https://github.com/JasSaini101/2026__Val_Champs_Model/actions/workflows/ci.yml)

An end-to-end ML system that estimates **P(Team A beats Team B) on each map**, rolls those odds up to Bo3/Bo5 series, and runs a **Monte Carlo simulation of the full Champions 2026 bracket** to give each team's chance of lifting the trophy. It updates while the event runs: finished matches are pulled in, ratings refresh, and the bracket is re-simulated from where it stands.

```
vlr.gg ─► scraper (rate-limited, cached) ─► parser ─► SQL (SQLite/Postgres)   [DVC]
        ─► point-in-time features ─► map model (Elo → LightGBM → PyTorch)       [MLflow]
        ─► series model (veto sim → Bo3/Bo5) ─► Monte Carlo bracket ─► odds
        ─► FastAPI ─► Streamlit dashboard
GitHub Actions cron: scrape new results → update → re-simulate → publish odds
```

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Scaffold: uv, ruff, pytest, pre-commit, CI, Docker | ✅ |
| 2 | Data: vlr.gg scraper, parser, SQL schema, DVC stage | ✅ |
| 3 | Point-in-time features (Elo, form, map pool, rosters) | ⏳ |
| 4 | Map model + calibration + MLflow tracking | ⏳ |
| 5 | Series model (veto simulation) | ⏳ |
| 6 | Monte Carlo bracket simulator | ⏳ |
| 7 | Scheduled live-update pipeline | ⏳ |
| 8 | FastAPI + Streamlit dashboard | ⏳ |

## Quick start

```bash
uv sync                       # install (Python 3.11+)
uv run pytest                 # run the test suite
uv run valchamps init-db      # create data/valchamps.db
uv run valchamps ingest --event 2097   # scrape one event (Champions 2024)
uv run valchamps ingest       # scrape every event in configs/events.yaml
```

Or with Docker:

```bash
docker build -t valchamps .
docker run --rm -v "$PWD/data:/app/data" valchamps ingest --event 2097
```

## Data layer

**Scraper** (`src/valchamps/data/scraper.py`): an `httpx` client that waits at least 2 s between requests, retries 429/5xx and connection errors with exponential backoff, and writes every page to `data/raw/vlr/`. Finished matches are served from that cache forever. Event match lists and live or upcoming matches are refetched, so re-running during an event only downloads what changed.

**Parser** (`src/valchamps/data/parser.py`): BeautifulSoup/lxml extraction of:

- event metadata and the event's match list (completed, live and upcoming)
- match header: teams, series score, best-of, UTC start time, stage
- map vetoes (`FNC ban Icebox; TH pick Lotus; …; Abyss remains`), with team tags mapped to team ids
- each map: name, who picked it, final rounds, attack/defence halves, duration
- round by round for each map: winner, side (attack/defence) and how the round ended (elimination, spike defused, spike detonated, time)
- per-player stats for each map: agent, rating, ACS, K/D/A, KAST, ADR, HS%, first kills/deaths *(these tables are missing from the pages vlr.gg currently serves to the scraper; being investigated)*

**Schema** (`src/valchamps/data/db.py`, SQLAlchemy Core, works on SQLite and Postgres):

```
events ─┐
        └─< matches >── teams
               ├─< maps ─< player_map_stats >── players
               │     └─< rounds
               └─< vetoes
scrape_log (audit of every fetch)
```

All writes are upserts, so re-ingesting is idempotent.

**Versioning**: `dvc.yaml` defines the `ingest` stage, and `dvc repro` rebuilds the data. Add a remote (`dvc remote add -d storage s3://…`) to share it.

### Configuration

Set with environment variables:

| Variable | Default |
|---|---|
| `VALCHAMPS_DB_URL` | `sqlite:///data/valchamps.db` |
| `VALCHAMPS_CACHE_DIR` | `data/raw/vlr` |
| `VALCHAMPS_REQUEST_INTERVAL` | `2.0` seconds |
| `VALCHAMPS_BASE_URL` | `https://www.vlr.gg` |

## Testing

The parser tests run against HTML fixtures in `tests/fixtures/vlr/`. The fixtures copy vlr.gg's markup, but their numbers and ids are **synthetic** (see the banner at the top of each file). HTTP is mocked with `respx`, so the suite never touches the network. Before relying on a full scrape, save a few real pages as extra fixtures and check that the selectors still match the live site.

Debug a page's structure with `uv run python scripts/inspect_vlr_page.py [page.html | --fetch /path]`.

## Project layout

```
src/valchamps/
  config.py         settings from env
  cli.py            `valchamps` command
  data/             scraper, parser, models, db, ingest
configs/events.yaml events to scrape
tests/              pytest suite + HTML fixtures
dvc.yaml            data pipeline
```
