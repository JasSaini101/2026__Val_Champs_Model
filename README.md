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
| 3 | Point-in-time features (Elo, form, map pool, rosters) | ✅ |
| 4 | Map model: Elo → linear → LightGBM → PyTorch, walk-forward backtests, MLflow on DagsHub | ✅ |
| 5 | Series model (veto simulation) | ⏳ |
| 6 | Monte Carlo bracket simulator | ⏳ |
| 7 | Scheduled live-update pipeline | ⏳ |
| 8 | FastAPI + Streamlit dashboard | ⏳ |

## Quick start

```bash
uv sync                       # install (Python 3.11+)
uv run pytest                 # run the test suite
uv run valchamps init-db      # create data/valchamps.db
uv run valchamps ingest --event 2274   # scrape one event (VCT 2025 Americas Kickoff)
uv run valchamps ingest       # scrape every event in configs/events.yaml
uv run valchamps build-features   # training table -> data/features/maps.parquet
uv run valchamps backtest     # walk-forward comparison of all models (logs to MLflow)
uv run valchamps train --model gbm   # holdout check, then fit on everything -> models/map_model.pkl
```

Or with Docker:

```bash
docker build -t valchamps .
docker run --rm -v "$PWD/data:/app/data" valchamps ingest --event 2274
```

## Data scope

Training data covers 2025 and 2026 (`configs/events.yaml`, 30 events):

- **VCT regional leagues:** Kickoff, Stage 1 and Stage 2 in Americas, EMEA, Pacific and China (24 events). They reflect current rosters and the current meta, and hold most of the tier-1 matches.
- **International events:** Masters Bangkok 2025, Masters Toronto 2025, Champions 2025, Masters Santiago 2026 and Masters London 2026 (5 events). These are the only matches between teams from different regions. Leagues alone show how a team ranks in its own region, not how the regions compare, and Champions is decided by cross-region matches. These results also show how each region's top teams actually do at international events.
- **Champions 2026 (event 2766):** the tournament being simulated. Its finished matches count as training data as they're played; its upcoming matches make up the bracket.

## Data layer

**Scraper** (`src/valchamps/data/scraper.py`): an `httpx` client that waits at least 2 s between requests, retries 429/5xx and connection errors with exponential backoff, and writes every page to `data/raw/vlr/`. Finished matches are served from that cache forever. Event match lists and live or upcoming matches are refetched, so re-running during an event only downloads what changed.

**Parser** (`src/valchamps/data/parser.py`): BeautifulSoup/lxml extraction of:

- event metadata (name, dates, location), final standings (place, circuit points, qualification note) and the event's match list (completed, live and upcoming)
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
placements (event × team final standings)
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

## Features

`valchamps build-features` replays every completed match oldest first. Before each match it records every team's current state as that match's features, then updates the state with the result. A row can therefore only reflect earlier matches, and a test checks that flipping every later result leaves earlier rows unchanged. Each map gives two rows, one from each team's point of view.

| Group | Features |
|---|---|
| Elo (`ratings.py`) | team rating, per-map adjustment, **region offset** (moved only by cross-region maps at Masters and Champions), win probability. Updates scale with round margin; ratings are pulled partway back to the mean each new year |
| Form (`form.py`) | map win rate and round difference over the last 5 matches, days of rest, matches in the last 30 days, experience, **international map win rate**, head-to-head record |
| Map pool (`map_pool.py`) | win rate on this map, shrunk toward the team's overall rate; pick and ban rate for this map; whether the map was the team's pick, the opponent's pick, or the decider |
| Roster (`roster.py`) | share of the lineup unchanged from the previous match; the lineup's average player rating over recent matches |
| Placement (`placement.py`) | finish in the team's most recent **completed** league event (and whether it won it), circuit points so far this season, finish at its most recent international event. Standings become visible only after the event's end date |

Hyper-parameters are in `params.yaml` and tracked by DVC. The same `FeatureBuilder` state will score upcoming Champions matches.

## Map model

`valchamps backtest` compares five models on the same **walk-forward folds**: for each event from mid-2025 on, a model is trained only on maps played before the event started and scored on the event. Masters London 2026 and the 2026 Stage 2 leagues are held out entirely for the final check in `valchamps train`.

| Model | What it is |
|---|---|
| `elo` | The Elo system's own probability (no training) |
| `elo_cal` | Elo rescaled by one learned factor (fixes over- or under-confidence) |
| `linear` | L2 logistic regression on all features, standardised and clipped at ±3 SD |
| `gbm` | LightGBM, with early stopping on the most recent 15% of training maps |
| `nn` | PyTorch MLP that is **antisymmetric by construction**: logit = g(A's view) − g(B's view) |

Inputs are the 54 point-in-time features plus `map_name` (categorical), `is_international` and `cross_region`. Every prediction is symmetric: P(A beats B) + P(B beats A) = 1. Scores (log loss, Brier, accuracy, calibration error) are reported **overall, on cross-region maps and on international events**, since Champions is decided by cross-region matches.

Runs go to MLflow. Set `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` to log to DagsHub; otherwise runs go to a local `mlflow.db` (`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`). `train` registers the final model as `map-model`. PyTorch is optional: `uv sync --extra nn`.

## Testing

The parser tests run against HTML fixtures in `tests/fixtures/vlr/`. The fixtures copy vlr.gg's markup, but their numbers and ids are **synthetic** (see the banner at the top of each file). HTTP is mocked with `respx`, so the suite never touches the network. Before relying on a full scrape, save a few real pages as extra fixtures and check that the selectors still match the live site.

Debug a page's structure with `uv run python scripts/inspect_vlr_page.py [page.html | --fetch /path]`.

## Project layout

```
src/valchamps/
  config.py         settings from env
  cli.py            `valchamps` command
  data/             scraper, parser, models, db, ingest
  features/         Elo, form, map pool, roster trackers; training-table builder
  models/           baselines, linear, LightGBM, PyTorch; walk-forward backtests; MLflow
configs/events.yaml events to scrape
tests/              pytest suite + HTML fixtures
dvc.yaml            pipeline (ingest -> features -> train)
params.yaml         feature and model hyper-parameters
```
