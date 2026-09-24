# VALORANT Champions 2026 — Title Odds

[![CI](https://github.com/JasSaini101/2026__Val_Champs_Model/actions/workflows/ci.yml/badge.svg)](https://github.com/JasSaini101/2026__Val_Champs_Model/actions/workflows/ci.yml)

An end-to-end ML system that estimates **P(Team A beats Team B) on each map**, rolls those odds up to Bo3/Bo5 series, and runs a **Monte Carlo simulation of the full Champions 2026 bracket** to give each team's chance of lifting the trophy. It updates while the event runs: finished matches are pulled in, ratings refresh, and the bracket is re-simulated from where it stands.

![Title odds in the dashboard](docs/images/dashboard_title_odds.png)

```mermaid
flowchart LR
    A[vlr.gg] -->|rate-limited, cached scraper| B[(SQL<br/>SQLite / Postgres)]
    B --> C[Point-in-time features<br/>Elo, form, map pool, roster]
    C --> D[Map model<br/>Elo → LightGBM → PyTorch]
    D --> E[Series model<br/>exact veto enumeration → Bo3/Bo5]
    E --> F[Monte Carlo bracket<br/>100k tournaments]
    F --> G[odds/ JSON + CSV]
    G --> H[FastAPI]
    G --> I[Streamlit dashboard]
    J([GitHub Actions, hourly]) -.->|scrape → re-simulate → commit| G
```

**Stack:** Python 3.11, httpx, BeautifulSoup, SQLAlchemy, pandas, LightGBM, PyTorch, scikit-learn, numpy, DVC + DagsHub, MLflow, FastAPI, Streamlit + Altair, GitHub Actions, Docker, pytest.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Scaffold: uv, ruff, pytest, pre-commit, CI, Docker | ✅ |
| 2 | Data: vlr.gg scraper, parser, SQL schema, DVC stage | ✅ |
| 3 | Point-in-time features (Elo, form, map pool, rosters) | ✅ |
| 4 | Map model: Elo → linear → LightGBM → PyTorch, walk-forward backtests, MLflow on DagsHub | ✅ |
| 5 | Series model: veto simulation, exact Bo3/Bo5 odds, series backtest | ✅ |
| 6 | Monte Carlo bracket simulator | ✅ |
| 7 | Scheduled live-update pipeline | ✅ |
| 8 | FastAPI + Streamlit dashboard | ✅ |

## Results

Every number below comes from **walk-forward backtests**: for each event from mid-2025 on, a model is trained only on maps played before that event and scored on it (15 events, 1,396 maps, 531 series). Lower log loss and Brier are better; a coin flip scores 0.693 and 0.25.

**Map level** (`valchamps backtest`):

| Model | Log loss | Brier | Accuracy | Log loss, cross-region maps (189) |
|---|---|---|---|---|
| Elo | 0.677 | 0.242 | 58.2% | 0.694 |
| Elo, calibrated | 0.675 | 0.241 | 58.2% | 0.687 |
| Logistic regression | 0.693 | 0.248 | 56.5% | 0.705 |
| LightGBM | 0.679 | 0.243 | 58.5% | 0.697 |
| **PyTorch (antisymmetric MLP)** | **0.674** | **0.241** | 56.9% | **0.685** |

**Series level** (`valchamps series-backtest`, 508 Bo3 and 23 Bo5):

| Method | Log loss | Brier | Accuracy |
|---|---|---|---|
| Simulated veto + map model | 0.657 | 0.232 | 62.7% |
| Real veto + map model | 0.656 | 0.232 | 62.9% |
| Raw Elo | 0.654 | 0.231 | 63.7% |

<img src="docs/images/backtest_calibration.png" alt="Calibration of the PyTorch map model in the backtest" width="380" align="right">

**What the numbers say**

- Pro VALORANT maps are close to coin flips: no model gets far below 0.67 log loss. The probabilities are **well calibrated** (right: predicted vs observed win rate, with the number of maps in each bin), which matters more for a bracket simulation than accuracy.
- The PyTorch model has the lowest log loss overall and on **cross-region maps**, where raw Elo is no better than a coin flip. Those are the matches that decide Champions, which is why it is the model used for the forecast.
- At series level the learned model **only matches Elo**; the map-level gain is too small to show through three maps. The simulated veto costs almost nothing against knowing the real veto (0.657 vs 0.656). The honest summary is that the features add little signal beyond a well-tuned Elo at this data size (2,774 maps from 1,086 tier-1 matches); the value of the rest of the system is turning map odds into calibrated bracket odds.
- The holdout events (Masters London 2026 and the 2026 Stage 2 leagues, 681 maps, never used for model selection) give 0.688 log loss and 54.9% accuracy for the final model.

<br clear="right">

## Quick start

```bash
uv sync                       # install (Python 3.11+)
uv run pytest                 # run the test suite
uv run valchamps init-db      # create data/valchamps.db
uv run valchamps ingest --event 2274   # scrape one event (VCT 2025 Americas Kickoff)
uv run valchamps ingest       # scrape every event in configs/events.yaml
uv run valchamps build-features   # training table -> data/features/maps.parquet
uv run valchamps backtest     # walk-forward comparison of all models (logs to MLflow)
uv sync --extra nn            # PyTorch, for the nn map model
uv run valchamps train --model nn    # holdout check, then fit on everything -> models/map_model.pkl
uv run valchamps series-backtest     # walk-forward Bo3/Bo5 odds: simulated veto vs actual maps vs Elo
uv run valchamps predict-match "G2 Esports" "Paper Rex" --best-of 3
uv run valchamps simulate            # Champions 2026 title odds (--odds elo for raw Elo)
uv run valchamps update              # scrape new results, re-simulate, publish to odds/2766/
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

Runs go to MLflow. Set `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` to log to DagsHub; otherwise runs go to a local `mlflow.db` (`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`). `train` registers the final model as `map-model`. PyTorch is optional: `uv sync --extra nn`. The DVC `train` stage trains `nn`, which is the model the series model uses.

## Series model

`src/valchamps/series/` turns per-map odds into Bo3/Bo5 odds.

**Veto simulation** (`veto.py`) uses the VCT formats, with seven maps in the pool and the teams alternating:

```
Bo3: ban, ban, pick, pick, ban, ban, decider
Bo5: ban, ban, pick, pick, pick, pick, decider
```

At each step the acting team picks or bans from the remaining maps in proportion to how often it has picked or banned each one before (counts from the map-pool tracker). `series.veto_prior` in `params.yaml` adds pseudo-vetoes to every map, which pulls the choice toward uniform. Who vetoes first is a coin flip. Every possible veto is enumerated exactly, not sampled, and reduced to what the series depends on: the maps in play order and who picked each.

**Map odds for unplayed maps**: `FeatureBuilder.hypothetical_rows` builds feature rows from the current state, without updating it, for every map in the pool in each pick context (team A's pick, team B's pick, decider). The map model scores those rows.

**Series odds** (`series.py`) are computed exactly from the per-map probabilities (Bo3 with a constant p gives p²(3 − 2p)) and averaged over the veto outcomes. Veto averaging and the map model are both symmetric, so P(A beats B) + P(B beats A) = 1.

`valchamps series-backtest` uses the map model's walk-forward folds. For each test event it trains the map model on earlier maps and scores every Bo3/Bo5 in the event three ways:

| Method | Maps used |
|---|---|
| `veto` | the simulated veto; this is what a forecast of an upcoming match uses |
| `actual_maps` | the maps the real veto produced, in order, including an unplayed decider |
| `elo` | the Elo probability without map information, the same on every map |

It also reports how much probability the simulated veto gave to the real map sequence, compared with a uniform veto.

`valchamps predict-match TEAM_A TEAM_B --best-of 3` prints series odds, the final-score distribution, a per-map table and the likeliest vetoes for an upcoming match. Teams can be given by name, tag or vlr.gg id. By default it uses the saved map model and the pool from the most recent vetoed match; `--maps` sets a different pool.

## Bracket simulator

`valchamps simulate` runs Champions 2026 (event 2766) 100,000 times and reports each team's chance of every final standing, summarised as playoffs, top 4, final and title.

**Format**: the format is data in `configs/bracket.yaml`: four GSL groups (Opening, Winner's, Elimination, Decider; the top two advance), then an 8-team double-elimination playoff with a Bo5 lower final and grand final. Each group's teams come from the event's Opening matches in the database. The playoff seeding copies Champions 2025's, because 2026's pairings weren't published yet. Replaying 2025 with `--event 2283` matches all 34 of its real results and reproduces its final standings.

**Odds**: series odds are computed once for every pair of teams at Bo3 and Bo5. `--odds model` (the default) uses the simulated veto and the map model, the same as `predict-match`. `--odds elo` uses raw Elo.

**Simulation**: all runs are simulated at once with numpy. A finished match is fixed to its real result in every run where the same two teams meet at that stage. A finished result that fits nowhere in the bracket (for example, when the configured seeding is wrong) is printed as a warning.

**Output**: the per-team table goes to `reports/bracket/odds_<event>_<odds>.csv`, with a JSON copy next to it.

## Live updates

`valchamps update` refreshes the forecast while the event runs. It scrapes the event (new results, and bracket slots as they fill in), re-runs the 100,000-tournament simulation from everything now known, and publishes to `odds/2766/`:

- `latest.json`: the current per-team odds, the time of the update, and the finished results behind them.
- `history.csv`: every published forecast, one row per team, so the odds can be followed through the event.
- `matchups.json`: the series prediction for every pairing of the event's teams at Bo3 and Bo5 (the same numbers as `predict-match`), so the hosted dashboard needs no API.

It publishes only when the set of finished results has changed, so running it every hour doesn't produce a commit every hour. The map model stays frozen for the event. Elo, form and the other features still update from every new result, because the whole history is replayed before each simulation.

**Schedule** (`.github/workflows/update-odds.yml`): the workflow runs hourly through September and October, and on demand from the Actions tab. Each run:

1. pulls only the database and the map model from DagsHub, not the page cache
2. restores the pages scraped by earlier runs from the Actions cache
3. runs `valchamps update`
4. commits `odds/` to `main` with `[skip ci]` when the odds changed

It needs the repository secrets `DAGSHUB_USERNAME` and `DAGSHUB_TOKEN`. It only reads from DagsHub and never pushes data back.

## API and dashboard

Install the extra first: `uv sync --extra serve --extra nn`. Then run the API and the dashboard in two terminals:

```bash
uv run valchamps serve        # FastAPI on http://127.0.0.1:8000 (interactive docs at /docs)
uv run valchamps dashboard    # Streamlit on http://127.0.0.1:8501, reading from the API
```

| Endpoint | Returns |
|---|---|
| `GET /health` | `{"status": "ok"}` |
| `GET /events/{id}/odds` | the latest published forecast (`odds/<id>/latest.json`) |
| `GET /events/{id}/odds/history` | every published forecast, one row per team per update |
| `GET /events/{id}/teams` | the event's teams and groups |
| `GET /predict?team_a=&team_b=&best_of=3` | series odds, score distribution, per-map odds and likeliest vetoes (as `predict-match`) |

**Data sources**: published odds come from `odds/` (`VALCHAMPS_ODDS_DIR`). Predictions replay the database and use `models/map_model.pkl` (`VALCHAMPS_MODEL_PATH`). That state loads on the first request, which takes a few seconds; after that a prediction takes well under a second. The API reloads it when the database or model file changes, so a scheduled update is picked up without a restart.

**Dashboard**: two tabs.

- **Title odds** shows the favourite, a bar chart of every team's title chance, a table of every stage, and the title odds over time with one team highlighted.
- **Match predictor** shows the series odds for any two teams, the final-score distribution, each team's chance on every map and the likeliest vetoes.

![Match predictor in the dashboard](docs/images/dashboard_match_predictor.png)

The chart colours come from a palette checked for colour blindness, in both light and dark mode. Both servers bind to localhost by default; pass `--host 0.0.0.0` to expose them.

### Hosted dashboard

With `VALCHAMPS_DATA_URL` set to a published `odds/` folder (a path or URL), the dashboard reads `latest.json`, `history.csv` and `matchups.json` from it instead of calling the API. The hosted version runs this way against the files the update job commits, so it needs no database, model or server of its own:

1. On [share.streamlit.io](https://share.streamlit.io), create an app from this repository, branch `main`, entrypoint `deploy/streamlit/streamlit_app.py`.
2. That's all: `deploy/streamlit/requirements.txt` installs only Streamlit, pandas, Altair and httpx, and the app reads `https://raw.githubusercontent.com/JasSaini101/2026__Val_Champs_Model/main/odds`. Each hourly commit shows up within a few minutes.

To try the same mode locally: `VALCHAMPS_DATA_URL=odds uv run streamlit run deploy/streamlit/streamlit_app.py`.

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
  series/           veto simulation, exact Bo3/Bo5 odds, series backtest, match predictions
  bracket/          tournament format, pairwise odds, Monte Carlo bracket simulation
  api/              FastAPI app: published odds, match predictions
  dashboard/        Streamlit dashboard and its charts
deploy/streamlit/   hosted dashboard entrypoint (reads the published odds from GitHub)
configs/events.yaml events to scrape
configs/bracket.yaml Champions format and playoff seeding
odds/<event>/       published live odds (latest.json, history.csv, matchups.json), committed by the update job
.github/workflows/  CI; update-odds.yml is the hourly live-update job
tests/              pytest suite + HTML fixtures
dvc.yaml            pipeline (ingest -> features -> train)
params.yaml         feature, model and series hyper-parameters
```
