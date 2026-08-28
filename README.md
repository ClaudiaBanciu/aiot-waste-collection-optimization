# AIoT Waste Collection Optimization — Sibiu

> **Research project** — predictive waste collection and route planning for three
> real collection routes in Sibiu, aimed at cutting driven kilometres.

632 container stops from three vehicle logs are cleaned, geocoded, and used to
answer two questions:

1. **Which containers need collecting?** A Random Forest, trained on a simulated
   30-day sawtooth fill history, predicts each container's fill level for
   tomorrow. Everything at or above `THRESHOLD` becomes a stop on the next round.
2. **In what order?** A nearest-neighbour TSP heuristic with a fixed depot start
   and a fixed landfill end reorders the stops, measured against the route as it
   was actually driven.

Results are shown in a Streamlit dashboard and written to disk (fill history,
predicted bins, collection rounds, distance matrices, route listings).

**Scope.** The container fill levels are **simulated**, not measured — there is no
live sensor feed, and the raw logs carry no fill data. Routing is single-vehicle
nearest-neighbour per route, not a capacity-constrained VRP. There are no tests.

---

## Architecture

```
data/raw/*.csv
   │ src/data_loader.py + data_preprocess.py — clean, build Address, add depot & landfill
   ▼
data/processed/data_combined.csv
   │ src/geocoding.py — Google Maps Geocoding
   ▼
data/processed/data_geocoded.csv          ← the app's stop list
   │ src/compute_distance_matrices.py — Google Maps Distance Matrix
   ▼
data/distance_matrices/google_maps/{vehicle}_distance_matrix.csv
   ├─ src/save_optimized_routes.py → outputs/routes/*.txt
   └─ app/main.py → interface.run(df) → outputs/predicted/, data/simulated/,
                                        data/distance_matrices/{standard,osrm,predicted}/
```

Layers: `src/` is the batch pipeline (run manually, writes files); `app/components/`
holds the domain classes; `app/ml/nn/` holds the solver; `config.py` holds every
constant; `app/main.py` loads the frame and delegates the whole UI to
`interface.run(df)`.

**The organising idea — `Point_k`.** Every distance is expressed over an N×N
matrix in km, where `Point_k` is the *k*-th row of that vehicle's slice of
`data_geocoded.csv`, in file order. `add_matrix_point()` (in `predicted_bins.py`)
reproduces that numbering as a `matrix_point` column before any filtering. That
is what lets `PredictedRoute` slice `depot → predicted bins → landfill` straight
out of the existing Google Maps matrix - real road distances for tomorrow's
round, no new API calls. `PredictedRoute.build()` raises if the matrix and the
CSV have drifted out of step.

The three distance sources share one `DistanceCalculator` interface (`matrix`,
`route_length`, `compare`), so the dashboard compares them side by side and the
solver runs unchanged on any of them:

| Class | Source |
|---|---|
| `StandardDistanceCalculator` | Haversine × 1.35 detour factor |
| `OSRMDistanceCalculator` | OSRM road network + drawable geometry; falls back to standard |
| `SequentialRouteDistanceCalculator` | Google Maps matrix, stops in recorded order — the "as driven" baseline |

### How the prediction works

Each container gets one fixed daily fill rate drawn from its capacity class
(`120L` → 3–6 %/day, `240L` → 4–7, `1.100L` → 6–10); day 0 is drawn uniformly in
0–100 %, past days follow by modular arithmetic so the sawtooth wraps, plus
σ = 1.5 pp of noise. A `RandomForestRegressor` (200 trees) predicts tomorrow from
today's level, yesterday's level, and the day-over-day growth — trained **only on
the ascending phase**, so it learns the fill rate rather than the post-collection
drop.

**One model per route.** The same algorithm, hyper-parameters and split run once
per route, so each of the three is its own dataset with its own scores. Route
identity is never a feature — it decides which model a container belongs to.

Scoring holds out the most recent ~20 % of the simulated days **chronologically**:
no model is trained on a day later than one it is scored on, matching how
`predict_next_day()` is actually used.

The predicted round is reported optimized-only: those bins have never been driven
as a round, so a "before" figure would be a route no driver ever took.

### Project structure

```
data/                      git-ignored in full — see "Generating the data files"
  raw/                     SB25SOM (334 pts) · SB30SOM (199) · SB45SOM (105)
  processed/               data_combined.csv → data_geocoded.csv
  simulated/               simulated_history.csv — written on app start
  distance_matrices/       google_maps/ standard/ osrm/ predicted/
src/
  data_preprocess.py       CSVPreprocessor — cleans a raw file, builds Address
  data_loader.py           DataLoader — merges routes, adds depot & landfill
  geocoding.py             CoordinatesCalculator — address → Lat/Lon
  compute_distance_matrices.py   Google Maps N×N matrix per vehicle
  save_optimized_routes.py NN route listings for all 3 methods
app/
  components/data_reader.py         auto-detects encoding and separator
  components/fill_predictor.py      sawtooth simulation + Random Forest
  components/distance_calculator.py Standard / OSRM / Sequential calculators
  components/predicted_bins.py      PredictedBins · PredictedRoute · PredictedBinsMap
  components/interface.py           full Streamlit UI — entry point run(df)
  ml/nn/nearest_neighbor.py         NearestNeighborSolver — fixed start 0, end n-1
  main.py                           loads + caches the data, calls run(df)
outputs/                 git-ignored · predicted/ routes/ exports/(legacy)
config.py                paths, THRESHOLD, N_DAYS, depots
```

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ (uses `X | None` syntax; developed on 3.13). A `.env` in the project
root, needed to build the data files (the app itself reads neither key):

```
GOOGLE_MAPS_API_KEY=...              # src/geocoding.py
GOOGLE_MAPS_DISTANCE_MATRIX_KEY=...  # src/compute_distance_matrices.py
```

`OSRM_URL` optionally points at your own OSRM instead of `router.project-osrm.org`.

---

## Generating the data files

**`data/` and `outputs/` are git-ignored**, so a clone gives you the code and
nothing else. Put the three raw vehicle logs in `data/raw/` (`SB25SOM.csv`,
`SB30SOM.csv`, `SB45SOM.csv` — the columns `data_preprocess.py` expects are
`Car, Datetime, Id, Capacity, Street, Number, City, Country`), then run steps 1–3
from the project root.

| # | Command | Produces | Needs |
|---|---|---|---|
| 1 | `python src/data_loader.py` | `data/processed/data_combined.csv` (638 rows) | — |
| 2 | `python src/geocoding.py` | `data/processed/data_geocoded.csv` | `GOOGLE_MAPS_API_KEY` |
| 3 | `python src/compute_distance_matrices.py` | one N×N matrix per vehicle | `GOOGLE_MAPS_DISTANCE_MATRIX_KEY` |
| 4 | `python src/save_optimized_routes.py` *(optional)* | `outputs/routes/*.txt` — one per vehicle × method | — |

Step 1 must be run **as a script**, not `python -m src.data_loader` — the module
imports `data_preprocess` as a top-level module. Step 4 skips the Google Maps
listing if step 3 was skipped, and the OSRM listing if the server is unreachable.

Step 3 costs ~162,000 billed matrix elements (334² + 199² + 105²), so run it once
and keep the CSVs. **It writes to `data/distance_matrices/`, but everything reads
from `data/distance_matrices/google_maps/`** — move them afterwards:

```bash
mkdir -p data/distance_matrices/google_maps && mv data/distance_matrices/*_distance_matrix.csv data/distance_matrices/google_maps/
```

Skipping step 3 still leaves a working app, but the Google Maps column reports
the missing matrix, the optimized map falls back to the haversine ordering, and
the predicted round reports 0 km.

The app itself writes, on every run: `data/simulated/simulated_history.csv`,
`outputs/predicted/{predicted_bins,collection_rounds}.csv`, and
`data/distance_matrices/predicted/`. It also fills `standard/` and `osrm/` the
first time those matrices are missing.

---

## Running the app

```bash
python -m streamlit run app/main.py
```

Then open <http://localhost:8501>.

The dashboard: pick a route in the sidebar, then get the stop table; the three
distance methods side by side (original vs NN-optimized, with % saved); the
as-driven and optimized routes on two maps drawn along real streets; *full
today* vs *full tomorrow*; a map of the bins predicted above the
threshold — numbered in NN order for a single vehicle, plain markers across all
three; and the 30-day simulated sawtooth for a selected container.

OSRM uses the public server and needs internet. If unreachable, that column says
so and no road line is drawn; the other two columns are unaffected.

---

## Configuration (`config.py`)

| Constant | Default | Description |
|---|---|---|
| `THRESHOLD` | `85` | Fill level (%) at or above which a container needs collection |
| `N_DAYS` | `30` | Days of simulated history per container |
| `INPUT_FILE` | `data/processed/data_geocoded.csv` | The app's input |
| `DEPOT_START` / `DEPOT_END` | Cisnădie / Cristian | Virtual first and last stop of every route |

Also there: `RAW_FILES`, `FINAL_COLUMNS`, and the three predicted-output paths.

---

## Known rough edges

- `src/compute_distance_matrices.py` writes one directory above where the
  matrices are read from (see step 3).
- `python -m src.data_loader` fails; so does the *"Pipeline: Load + Clean"* entry
  in `.vscode/launch.json`.
- The matrices under `distance_matrices/standard/` and `osrm/` are written but
  never read back. A comment in `save_optimized_routes.py` claims the app reuses
  the OSRM one instead of re-querying; it does not — `OSRMDistanceCalculator`
  always goes to the server. Wiring that cache up would save many requests.
- `outputs/exports/*.html` came from a route viewer that has since been removed;
  nothing writes there any more.
- `config.THRESHOLDS` and `FillLevelPredictor.THRESHOLDS` (per-capacity
  thresholds) are defined but unused — only the global `THRESHOLD` is applied.
