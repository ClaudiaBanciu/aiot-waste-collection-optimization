# AIoT Waste Collection Optimization — Sibiu

> **Research project** — AIoT-enabled multi-objective optimization for predictive
> waste collection and real-time dynamic route planning, with the goal of reducing
> operational costs and environmental impact.

---

## Objectives

1. **Predictive analytics** — forecast the fill level of each waste container for
   the next day using a machine-learning model (Random Forest), trained on a
   simulated 30-day sawtooth history.
2. **Route optimization** — apply the nearest-neighbour heuristic to the current
   collection order and compare the result against the as-driven route, using both
   straight-line (Haversine) and real road distances (OSRM).
3. **Decision support** — surface the rule-based (current fill ≥ threshold) and
   model-based (predicted fill ≥ threshold tomorrow) container sets side-by-side,
   so dispatchers can proactively schedule collection before a bin overflows. The
   predicted set is mapped, exported as CSV, and costed with its own distance
   matrix, so tomorrow's round is a concrete plan rather than a count.
4. **Impact evaluation** — quantify distance savings (km and %) between the
   unoptimized and optimized routes, with road geometry visualized on an
   interactive map.

---

## Project structure

```
aiot-waste-collection-optimization/
│
├── data/
│   ├── raw/                      Original CSVs from the 3 collection vehicles
│   │   ├── SB25SOM.csv           Route 1 — SB25
│   │   ├── SB30SOM.csv           Route 2 — SB30
│   │   └── SB45SOM.csv           Route 3 — SB45
│   ├── processed/                Pipeline outputs (ready for the app)
│   │   ├── data_combined.csv     Merged routes + depots
│   │   └── data_geocoded.csv     + Latitude / Longitude (Google Maps)
│   ├── simulated/
│   │   └── simulated_history.csv 30-day sawtooth history (auto-generated on startup)
│   └── distance_matrices/        One N×N matrix (km) per vehicle
│       ├── google_maps/          Real road distances — src/compute_distance_matrices.py
│       ├── standard/             Haversine × 1.35 (auto-generated on startup)
│       ├── osrm/                 OSRM road network (auto-generated on startup)
│       └── predicted/            Google Maps matrix sliced to the predicted bins
│
├── src/                          Data preparation pipeline (run once)
│   ├── data_preprocess.py        CSVPreprocessor — cleans raw files, builds Address
│   ├── data_loader.py            load_and_combine — merges routes and adds depots
│   └── geocoding.py              CoordinatesCalculator — address → Lat/Lon via Google Maps
│
├── app/
│   ├── components/               Reusable, independently testable classes
│   │   ├── data_reader.py        DataReader — auto-detects CSV encoding and separator
│   │   ├── fill_predictor.py     FillLevelPredictor — sawtooth simulation + Random Forest
│   │   ├── predicted_bins.py     PredictedBins / PredictedRoute / PredictedBinsMap —
│   │   │                         bins predicted above threshold: CSV, map, distance matrix
│   │   ├── distance_calculator.py  StandardDistanceCalculator (Haversine) +
│   │   │                           OSRMDistanceCalculator (road network)
│   │   ├── interface.py          Main Streamlit UI — entry point is run(df)
│   │   └── webscript.py          Legacy route viewer (kept for reference)
│   └── main.py                   App entry point — loads data, calls interface.run(df)
│
├── outputs/
│   ├── exports/                  Static HTML map exports (exploratory)
│   └── predicted/                Auto-generated on every app run
│       ├── predicted_bins.csv    Bins predicted ≥ threshold + coordinates
│       └── collection_rounds.csv Per-vehicle bin count and round length
│
├── config.py                     Central constants (paths, threshold, N_DAYS, …)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment variables (optional — only needed to re-run geocoding)

Create a `.env` file in the project root:

```
GOOGLE_MAPS_API_KEY=your_key_here
```

The app reads `data/processed/data_geocoded.csv`, which is already included in the
repository, so **you can skip this step and run the app directly**.

---

## Running the app

```bash
python -m streamlit run app/main.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

> **Road distances (OSRM)** — the app uses the public server
> `router.project-osrm.org` (requires internet). If it is unavailable, the app
> falls back to straight-line distances automatically — no configuration needed.

---

## Re-running the data pipeline (optional)

Only necessary if you want to regenerate the processed CSVs from scratch.

```bash
# Step 1 — clean raw files and combine routes
python -m src.data_loader

# Step 2 — geocode addresses (requires GOOGLE_MAPS_API_KEY in .env)
python -m src.geocoding
```

---

## Key components

### `FillLevelPredictor` (`app/components/fill_predictor.py`)

Simulates a 30-day **sawtooth fill history** for each container:

- Each container fills at a fixed daily rate drawn from its capacity class
  (`120L` → 3–6 %/day, `240L` → 4–7 %/day, `1.100L` → 6–10 %/day).
- The fill level resets to ~0 % when it reaches 100 % (collection event).
- Day 0 is drawn uniformly in the 0–100 % range, so each container starts at
  an independent point in its own fill cycle.
- A **RandomForestRegressor** is trained **only on the ascending phase** of the
  sawtooth, so it learns the fill rate — not the post-collection drop.
- The simulated history is saved automatically to `data/simulated/simulated_history.csv`.

### `PredictedBins` / `PredictedRoute` / `PredictedBinsMap` (`app/components/predicted_bins.py`)

Turns the Random Forest output into the next round's work order. Every container
whose `fill_predicted` reaches `THRESHOLD` becomes a stop.

The app shows **the map**: each predicted bin as a marker coloured by predicted
fill level (yellow at the threshold through dark red for overflowing), with the
depot and the landfill for context. For a single vehicle the stops are drawn as
numbered pins in nearest-neighbour order, joined by a line that follows the
actual streets (see *Route lines* below); across all three vehicles they are
plain markers with no line, and hovering one names the route and vehicle it
belongs to.

The detail is written to disk rather than shown as tables, so a run leaves
something behind to work from:

- `outputs/predicted/predicted_bins.csv` — route, vehicle, container Id,
  capacity, address, latitude/longitude, fill level today, predicted fill level
  tomorrow, and the matrix row the bin came from.
- `outputs/predicted/collection_rounds.csv` — one row per vehicle: bins to
  collect, nearest-neighbour round length, and the matrix it was measured on.
- `data/distance_matrices/predicted/{vehicle}_distance_matrix.csv` — an N×N
  matrix over `depot → predicted bins → landfill`.

There is deliberately **no unoptimized baseline** here. The predicted bins are a
subset of the vehicle's stops that has never been driven as a round, so ordering
them the way they happen to sit in the source data would invent a route no
driver would take and compare against it. The optimized round is the only route
the section reports.

The matrix is **sliced out of the vehicle's existing Google Maps matrix** rather
than measured again, so the numbers are the same real road distances the
full-route figures use and no new API calls are spent. This works because
`Point_k` in `{vehicle}_distance_matrix.csv` is the *k*-th row of that vehicle's
slice of `data_geocoded.csv` in file order — `add_matrix_point()` reproduces that
numbering so any container can be traced back to its row in the source matrix.

> Re-run `python src/compute_distance_matrices.py` whenever `data_geocoded.csv`
> changes: the slice assumes the matrix and the CSV are still in step, and raises
> a `ValueError` if it can tell they are not.

### Route lines on the maps

Every map draws its route with `OSRMDistanceCalculator.road_route()`, whose
geometry follows the real street network — the line bends round corners and
takes the one-way system rather than cutting between stops.

If OSRM is unreachable, **no line is drawn at all**. A straight line between
consecutive stops would trace a path no truck can take, so an empty map is the
more honest fallback; the markers still show every stop. The same applies to the
all-routes view of the predicted bins, which has no single round to trace.

> The drawn path comes from OSRM while the reported kilometres come from the
> Google Maps matrix, so the two can differ by a few percent. Google Maps has no
> free geometry endpoint here, and OSRM is the only source of a drawable outline
> that costs nothing per call.

### `StandardDistanceCalculator` / `OSRMDistanceCalculator` (`app/components/distance_calculator.py`)

Both expose the same interface: `route_length`, `optimize` (nearest-neighbour
with fixed depot start and landfill end), and `compare` (unoptimized vs
optimized). The OSRM calculator additionally returns road geometry for map
rendering and uses the `/table/v1` endpoint for the full N×N distance matrix.

---

## Configuration (`config.py`)

| Constant | Default | Description |
|---|---|---|
| `THRESHOLD` | `85` | Fill level (%) above which a container needs collection |
| `N_DAYS` | `30` | Days of simulated history per container |
| `SIMULATED_HISTORY_FILE` | `data/simulated/simulated_history.csv` | History output path |
| `INPUT_FILE` | `data/processed/data_geocoded.csv` | App input data |
| `PREDICTED_BINS_FILE` | `outputs/predicted/predicted_bins.csv` | Predicted bins + coordinates |
| `PREDICTED_ROUNDS_FILE` | `outputs/predicted/collection_rounds.csv` | Per-vehicle round summary |
| `PREDICTED_MATRIX_DIR` | `data/distance_matrices/predicted` | Sliced matrices, one per vehicle |

---