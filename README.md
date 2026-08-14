# AIoT Waste Collection Optimization — Sibiu

> **Research project** — AIoT-enabled multi-objective optimization for predictive
> waste collection and real-time dynamic route planning, with the goal of reducing
> operational costs and environmental impact.

---

## Objectives

1. **Predictive analytics** — forecast the fill level of each waste container for
   the next day using a machine-learning model (Random Forest), trained on a
   simulated 30-day sawtooth history anchored to real sensor measurements.
2. **Route optimization** — apply the nearest-neighbour heuristic to the current
   collection order and compare the result against the as-driven route, using both
   straight-line (Haversine) and real road distances (OSRM).
3. **Decision support** — surface the rule-based (current fill ≥ threshold) and
   model-based (predicted fill ≥ threshold tomorrow) container sets side-by-side,
   so dispatchers can proactively schedule collection before a bin overflows.
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
│   │   ├── data_combined.csv     Merged routes + depots + fill_level
│   │   └── data_geocoded.csv     + Latitude / Longitude (Google Maps)
│   └── simulated/
│       └── simulated_history.csv 30-day sawtooth history (auto-generated on startup)
│
├── src/                          Data preparation pipeline (run once)
│   ├── data_preprocess.py        CSVPreprocessor — cleans raw files, builds Address
│   ├── data_loader.py            load_and_combine — merges routes, adds depots & fill_level
│   └── geocoding.py              CoordinatesCalculator — address → Lat/Lon via Google Maps
│
├── app/
│   ├── components/               Reusable, independently testable classes
│   │   ├── data_reader.py        DataReader — auto-detects CSV encoding and separator
│   │   ├── fill_level.py         FillLevel — normalises the fill_level column to Fill_num (0-100)
│   │   ├── fill_predictor.py     FillLevelPredictor — sawtooth simulation + Random Forest
│   │   ├── distance_calculator.py  StandardDistanceCalculator (Haversine) +
│   │   │                           OSRMDistanceCalculator (road network)
│   │   ├── interface.py          Main Streamlit UI — entry point is run(df)
│   │   └── webscript.py          Legacy route viewer (kept for reference)
│   └── main.py                   App entry point — loads data, calls interface.run(df)
│
├── outputs/
│   └── exports/                  Static HTML map exports (exploratory)
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
- Day 0 is anchored to the real measured value so the simulated past is
  consistent with the actual observation.
- A **RandomForestRegressor** is trained **only on the ascending phase** of the
  sawtooth, so it learns the fill rate — not the post-collection drop.
- The simulated history is saved automatically to `data/simulated/simulated_history.csv`.

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

---