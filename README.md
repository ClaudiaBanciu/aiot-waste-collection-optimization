# aiot-waste-collection-optimization

AIoT-powered optimization for predictive waste collection and route planning in
Sibiu — reducing distance, cost and environmental impact.

The project starts from real data collected by three waste-collection vehicles
(routes SB25 / SB30 / SB45) and combines three ideas: **prediction** (which
containers will need collecting), **optimization** (a shorter vehicle route) and
**impact evaluation** (non-optimized vs optimized distance).

## Project structure

```
aiot-main/
├── data/
│   ├── raw/              the 3 raw CSV files (untouched)
│   └── processed/        data_combined.csv, data_geocoded.csv
├── src/                  data preparation pipeline
│   ├── data_preprocess.py   CSVPreprocessor: cleans a raw file, builds Address
│   ├── data_loader.py       load_and_combine: merges routes, adds depots + fill_level
│   └── geocoding.py         CoordinatesCalculator: address -> Latitude/Longitude
├── app/
│   ├── components/       reusable classes
│   │   ├── distance_calculator.py   DistanceCalculator, StandardDistanceCalculator,
│   │   │                            OSRMDistanceCalculator
│   │   └── fill_predictor.py        FillLevelPredictor (Random Forest)
│   └── main.py          Streamlit web interface
└── requirements.txt
```

## Data pipeline (`src/`)

1. **`data_preprocess.py` → `CSVPreprocessor`** — cleans a raw file: removes rows
   without a street/number or with an invalid number, and builds a single
   `Address` column.
2. **`data_loader.py` → `load_and_combine`** — merges the 3 routes, assigns a
   `route_id` (1/2/3), adds a departure and an arrival depot for each route, and
   generates a random `fill_level` (0–100%) for every container. → `data_combined.csv`
3. **`geocoding.py` → `CoordinatesCalculator`** — turns text addresses into
   coordinates (Latitude/Longitude) via Google Maps. → `data_geocoded.csv`

## Components (`app/components/`)

- **`DistanceCalculator`** (base class) — works on a list of `(lat, lon)` points:
  route length in the given order, nearest-neighbour optimization, and a
  neoptimized-vs-optimized comparison.
- **`StandardDistanceCalculator`** — straight-line distance (haversine).
- **`OSRMDistanceCalculator`** — real road distance via the public OSRM server
  (no Docker), including the street-following geometry drawn on the map; falls
  back to the standard calculator if OSRM is unavailable.
- **`FillLevelPredictor`** — next-day fill-level prediction with a
  **Random Forest** (decision trees), trained on a simulated 7-day history.

## Web interface (`app/main.py`)

For a selected route it shows: filters, statistics, the container table, the
**non-optimized vs optimized route comparison** (straight-line and road distance,
side-by-side maps with hover details) and the **Random Forest prediction**
(test MAE, rule-today vs prediction-tomorrow).

## Setup & run

```bash
pip install -r requirements.txt
python -m streamlit run app/main.py
```

The app reads `data/processed/data_geocoded.csv`, which is already included, so
you can run it directly. The OSRM road distance uses the public server
`router.project-osrm.org` (needs internet); without it, the app automatically
uses the straight-line distance.
