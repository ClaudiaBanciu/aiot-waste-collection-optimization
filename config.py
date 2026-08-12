"""Central configuration — all project-wide constants live here.

Import from any module with:
    from config import THRESHOLD, INPUT_FILE, ...
"""

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

# Fill level (%) above which a container "needs collection"
THRESHOLD: int = 80

# Path to the geocoded CSV produced by src/geocoding.py
INPUT_FILE: str = "data/processed/data_geocoded.csv"

# ---------------------------------------------------------------------------
# Data loading (src/data_loader.py)
# ---------------------------------------------------------------------------

# Raw CSV files and their route IDs
RAW_FILES: list[tuple[str, int]] = [
    ("data/raw/SB25SOM.csv", 1),
    ("data/raw/SB30SOM.csv", 2),
    ("data/raw/SB45SOM.csv", 3),
]

# Columns retained in the combined output
FINAL_COLUMNS: list[str] = [
    "route_id",
    "Car",
    "Datetime",
    "Id",
    "Capacity",
    "Address",
    "fill_level",
]

# Depot locations added as virtual stops at the start and end of each route
DEPOT_START: str = "Strada Șelimbărului 90, Cisnădie, Romania"
DEPOT_END: str = "DN1 FN, Cristian 557085"
