"""Predicted bins — the containers the Random Forest expects to be full tomorrow.

Once ``FillLevelPredictor`` has produced a ``fill_predicted`` value per
container, the bins at or above the collection threshold are exactly the stops
a truck should serve on the next round. This module turns that prediction into
the three artefacts the app exposes:

  1. ``PredictedBins``  — the bin list with coordinates, exportable as CSV.
  2. ``PredictedRoute`` — depot → predicted bins → landfill, together with an
     N×N distance matrix **sliced out of the precomputed Google Maps matrix**
     for that car, so the distances are real road distances rather than a new
     approximation.
  3. ``PredictedBinsMap`` — a folium map of those bins, optionally with the
     nearest-neighbour collection order drawn on top.

Matrix provenance
-----------------
``src/compute_distance_matrices.py`` writes one N×N CSV per car, where row/column
``Point_k`` is the *k*-th row of that car's slice of ``data_geocoded.csv``, in
file order. ``add_matrix_point`` reproduces that numbering so any container row
can be traced back to its row in the Google Maps matrix - which is what makes
slicing a smaller matrix out of it possible without spending new API calls.
"""

from __future__ import annotations

import os
from pathlib import Path

import folium
import numpy as np
import pandas as pd

from app.ml.nn.nearest_neighbor import NearestNeighborSolver

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOOGLE_MAPS_MATRIX_DIR = PROJECT_ROOT / "data" / "distance_matrices" / "google_maps"

# Marker colour by predicted fill level (%) — checked from the top down.
_FILL_COLORS: list[tuple[float, str]] = [
    (100, "#7f0000"),   # overflowing
    (95,  "#c62828"),
    (90,  "#ef6c00"),
    (0,   "#f9a825"),   # at threshold
]

_DEPOT_COLOR = "#546e7a"
_LANDFILL_COLOR = "#212121"

# Shared style for every route line the app draws, here and in RouteMap.
# Thin enough to read the street names underneath, with rounded joins and a low
# smooth_factor so the line keeps the curve of the road instead of being
# simplified into visible corners (Leaflet simplifies *more* as the factor rises).
ROUTE_LINE_STYLE: dict = {
    "color":         "#1a1a1a",
    "weight":        2,
    "opacity":       0.9,
    "smooth_factor": 0.5,
    "line_join":     "round",
    "line_cap":      "round",
}


def add_matrix_point(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ``matrix_point`` column linking each row to the Google Maps matrix.

    ``Point_k`` in ``{car}_distance_matrix.csv`` is the *k*-th row of that car's
    slice of the geocoded CSV, so the numbering is simply a per-car running
    count in file order. Call this before any filtering or sorting, while
    the frame is still in file order — otherwise the numbering drifts out of
    step with the saved matrices.
    """
    df = df.copy()
    df["matrix_point"] = df.groupby("Car").cumcount() + 1
    return df


def fill_color(fill: float) -> str:
    """Marker colour for a predicted fill level (%)."""
    for limit, color in _FILL_COLORS:
        if fill >= limit:
            return color
    return _FILL_COLORS[-1][1]


class PredictedBins:
    """Containers predicted to reach the collection threshold tomorrow.

    Usage:
        bins = PredictedBins.from_predictions(df, predictions, THRESHOLD)
        bins.for_route(1).to_csv("outputs/predicted/route_1.csv")
    """

    #: Columns of the exported CSV, in order.
    COLUMNS: list[str] = [
        "route_id",
        "Car",
        "Id",
        "Capacity",
        "Address",
        "Latitude",
        "Longitude",
        "fill_current",
        "fill_predicted",
        "matrix_point",
    ]

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    @classmethod
    def from_predictions(
        cls,
        df: pd.DataFrame,
        predictions: pd.DataFrame,
        threshold: float,
    ) -> "PredictedBins":
        """Select predictions at/above *threshold* and attach their location.

        ``predictions`` must carry a ``source_index`` column pointing back at
        the row of *df* each prediction came from — that is what supplies the
        address, coordinates, and ``matrix_point`` the map and matrix need.
        """
        if "source_index" not in predictions.columns:
            raise KeyError(
                "predictions must carry a 'source_index' column linking each "
                "row back to the container it was predicted from."
            )
        if "matrix_point" not in df.columns:
            df = add_matrix_point(df)

        above = predictions[predictions["fill_predicted"] >= threshold]
        located = df.loc[above["source_index"]].reset_index(drop=True)
        above = above.reset_index(drop=True)

        frame = pd.DataFrame({
            "route_id":       located["route_id"],
            "Car":            located["Car"],
            "Id":             located["Id"],
            "Capacity":       located["Capacity"],
            "Address":        located["Address"],
            "Latitude":       located["Latitude"],
            "Longitude":      located["Longitude"],
            "fill_current":   above["fill_current"].astype(float),
            "fill_predicted": above["fill_predicted"].astype(float),
            "matrix_point":   located["matrix_point"].astype(int),
        })
        # Sorted by matrix row so the frame lines up with the Google Maps
        # matrix. This is a bookkeeping order, not a driving order — the round
        # itself is decided by the nearest-neighbour solver.
        frame = frame.sort_values(["Car", "matrix_point"]).reset_index(drop=True)
        return cls(frame[cls.COLUMNS])

    def for_route(self, route_id) -> "PredictedBins":
        """The subset of bins on a single route."""
        return PredictedBins(
            self.frame[self.frame["route_id"] == route_id].reset_index(drop=True)
        )

    def for_car(self, car: str) -> "PredictedBins":
        """The subset of bins served by a single vehicle."""
        return PredictedBins(
            self.frame[self.frame["Car"] == car].reset_index(drop=True)
        )

    @property
    def cars(self) -> list[str]:
        """Vehicles that have at least one predicted bin."""
        return sorted(self.frame["Car"].dropna().unique().tolist())

    @property
    def empty(self) -> bool:
        return self.frame.empty

    def __len__(self) -> int:
        return len(self.frame)

    def as_nodes(self) -> pd.DataFrame:
        """The bins in the node shape ``PredictedBinsMap`` draws.

        Used when there is no Google Maps matrix to build a ``PredictedRoute``
        from: the bins can still be mapped, just without depot, landfill, or a
        distance.
        """
        nodes = self.frame.copy()
        nodes["kind"] = "bin"
        nodes["label"] = [
            f"{row.Id} (Point_{row.matrix_point})" for row in nodes.itertuples()
        ]
        return nodes

    def to_csv(self, path: str | Path) -> Path:
        """Write the bins to *path* (parent directories created). Returns the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(path, index=False)
        return path


class PredictedRoute:
    """A collection round over one car's predicted bins.

    Nodes are ``depot → predicted bins → landfill``: the nearest-neighbour
    solver takes index 0 as the fixed start and index n-1 as the fixed end, and
    a real round starts at the depot and finishes at the landfill.

    The distance matrix is a slice of the car's Google Maps matrix — the same
    road distances the full-route figures use, just restricted to the stops that
    actually need collecting. No new API calls are made.

    Usage:
        route = PredictedRoute.build(bins.for_car(car), df, car)
        route.save_matrix("data/distance_matrices/predicted/CAR_distance_matrix.csv")
        order, km = route.optimized()
    """

    def __init__(self, nodes: pd.DataFrame, matrix: pd.DataFrame, car: str):
        self.nodes = nodes
        self.matrix = matrix
        self.car = car

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def matrix_path(cls, car: str) -> Path:
        """Path of the source Google Maps matrix for *car*."""
        return GOOGLE_MAPS_MATRIX_DIR / f"{car}_distance_matrix.csv"

    @classmethod
    def available(cls, car: str) -> bool:
        """True if the Google Maps matrix this slice is cut from exists."""
        return cls.matrix_path(car).exists()

    @classmethod
    def build(
        cls,
        bins: PredictedBins,
        df: pd.DataFrame,
        car: str,
    ) -> "PredictedRoute | None":
        """Slice a matrix for *car*'s predicted bins out of its Google Maps matrix.

        *df* is the full geocoded frame — the car's first and last rows are the
        departure depot and the landfill. Returns ``None`` when the Google Maps
        matrix for that car has not been generated yet.
        """
        source = cls.matrix_path(car)
        if not source.exists():
            return None

        if "matrix_point" not in df.columns:
            df = add_matrix_point(df)
        car_rows = df[df["Car"] == car]
        if len(car_rows) < 2:
            return None

        gm = pd.read_csv(source, index_col=0)
        bin_rows = bins.for_car(car).frame

        nodes = pd.concat([
            cls._node(car_rows.iloc[0], "depot"),
            *(cls._node(r, "bin") for _, r in bin_rows.iterrows()),
            cls._node(car_rows.iloc[-1], "landfill"),
        ], ignore_index=True)

        # matrix_point is 1-based; the matrix rows are 0-based positions.
        positions = [p - 1 for p in nodes["matrix_point"]]
        if max(positions) >= len(gm):
            raise ValueError(
                f"{source.name} has {len(gm)} points but the data references "
                f"Point_{max(positions) + 1}. Re-run "
                "`python src/compute_distance_matrices.py` — the matrix is "
                "out of step with data_geocoded.csv."
            )

        matrix = gm.iloc[positions, positions].copy()
        matrix.index = nodes["label"].tolist()
        matrix.columns = nodes["label"].tolist()
        return cls(nodes, matrix, car)

    @staticmethod
    def _node(row: pd.Series, kind: str) -> pd.DataFrame:
        """One matrix node — a depot, a predicted bin, or the landfill."""
        point = int(row["matrix_point"])
        if kind == "depot":
            label = f"Depot (Point_{point})"
        elif kind == "landfill":
            label = f"Landfill (Point_{point})"
        else:
            # The point number keeps the label unique even when a container Id
            # is collected more than once on the same round.
            label = f"{row['Id']} (Point_{point})"
        return pd.DataFrame([{
            "label":          label,
            "kind":           kind,
            "route_id":       row.get("route_id"),
            "Car":            row.get("Car"),
            "Id":             row.get("Id"),
            "Capacity":       row.get("Capacity"),
            "Address":        row.get("Address"),
            "Latitude":       float(row["Latitude"]),
            "Longitude":      float(row["Longitude"]),
            "fill_current":   row.get("fill_current", np.nan),
            "fill_predicted": row.get("fill_predicted", np.nan),
            "matrix_point":   point,
        }])

    # ------------------------------------------------------------------
    # Distances
    # ------------------------------------------------------------------
    @property
    def n_bins(self) -> int:
        """Number of predicted bins (matrix nodes minus depot and landfill)."""
        return int((self.nodes["kind"] == "bin").sum())

    def length_km(self, order: list[int]) -> float:
        """Distance driven visiting the nodes in *order*, depot → landfill."""
        m = self.matrix.to_numpy(dtype=float)
        return float(sum(m[order[i]][order[i + 1]] for i in range(len(order) - 1)))

    def optimized(self) -> tuple[list[int], float]:
        """Nearest-neighbour order over the bins, with depot and landfill fixed.

        Returns ``(order, km)`` where *order* indexes into ``self.nodes``.
        If the solver cannot find a path, falls back to the matrix's own node
        order — the bins still get collected, just without the optimisation.
        """
        labels = self.matrix.index.tolist()
        path, km = NearestNeighborSolver().solve(
            self.matrix.to_numpy(dtype=float), labels, verbose=False
        )
        if path is None:
            order = list(range(len(labels)))
            return order, self.length_km(order)
        position = {label: i for i, label in enumerate(labels)}
        return [position[label] for label in path], float(km)

    def ordered_nodes(self, order: list[int]) -> pd.DataFrame:
        """The node table rearranged into *order*."""
        return self.nodes.iloc[order].reset_index(drop=True)

    def save_matrix(self, path: str | Path) -> Path:
        """Write the sliced matrix to *path* (parent directories created)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.matrix.to_csv(path)
        return path


class PredictedBinsMap:
    """folium map of predicted bins, optionally with the collection order drawn.

    Usage:
        folium_map = PredictedBinsMap(route.nodes).draw()
        folium_map = PredictedBinsMap(route.ordered_nodes(order)).draw(numbered=True)
    """

    def __init__(self, nodes: pd.DataFrame):
        self.nodes = nodes

    def draw(
        self,
        numbered: bool = False,
        geometry: list[tuple] | None = None,
    ) -> folium.Map:
        """Build the map.

        Parameters
        ----------
        numbered : draw each stop as a numbered pin in row order rather than a
                   plain circle — use it when ``nodes`` is already in visiting order.
        geometry : OSRM road outline for the round, drawn as it follows the
                   streets. Without one no line is drawn: joining the stops
                   straight would show a path the truck cannot take.
        """
        lats = self.nodes["Latitude"].astype(float)
        lons = self.nodes["Longitude"].astype(float)
        map_ = folium.Map(location=[lats.mean(), lons.mean()], zoom_start=13)

        if geometry:
            folium.PolyLine(geometry, **ROUTE_LINE_STYLE).add_to(map_)

        for i, (_, row) in enumerate(self.nodes.iterrows(), start=1):
            details = self._details(row, i if numbered else None)
            location = [float(row["Latitude"]), float(row["Longitude"])]
            color = self._color(row)
            marker = folium.Marker(
                location=location,
                tooltip=folium.Tooltip(details, sticky=True),
                popup=folium.Popup(details, max_width=300),
                icon=self._pin(i, color),
            ) if numbered else folium.CircleMarker(
                location=location,
                radius=7 if row["kind"] == "bin" else 9,
                color="#ffffff", weight=2,
                fill=True, fill_color=color, fill_opacity=0.95,
                tooltip=folium.Tooltip(details, sticky=True),
                popup=folium.Popup(details, max_width=300),
            )
            marker.add_to(map_)

        return map_

    @staticmethod
    def _color(row: pd.Series) -> str:
        if row["kind"] == "depot":
            return _DEPOT_COLOR
        if row["kind"] == "landfill":
            return _LANDFILL_COLOR
        return fill_color(float(row["fill_predicted"]))

    @staticmethod
    def _pin(number: int, color: str) -> folium.DivIcon:
        """Teardrop pin with the stop number inside."""
        html = (
            f'<div style="position:relative;width:28px;height:28px;">'
            f'<div style="width:24px;height:24px;margin:0 2px;background:{color};'
            f'border:2px solid #fff;border-radius:50% 50% 50% 0;'
            f'transform:rotate(-45deg);box-shadow:0 1px 4px rgba(0,0,0,.45);"></div>'
            f'<div style="position:absolute;left:0;top:0;width:28px;height:26px;'
            f'display:flex;align-items:center;justify-content:center;color:#fff;'
            f'font-family:sans-serif;font-size:11px;font-weight:700;">{number}</div>'
            f'</div>'
        )
        return folium.DivIcon(html=html, icon_size=(28, 28), icon_anchor=(14, 28))

    @staticmethod
    def _details(row: pd.Series, stop_order: int | None) -> str:
        """HTML tooltip/popup content for one node.

        The route and vehicle are always shown: on the all-routes map the bins
        of three vehicles are mixed together (and their depots sit on the same
        spot), so which round a marker belongs to is the first thing you need.
        """
        head = f"Stop {stop_order} &middot; " if stop_order else ""
        origin = f"Route {row['route_id']} &middot; {row['Car']}"
        if row["kind"] == "depot":
            return f"<b>{head}Depot</b><br>{origin}<br>{row['Address']}"
        if row["kind"] == "landfill":
            return f"<b>{head}Landfill</b><br>{origin}<br>{row['Address']}"
        return (
            f"<b>{head}{row['Id']}</b><br>"
            f"{origin}<br>"
            f"{row['Address']}<br>"
            f"Capacity: {row['Capacity']}<br>"
            f"Fill level today: {float(row['fill_current']):.0f}%<br>"
            f"<b>Predicted tomorrow: {float(row['fill_predicted']):.0f}%</b><br>"
            f"Matrix row: Point_{int(row['matrix_point'])}"
        )

    @staticmethod
    def legend_html() -> str:
        """A small inline legend matching the marker colours."""
        swatch = (
            '<span style="display:inline-block;width:11px;height:11px;'
            'border-radius:50%;background:{};border:1px solid #fff;'
            'vertical-align:middle;margin-right:5px"></span>{}'
        )
        items = [
            swatch.format("#7f0000", "≥ 100%"),
            swatch.format("#c62828", "≥ 95%"),
            swatch.format("#ef6c00", "≥ 90%"),
            swatch.format("#f9a825", "at threshold"),
            swatch.format(_DEPOT_COLOR, "depot"),
            swatch.format(_LANDFILL_COLOR, "landfill"),
        ]
        return (
            '<div style="font-size:0.85em;color:gray;margin:0.3em 0">'
            + " &nbsp; ".join(items)
            + "</div>"
        )
