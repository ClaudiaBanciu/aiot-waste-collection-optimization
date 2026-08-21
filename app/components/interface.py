"""Full Streamlit interface — entry point is run(df).

main.py stays thin and only calls run(df) from here.

Classes:
  RouteMap      — builds and draws folium maps for a route.
  RouteAnalyzer — compares distance methods (standard vs OSRM).

Module-level cached helpers are kept as functions because Streamlit's
@st.cache_data / @st.cache_resource decorators work most reliably on
module-level functions.
"""
import os
import sys

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.components.distance_calculator import (
    OSRMDistanceCalculator,
    SequentialRouteDistanceCalculator,
    StandardDistanceCalculator,
)
from app.components.fill_predictor import FillLevelPredictor
from app.ml.nn.nearest_neighbor import NearestNeighborSolver
from config import THRESHOLD, THRESHOLDS, N_DAYS, SIMULATED_HISTORY_FILE

_DIST_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "distance_matrices")
_MATRIX_DIR = os.path.join(_DIST_DIR, "google_maps")
_STD_DIR    = os.path.join(_DIST_DIR, "standard")
_OSRM_DIR   = os.path.join(_DIST_DIR, "osrm")


# =====================================================================
# Streamlit-cached helpers (module-level for decorator compatibility)
# =====================================================================

@st.cache_data(show_spinner="Running Nearest Neighbour algorithm...")
def _nn_results_cached(car: str) -> dict:
    """Run NearestNeighborSolver on the precomputed Google Maps matrix."""
    solver = NearestNeighborSolver()
    matrix_path = os.path.join(_MATRIX_DIR, f"{car}_distance_matrix.csv")
    if not os.path.exists(matrix_path):
        return {"google_maps": None}
    path, dist = solver.from_file(matrix_path, verbose=False)
    return {"google_maps": {"optimized_km": dist, "path": path}}


@st.cache_data(show_spinner=False)
def _sequential_distance_cached(car: str) -> float | None:
    if not SequentialRouteDistanceCalculator.available(car):
        return None
    calc = SequentialRouteDistanceCalculator.for_car(car)
    return calc.distance_for_first_file()


def _save_matrix(matrix: list[list[float]], car: str, directory: str) -> None:
    """Save an N×N matrix to directory/{car}_distance_matrix.csv. Skips if already exists."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{car}_distance_matrix.csv")
    if os.path.exists(path):
        return
    n = len(matrix)
    labels = [f"Point_{i + 1}" for i in range(n)]
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(path)


@st.cache_data(show_spinner="Computing distances (standard + OSRM on road)...")
def _compare_distances_cached(points_tuple: tuple, car: str) -> dict:
    points = list(points_tuple)

    # Standard — compute matrix explicitly so we can save it
    std_calc = StandardDistanceCalculator()
    std_matrix = std_calc.matrix(points)
    if car:
        _save_matrix(std_matrix, car, _STD_DIR)
    unoptimized = std_calc.route_length(points, std_matrix)
    n = len(points)
    path_names, optimized = NearestNeighborSolver().solve(
        np.array(std_matrix), [f"P{i}" for i in range(n)], verbose=False
    )
    order = [int(name[1:]) for name in path_names]
    savings = (1 - optimized / unoptimized) * 100 if unoptimized else 0.0
    standard = {
        "method": "standard",
        "unoptimized_km": unoptimized,
        "optimized_km": optimized,
        "savings_%": savings,
        "optimal_order": order,
    }

    # OSRM — compare() uses road_route() for actual measurements;
    # we separately fetch the table matrix to save it (fails silently for >100 pts)
    osrm_calc = OSRMDistanceCalculator()
    osrm_ok = osrm_calc.available()
    osrm = osrm_calc.compare(points) if osrm_ok else None
    if osrm_ok and car:
        try:
            osrm_matrix = osrm_calc.matrix(points)
            _save_matrix(osrm_matrix, car, _OSRM_DIR)
        except Exception:
            pass

    return {"standard": standard, "osrm": osrm, "osrm_ok": osrm_ok}


@st.cache_resource(show_spinner=False)
def _train_predictor_cached(
    _df: pd.DataFrame,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Train the Random Forest once on all containers and save the history CSV.

    Leading underscore tells Streamlit not to hash the DataFrame.

    Returns
    -------
    (mae, predictions, history)
      mae         — float, test-set Mean Absolute Error
      predictions — DataFrame with fill_current / fill_predicted per container
      history     — DataFrame with 30-day simulated sawtooth history per container
    """
    containers = _df[
        _df["Id"].notna() & _df["Capacity"].notna() & _df["Fill_num"].notna()
    ][["Id", "Capacity", "route_id"]].copy()
    containers["fill_level"] = _df.loc[containers.index, "Fill_num"].astype(float)

    predictor = FillLevelPredictor(n_days=N_DAYS)
    predictor.train(containers)
    predictor.save_history(SIMULATED_HISTORY_FILE)

    predictions = predictor.predict_next_day()
    return predictor.mae_, predictions, predictor.history_


# =====================================================================
# RouteMap — builds folium maps for a route
# =====================================================================

class RouteMap:
    """Builds and draws folium maps for a single route.

    Usage:
        route_map = RouteMap.from_route(df_route)
        folium_map = route_map.draw(color="red", popup_title="unoptimized route")

    Or, for a custom set of points:
        route_map = RouteMap(points, ordered_df)
        folium_map = route_map.draw(color="green", popup_title="optimized route",
                                    geometry=osrm_geometry)
    """

    def __init__(self, points: list[tuple], ordered: pd.DataFrame):
        self.points = points
        self.ordered = ordered

    @classmethod
    def from_route(cls, df_route: pd.DataFrame) -> "RouteMap":
        """Factory: extract ordered (depot → stops → dump) points from a route DataFrame.

        The first depot row is the departure point (fixed start).
        The second depot row is the landfill/dump (fixed end).
        Intermediate stops are sorted chronologically and can be reordered.
        """
        depots = df_route[df_route["Id"].isna()]
        stops = df_route[df_route["Id"].notna()].sort_values("Datetime")
        parts = []
        if len(depots) >= 1:
            parts.append(depots.iloc[[0]])    # departure depot (fixed start)
        parts.append(stops)
        if len(depots) >= 2:
            parts.append(depots.iloc[[-1]])   # landfill / dump (fixed end)
        ordered = pd.concat(parts).reset_index(drop=True)
        points = list(zip(ordered["Latitude"], ordered["Longitude"]))
        return cls(points, ordered)

    def reorder(self, order: list[int]) -> "RouteMap":
        """Return a new RouteMap with points reordered according to order indices."""
        new_points = [self.points[i] for i in order]
        new_ordered = self.ordered.iloc[order].reset_index(drop=True)
        return RouteMap(new_points, new_ordered)

    def draw(
        self,
        color: str,
        popup_title: str,
        geometry: list[tuple] | None = None,
    ) -> folium.Map:
        """Build and return a folium map for this route.

        If geometry is provided (OSRM road outline), the dotted line follows
        the streets; otherwise it connects stops with straight lines.
        """
        center_lat = sum(p[0] for p in self.points) / len(self.points)
        center_lon = sum(p[1] for p in self.points) / len(self.points)
        map_ = folium.Map(location=[center_lat, center_lon], zoom_start=13)

        line = geometry if geometry else self.points
        folium.PolyLine(
            line, color="black", weight=3, opacity=0.7, dash_array="2, 8"
        ).add_to(map_)

        n = len(self.points)
        stop_order = 0
        for i, (lat, lon) in enumerate(self.points):
            row = self.ordered.iloc[i]
            if i == 0 and pd.isna(row["Id"]):
                # Fixed start — depot
                stop_order += 1
                details = self._point_details(row, popup_title, stop_order, kind="depot")
                icon = self._numbered_pin(stop_order, "gray")
            elif i == n - 1 and pd.isna(row["Id"]):
                # Fixed end — dump / landfill
                stop_order += 1
                details = self._point_details(row, popup_title, stop_order, kind="dump")
                icon = self._numbered_pin(stop_order, "black")
            else:
                stop_order += 1
                details = self._point_details(row, popup_title, stop_order)
                icon = self._numbered_pin(stop_order, color)
            folium.Marker(
                location=[lat, lon],
                tooltip=folium.Tooltip(details, sticky=True),
                popup=folium.Popup(details, max_width=300),
                icon=icon,
            ).add_to(map_)
        return map_

    @staticmethod
    def _numbered_pin(number: int, color: str) -> folium.DivIcon:
        """Teardrop pin with the stop number inside — no extra icon library needed."""
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
    def _point_details(
        row: pd.Series,
        title: str,
        stop_order: int,
        kind: str = "stop",
    ) -> str:
        """HTML tooltip/popup content for a single stop, depot, or dump."""
        if kind == "depot":
            return (
                f"<b>Stop {stop_order} &middot; Depot</b><br>"
                f"{row['Address']}<br><i>{title}</i>"
            )
        if kind == "dump":
            return (
                f"<b>Stop {stop_order} &middot; Landfill</b><br>"
                f"{row['Address']}<br><i>{title}</i>"
            )
        fill = f"{row['Fill_num']:.0f}%" if pd.notna(row["Fill_num"]) else "-"
        time = row["time"] if pd.notna(row["time"]) else "-"
        return (
            f"<b>Stop {stop_order} &middot; {row['Id']}</b><br>"
            f"{row['Address']}<br>"
            f"Capacity: {row['Capacity']}<br>"
            f"Fill level: {fill}<br>"
            f"Time: {time}<br>"
            f"Vehicle: {row['Car']}"
        )


# =====================================================================
# RouteAnalyzer — compares distance methods for a route
# =====================================================================

class RouteAnalyzer:
    """Compares standard (straight-line) vs OSRM (road) distances for a route.

    Results are Streamlit-cached via a private module-level function so that
    expensive OSRM calls are never repeated within a session.

    Usage:
        analyzer = RouteAnalyzer(points).compute()
        print(analyzer.standard["unoptimized_km"])
        print(analyzer.osrm_available)
    """

    def __init__(self, points: list[tuple]):
        self.points = points
        self._result: dict | None = None

    def compute(self, car: str = "") -> "RouteAnalyzer":
        """Run both distance methods and cache the result. Returns self."""
        self._result = _compare_distances_cached(tuple(self.points), car)
        return self

    def _require_computed(self) -> None:
        if self._result is None:
            raise RuntimeError("Call compute() before accessing results.")

    @property
    def standard(self) -> dict:
        """Standard (straight-line) distance results."""
        self._require_computed()
        return self._result["standard"]

    @property
    def osrm(self) -> dict | None:
        """OSRM (road network) distance results, or None if server unavailable."""
        self._require_computed()
        return self._result["osrm"]

    @property
    def osrm_available(self) -> bool:
        """True if the OSRM server responded successfully."""
        self._require_computed()
        return self._result["osrm_ok"]


# =====================================================================
# Entry point — called from main.py
# =====================================================================

def run(df: pd.DataFrame) -> None:
    """Render the full Streamlit interface."""
    st.title("🚛 Waste Management - Sibiu")

    # --- Sidebar filters ---
    st.sidebar.header("Filters")
    available_routes = sorted(df["route_id"].unique())
    selected_route = st.sidebar.selectbox("Route", available_routes)
    level_min, level_max = st.sidebar.slider("Fill level (%) — range", 0, 100, (0, 100))

    df_route = df[df["route_id"] == selected_route].copy()
    route_vehicles = sorted(df_route["Car"].dropna().unique())
    df_filtered = df_route[df_route["Fill_num"].between(level_min, level_max)].copy()

    st.caption(f"Route {selected_route} is served by: {', '.join(route_vehicles)}")

    # --- Quick metrics ---
    col1, col2 = st.columns(2)
    col1.metric("Displayed containers", int(df_filtered["Id"].notna().sum()))
    col2.metric("Vehicles on route", len(route_vehicles))

    # --- Container table ---
    st.subheader("Containers on route")
    with st.expander("View data as table", expanded=False):
        table = df_filtered[df_filtered["Id"].notna()][
            ["Id", "Car", "Address", "time", "Fill_num", "Capacity"]
        ].copy()
        table = table.rename(
            columns={"Fill_num": "Fill level (%)", "time": "Time",
                     "Car": "Vehicle", "Capacity": "Capacity"}
        ).sort_values("Time")
        st.dataframe(table, width="stretch", hide_index=True)

    # -----------------------------------------------------------------
    # ROUTE DISTANCES: original vs NN-optimized
    # -----------------------------------------------------------------
    st.subheader("Route distances: original vs optimized")

    route_map = RouteMap.from_route(df_route)

    if len(route_map.points) < 2:
        st.warning("The route does not have enough points to calculate distance.")
        return

    car = route_vehicles[0] if route_vehicles else None
    analyzer = RouteAnalyzer(route_map.points).compute(car or "")
    std = analyzer.standard
    nn_results = _nn_results_cached(car) if car else {}

    gm_seq = _sequential_distance_cached(car) if car else None
    gm_nn = nn_results.get("google_maps")

    col_gm, col_std_nn, col_osrm_nn = st.columns(3)

    _cap_style = "min-height:52px;font-size:0.85em;color:gray;margin-bottom:0.5em"

    with col_gm:
        st.markdown("#### Google Maps")
        st.markdown(
            f'<div style="{_cap_style}">Precomputed matrix — actual road distances, '
            "stops in recorded order vs NN-optimized order.</div>",
            unsafe_allow_html=True,
        )
        if gm_seq is None:
            st.info("Matrix file not found. Run `python src/compute_distance_matrices.py`.")
        else:
            savings = (1 - gm_nn["optimized_km"] / gm_seq) * 100 if (gm_nn and gm_seq) else 0
            st.metric("Original (sequential)", f"{gm_seq:.2f} km")
            if gm_nn:
                st.metric(
                    "NN optimized",
                    f"{gm_nn['optimized_km']:.2f} km",
                    delta=f"-{savings:.1f}%" if savings >= 0 else f"+{abs(savings):.1f}%",
                    delta_color="inverse",
                )

    with col_std_nn:
        st.markdown("#### Standard (haversine)")
        st.markdown(
            f'<div style="{_cap_style}">Straight-line distance × 1.35 detour factor. '
            "Ignores roads and one-way streets.</div>",
            unsafe_allow_html=True,
        )
        savings_std = std["savings_%"]
        st.metric("Original", f"{std['unoptimized_km']:.2f} km")
        st.metric(
            "NN optimized",
            f"{std['optimized_km']:.2f} km",
            delta=f"-{savings_std:.1f}%" if savings_std >= 0 else f"+{abs(savings_std):.1f}%",
            delta_color="inverse",
        )

    with col_osrm_nn:
        st.markdown("#### OSRM (road network)")
        st.markdown(
            f'<div style="{_cap_style}">Real road distances via OSRM routing server.</div>',
            unsafe_allow_html=True,
        )
        if not analyzer.osrm_available:
            st.info("OSRM server unavailable.")
        else:
            osrm = analyzer.osrm
            savings_osrm = (
                (1 - osrm["optimized_km"] / osrm["unoptimized_km"]) * 100
                if osrm["unoptimized_km"] else 0
            )
            st.metric("Original", f"{osrm['unoptimized_km']:.2f} km")
            st.metric(
                "NN optimized",
                f"{osrm['optimized_km']:.2f} km",
                delta=f"-{savings_osrm:.1f}%" if savings_osrm >= 0 else f"+{abs(savings_osrm):.1f}%",
                delta_color="inverse",
            )

    # --- Side-by-side maps ---
    st.markdown("**Visual on map** (left: real order / right: NN-optimized via Google Maps)")

    # Use Google Maps NN path for the optimized map order; fall back to standard haversine.
    if gm_nn and gm_nn.get("path"):
        optimal_order = [int(name.split("_")[1]) - 1 for name in gm_nn["path"]]
    else:
        optimal_order = std["optimal_order"]
    optimized_map = route_map.reorder(optimal_order)

    col_left, col_right = st.columns(2)
    with col_left:
        if gm_seq is not None:
            unopt_label = f"{gm_seq:.2f} km · Google Maps · original order"
        else:
            unopt_label = f"{std['unoptimized_km']:.2f} km · straight-line"
        st.caption(f"Unoptimized — {unopt_label}")
        st_folium(
            route_map.draw("red", "unoptimized route"),
            height=430, width="stretch", key="map_unopt",
        )
    with col_right:
        if gm_nn:
            opt_label = f"{gm_nn['optimized_km']:.2f} km · Google Maps NN"
        else:
            opt_label = f"{std['optimized_km']:.2f} km · straight-line NN"
        st.caption(f"Optimized (NN) — {opt_label}")
        st_folium(
            optimized_map.draw("green", "optimized route (Google Maps NN)"),
            height=430, width="stretch", key="map_opt",
        )

    # -----------------------------------------------------------------
    # FILL LEVEL PREDICTION with Random Forest
    # -----------------------------------------------------------------
    st.subheader("Fill level prediction — Random Forest (decision trees)")

    mae, predictions, history = _train_predictor_cached(df)

    pred_route = predictions[predictions["route_id"] == selected_route].copy()
    route_container_ids = pred_route["Id"].dropna().unique().tolist()
    hist_route = history[history["Id"].isin(route_container_ids)].copy()

    # Pre-compute all three tables once
    rule_df = (
        pred_route[pred_route["fill_current"] >= THRESHOLD][["Id", "Capacity", "fill_current"]]
        .rename(columns={"fill_current": "Fill level today (%)"})
        .sort_values("Fill level today (%)", ascending=False)
        .reset_index(drop=True)
    )
    above_pred = (
        pred_route[pred_route["fill_predicted"] >= THRESHOLD][
            ["Id", "Capacity", "fill_current", "fill_predicted"]
        ]
        .rename(columns={
            "fill_current":   "Fill level today (%)",
            "fill_predicted": "Predicted tomorrow (%)",
        })
        .sort_values("Predicted tomorrow (%)", ascending=False)
        .reset_index(drop=True)
    )
    all_pred = (
        pred_route[["Id", "Capacity", "fill_current", "fill_predicted"]]
        .rename(columns={
            "fill_current":   "Fill level today (%)",
            "fill_predicted": "Predicted tomorrow (%)",
        })
        .sort_values("Predicted tomorrow (%)", ascending=False)
        .reset_index(drop=True)
    )

    st.write(
        f"Threshold: **{THRESHOLD}%** · {N_DAYS}-day sawtooth history · "
        f"Model: **RandomForestRegressor** · MAE: **{mae:.2f}** pp"
    )

    # Stage 1 & Stage 2 side by side
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("#### Stage 1 — Rule (today)")
        st.metric(f"Containers ≥ {THRESHOLD}%", len(rule_df))
        with st.expander("See table"):
            if rule_df.empty:
                st.info("No containers above threshold today.")
            else:
                st.dataframe(rule_df, width="stretch", hide_index=True)

    with col_s2:
        st.markdown("#### Stage 2 — Prediction (tomorrow)")
        st.metric(f"Containers predicted ≥ {THRESHOLD}%", len(above_pred))
        with st.expander("See table"):
            if above_pred.empty:
                st.info("No containers predicted above threshold tomorrow.")
            else:
                st.dataframe(above_pred, width="stretch", hide_index=True)

    with st.expander("All containers — today vs tomorrow"):
        st.dataframe(all_pred, width="stretch", hide_index=True)

    st.caption(
        f"Note: history is simulated (sawtooth pattern), not real sensor data. "
        f"Saved to `{SIMULATED_HISTORY_FILE}`."
    )

    # -----------------------------------------------------------------
    # 30-DAY FILL HISTORY PER CONTAINER
    # -----------------------------------------------------------------
    st.subheader("Fill level history — last 30 days (simulated)")

    if hist_route.empty:
        st.info("No history available for this route.")
    else:
        id_options = sorted(hist_route["Id"].dropna().unique().tolist())
        selected_id = st.selectbox(
            "Select container ID", id_options, key="hist_container_id"
        )

        hist_sel = hist_route[hist_route["Id"] == selected_id].sort_values("day").copy()
        capacity  = hist_sel["Capacity"].iloc[0]
        hist_sel["date_label"] = hist_sel["day"].apply(
            lambda d: "Today" if d == 0 else f"{-d}d ago"
        )

        line = (
            alt.Chart(hist_sel)
            .mark_line(point=alt.OverlayMarkDef(size=40))
            .encode(
                x=alt.X(
                    "day:Q",
                    title="Day (0 = today)",
                    axis=alt.Axis(labelExpr="datum.value === 0 ? 'Today' : datum.value + 'd'"),
                ),
                y=alt.Y(
                    "fill_level:Q",
                    title="Fill level (%)",
                    scale=alt.Scale(domain=[0, 105]),
                ),
                tooltip=[
                    alt.Tooltip("date_label:N", title="Day"),
                    alt.Tooltip("fill_level:Q", title="Fill level (%)", format=".1f"),
                ],
                color=alt.value("#1f77b4"),
            )
            .properties(
                title=f"Container {selected_id}  ·  {capacity}  ·  threshold {THRESHOLD}%",
                height=320,
            )
        )

        threshold_line = (
            alt.Chart(pd.DataFrame({"thr": [THRESHOLD]}))
            .mark_rule(color="red", strokeDash=[6, 3], strokeWidth=1.5)
            .encode(y="thr:Q")
        )

        threshold_label = (
            alt.Chart(pd.DataFrame({
                "thr":   [THRESHOLD],
                "day":   [-(N_DAYS - 2)],
                "label": [f"Threshold {THRESHOLD}%"],
            }))
            .mark_text(align="left", color="red", fontSize=11, dy=-6)
            .encode(x="day:Q", y="thr:Q", text="label:N")
        )

        st.altair_chart(
            (line + threshold_line + threshold_label).interactive(),
            width="stretch",
        )
