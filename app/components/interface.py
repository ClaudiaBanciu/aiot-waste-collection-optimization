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
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.components.distance_calculator import (
    OSRMDistanceCalculator,
    StandardDistanceCalculator,
)
from app.components.fill_predictor import FillLevelPredictor
from config import THRESHOLD


# =====================================================================
# Streamlit-cached helpers (module-level for decorator compatibility)
# =====================================================================

@st.cache_data(show_spinner="Computing distances (standard + OSRM on road)...")
def _compare_distances_cached(points_tuple: tuple) -> dict:
    points = list(points_tuple)
    standard = StandardDistanceCalculator().compare(points)
    osrm_calc = OSRMDistanceCalculator()
    osrm_ok = osrm_calc.available()
    osrm = osrm_calc.compare(points) if osrm_ok else None
    return {"standard": standard, "osrm": osrm, "osrm_ok": osrm_ok}


@st.cache_resource(show_spinner=False)
def _train_predictor_cached(_df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Train the Random Forest once on all containers.
    Leading underscore tells Streamlit not to hash the DataFrame."""
    containers = _df[
        _df["Id"].notna() & _df["Capacity"].notna() & _df["Fill_num"].notna()
    ][["Id", "Capacity", "route_id"]].copy()
    containers["fill_level"] = _df.loc[containers.index, "Fill_num"].astype(float)
    predictor = FillLevelPredictor(n_days=7)
    predictor.train(containers)
    predictions = predictor.predict_next_day()
    return predictor.mae_, predictions


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
        """Factory: extract ordered (depot → stops) points from a route DataFrame."""
        depots = df_route[df_route["Id"].isna()]
        stops = df_route[df_route["Id"].notna()].sort_values("Datetime")
        parts = []
        if len(depots):
            parts.append(depots.iloc[[0]])   # departure depot only (no return)
        parts.append(stops)
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

        stop_order = 0
        for i, (lat, lon) in enumerate(self.points):
            row = self.ordered.iloc[i]
            if pd.isna(row["Id"]):
                details = self._point_details(row, popup_title, None)
                icon = folium.Icon(color="gray", icon="home", prefix="fa")
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
    def _point_details(row: pd.Series, title: str, stop_order: int | None) -> str:
        """HTML tooltip/popup content for a single stop or depot."""
        if pd.isna(row["Id"]):
            return f"<b>Depot</b><br>{row['Address']}<br><i>{title}</i>"
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

    def compute(self) -> "RouteAnalyzer":
        """Run both distance methods and cache the result. Returns self."""
        self._result = _compare_distances_cached(tuple(self.points))
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
    col1, col2, col3 = st.columns(3)
    col1.metric("Displayed containers", int(df_filtered["Id"].notna().sum()))
    col2.metric(
        "Average fill level",
        f"{df_filtered['Fill_num'].mean():.1f}%"
        if df_filtered["Fill_num"].notna().any()
        else "-",
    )
    col3.metric("Vehicles on route", len(route_vehicles))

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
        st.dataframe(table, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------
    # ROUTE COMPARISON: unoptimized vs optimized
    # -----------------------------------------------------------------
    st.subheader("Route comparison: unoptimized vs optimized")

    route_map = RouteMap.from_route(df_route)

    if len(route_map.points) < 2:
        st.warning("The route does not have enough points to calculate distance.")
        return

    analyzer = RouteAnalyzer(route_map.points).compute()
    std = analyzer.standard

    # --- Standard distance (straight line) ---
    st.markdown("**Standard distance (straight line):**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Unoptimized", f"{std['unoptimized_km']:.2f} km")
    c2.metric("Optimized", f"{std['optimized_km']:.2f} km")
    c3.metric("Savings", f"{std['savings_%']:.1f}%")

    chart_rows = [
        {"method": "straight line", "route": "unoptimized", "km": std["unoptimized_km"]},
        {"method": "straight line", "route": "optimized",   "km": std["optimized_km"]},
    ]

    # --- OSRM road distance ---
    if not analyzer.osrm_available:
        st.info(
            "The public OSRM server is not responding — only the standard distance "
            "is shown. Road distance will appear when the server is reachable."
        )
    else:
        osrm = analyzer.osrm
        st.markdown("**Road distance (OSRM):**")
        o1, o2, o3 = st.columns(3)
        o1.metric("Unoptimized", f"{osrm['unoptimized_km']:.2f} km")
        o2.metric("Optimized",   f"{osrm['optimized_km']:.2f} km")
        factor = (
            osrm["unoptimized_km"] / std["unoptimized_km"]
            if std["unoptimized_km"] else 0
        )
        o3.metric("Road / straight-line factor", f"{factor:.2f}x")
        chart_rows += [
            {"method": "on road (OSRM)", "route": "unoptimized", "km": osrm["unoptimized_km"]},
            {"method": "on road (OSRM)", "route": "optimized",   "km": osrm["optimized_km"]},
        ]

    chart = (
        alt.Chart(pd.DataFrame(chart_rows))
        .mark_bar()
        .encode(
            x=alt.X("route:N", title=None),
            y=alt.Y("km:Q", title="Distance (km)"),
            color=alt.Color("route:N", legend=None),
            column=alt.Column("method:N", title=None),
            tooltip=["method", "route", alt.Tooltip("km:Q", format=".2f")],
        )
        .properties(width=180, height=280)
    )
    st.altair_chart(chart, use_container_width=False)

    if analyzer.osrm_available:
        st.caption(
            "Road distance (OSRM) is greater than straight-line because roads are not "
            "straight. Note: nearest-neighbour optimization minimizes the straight-line "
            "distance — measured on the road, the optimized order is not always shorter."
        )

    # --- Side-by-side maps ---
    st.markdown("**Visual on map** (left: real order / right: optimized order)")

    optimal_order = std["optimal_order"]
    optimized_map = route_map.reorder(optimal_order)

    geom_unopt = geom_opt = None
    if analyzer.osrm_available and analyzer.osrm.get("geom_unopt"):
        geom_unopt = analyzer.osrm["geom_unopt"]
        geom_opt   = analyzer.osrm["geom_opt"]

    col_left, col_right = st.columns(2)
    with col_left:
        label = (
            f"{analyzer.osrm['unoptimized_km']:.2f} km on road"
            if geom_unopt else f"{std['unoptimized_km']:.2f} km straight line"
        )
        st.caption(f"Unoptimized — {label}")
        st_folium(
            route_map.draw("red", "unoptimized route", geom_unopt),
            height=430, use_container_width=True, key="map_unopt",
        )
    with col_right:
        label = (
            f"{analyzer.osrm['optimized_km']:.2f} km on road"
            if geom_opt else f"{std['optimized_km']:.2f} km straight line"
        )
        st.caption(f"Optimized (nearest-neighbour) — {label}")
        st_folium(
            optimized_map.draw("green", "optimized route", geom_opt),
            height=430, use_container_width=True, key="map_opt",
        )

    st.caption(
        "Optimization uses the nearest-neighbour heuristic: starting from the depot, "
        "always choose the closest unvisited stop. Does not guarantee the absolute "
        "optimal route, but is fast and widely used as a starting point."
    )

    # -----------------------------------------------------------------
    # FILL LEVEL PREDICTION with Random Forest
    # -----------------------------------------------------------------
    st.subheader("Fill level prediction — Random Forest (decision trees)")

    mae, predictions = _train_predictor_cached(df)
    st.write(
        "The model predicts **tomorrow's** fill level for each container, based on "
        "the current level, the previous day's level, and the growth rate "
        "(simulated 7-day history). Model: **RandomForestRegressor**."
    )
    st.metric("Mean Absolute Error (MAE) on test set", f"{mae:.2f} percentage points")

    pred_route = predictions[predictions["route_id"] == selected_route].copy()
    count_today    = int((pred_route["fill_current"]   >= THRESHOLD).sum())
    count_tomorrow = int((pred_route["fill_predicted"] >= THRESHOLD).sum())

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric(f"Containers >= {THRESHOLD}% TODAY (rule)", count_today)
    cc2.metric(f"Containers >= {THRESHOLD}% TOMORROW (prediction)", count_tomorrow)
    cc3.metric(
        "Extra vs rule", count_tomorrow - count_today,
        help="Containers below threshold today but predicted to exceed it tomorrow.",
    )

    with st.expander("View predictions for this route", expanded=False):
        display = pred_route.rename(columns={
            "fill_current":   "Fill level today (%)",
            "fill_predicted": "Predicted fill tomorrow (%)",
        })
        st.dataframe(
            display[["Id", "Capacity", "Fill level today (%)", "Predicted fill tomorrow (%)"]],
            use_container_width=True, hide_index=True,
        )

    st.caption(
        "Difference from a fixed rule: the model catches containers that are below "
        "the threshold TODAY but WILL exceed it tomorrow — so they can be collected "
        "proactively in a single pass. Note: the history is simulated, not real."
    )
