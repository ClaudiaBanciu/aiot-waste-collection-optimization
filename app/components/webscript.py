"""Legacy route viewer — superseded by interface.py.

interface.py is the canonical UI (Random Forest prediction, OSRM road
distances, depot-aware route maps).  This module is kept for reference
because it offers a complementary view: per-vehicle distance breakdown
and a simple linear-regression fill-level trend.

Entry point:  run_legacy(df)  — mirrors run(df) in interface.py.
"""
import datetime as dt

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from app.components.distance_calculator import StandardDistanceCalculator


# =====================================================================
# LegacyRouteViewer — per-vehicle breakdown + linear-regression trend
# =====================================================================

class LegacyRouteViewer:
    """Renders the legacy route view for a single selected route.

    This viewer provides:
      - A chronological folium map coloured by vehicle.
      - A per-vehicle unoptimized vs optimized distance table.
      - A simple linear-regression fill-level trend over time.

    Usage:
        viewer = LegacyRouteViewer(df, selected_route)
        viewer.render()
    """

    COLORS = ["blue", "red", "green", "purple", "orange", "darkred", "cadetblue"]

    def __init__(
        self,
        df: pd.DataFrame,
        selected_route: str,
        level_min: int = 0,
        level_max: int = 100,
    ):
        self.df = df
        self.selected_route = selected_route
        self.level_min = level_min
        self.level_max = level_max
        self.df_route = df[df["route_id"] == selected_route].copy()
        self.route_vehicles = sorted(self.df_route["Car"].dropna().unique())
        self._calc = StandardDistanceCalculator()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Render all sections for the selected route."""
        self._render_metrics()
        self._render_map()
        self._render_table()
        self._render_distances()
        self._render_prediction()

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _render_metrics(self) -> None:
        """Quick summary metrics at the top."""
        df_f = self._filtered_df()
        col1, col2, col3 = st.columns(3)
        col1.metric("Listed Containers", len(df_f))
        col2.metric(
            "Average Fill Level",
            f"{df_f['Fill_num'].mean():.1f}%" if len(df_f) else "-",
        )
        col3.metric("Vehicles on Route", len(self.route_vehicles))
        st.caption(
            f"Route {self.selected_route} is served by: "
            f"{', '.join(self.route_vehicles)}"
        )

    def _render_map(self) -> None:
        """Folium map — stops coloured by vehicle, in chronological order."""
        st.subheader("Map — the route, in chronological order")
        df_f = self._filtered_df()
        if df_f.empty:
            st.warning("No container matches the selected filters.")
            return

        vehicle_color = self._vehicle_color_map()
        map_ = folium.Map(
            location=[df_f["Latitude"].mean(), df_f["Longitude"].mean()],
            zoom_start=13,
        )
        for vehicle in self.route_vehicles:
            group = (
                df_f[df_f["Car"] == vehicle]
                .sort_values("Datetime")
                .reset_index(drop=True)
            )
            if group.empty:
                continue
            color = vehicle_color[vehicle]
            points = list(zip(group["Latitude"], group["Longitude"]))
            folium.PolyLine(points, color=color, weight=2, opacity=0.6).add_to(map_)
            for i, row in group.iterrows():
                self._add_stop_marker(map_, i, row, vehicle, color)

        st_folium(map_, width=1100, height=550, key="map_route_legacy")

    def _render_table(self) -> None:
        """Expandable data table with containers on route."""
        st.subheader("Containers on route")
        df_f = self._filtered_df()
        with st.expander("View data as table", expanded=True):
            table = df_f[["Id", "Car", "Address", "time", "Fill_num", "Capacity"]].copy()
            table = table.rename(columns={"Fill_num": "Fill Level (%)", "time": "Time"})
            table = table.sort_values("Time")
            st.dataframe(table, width="stretch", hide_index=True)

    def _render_distances(self) -> None:
        """Per-vehicle unoptimized vs optimized distance comparison."""
        st.subheader("Distance: unoptimized vs optimized")
        total_unopt = total_opt = 0.0

        for vehicle in self.route_vehicles:
            vehicle_df = self.df_route[self.df_route["Car"] == vehicle]
            depots = vehicle_df[vehicle_df["Id"].isna()]
            stops  = vehicle_df[vehicle_df["Id"].notna()].sort_values("Datetime")

            # Rebuild ordered sequence: depot (fixed start) → stops → dump (fixed end)
            parts: list[pd.DataFrame] = []
            if len(depots) >= 1:
                parts.append(depots.iloc[[0]])
            parts.append(stops)
            if len(depots) >= 2:
                parts.append(depots.iloc[[-1]])
            group = pd.concat(parts).reset_index(drop=True)

            if len(group) < 2:
                continue
            has_fixed_end = len(depots) >= 2
            points = list(zip(group["Latitude"], group["Longitude"]))
            dist_unopt = self._calc.route_length(points)
            order, dist_opt = self._calc.optimize(points, fixed_last=has_fixed_end)

            total_unopt += dist_unopt
            total_opt += dist_opt
            savings = (1 - dist_opt / dist_unopt) * 100 if dist_unopt > 0 else 0

            c1, c2, c3 = st.columns(3)
            c1.metric(f"{vehicle} — unoptimized", f"{dist_unopt:.2f} km")
            c2.metric(f"{vehicle} — optimized", f"{dist_opt:.2f} km")
            c3.metric(f"{vehicle} — savings", f"{savings:.1f}%")

        if total_unopt > 0:
            savings_pct = (1 - total_opt / total_unopt) * 100
            st.markdown(
                f"**Total route {self.selected_route}:** "
                f"{total_unopt:.2f} km unoptimized → {total_opt:.2f} km optimized "
                f"({savings_pct:.1f}% savings)"
            )
        st.caption(
            "Optimization uses the nearest-neighbour heuristic: starting from the "
            "first stop, always choose the closest unvisited stop. Does not guarantee "
            "the absolute optimal route, but is a simple, fast algorithm."
        )

    def _render_prediction(self) -> None:
        """Linear-regression fill-level trend over time of day."""
        st.subheader("Fill level prediction — linear regression")
        st.write(
            "The fill level tends to increase with the time of day (containers fill "
            "up during the day). Choose a time to estimate the average fill level for "
            "the route at that moment."
        )

        if len(self.df_route) < 2:
            st.info("Not enough data on this route for a prediction.")
            return

        fit_data = self.df_route[["time_numeric", "Fill_num"]].dropna()
        x = fit_data["time_numeric"].values
        y = fit_data["Fill_num"].values

        if len(set(x)) < 2:
            st.info("Not enough time variation on this route for a prediction.")
            return

        slope, intercept = np.polyfit(x, y, 1)

        chosen_time = st.slider(
            "Time for prediction (whole route)",
            min_value=dt.time(0, 0),
            max_value=dt.time(23, 50),
            value=dt.time(12, 0),
            step=dt.timedelta(minutes=10),
        )
        chosen_hour = chosen_time.hour + chosen_time.minute / 60
        predicted = float(np.clip(slope * chosen_hour + intercept, 0, 100))

        st.metric(
            f"Estimated fill level for route {self.selected_route} "
            f"at {chosen_time.strftime('%H:%M')}",
            f"{predicted:.1f}%",
        )
        st.caption(
            f"Trend: +{slope:.2f}% fill / hour "
            f"(linear regression slope for the whole route)."
        )

        chart_data = self.df_route[
            ["time_numeric", "Fill_num", "time", "Address"]
        ].copy().rename(columns={"Fill_num": "Fill Level (%)"})

        regression_line = pd.DataFrame({"time_numeric": [x.min(), x.max()]})
        regression_line["Fill Level (%)"] = (
            slope * regression_line["time_numeric"] + intercept
        )

        scatter = (
            alt.Chart(chart_data)
            .mark_circle(size=60, opacity=0.6)
            .encode(
                x=alt.X("time_numeric:Q", title="Time of day (decimal hours)"),
                y=alt.Y("Fill Level (%):Q", scale=alt.Scale(domain=[0, 100])),
                tooltip=[
                    alt.Tooltip("time:N", title="Time"),
                    alt.Tooltip("Fill Level (%):Q", title="Fill Level (%)"),
                    alt.Tooltip("Address:N", title="Address"),
                ],
            )
        )
        line = (
            alt.Chart(regression_line)
            .mark_line(color="red")
            .encode(x="time_numeric:Q", y="Fill Level (%):Q")
        )
        st.altair_chart(
            (scatter + line).properties(height=350), width="stretch"
        )
        st.caption(
            "Time on the axis is a decimal number (e.g. 7.85 = 07:51). "
            "Hover over a point to see the exact time and address."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _filtered_df(self) -> pd.DataFrame:
        """Return df_route filtered by the fill-level range set at construction."""
        return self.df_route[
            self.df_route["Fill_num"].between(self.level_min, self.level_max)
        ].copy()

    def _vehicle_color_map(self) -> dict[str, str]:
        """Map each vehicle to a CSS color string."""
        return {
            v: self.COLORS[i % len(self.COLORS)]
            for i, v in enumerate(self.route_vehicles)
        }

    @staticmethod
    def _add_stop_marker(
        map_: folium.Map,
        index: int,
        row: pd.Series,
        vehicle: str,
        color: str,
    ) -> None:
        """Add a circle marker + sequence number to the folium map."""
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=(
                f"#{index + 1} — {row['Address']}<br>"
                f"Id: {row['Id']}<br>Vehicle: {vehicle}<br>"
                f"Time: {row['time']}<br>Fill level: {row['Fill_num']}%"
            ),
        ).add_to(map_)
        folium.map.Marker(
            [row["Latitude"], row["Longitude"]],
            icon=folium.DivIcon(
                html=(
                    f'<div style="font-size:9px;color:white;font-weight:bold;'
                    f'transform:translate(6px,-6px);">{index + 1}</div>'
                )
            ),
        ).add_to(map_)


# =====================================================================
# Entry point
# =====================================================================

def run_legacy(df: pd.DataFrame) -> None:
    """Render the legacy route view.

    Intended to be called from main.py as an alternative to run().
    Adds the sidebar controls itself and delegates rendering to
    LegacyRouteViewer.
    """
    st.title("🚛 Waste Management - Sibiu (legacy view)")

    st.sidebar.header("Filters")
    available_routes = sorted(df["route_id"].unique())
    selected_route = st.sidebar.selectbox("Route", available_routes)

    level_min, level_max = st.sidebar.slider(
        "Fill level (%) — range", 0, 100, (0, 100)
    )

    LegacyRouteViewer(df, selected_route, level_min, level_max).render()
