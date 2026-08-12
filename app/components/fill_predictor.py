import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


class FillLevelPredictor:
    """Fill-level prediction based on Random Forest (decision trees).

    We only have one real fill_level measurement per container from a single
    day. A model cannot learn an evolution from a single isolated point, so we
    simulate a history of N days (7 by default) starting from the real value as
    the LAST known day. We then train a RandomForestRegressor to predict
    TOMORROW's level from (current level, previous day's level, growth rate).

    Usage:
        predictor = FillLevelPredictor(n_days=7)
        mae = predictor.train(containers_df)
        predictions = predictor.predict_next_day()
    """

    # Plausible daily fill rate (%/day) per capacity type.
    RATE: dict[str, tuple[int, int]] = {
        "120L": (8, 15),
        "240L": (10, 18),
        "1.100L": (12, 22),
    }

    def __init__(self, n_days: int = 7, seed: int = 42, n_estimators: int = 200):
        self.n_days = n_days
        self.seed = seed
        self.n_estimators = n_estimators
        self.rng = np.random.default_rng(seed)
        self.model: RandomForestRegressor | None = None
        self.mae_: float | None = None
        self.history_: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Simulated N-day history
    # ------------------------------------------------------------------
    def simulate_history(self, containers: pd.DataFrame) -> pd.DataFrame:
        """Generate a synthetic N-day history for each container.

        containers must have columns: Id, Capacity, fill_level (+ optional route_id).
        Id is not always unique, so we build our own per-row key.
        """
        c = containers.reset_index(drop=True).copy()
        c["key"] = c.index

        rows = []
        for _, r in c.iterrows():
            lo, hi = self.RATE.get(str(r["Capacity"]), (8, 15))
            series = [None] * self.n_days
            series[-1] = float(r["fill_level"])          # last day = the real value
            for day in range(self.n_days - 2, -1, -1):   # reconstruct backwards
                step = self.rng.uniform(lo, hi) * self.rng.uniform(0.7, 1.3)
                series[day] = max(0.0, series[day + 1] - step)
            for day, level in enumerate(series):
                rows.append({
                    "key": r["key"],
                    "Id": r["Id"],
                    "route_id": r.get("route_id"),
                    "Capacity": r["Capacity"],
                    "day": day,
                    "fill_level": round(min(100.0, level), 1),
                })
        self.history_ = pd.DataFrame(rows)
        return self.history_

    @staticmethod
    def _features(history: pd.DataFrame) -> pd.DataFrame:
        """Add prev_level (previous day), rate (growth) and target (next day).

        Pure transformation — no instance state is needed, hence @staticmethod.
        """
        h = history.sort_values(["key", "day"]).copy()
        h["prev_level"] = h.groupby("key")["fill_level"].shift(1)
        h["rate"] = h["fill_level"] - h["prev_level"]
        h["target"] = h.groupby("key")["fill_level"].shift(-1)
        return h

    # ------------------------------------------------------------------
    # Training + evaluation
    # ------------------------------------------------------------------
    def train(self, containers: pd.DataFrame) -> float:
        """Simulate history, train the Random Forest, and return the test MAE."""
        history = self.simulate_history(containers)
        h = self._features(history)
        train_data = h.dropna(subset=["prev_level", "rate", "target"])

        X = train_data[["fill_level", "prev_level", "rate"]]
        y = train_data["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.seed
        )
        self.model.fit(X_train, y_train)
        self.mae_ = float(mean_absolute_error(y_test, self.model.predict(X_test)))
        return self.mae_

    # ------------------------------------------------------------------
    # Next-day prediction, per container
    # ------------------------------------------------------------------
    def predict_next_day(self) -> pd.DataFrame:
        """For each container, predict tomorrow's fill level from the last
        day of the simulated history.
        Returns: key, Id, route_id, Capacity, fill_current, fill_predicted."""
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train(...) first.")
        if self.history_ is None:
            raise RuntimeError("No history available. Call train(...) first.")
        h = self._features(self.history_)
        last = h[h["day"] == self.n_days - 1].copy()
        X = last[["fill_level", "prev_level", "rate"]].fillna(0)
        last["fill_predicted"] = self.model.predict(X).round(1)
        return last.rename(columns={"fill_level": "fill_current"})[
            ["key", "Id", "route_id", "Capacity", "fill_current", "fill_predicted"]
        ]
