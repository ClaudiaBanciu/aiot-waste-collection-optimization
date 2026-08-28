import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


class FillLevelPredictor:
    """Fill-level prediction based on Random Forest.

    We simulate a 30-day *sawtooth* history for each container:
      - Each container fills at a fixed daily rate (chosen once per container,
        uniform within the range for its capacity class).
      - When the fill level reaches 100 % the container is emptied and the
        cycle restarts — producing the characteristic sawtooth / saw-wave
        pattern of real waste containers.
      - Day 0 is drawn uniformly in the 0-100 % range, giving each container
        an independent starting point in its own fill cycle.

    A RandomForestRegressor is then trained to predict TOMORROW's fill level
    from three features: current level, previous day's level, and the
    day-over-day growth rate. Training and scoring are split *chronologically* —
    the oldest days train the model, the most recent ones score it.

    Usage
    -----
        predictor = FillLevelPredictor(n_days=30)
        mae = predictor.train(containers_df)
        predictor.save_history("data/simulated/simulated_history.csv")
        predictions = predictor.predict_next_day()
    """

    # Plausible daily fill rate (%/day) per capacity class.
    # Rates are lower for larger containers (they serve more addresses but
    # their absolute capacity is much bigger).
    RATE: dict[str, tuple[float, float]] = {
        "120L":   (3.0,  6.0),
        "240L":   (4.0,  7.0),
        "1.100L": (6.0, 10.0),
    }

    # Collection threshold (%) per capacity class — the level above which a
    # container is considered ready for collection.
    THRESHOLDS: dict[str, int] = {
        "120L":   85,
        "240L":   80,
        "1.100L": 70,
    }

    # Share of the most recent days held out for testing. The split is
    # chronological, not random: the task is to forecast tomorrow, so the model
    # must be scored on days that come *after* everything it trained on.
    TEST_FRACTION: float = 0.2

    def __init__(self, n_days: int = 30, seed: int = 42, n_estimators: int = 200):
        self.n_days = n_days
        self.seed = seed
        self.n_estimators = n_estimators
        self.rng = np.random.default_rng(seed)
        self.model: RandomForestRegressor | None = None
        self.mae_: float | None = None
        self.history_: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Sawtooth simulated N-day history
    # ------------------------------------------------------------------
    def simulate_history(self, containers: pd.DataFrame) -> pd.DataFrame:
        """Generate a synthetic N-day sawtooth history for each container.

        Parameters
        ----------
        containers : DataFrame with columns Id, Capacity
                     (plus optional route_id, Car, Address, …).

        Returns
        -------
        DataFrame with one row per (container, day).
        Columns: key, Id, route_id, Capacity, day, fill_level.
        ``day = 0`` is today; ``day = -(n_days-1)`` is the oldest day.

        Algorithm
        ---------
        1. Draw a fixed daily rate ``r`` uniformly from the capacity range.
        2. The cycle duration is ``100 / r`` days (time to fill from 0 → 100 %).
        3. Draw today's level uniformly in [0, 100] and derive how many days
           have elapsed since the last emptying as ``level_today / r``.
        4. For each past day, compute ``position_in_cycle`` using modular
           arithmetic so the sawtooth wraps naturally.
        5. Add Gaussian noise (σ = 1.5 %) and clip to [0, 100].
        """
        c = containers.reset_index(drop=True).copy()
        c["key"] = c.index

        rows = []
        for _, r in c.iterrows():
            lo, hi = self.RATE.get(str(r["Capacity"]), (3.0, 6.0))

            # Fixed daily rate for this container (drawn once)
            rate = float(self.rng.uniform(lo, hi))
            cycle_duration = 100.0 / rate          # days for a full fill cycle

            level_today = float(self.rng.integers(0, 101))
            # Estimate phase: how many days since the last emptying at day 0
            days_since_empty_today = level_today / rate

            for day in range(-(self.n_days - 1), 1):   # -(n_days-1) … 0
                days_past = -day                         # 0 for today, >0 for past
                position = (days_since_empty_today - days_past) % cycle_duration
                noise = float(self.rng.normal(0, 1.5))
                level = min(max(rate * position + noise, 0.0), 100.0)

                rows.append({
                    "key":      int(r["key"]),
                    "Id":       r["Id"],
                    "route_id": r.get("route_id"),
                    "Capacity": r["Capacity"],
                    "day":      day,
                    "fill_level": round(level, 1),
                })

        self.history_ = pd.DataFrame(rows)
        return self.history_

    def save_history(self, path: str) -> None:
        """Save the simulated history DataFrame to *path* as a CSV.

        Intermediate directories are created automatically.
        Raises RuntimeError if called before ``train()``.
        """
        if self.history_ is None:
            raise RuntimeError("No history to save. Call train(...) first.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.history_.to_csv(path, index=False)

    @staticmethod
    def _features(history: pd.DataFrame) -> pd.DataFrame:
        """Add prev_level, rate (growth), and target (next-day level).

        Pure transformation — no instance state is needed, hence @staticmethod.
        ``history`` is sorted by (key, day) ascending before shifting so that
        shift(1) always gives the previous *calendar* day.
        """
        h = history.sort_values(["key", "day"]).copy()
        h["prev_level"] = h.groupby("key")["fill_level"].shift(1)
        h["rate"]       = h["fill_level"] - h["prev_level"]
        h["target"]     = h.groupby("key")["fill_level"].shift(-1)
        return h

    # ------------------------------------------------------------------
    # Training + evaluation
    # ------------------------------------------------------------------
    def train(self, containers: pd.DataFrame) -> float:
        """Simulate sawtooth history, train the Random Forest, return test MAE.

        The last ``TEST_FRACTION`` of the simulated days is held out; the model
        never sees them while fitting, so the score reflects forecasting a day
        that comes after all its training data — the way predict_next_day()
        actually runs.

        Parameters
        ----------
        containers : DataFrame with Id, Capacity (+ route_id).

        Returns
        -------
        float — Mean Absolute Error (percentage points) on the held-out days.

        Raises
        ------
        ValueError — if the split leaves either side empty.
        """
        history = self.simulate_history(containers)
        h       = self._features(history)
        usable  = h.dropna(subset=["prev_level", "rate", "target"])

        # Keep only the ascending phase (exclude post-emptying resets).
        # This way the model learns "how fast does a container fill?"
        # rather than "containers get emptied after they're full."
        # Result: predict fill level tomorrow *assuming no collection today*.
        usable = usable[usable["target"] >= usable["fill_level"]]

        # Chronological split: train on the older days, test on the most recent
        # ones. A random split would let the model train on day -13 and be
        # scored on day -14 — fitting the future to predict the past, which is
        # never how predict_next_day() is used. Because every day is cut at the
        # same point, the test share is only approximately TEST_FRACTION: the
        # ascending-phase filter above removes a different number of rows from
        # each day.
        cutoff     = -max(1, round(self.n_days * self.TEST_FRACTION))
        train_data = usable[usable["day"] <  cutoff]
        test_data  = usable[usable["day"] >= cutoff]

        if train_data.empty or test_data.empty:
            raise ValueError(
                f"Chronological split at day {cutoff} leaves "
                f"{len(train_data)} training and {len(test_data)} test rows. "
                f"Increase n_days (currently {self.n_days})."
            )

        features = ["fill_level", "prev_level", "rate"]

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.seed
        )
        self.model.fit(train_data[features], train_data["target"])
        self.mae_ = float(
            mean_absolute_error(
                test_data["target"], self.model.predict(test_data[features])
            )
        )
        return self.mae_

    # ------------------------------------------------------------------
    # Next-day prediction, per container
    # ------------------------------------------------------------------
    def predict_next_day(self) -> pd.DataFrame:
        """Predict tomorrow's fill level for each container from today's state.

        Uses ``day = 0`` (today) from the simulated history as the input row.

        Returns
        -------
        DataFrame with columns: key, Id, route_id, Capacity,
        fill_current (today), fill_predicted (tomorrow).

        Raises
        ------
        RuntimeError — if called before ``train()``.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train(...) first.")
        if self.history_ is None:
            raise RuntimeError("No history available. Call train(...) first.")

        h     = self._features(self.history_)
        today = h[h["day"] == 0].copy()
        X     = today[["fill_level", "prev_level", "rate"]].fillna(0)
        today["fill_predicted"] = self.model.predict(X).round(1)

        return today.rename(columns={"fill_level": "fill_current"})[
            ["key", "Id", "route_id", "Capacity", "fill_current", "fill_predicted"]
        ]
