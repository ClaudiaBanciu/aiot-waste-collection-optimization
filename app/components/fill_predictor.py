
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


class FillLevelPredictor:
    """Fill-level prediction based on decision trees (Random Forest).

    We only have one real fill_level measurement per container, from a single
    day. A model cannot learn an evolution from a single isolated point, so we
    simulate a history of N days (7 by default) starting from the real value as
    the LAST known day. We then train a RandomForestRegressor to predict
    TOMORROW's level from (current level, previous day's level, growth rate).
    """

    # plausible daily fill rate (%/day) per capacity type: large commercial
    # containers fill faster than small residential ones.
    RATE = {"120L": (8, 15), "240L": (10, 18), "1.100L": (12, 22)}

    def __init__(self, n_zile=7, seed=42, n_estimators=200):
        self.n_zile = n_zile
        self.seed = seed
        self.n_estimators = n_estimators
        self.rng = np.random.default_rng(seed)
        self.model = None
        self.mae_ = None
        self.istoric_ = None

    # ------------------------------------------------------------------
    # simulated N-day history
    # ------------------------------------------------------------------
    def simuleaza_istoric(self, containere: pd.DataFrame) -> pd.DataFrame:
        """containere: DataFrame with Id, Capacity, fill_level (+ optional route_id).
        Id is not always unique, so we use our own key, unique per row."""
        c = containere.reset_index(drop=True).copy()
        c["cheie"] = c.index

        randuri = []
        for _, r in c.iterrows():
            lo, hi = self.RATE.get(str(r["Capacity"]), (8, 15))
            serie = [None] * self.n_zile
            serie[-1] = float(r["fill_level"])                 # last day = the real value
            for zi in range(self.n_zile - 2, -1, -1):          # go backwards in time
                pas = self.rng.uniform(lo, hi) * self.rng.uniform(0.7, 1.3)
                serie[zi] = max(0.0, serie[zi + 1] - pas)
            for zi, niv in enumerate(serie):
                randuri.append({
                    "cheie": r["cheie"], "Id": r["Id"],
                    "route_id": r.get("route_id"), "Capacity": r["Capacity"],
                    "zi": zi, "fill_level": round(min(100.0, niv), 1),
                })
        self.istoric_ = pd.DataFrame(randuri)
        return self.istoric_

    def _features(self, istoric: pd.DataFrame) -> pd.DataFrame:
        """Add nivel_ant (previous day), rata (growth) and tinta (next day)."""
        h = istoric.sort_values(["cheie", "zi"]).copy()
        h["nivel_ant"] = h.groupby("cheie")["fill_level"].shift(1)
        h["rata"] = h["fill_level"] - h["nivel_ant"]
        h["tinta"] = h.groupby("cheie")["fill_level"].shift(-1)
        return h

    # ------------------------------------------------------------------
    # training + evaluation
    # ------------------------------------------------------------------
    def antreneaza(self, containere: pd.DataFrame) -> float:
        """Simulate the history, train the Random Forest and return the test MAE."""
        istoric = self.simuleaza_istoric(containere)
        h = self._features(istoric)
        train = h.dropna(subset=["nivel_ant", "rata", "tinta"])  # drop the first/last day

        X = train[["fill_level", "nivel_ant", "rata"]]
        y = train["tinta"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=self.seed)

        self.model = RandomForestRegressor(n_estimators=self.n_estimators,
                                           random_state=self.seed)
        self.model.fit(Xtr, ytr)
        self.mae_ = float(mean_absolute_error(yte, self.model.predict(Xte)))
        return self.mae_

    # ------------------------------------------------------------------
    # next-day prediction, per container
    # ------------------------------------------------------------------
    def prezice_ziua_urmatoare(self) -> pd.DataFrame:
        """For each container: take the last day of history and predict the next day.
        Returns the columns cheie, Id, route_id, Capacity, fill_curent, fill_prezis."""
        if self.model is None:
            raise RuntimeError("Model is not trained. Call antreneaza(...) first.")
        h = self._features(self.istoric_)
        ultima = h[h["zi"] == self.n_zile - 1].copy()
        X = ultima[["fill_level", "nivel_ant", "rata"]].fillna(0)
        ultima["fill_prezis"] = self.model.predict(X).round(1)
        return ultima.rename(columns={"fill_level": "fill_curent"})[
            ["cheie", "Id", "route_id", "Capacity", "fill_curent", "fill_prezis"]]
