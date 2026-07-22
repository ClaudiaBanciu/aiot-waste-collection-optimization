
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


class FillLevelPredictor:
   

    
    RATE = {"120L": (8, 15), "240L": (10, 18), "1.100L": (12, 22)}

    def __init__(self, n_zile=7, seed=42, n_estimators=200):
        self.n_zile = n_zile
        self.seed = seed
        self.n_estimators = n_estimators
        self.rng = np.random.default_rng(seed)
        self.model = None
        self.mae_ = None
        self.istoric_ = None

    
    def simuleaza_istoric(self, containere: pd.DataFrame) -> pd.DataFrame:
        """containere: DataFrame cu Id, Capacity, fill_level (+ route_id optional).
        Id-ul nu e mereu unic, asa ca folosim o cheie proprie, unica pe rand."""
        c = containere.reset_index(drop=True).copy()
        c["cheie"] = c.index

        randuri = []
        for _, r in c.iterrows():
            lo, hi = self.RATE.get(str(r["Capacity"]), (8, 15))
            serie = [None] * self.n_zile
            serie[-1] = float(r["fill_level"])                 
            for zi in range(self.n_zile - 2, -1, -1):         
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
        """Adauga nivel_ant (ziua anterioara), rata (crestere) si tinta (ziua urmatoare)."""
        h = istoric.sort_values(["cheie", "zi"]).copy()
        h["nivel_ant"] = h.groupby("cheie")["fill_level"].shift(1)
        h["rata"] = h["fill_level"] - h["nivel_ant"]
        h["tinta"] = h.groupby("cheie")["fill_level"].shift(-1)
        return h

    # ------------------------------------------------------------------
    # Etapa 2: antrenare + evaluare
    # ------------------------------------------------------------------
    def antreneaza(self, containere: pd.DataFrame) -> float:
        """Simuleaza istoricul, antreneaza Random Forest si intoarce MAE pe test."""
        istoric = self.simuleaza_istoric(containere)
        h = self._features(istoric)
        train = h.dropna(subset=["nivel_ant", "rata", "tinta"])  # scoatem prima/ultima zi

        X = train[["fill_level", "nivel_ant", "rata"]]
        y = train["tinta"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=self.seed)

        self.model = RandomForestRegressor(n_estimators=self.n_estimators,
                                           random_state=self.seed)
        self.model.fit(Xtr, ytr)
        self.mae_ = float(mean_absolute_error(yte, self.model.predict(Xte)))
        return self.mae_

    
    def prezice_ziua_urmatoare(self) -> pd.DataFrame:
        """Pentru fiecare container: ia ultima zi din istoric si prezice ziua urmatoare.
        Returneaza coloanele cheie, Id, route_id, Capacity, fill_curent, fill_prezis."""
        if self.model is None:
            raise RuntimeError("Modelul nu e antrenat. Apeleaza intai antreneaza(...).")
        h = self._features(self.istoric_)
        ultima = h[h["zi"] == self.n_zile - 1].copy()
        X = ultima[["fill_level", "nivel_ant", "rata"]].fillna(0)
        ultima["fill_prezis"] = self.model.predict(X).round(1)
        return ultima.rename(columns={"fill_level": "fill_curent"})[
            ["cheie", "Id", "route_id", "Capacity", "fill_curent", "fill_prezis"]]
