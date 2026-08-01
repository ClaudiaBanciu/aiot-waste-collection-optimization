"""
fill_level_model.py — Logica de simulare istoric + model predictiv de
fill_level, extrasă din fill_level_prediction_draft.ipynb (Etapele 2-6),
reutilizabilă și din interfața Streamlit (app.py).

Notă: notebook-ul rămâne independent (nu trebuie neapărat să importe de
aici — cerința e ca notebook-ul să nu ajungă pe Git oricum), dar acest
modul reproduce exact aceeași logică, ca rezultatele din aplicație să fie
consistente cu cele din notebook.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

THRESHOLDS = {
    "120L": 85,
    "240L": 80,
    "1.100L": 70,
}

DAILY_RATES = {
    "120L": (3, 6),
    "240L": (4, 7),
    "1.100L": (6, 10),
}

FEATURES = ["simulated_fill_level", "previous_day_level", "growth_rate"]
TARGET = "next_day_level"

N_DAYS = 30


def prepare_containers(df):
    """Reproduce Etapa 2: marchează punctele structurale (depozit
    plecare/sosire = primul/ultimul rând al fiecărei rute) și returnează
    doar subsetul de containere reale, cu o cheie unică 'uid' și pragul
    de colectare pe capacitate."""
    df = df.copy()
    df["point_type"] = "container"

    for route_id, grup in df.groupby("route_id", sort=False):
        df.loc[grup.index[0], "point_type"] = "depot_departure"
        df.loc[grup.index[-1], "point_type"] = "depot_arrival"

    containers = df[df["point_type"] == "container"].copy()
    containers = containers.dropna(subset=["Id", "Capacity", "fill_level"])
    containers = containers.reset_index(drop=True)
    containers["uid"] = containers.index
    containers["threshold"] = containers["Capacity"].map(THRESHOLDS)

    return containers


def simulate_history(containers, n_days=N_DAYS, seed=42):
    """Reproduce Etapa 4: istoric simulat de n_days zile per container.

    Model: un container se umple cu o rată zilnică ~constantă (specifică
    lui, aleasă o singură dată, nu reales în fiecare zi) și e golit
    periodic (colectare) — deci istoricul e un tipar de tip 'fierăstrău'
    (crește constant, revine la ~0 la fiecare ciclu de golire), nu o
    linie care scade și rămâne blocată la 0 timp de zeci de zile (ceea ce
    nu ar fi realist pentru un container dintr-un oraș).

    Ancorăm ziua 0 la valoarea reală măsurată: calculăm de câte zile e
    "de la ultima golire" la ziua 0 (nivel_azi / rată), apoi reconstruim
    tiparul de fierăstrău înapoi în timp, păstrând acel punct de ancoraj.
    """
    np.random.seed(seed)
    randuri = []

    for _, rand in containers.iterrows():
        capacitate = rand["Capacity"]
        rata_min, rata_max = DAILY_RATES[capacitate]

        nivel_azi = rand["fill_level"]
        rata = np.random.uniform(rata_min, rata_max)  # rată fixă, per container
        durata_ciclu = 100 / rata  # zile ca să se umple complet, de la 0 la 100%

        zile_de_la_ultima_golire_azi = nivel_azi / rata

        for zi in range(0, -n_days, -1):
            zile_trecute = -zi  # câte zile în urmă e față de azi
            pozitie_in_ciclu = (zile_de_la_ultima_golire_azi - zile_trecute) % durata_ciclu
            zgomot = np.random.normal(0, 1.5)
            nivel = min(max(rata * pozitie_in_ciclu + zgomot, 0), 100)

            randuri.append({
                "uid": rand["uid"],
                "Id": rand["Id"],
                "route_id": rand["route_id"],
                "Car": rand["Car"],
                "Capacity": capacitate,
                "day": zi,
                "simulated_fill_level": nivel,
            })

    return pd.DataFrame(randuri)


def train_model(history):
    """Reproduce Etapa 5: construiește features/target, împarte
    train/test, antrenează regresia liniară, calculează MAE."""
    sorted_history = history.sort_values(["uid", "day"]).reset_index(drop=True)

    sorted_history["previous_day_level"] = sorted_history.groupby("uid")["simulated_fill_level"].shift(1)
    sorted_history["next_day_level"] = sorted_history.groupby("uid")["simulated_fill_level"].shift(-1)
    sorted_history["growth_rate"] = (
        sorted_history["simulated_fill_level"] - sorted_history["previous_day_level"]
    )

    model_data = sorted_history.dropna(subset=["previous_day_level", "next_day_level", "growth_rate"])

    X = model_data[FEATURES]
    y = model_data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))

    return model, mae, sorted_history


def predict_current_state(containers, sorted_history, model):
    """Reproduce Etapa 6: starea curentă (ziua 0 + ziua -1) per
    container, cu predicția modelului pentru mâine."""
    ziua_0 = sorted_history[sorted_history["day"] == 0][["uid", "Id", "simulated_fill_level"]]
    ziua_minus1 = sorted_history[sorted_history["day"] == -1][["uid", "simulated_fill_level"]]
    ziua_minus1 = ziua_minus1.rename(columns={"simulated_fill_level": "previous_day_level"})

    stare = ziua_0.merge(ziua_minus1, on="uid", how="inner")
    stare["growth_rate"] = stare["simulated_fill_level"] - stare["previous_day_level"]

    stare["predicted_fill_level"] = model.predict(stare[FEATURES]).clip(0, 100)

    stare = stare.merge(
        containers[["uid", "Capacity", "threshold", "Address", "route_id", "Car",
                    "Latitude", "Longitude"]],
        on="uid", how="left"
    )
    return stare


def select_by_rule(containers):
    return containers[containers["fill_level"] >= containers["threshold"]].copy()


def select_by_prediction(current_state):
    return current_state[current_state["predicted_fill_level"] >= current_state["threshold"]].copy()


def run_full_pipeline(df, n_days=N_DAYS, seed=42):
    """Rulează tot fluxul dintr-o dată (util pentru cache în Streamlit):
    returnează containers, history, model, mae, current_state,
    rule_collection, predictive_collection."""
    containers = prepare_containers(df)
    history = simulate_history(containers, n_days=n_days, seed=seed)
    model, mae, sorted_history = train_model(history)
    current_state = predict_current_state(containers, sorted_history, model)
    rule_collection = select_by_rule(containers)
    predictive_collection = select_by_prediction(current_state)

    return {
        "containers": containers,
        "history": history,
        "sorted_history": sorted_history,
        "model": model,
        "mae": mae,
        "current_state": current_state,
        "rule_collection": rule_collection,
        "predictive_collection": predictive_collection,
    }