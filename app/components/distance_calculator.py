
import os

import numpy as np
import requests


class DistanceCalculator:
    """Interfata comuna. Toate metodele lucreaza cu puncte = [(lat, lon), ...]."""

    metoda = "necunoscuta"

   
    def matrice(self, puncte):
        raise NotImplementedError

   
    def lungime_traseu(self, puncte, matrice=None):
        """Suma distantelor consecutive (km), in ordinea DATA a punctelor."""
        m = matrice if matrice is not None else self.matrice(puncte)
        return sum(m[i][i + 1] for i in range(len(puncte) - 1))

    def optimizeaza(self, puncte, matrice=None):
        
        n = len(puncte)
        if n < 2:
            return list(range(n)), 0.0
        m = matrice if matrice is not None else self.matrice(puncte)
        ramase = set(range(1, n))
        ordine, curent, total = [0], 0, 0.0
        while ramase:
            urm = min(ramase, key=lambda j: m[curent][j])
            total += m[curent][urm]
            ordine.append(urm)
            ramase.discard(urm)
            curent = urm
        return ordine, total

    def compara(self, puncte):
        n = len(puncte)
        if n < 2:
            return {"metoda": self.metoda, "neoptimizat_km": 0.0, "optimizat_km": 0.0,
                    "economie_%": 0.0, "ordine_optima": list(range(n))}
        m = self.matrice(puncte)
        neopt = self.lungime_traseu(puncte, m)
        ordine, opt = self.optimizeaza(puncte, m)
        economie = (1 - opt / neopt) * 100 if neopt else 0.0
        return {"metoda": self.metoda, "neoptimizat_km": neopt, "optimizat_km": opt,
                "economie_%": economie, "ordine_optima": ordine}


class StandardDistanceCalculator(DistanceCalculator):
    
    metoda = "standard"
    R = 6371.0  # raza Pamantului (km)

    def matrice(self, puncte):
        pts = np.radians(np.asarray(puncte, dtype=float))
        lat = pts[:, 0][:, None]
        lon = pts[:, 1][:, None]
        dlat = lat - lat.T
        dlon = lon - lon.T
        a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
        return (2 * self.R * np.arcsin(np.sqrt(a))).tolist()


class OSRMDistanceCalculator(DistanceCalculator):
    """Distanta reala pe reteaua rutiera, calculata prin OSRM (serverul public
    router.project-osrm.org), fara Docker si fara instalare.

    Foloseste serviciul Route (nu Table): dintr-o cerere obtinem lungimea pe
    sosea a unui traseu care trece prin punctele date, in ordinea data. Pentru
    trasee lungi impartim in bucati (chunk-uri) consecutive care se suprapun
    intr-un punct, ca sa nu depasim lungimea maxima a unui URL.

    Ordinea optimizata e calculata geometric (nearest-neighbour pe distante in
    linie dreapta, rapid si local), iar apoi ii masuram lungimea REALA pe sosea.
    Daca serverul OSRM nu raspunde, totul cade pe calculatorul standard.
    """

    metoda = "osrm"

    def __init__(self, url=None, fallback=None, chunk=90):
        self.url = (url or os.environ.get("OSRM_URL", "https://router.project-osrm.org")).rstrip("/")
        self.fallback = fallback or StandardDistanceCalculator()
        self.chunk = chunk
        self._disponibil = None

    def disponibil(self) -> bool:
        """Verifica o singura data daca serverul OSRM raspunde."""
        if self._disponibil is None:
            try:
                r = requests.get(f"{self.url}/route/v1/driving/24.15,45.79;24.16,45.80"
                                 "?overview=false", timeout=6)
                self._disponibil = r.status_code == 200 and r.json().get("code") == "Ok"
            except Exception:
                self._disponibil = False
        return self._disponibil

    def _segment(self, puncte):
        """Printr-o singura cerere Route: (lungimea pe sosea in km, geometria drumului).
        Cerem overview=full + geometries=geojson ca sa primim conturul real pe strazi."""
        coord = ";".join(f"{lon},{lat}" for lat, lon in puncte)
        url = f"{self.url}/route/v1/driving/{coord}?overview=full&geometries=geojson"
        r = requests.get(url, timeout=40)
        r.raise_for_status()
        ruta = r.json()["routes"][0]
        distanta = ruta["distance"] / 1000
        # GeoJSON da coordonate [lon, lat] -> le intoarcem ca (lat, lon) pentru folium
        geometrie = [(lat, lon) for lon, lat in ruta["geometry"]["coordinates"]]
        return distanta, geometrie

    def traseu_pe_sosea(self, puncte):
        """Traseul complet pe sosea in ordinea DATA, impartit in chunk-uri consecutive
        care se suprapun intr-un punct. Returneaza (distanta_totala_km, geometrie)."""
        total, geometrie, pas = 0.0, [], self.chunk - 1
        for start in range(0, len(puncte) - 1, pas):
            bucata = puncte[start:start + self.chunk]
            if len(bucata) >= 2:
                dist, geom = self._segment(bucata)
                total += dist
                geometrie.extend(geom)
        return total, geometrie

    def compara(self, puncte):
        n = len(puncte)
        if n < 2:
            return {"metoda": self.metoda, "neoptimizat_km": 0.0, "optimizat_km": 0.0,
                    "economie_%": 0.0, "ordine_optima": list(range(n)),
                    "geom_neopt": [], "geom_opt": []}
        # ordinea optima o gasim geometric (rapid, local)
        ordine, _ = self.fallback.optimizeaza(puncte)
        try:
            neopt, geom_neopt = self.traseu_pe_sosea(puncte)
            opt, geom_opt = self.traseu_pe_sosea([puncte[i] for i in ordine])
            self.metoda = "osrm"
        except Exception:
            # orice problema la OSRM -> cadem pe distanta standard (fara geometrie)
            rez = self.fallback.compara(puncte)
            rez["metoda"] = "standard (fallback)"
            rez["geom_neopt"] = rez["geom_opt"] = []
            return rez
        economie = (1 - opt / neopt) * 100 if neopt else 0.0
        return {"metoda": self.metoda, "neoptimizat_km": neopt, "optimizat_km": opt,
                "economie_%": economie, "ordine_optima": ordine,
                "geom_neopt": geom_neopt, "geom_opt": geom_opt}
