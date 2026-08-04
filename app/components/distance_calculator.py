
import os

import numpy as np
import requests


class DistanceCalculator:
    """Common interface. All methods work with points = [(lat, lon), ...]."""

    metoda = "necunoscuta"

    def matrice(self, puncte):
        raise NotImplementedError

    def lungime_traseu(self, puncte, matrice=None):
        """Sum of consecutive distances (km), in the GIVEN order of the points."""
        m = matrice if matrice is not None else self.matrice(puncte)
        return sum(m[i][i + 1] for i in range(len(puncte) - 1))

    def optimizeaza(self, puncte, matrice=None):
        # Nearest-neighbour: start from the first point and always move to the
        # closest not-yet-visited point. Returns (order_indices, distance_km).
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
        # Build the matrix ONCE and return the non-optimized distance, the
        # optimized distance, the saving (%) and the optimal order of points.
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
    """Standard, straight-line distance (haversine)."""

    metoda = "standard"
    R = 6371.0  # Earth's radius (km)

    def matrice(self, puncte):
        pts = np.radians(np.asarray(puncte, dtype=float))
        lat = pts[:, 0][:, None]
        lon = pts[:, 1][:, None]
        dlat = lat - lat.T
        dlon = lon - lon.T
        a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
        return (2 * self.R * np.arcsin(np.sqrt(a))).tolist()


class OSRMDistanceCalculator(DistanceCalculator):
    """Real road-network distance, computed via OSRM (the public server
    router.project-osrm.org), with no Docker and no installation.

    Uses the Route service (not Table): a single request gives the road length
    of a trip passing through the given points, in the given order. For long
    trips we split into consecutive overlapping chunks (sharing one point) so we
    don't exceed the maximum URL length.

    The optimized order is computed geometrically (nearest-neighbour on
    straight-line distances, fast and local), then we measure its REAL road
    length. If the OSRM server does not respond, everything falls back to the
    standard calculator.
    """

    metoda = "osrm"

    def __init__(self, url=None, fallback=None, chunk=90):
        self.url = (url or os.environ.get("OSRM_URL", "https://router.project-osrm.org")).rstrip("/")
        self.fallback = fallback or StandardDistanceCalculator()
        self.chunk = chunk
        self._disponibil = None

    def disponibil(self) -> bool:
        """Check only once whether the OSRM server responds."""
        if self._disponibil is None:
            try:
                r = requests.get(f"{self.url}/route/v1/driving/24.15,45.79;24.16,45.80"
                                 "?overview=false", timeout=6)
                self._disponibil = r.status_code == 200 and r.json().get("code") == "Ok"
            except Exception:
                self._disponibil = False
        return self._disponibil

    def _segment(self, puncte):
        """With a single Route request: (road length in km, road geometry).
        We ask for overview=full + geometries=geojson to get the real
        street-following outline."""
        coord = ";".join(f"{lon},{lat}" for lat, lon in puncte)
        url = f"{self.url}/route/v1/driving/{coord}?overview=full&geometries=geojson"
        r = requests.get(url, timeout=40)
        r.raise_for_status()
        ruta = r.json()["routes"][0]
        distanta = ruta["distance"] / 1000
        # GeoJSON gives [lon, lat] coordinates -> we return them as (lat, lon) for folium
        geometrie = [(lat, lon) for lon, lat in ruta["geometry"]["coordinates"]]
        return distanta, geometrie

    def traseu_pe_sosea(self, puncte):
        """Full road trip in the GIVEN order, split into consecutive chunks that
        overlap by one point. Returns (total_distance_km, geometry)."""
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
        # find the optimal order geometrically (fast, local)
        ordine, _ = self.fallback.optimizeaza(puncte)
        try:
            neopt, geom_neopt = self.traseu_pe_sosea(puncte)
            opt, geom_opt = self.traseu_pe_sosea([puncte[i] for i in ordine])
            self.metoda = "osrm"
        except Exception:
            # any OSRM problem -> fall back to the standard distance (no geometry)
            rez = self.fallback.compara(puncte)
            rez["metoda"] = "standard (fallback)"
            rez["geom_neopt"] = rez["geom_opt"] = []
            return rez
        economie = (1 - opt / neopt) * 100 if neopt else 0.0
        return {"metoda": self.metoda, "neoptimizat_km": neopt, "optimizat_km": opt,
                "economie_%": economie, "ordine_optima": ordine,
                "geom_neopt": geom_neopt, "geom_opt": geom_opt}
