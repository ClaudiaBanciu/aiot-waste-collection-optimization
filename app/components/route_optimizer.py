from geopy.distance import geodesic

def distanta_traseu(puncte):
    """Suma distanțelor consecutive (km), în ordinea dată a punctelor."""
    total = 0.0
    for i in range(len(puncte) - 1):
        total += geodesic(puncte[i], puncte[i + 1]).km
    return total

def optimizeaza_nearest_neighbor(puncte):
    """Euristică greedy 'cel mai apropiat vecin'."""
    if len(puncte) < 2:
        return puncte, 0.0

    ramase = list(range(1, len(puncte)))
    ordine = [0]
    curent = 0
    total = 0.0

    while ramase:
        distante = [(j, geodesic(puncte[curent], puncte[j]).km) for j in ramase]
        urmator, dist = min(distante, key=lambda t: t[1])
        total += dist
        ordine.append(urmator)
        ramase.remove(urmator)
        curent = urmator

    traseu_optimizat = [puncte[i] for i in ordine]
    return traseu_optimizat, total