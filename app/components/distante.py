import datetime as dt
import numpy as np
import pandas as pd
import altair as alt
import folium
from geopy.distance import geodesic
from streamlit_folium import st_folium
import streamlit as st
import sys, os


class Distante:
    def __init__(self, file_path):
        self.file_path = file_path
        #self.df = self.distante(file_path)

    @staticmethod
    def distanta_traseu(puncte):
    
        total = 0.0
        for i in range(len(puncte) - 1):
            total += geodesic(puncte[i], puncte[i + 1]).km
        return total

    @staticmethod
    def optimizeaza_nearest_neighbor(puncte):
        """Euristică greedy 'cel mai apropiat vecin': pornește din primul punct,
        la fiecare pas sare la cel mai apropiat punct nevizitat. Returnează
        traseul reordonat și distanța totală."""
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