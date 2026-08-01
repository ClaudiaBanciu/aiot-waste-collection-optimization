# 📦 Optimizarea Colectării Deșeurilor — Fill Level Forecasting

Acest proiect conține analiza, simularea și modelul predictiv dezvoltate în notebook-ul `fill_level_prediction_draft.ipynb` pentru optimizarea rutelor de colectare a deșeurilor în Sibiu.

---

## 📌 Obiectivul Proiectului

Proiectul compară două metodologii de decizie pentru identificarea containerelor care necesită colectare în ziua curentă:

1. **Regula Fixă (Baseline - Fără AI):** Reacționează la nivelul de umplere curent ($t_0$). Dacă valoarea depășește pragul prestabilit, containerul este inclus pe rută.
2. **Model Predictiv (Machine Learning):** Învață dinamica de umplere din date istorice, estimează nivelul de umplere pentru ziua următoare ($t_{+1}$) și aplică pragul pe valoarea prezisă.

> **Beneficiu practic:** Regula simplă este *reactivă* (intervine după ce containerul s-a umplut), în timp ce modelul predictiv este *proactiv* (identifică din timp containerele care vor deveni critice până la următoarea cursă scheduled, prevenind depășirile și mirosurile neplăcute).

---

## 🛠️ Structura Notebook-ului (`fill_level_prediction_draft.ipynb`)

Notebook-ul este structurat pe 7 etape principale:

* **Etapa 1 — Cadrul Conceptual:** Clarificarea diferenței dintre o regulă scrisă manual (deterministă) și un sistem predictiv ce descoperă tipare din date.
* **Etapa 2 — Curățarea și Structurarea Datelor:**
  * Încărcarea `data_geocoded.csv` ($638$ rânduri).
  * Identificarea celor $6$ rânduri cu `fill_level` lipsă ca fiind depozitele de start/sosire ($3 \text{ rute} \times 2 \text{ puncte}$) și marcarea lor în coloana `point_type` (`depot_departure`, `depot_arrival`, `container`).
  * Extragerea subsetului de $632$ containere și adăugarea unei chei sintetice unice `uid` pentru prevenirea erorilor legate de ID-uri duplicate.
* **Etapa 3 — Regula Simplă (Baseline):**
  * Praguri pe capacități: **120L** (85%), **240L** (80%), **1.100L** (70%).
  * Rezultat: **138 containere** selectate.
* **Etapa 4 — Simularea Istoricului de Umplere:**
  * Simularea a 30 de zile retrograde per container, aplicând o rată zilnică stocastică de umplere în funcție de capacitate (120L: 3-6%, 240L: 4-7%, 1.100L: 6-10%).
  * Generarea setului istoric ($18.960$ rânduri: $632 \text{ containere} \times 30 \text{ zile}$).
* **Etapa 5 — Antrenarea Modelului:**
  * Feature-uri: `simulated_fill_level`, `previous_day_level`, `growth_rate`.
  * Target: `next_day_level`.
  * Split Train/Test (80/20) și antrenare **Regresie Liniară** ($\text{MAE} \approx 1.00$ p.p.).
* **Etapa 6 — Aplicarea Predicției:**
  * Rularea modelului pe starea din ziua $0$ și aplicarea pragurilor pe valoarea prezisă.
  * Rezultat: **150 containere** selectate.
* **Etapa 7 — Analiză Comparativă:**
  * **138 vs. 150 containere**.
  * Modelul a identificat **12 containere suplimentare** care erau sub prag azi, dar vor depăși pragul mâine.

---

## 📂 Fișiere Generat / Output-uri (`data/processed/`)

Seturile de date generate de notebook și salvate în folderul `data/processed/`:

| Nume Fișier | Mărime | Descriere |
| :--- | :---: | :--- |
| 📄 `collection_simple_rule.csv` | ~19 KB | Lista celor **138 containere** selectate prin regula simplă (bazată pe nivelul curent). |
| 📄 `simulated_history.csv` | ~522 KB | Istoricul simulat pe 30 de zile pentru cele 632 containere ($18.960$ rânduri). |
| 📄 `collection_predictive.csv` | ~17 KB | Lista celor **150 containere** selectate prin modelul predictiv (bazat pe nivelul estimat). |

---

## 🚀 Cum se rulează

1. Asigură-te că fișierul de intrare `data_geocoded.csv` se află în directorul corespunzător.
2. Deschide notebook-ul `fill_level_prediction_draft.ipynb` în VS Code sau Jupyter Lab.
3. Rulează celulele secvențial. Fișierele `.csv` rezultate vor fi salvate automat în `data/processed/`.