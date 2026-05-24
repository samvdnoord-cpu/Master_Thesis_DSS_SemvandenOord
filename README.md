# Master_Thesis_SemvandenOord_v2

## Stappenplan

De scripts moeten in de volgende volgorde worden uitgevoerd. Latere stappen zijn afhankelijk van de uitvoer van eerdere stappen.

---

### Stap 1 — RQ1: Exploratory Data Analysis

Voer eerst de EDA uit om het dataset te verkennen voordat de modellen worden getraind.

```bash
python RQ1/EDA.py
```

---

### Stap 2 — RQ1: Modeltraining (afzonderlijk uitvoeren)

Train elk van de vier classificatiemodellen afzonderlijk. De scripts slaan de resultaten en modelbestanden op die nodig zijn voor de volgende stappen.

```bash
python RQ1/RQ1_naive_bayes.py
python RQ1/RQ1_Logistic_Regression.py
python RQ1/RQ1_random_forest.py
python RQ1/RQ1_XGBoost.py
```

> Let op: elk script slaat resultaten op als CSV-bestand. Zorg dat alle vier scripts succesvol zijn afgerond voordat je verdergaat met stap 3.

---

### Stap 3 — RQ1: Statistische vergelijking van modellen

Vergelijk de prestaties van de vier modellen statistisch. Dit script laadt de CSV-resultaten van stap 2 en vereist dat alle vier modellen zijn gerund.

```bash
python RQ1/RQ1_statistical_comparison.py
```

---

### Stap 4 — RQ2: SHAP-analyse (modelinterpretatie)

Voer de SHAP-analyse uit op het beste model (XGBoost) om de modelvoorspellingen te verklaren. Dit script laadt de opgeslagen XGBoost-pipeline en de bijbehorende datasplits uit stap 2.

```bash
python RQ2/RQ2_SHAP_analysis.py
```

---

### Stap 5 — RQ3: Fairness-analyse

Evalueer de eerlijkheid en subgroepprestaties van het XGBoost-model. Dit script test het model op een vaste held-out testset en berekent fairness-metrics per subgroep.

```bash
python RQ3/RQ3_fairness_analyses.py
```

---

## Afhankelijkheden tussen stappen

```
EDA (stap 1)
    └── Modeltraining RQ1 (stap 2)
            ├── Statistische vergelijking RQ1 (stap 3)
            ├── SHAP-analyse RQ2 (stap 4)
            └── Fairness-analyse RQ3 (stap 5)
```

Stappen 3, 4 en 5 kunnen pas worden uitgevoerd nadat stap 2 volledig is afgerond.
