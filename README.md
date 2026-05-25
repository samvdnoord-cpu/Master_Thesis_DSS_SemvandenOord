# Master_Thesis_SemvandenOord_v2

## Step-by-step guide

The scripts must be executed in the following order. Later steps depend on the output of earlier steps.

---

### Step 1 — RQ1: Exploratory Data Analysis

Run the EDA first to explore the dataset before training the models.

```bash
python RQ1/EDA.py
```

---

### Step 2 — RQ1: Model training (run separately)

Train each of the four classification models separately. The scripts save the results and model files needed for subsequent steps.

```bash
python RQ1/RQ1_naive_bayes.py
python RQ1/RQ1_Logistic_Regression.py
python RQ1/RQ1_random_forest.py
python RQ1/RQ1_XGBoost.py
```

> Note: each script saves its results as a CSV file. Make sure all four scripts have completed successfully before proceeding to step 3.

---

### Step 3 — RQ1: Statistical comparison of models

Statistically compare the performance of the four models. This script loads the CSV results from step 2 and requires all four models to have been run.

```bash
python RQ1/RQ1_statistical_comparison.py
```

---

### Step 4 — RQ2: SHAP analysis (model interpretation)

Run the SHAP analysis on the best model (XGBoost) to explain model predictions. This script loads the saved XGBoost pipeline and corresponding data splits from step 2.

```bash
python RQ2/RQ2_SHAP_analysis.py
```

---

### Step 5 — RQ3: Fairness analysis

Evaluate the fairness and subgroup performance of the XGBoost model. This script tests the model on a fixed held-out test set and computes fairness metrics per subgroup.

```bash
python RQ3/RQ3_fairness_analyses.py
```

---

## Dependencies between steps

```
EDA (step 1)
    └── Model training RQ1 (step 2)
            ├── Statistical comparison RQ1 (step 3)
            ├── SHAP analysis RQ2 (step 4)
            └── Fairness analysis RQ3 (step 5)
```

Steps 3, 4, and 5 can only be run after step 2 has fully completed.
