# --- RQ1 Model: Random Forest ---

# --- Imports ---
import openml
import numpy as np
import pandas as pd
import time
from datetime import datetime
import os

from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)

def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

log("script started")

# --- Create output directory for results ---
results_dir = "results_random_forest"

if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Map '{results_dir}' aangemaakt", flush=True)
else:
    print(f"Map '{results_dir}' bestaat al", flush=True)

# --- Load dataset from OpenML ---
log("OpenML dataset loading: 45547")


dataset = openml.datasets.get_dataset(45547)

df, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe"
)

log(f"Dataset loaded with shape: {df.shape}")

# --- Preprocessing ---
# Random Forest is tree-based, so minimal preprocessing is required

# Working copy of the dataset
df_CVD = df.copy()

# 1. Basic transformations
log("Basis transformation started")

# Convert age from days to years
df_CVD["age"] = df_CVD["age"] / 365.25

# Correct data types
for col in ["height", "weight", "ap_hi", "ap_lo", "gender", "cardio"]:
    df_CVD[col] = pd.to_numeric(df_CVD[col], errors="coerce").astype("int64")

# 2. Remove implausible blood pressure values
df_CVD_cleaned = df_CVD[
    (df_CVD["ap_hi"] >= 40) & (df_CVD["ap_hi"] <= 300) &
    (df_CVD["ap_lo"] >= 40) & (df_CVD["ap_lo"] <= 200) &
    (df_CVD["ap_hi"] > df_CVD["ap_lo"])
]

# 3. Remove implausible height values
df_CVD_cleaned = df_CVD_cleaned[
    (df_CVD_cleaned["height"] >= 120) & (df_CVD_cleaned["height"] <= 220)
    ]

# 4. Remove implausible weight values
df_CVD_cleaned = df_CVD_cleaned[
    (df_CVD_cleaned["weight"] >= 20) & (df_CVD_cleaned["weight"] <= 250)
    ]

# 5. Recode gender: original encoding 1 = women, 2 = men → recoded to 1 = women, 0 = men
df_CVD_cleaned["gender"] = df_CVD_cleaned["gender"].replace({1: 1, 2: 0})

# Sanity check
print(df_CVD_cleaned.describe())
print(df_CVD_cleaned.head())

# Separate features and target
X = df_CVD_cleaned.drop(columns=["cardio"])
y = df_CVD_cleaned["cardio"]

# --- Preprocessing pipeline ---

# 1. Define column groups
numeric_features = ["age", "height", "ap_hi", "ap_lo", "weight"]
binary_features = ["gender", "smoke", "alco", "active"]
ordinal_features = ["cholesterol", "gluc"]

# 2. Build column transformer (passthrough — Random Forest handles raw values natively)
preprocessor_randomforest = ColumnTransformer(
    transformers=[
        ("num" , "passthrough", numeric_features),
        ("bin", "passthrough", binary_features),
        ("ordinal", "passthrough", ordinal_features)
    ],
    remainder="drop"
)

# --- Modeling pipeline: Random Forest with repeated nested cross-validation ---

# Hyperparameter search space
param_dist = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [5, 10, 20, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", None],
    "model__bootstrap": [True, False],
    }

# Four outer seeds to repeat the nested CV and reduce variance in performance estimates
outer_seeds = [11, 22, 33, 44]

all_results_randomforest = []
best_params_randomforest = []

for seed in outer_seeds:
    # Outer loop: 5-fold stratified CV for unbiased performance estimation
    log(f"starting outer CV voor seed={seed}")
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        randomforest_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor_randomforest),
            ("model", RandomForestClassifier(random_state=seed, n_jobs=1))])

        log(f"starting inner CV voor seed={seed}, fold={fold_idx}")
        # Inner loop: 3-fold CV for hyperparameter selection
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

        search = RandomizedSearchCV(
            estimator=randomforest_pipeline,
            param_distributions=param_dist,
            n_iter=25,
            scoring="f1",
            n_jobs=-1,
            cv=inner_cv,
            random_state=seed,
            refit=True
        )

        search.fit(X_train, y_train)

        # Best model from inner CV, evaluated on the outer test fold
        best_model_randomforest = search.best_estimator_

        y_pred_rf = best_model_randomforest.predict(X_test)
        y_proba_rf = best_model_randomforest.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()

        fold_result = {
            "seed": seed,
            "fold": fold_idx,
            "accuracy": accuracy_score(y_test, y_pred_rf),
            "f1": f1_score(y_test, y_pred_rf),
            "precision": precision_score(y_test, y_pred_rf, zero_division=0),
            "recall": recall_score(y_test, y_pred_rf, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba_rf),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "best_params": search.best_params_
        }

        all_results_randomforest.append(fold_result)
        best_params_randomforest.append({
            "seed": seed,
            "fold": fold_idx,
            **search.best_params_
        })


# --- Summarize nested CV results ---
log("summarizing of results")
results_df_rf = pd.DataFrame(all_results_randomforest)

summary_rf = results_df_rf[
    ["accuracy", "f1", "precision", "recall", "roc_auc", "specificity"]
].agg(["mean", "std"]).T

print("\nRandom Forest - Repeated Nested CV Summary")
print(summary_rf)

print("\nAverage confusion matrix components per outer test fold")
print(results_df_rf[["tp", "fp", "tn", "fn"]].mean())

print("\nPer-seed average performance")
per_seed_summary = results_df_rf.groupby("seed")[
    ["accuracy", "f1", "precision", "recall", "roc_auc", "specificity"]
].mean()
print(per_seed_summary)

print("\nExample of best parameter settings frequency")
best_params_df = pd.DataFrame(best_params_randomforest)

param_cols = [col for col in best_params_df.columns if col not in ["seed", "fold"]]
for col in param_cols:
    print(f"\n{col}")
    print(best_params_df[col].value_counts().head(10))


# --- Save nested CV results to CSV ---
#results_df_rf.to_csv("results_randomforest_v2.csv", index=False)

results_df_rf.to_csv(os.path.join(results_dir, "results_randomforest_v2.csv"), index=False)
summary_rf.to_csv(os.path.join(results_dir, "summary_randomforest_v2.csv"))
best_params_df.to_csv(os.path.join(results_dir, "best_params_randomforest_v2.csv"), index=False)

log("Bestanden opgeslagen: results_randomforest_v2.csv, summary_randomforest_v2.csv, best_params_randomforest_v2.csv")
