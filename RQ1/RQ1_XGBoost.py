# --- Imports ---

# Data handling
import numpy as np
import pandas as pd
import time
from datetime import datetime
import os

# Data loading
import openml
import openml

# Scikit-learn - preprocessing & pipelines
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Scikit-learn - model selection
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV
)

# Scikit-learn - metrics
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

# XGBoost
from xgboost import XGBClassifier

# Hyperparameter distributions
from scipy.stats import randint, uniform

import joblib
from sklearn.model_selection import train_test_split

def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

log("script started")

# --- Create output directory for results ---
results_dir = "results_XGboost"

if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Map '{results_dir}' aangemaakt", flush=True)
else:
    print(f"Map '{results_dir}' bestaat al", flush=True)

# --- Load dataset from OpenML ---
dataset = openml.datasets.get_dataset(45547)

df, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe"
)

# --- Preprocessing ---
# XGBoost is tree-based, so minimal preprocessing is required

# Working copy of the dataset
df_CVD = df.copy()

# 1. Convert age from days to years
df_CVD["age"] = df_CVD["age"] / 365.25

# 2. Correct data types
for col in ["height", "weight", "ap_hi", "ap_lo", "gender", "cardio"]:
    df_CVD[col] = pd.to_numeric(df_CVD[col], errors="coerce").astype("int64")

# 3. Remove implausible blood pressure values
df_CVD_cleaned = df_CVD[
    (df_CVD["ap_hi"] >= 40) & (df_CVD["ap_hi"] <= 300) &
    (df_CVD["ap_lo"] >= 40) & (df_CVD["ap_lo"] <= 200) &
    (df_CVD["ap_hi"] > df_CVD["ap_lo"])
]

# 4. Remove implausible height values
df_CVD_cleaned = df_CVD_cleaned[
    (df_CVD_cleaned["height"] >= 120) & (df_CVD_cleaned["height"] <= 220)
    ]

# 5. Remove implausible weight values
df_CVD_cleaned = df_CVD_cleaned[
    (df_CVD_cleaned["weight"] >= 20) & (df_CVD_cleaned["weight"] <= 250)
    ]

# 6. Recode gender: original encoding 1 = women, 2 = men → recoded to 1 = women, 0 = men
df_CVD_cleaned["gender"] = df_CVD_cleaned["gender"].replace({1: 1, 2: 0})

# Sanity check
print(df_CVD_cleaned.describe())
print(df_CVD_cleaned.head())

# Separate features and target
X = df_CVD_cleaned.drop(columns=["cardio"])
y = df_CVD_cleaned["cardio"]

#-----------------------------------------------------------------------

# --- Preprocessing pipeline ---

# 1. Define column groups
numeric_features = ["age", "height", "ap_hi", "ap_lo", "weight"]
binary_features = ["gender", "smoke", "alco", "active"]
ordinal_features = ["cholesterol", "gluc"]

# 2. Build column transformer (passthrough — XGBoost handles raw values natively)
preprocessor_XGboost = ColumnTransformer(
    transformers=[
        ("num" , "passthrough", numeric_features),
        ("bin", "passthrough", binary_features),
        ("ordinal", "passthrough", ordinal_features)
    ],
    remainder="drop"
)


# --- Modeling pipeline: XGBoost with repeated nested cross-validation ---

# Hyperparameter search space
param_dist_xgb = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [3, 6, 9],
    "model__learning_rate": [0.05, 0.1, 0.3],
    "model__subsample": [0.75, 0.85, 1.0],
    "model__colsample_bytree": [0.7, 0.85, 1.0],
    "model__min_child_weight": [1, 3, 5],
    "model__gamma": [0, 1, 3],
    "model__reg_alpha": [0, 0.5, 1],
    "model__reg_lambda": [0.5, 1, 2],
}

# Four outer seeds to repeat the nested CV and reduce variance in performance estimates
outer_seeds = [11, 22, 33, 44]

all_results_xgb = []
best_params_xgb = []

for seed in outer_seeds:
    # Outer loop: 5-fold stratified CV for unbiased performance estimation
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        xgb_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor_XGboost),   # zelfde als RF
            ("model", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=1,
                tree_method="hist",   # sneller
                verbosity=0
            ))
        ])

        # Inner loop: 3-fold CV for hyperparameter selection
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

        search = RandomizedSearchCV(
            estimator=xgb_pipeline,
            param_distributions=param_dist_xgb,
            n_iter=25,
            scoring="f1",
            n_jobs=-1,
            cv=inner_cv,
            random_state=seed,
            refit=True
        )

        search.fit(X_train, y_train)

        # Best model from inner CV, evaluated on the outer test fold
        best_model_xgb = search.best_estimator_

        y_pred_xgb = best_model_xgb.predict(X_test)
        y_proba_xgb = best_model_xgb.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_xgb).ravel()

        fold_result = {
            "seed": seed,
            "fold": fold_idx,
            "accuracy": accuracy_score(y_test, y_pred_xgb),
            "f1": f1_score(y_test, y_pred_xgb),
            "precision": precision_score(y_test, y_pred_xgb, zero_division=0),
            "recall": recall_score(y_test, y_pred_xgb, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba_xgb),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "best_params": search.best_params_
        }

        all_results_xgb.append(fold_result)
        best_params_xgb.append({
            "seed": seed,
            "fold": fold_idx,
            **search.best_params_
        })


# --- Summarize nested CV results ---
results_df_xgb = pd.DataFrame(all_results_xgb)

summary_xgb = results_df_xgb[
    ["accuracy", "f1", "precision", "recall", "roc_auc", "specificity"]
].agg(["mean", "std"]).T

print("\nXGBoost - Repeated Nested CV Summary")
print(summary_xgb)

print("\nAverage confusion matrix components per outer test fold")
print(results_df_xgb[["tp", "fp", "tn", "fn"]].mean())

print("\nPer-seed average performance")
per_seed_summary_xgb = results_df_xgb.groupby("seed")[
    ["accuracy", "f1", "precision", "recall", "roc_auc", "specificity"]
].mean()
print(per_seed_summary_xgb)

print("\nExample of best parameter settings frequency")
best_params_df_xgb = pd.DataFrame(best_params_xgb)

param_cols = [col for col in best_params_df_xgb.columns if col not in ["seed", "fold"]]
for col in param_cols:
    print(f"\n{col}")
    print(best_params_df_xgb[col].value_counts().head(10))


# --- Save nested CV results to CSV ---
results_df_xgb.to_csv(os.path.join(results_dir, "results_xgboost_v2.csv"), index=False)
summary_xgb.to_csv(os.path.join(results_dir, "summary_xgboost_v2.csv"))
best_params_df_xgb.to_csv(os.path.join(results_dir, "best_params_xgboost_v2.csv"), index=False)


log("Bestanden opgeslagen: results_xgboost_v2.csv, summary_xgboost_v2.csv, best_params_xgboost_v2.csv")


# --- Final model: train on a fixed split for use in RQ2 (SHAP analysis) ---

import os
import joblib
from sklearn.model_selection import train_test_split

# 0. Create output folder
final_model_dir = "final_model_rq1"
os.makedirs(final_model_dir, exist_ok=True)

# 1. Make one fixed train-test split for interpretation
# Use a fixed random_state for reproducibility
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 2. Fill in the most frequently selected hyperparameters from nested CV
# Note: max_depth had a tie between 3 and 6; 6 is chosen here as a sensible final choice
final_xgb_params = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.10,
    "subsample": 0.75,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
    "gamma": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 2.0
}

# 3. Build final pipeline
final_xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor_XGboost),
    ("model", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
        **final_xgb_params
    ))
])

# 4. Fit final model on the fixed training set
final_xgb_pipeline.fit(X_train_final, y_train_final)

# 5. Optional: generate predictions now already for later use in RQ2
y_pred_final = final_xgb_pipeline.predict(X_test_final)
y_proba_final = final_xgb_pipeline.predict_proba(X_test_final)[:, 1]

# 6. Save final model and corresponding datasets for RQ2
joblib.dump(final_xgb_pipeline, os.path.join(final_model_dir, "final_xgb_pipeline.pkl"))
joblib.dump(X_train_final, os.path.join(final_model_dir, "X_train_final.pkl"))
joblib.dump(X_test_final, os.path.join(final_model_dir, "X_test_final.pkl"))
joblib.dump(y_train_final, os.path.join(final_model_dir, "y_train_final.pkl"))
joblib.dump(y_test_final, os.path.join(final_model_dir, "y_test_final.pkl"))

# Save predictions too (handy for RQ2 / fairness checks)
joblib.dump(y_pred_final, os.path.join(final_model_dir, "y_pred_final.pkl"))
joblib.dump(y_proba_final, os.path.join(final_model_dir, "y_proba_final.pkl"))

# Save metadata about the final model choice
final_model_info = {
    "model_name": "XGBoost",
    "selection_basis": "Most frequently selected hyperparameters from repeated nested CV",
    "random_state_final_split": 42,
    "test_size": 0.2,
    "chosen_params": final_xgb_params,
    "note": "max_depth showed a tie between 3 and 6; 6 was selected as final value."
}
joblib.dump(final_model_info, os.path.join(final_model_dir, "final_model_info.pkl"))

print("\nFinal XGBoost pipeline and RQ2 datasets saved in folder:", final_model_dir)
