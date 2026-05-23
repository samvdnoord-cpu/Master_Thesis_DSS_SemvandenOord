##### RQ1: logistic regression #####

## nessacary packages 
import numpy as np
import pandas as pd
import openml
import os
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

## alle log fingen zijn met chat: 
def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

log("script started")


## ---------map aanmaken voor resultaten ----------------
results_dir = "resultaten_logistische_regressie"

if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Map '{results_dir}' aangemaakt", flush=True)
else:
    print(f"Map '{results_dir}' bestaat al", flush=True)

## loading in the data 
# Data loading in 
log("OpenML dataset loading: 45547")


dataset = openml.datasets.get_dataset(45547)

df, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe"
)

log(f"Dataset loaded with shape: {df.shape}")

## ----------------Building preprocessing --------------
## Building preprocessing
log("Preprocessing started")

##-- making a copy of the dataset 
df_CVD = df.copy()

## 1. basic: Outliers, transformation of features 

## Transforming age into years instead of days. 
df_CVD["age"] = (df_CVD["age"] / 365.25)

## Datatype correcting -> height, weight, ap_hi, ap_lo, age to intergers. 
df_CVD["height"] = df_CVD["height"].astype(int)
df_CVD["weight"] = df_CVD["weight"].astype(int)
df_CVD["ap_hi"] = df_CVD["ap_hi"].astype(int)
df_CVD["ap_lo"] = df_CVD["ap_lo"].astype(int)
df_CVD["gender"] = df_CVD["gender"].astype(int)
df_CVD["cardio"] = df_CVD["cardio"].astype(int)

log("Basic transformations completed")

## 2. Bloodpressure cleaning: 
####--- Systolic blood pressure: below 40 and above 300 is unlikely-> all adults so can use adult ranges
df_CVD_cleaned = df_CVD[
    (df_CVD["ap_hi"] >= 40) & (df_CVD["ap_hi"] <= 300) &
    (df_CVD["ap_lo"] >= 40) & (df_CVD["ap_lo"] <= 200) &
    (df_CVD["ap_hi"] > df_CVD["ap_lo"])
]


log(f"Shape after blood pressure cleaning: {df_CVD_cleaned.shape}")

## First there were: 70000 rows and now 68672 row when outliers are deleted. 

## 3. height correction outliers:

log("Height cleaning started")

df_CVD_cleaned = df_CVD_cleaned[
    (df_CVD_cleaned["height"] >= 120) & (df_CVD_cleaned["height"] <= 220)
    ]

log(f"Shape after height cleaning: {df_CVD_cleaned.shape}")


## 4. weight correction for not reliable outliers --> why exclude the lower values ?? 
log("Weight cleaning started")



df_CVD_cleaned = df_CVD_cleaned[df_CVD_cleaned["weight"] >= 30]
## checking weigth skwed for maybe log in preprocessing 
print("skewness of weight")
print(df_CVD_cleaned["weight"].skew())

log(f"Shape after weight cleaning: {df_CVD_cleaned.shape}")

## 5. Gender adjusten from 2 and 1 to 0 and 1: (1 women, 2 men)
log("Gender recoding started")



df_CVD_cleaned["gender"] = df_CVD_cleaned["gender"].replace({2: 0, 1: 1})

## checking if transformation of all numerical values went well



print(df_CVD_cleaned.describe())
print(df_CVD_cleaned.head())

## deviding dataset into featuers and target variable 
X = df_CVD_cleaned.drop(columns=["cardio"])
y = df_CVD_cleaned["cardio"]

log(f"Features and target created. X shape: {X.shape}, y shape: {y.shape}")

#####--------preprocessing pipeline -----### 

log("Preprocessing pipeline definition started") 

#1) define columns 
numeric_features = ["age", "height", "ap_hi", "ap_lo", "weight"]
binary_features = ["gender", "smoke", "alco", "active"] 
ordinal_features = ["cholesterol", "gluc"]

# 2) define pipeline 
preprocessor_logistic = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("bin", "passthrough", binary_features),
        ("ordinal", StandardScaler(), ordinal_features)
    ],
    remainder="drop"
)

log("Preprocessing pipeline defined")

#### ------- modeling pipeline logistic regression----###3

log("Model setup started") 
### hyperparameters 
param__dist_logistic = {
    "model__C": [0.1, 1, 10],
    "model__solver": ["lbfgs", "liblinear"],
    "model__class_weight": [None, "balanced"]
}

## repeated nested cross validation
outer_seeds = [11, 22, 33, 44]

all_results_logistic = []
best_params_logistic = []

log("Repeated nested cross-validation started")

for seed in outer_seeds: 
    #### outer loop 
    log(f"Starting outer CV for seed={seed}")
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        log(f"Starting fold {fold_idx} for seed={seed}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        logistic_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor_logistic),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=seed,))
        ])
        
        log(f"Starting inner CV for seed={seed}, fold={fold_idx}")
        ### inner loop
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

        search = RandomizedSearchCV(
            estimator=logistic_pipeline,
            param_distributions=param__dist_logistic,
            n_iter=10,
            scoring="f1",
            n_jobs=-1,
            cv=inner_cv,
            random_state=seed,
            refit=True
        )

        search.fit(X_train, y_train)

        log(f"Inner CV completed for seed={seed}, fold={fold_idx}")
        log(f"Best params for seed={seed}, fold={fold_idx}: {search.best_params_}")

        #### best model refit on outer Cv 
        best_model_logistic = search.best_estimator_

        ### evaluate on test fold of outer cv 
        y_pred_log = best_model_logistic.predict(X_test)
        y_proba_log = best_model_logistic.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_log).ravel()

        fold_result = {
            "seed": seed,
            "fold": fold_idx,
            "accuracy": accuracy_score(y_test, y_pred_log),
            "f1": f1_score(y_test, y_pred_log, zero_division=0),
            "precision": precision_score(y_test, y_pred_log, zero_division=0),
            "recall": recall_score(y_test, y_pred_log, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba_log),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "best_params": search.best_params_
        }

        all_results_logistic.append(fold_result)
        best_params_logistic.append({
            "seed": seed,
            "fold": fold_idx,
            **search.best_params_
        })

        log(f"Fold {fold_idx} for seed={seed} completed")

### summarize the results 
log("Summarizing results")


results_df_logistic = pd.DataFrame(all_results_logistic)

###vanaf hier chat 
summary_logistic = results_df_logistic[
    ["accuracy", "f1", "precision", "recall", "roc_auc", "specificity"]
].agg(["mean", "std"]).T

print("\nLogistic Regression - Repeated Nested CV Summary")
print(summary_logistic)

print("\nAverage confusion matrix components per outer test fold")
print(results_df_logistic[["tp", "fp", "tn", "fn"]].mean())

print("\nPer-seed average performance")
per_seed_summary_logistic = results_df_logistic.groupby("seed")[
    ["accuracy", "f1", "precision", "recall", "roc_auc", "specificity"]
].mean()
print(per_seed_summary_logistic)

print("\nExample of best parameter settings frequency")
best_params_df_logistic = pd.DataFrame(best_params_logistic)

param_cols = [col for col in best_params_df_logistic.columns if col not in ["seed", "fold"]]
for col in param_cols:
    print(f"\n{col}")
    print(best_params_df_logistic[col].value_counts().head(10))

## save results
log("Saving output files")

results_df_logistic.to_csv(
    os.path.join(results_dir, "results_logistic_regression.csv"),
    index=False
)

summary_logistic.to_csv(
    os.path.join(results_dir, "summary_logistic_regression.csv")
)

best_params_df_logistic.to_csv(
    os.path.join(results_dir, "best_params_logistic_regression.csv"),
    index=False
)

per_seed_summary_logistic.to_csv(
    os.path.join(results_dir, "per_seed_summary_logistic_regression.csv")
)

log("Bestanden opgeslagen: results_logistic_regression.csv, summary_logistic_regression.csv, best_params_logistic_regression.csv, per_seed_summary_logistic_regression.csv")
log("script finished")