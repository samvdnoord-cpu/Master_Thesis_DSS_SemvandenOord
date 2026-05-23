####### -------------------- Thesis RQ1: Naive Bayes code base ----------#####
# importing required packages
###----------------------------------
# This script uses OpenMl to retrieve the cardiovascular disease dataset.
# pandas and Numpy are used for daat handeling.
# Scikit-learn is used for preprocessing.
###----------------------------------
import openml
import numpy as np
import pandas as pd
import os
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

#### ------------------------------------------
# Creating output directions for model results 
#### ------------------------------------------
# The results of each cross-validation fold and the summary statistics are saved as CSV files. 
# Creating a result folder for anive bayes keeps the output of this model organized.
# Furthermore, the folder makes it also easier to compare later on the different models together. 


results_dir = "results_naive_bayes"

if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Map '{results_dir}' aangemaakt", flush=True)
else:
    print(f"Map '{results_dir}' bestaat al", flush=True)



####---------------------------------------------------
# Load the cardiovascular disease dataset from OpenML
####----------------------------------------------------
#The dataset is retrieved directly from OpenML by entering its ID.
# The dataset is loaded as a pandas Dataframe. 
# The target variable is still included in the dataframe at this point. Later on the target variable will be separated. 

dataset = openml.datasets.get_dataset(45547)

df, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe"
)



####---------------------------------------------
# Build the preprocessing pipeline
#### --------------------------------------------
# Firt making a copy of the dataset to keep the original dataset intact. 
df_CVD = df.copy()

## ------------------step 1. basic: Outliers, transformation of features -----------------
# In this step the age variable is transformed from days into years. This makes the feature more interpretable and also more in line with how age is usually represented in the medical domain

## Transforming age into years instead of days. 
df_CVD["age"] = (df_CVD["age"] / 365).astype(int)

## Datatype correcting -> height, weight, ap_hi, ap_lo, age to intergers. 
df_CVD["height"] = df_CVD["height"].astype(int)
df_CVD["weight"] = df_CVD["weight"].astype(int)
df_CVD["ap_hi"] = df_CVD["ap_hi"].astype(int)
df_CVD["ap_lo"] = df_CVD["ap_lo"].astype(int)
df_CVD["gender"] = df_CVD["gender"].astype(int)
df_CVD["cardio"] = df_CVD["cardio"].astype(int)


## -------------- step 2: Cleaning implausible blood pressure values---------------

#--- Systolic blood pressure: below 40 and above 300 is unlikely-> all adults so can use adult ranges
df_CVD_cleaned = df_CVD[
    (df_CVD["ap_hi"] >= 40) & (df_CVD["ap_hi"] <= 300) &
    (df_CVD["ap_lo"] >= 40) & (df_CVD["ap_lo"] <= 200) &
    (df_CVD["ap_hi"] > df_CVD["ap_lo"])
]
 

##----------------- step 3. height correction outliers--------------
df_CVD_cleaned = df_CVD_cleaned[
    (df_CVD_cleaned["height"] >= 120) & (df_CVD_cleaned["height"] <= 220)
    ]

##----------------- step 4. weight correction for not reliable outliers ---------------
df_CVD_cleaned = df_CVD_cleaned[df_CVD_cleaned["weight"] >= 20]
## checking weigth skwed for maybe log in preprocessing 
print("skewness of weight")
print(df_CVD_cleaned["weight"].skew())



##----------------- step 5. Gender adjusten from 2 and 1 to 0 and 1: (1 women, 2 men)
df_CVD_cleaned["gender"] = df_CVD_cleaned["gender"].replace({2: 0, 1: 1})

## checking if transformation of all numerical values went well
print(df_CVD_cleaned.describe())
print(df_CVD_cleaned.head())

## deviding dataset into featuers and target variable 
X = df_CVD_cleaned.drop(columns=["cardio"])
y = df_CVD_cleaned["cardio"]




####-------------------------------------------
# preprocessing pipeline 
####-------------------------------------------


#1) define columns 
numeric_features = ["age", "height", "ap_hi", "ap_lo"]
binary_features = ["gender", "smoke", "alco", "active"] 
ordinal_features = ["cholesterol", "gluc"]

#2) log-transform for weight 
log_transformer = FunctionTransformer(np.log, validate=False)

#3) preprocessing pipeline definition 
preprocessor_baseline = ColumnTransformer(
    transformers=[
        ("weight", Pipeline([
            ("log", log_transformer),
            ("scale", MinMaxScaler())]), ["weight"]),
            ("num", MinMaxScaler(), numeric_features),
            ("ordinal", MinMaxScaler(), ordinal_features),
            ("binary", "passthrough", binary_features),
    ],
    remainder="drop"
)

####--------------------------------------
##  Naive bayes model 
####-----------------------------------------
# 1) model pipeline making 
outer_seeds = [11, 22, 33, 44]

all_results_baseline = []

for seed in outer_seeds:
    ### outer loop
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        baseline_naive_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor_baseline),
            ("baseline_model", GaussianNB())
        ])

        ## inner loop 
        ## naive bayes has no hyperparamter tuning, but we keep the same nested structure to be consistent with the other models. 
        

        ## fit on outer train fold 
        baseline_naive_pipeline.fit(X_train, y_train)

        ## evaluate on outer test fold 
        y_pred_nb = baseline_naive_pipeline.predict(X_test)
        y_proba_nb = baseline_naive_pipeline.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_nb).ravel()

        fold_result = {
            "seed": seed,
            "fold": fold_idx,
            "accuracy": accuracy_score(y_test, y_pred_nb),
            "f1": f1_score(y_test, y_pred_nb, zero_division=0),
            "precision": precision_score(y_test, y_pred_nb, zero_division=0),
            "recall": recall_score(y_test, y_pred_nb, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba_nb),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        }

        all_results_baseline.append(fold_result)




####------------------------------------------
# summarize the results
####-----------------------------------------


results_df_baseline = pd.DataFrame(all_results_baseline)

summary_baseline = results_df_baseline[
    ["accuracy", "f1", "precision", "recall", "roc_auc", "specificity"]
].agg(["mean", "std"]).T

print("\nNaive Bayes - Repeated Nested CV Style Summary")
print(summary_baseline)

print("\nAverage confusion matrix components per outer test fold")
print(results_df_baseline[["tp", "fp", "tn", "fn"]].mean())

print("\nPer-seed average performance")
per_seed_summary_baseline = results_df_baseline.groupby("seed")[
    ["accuracy", "f1", "precision", "recall", "roc_auc", "specificity"]
].mean()
print(per_seed_summary_baseline)

results_df_baseline.to_csv(
    os.path.join(results_dir, "results_naive_bayes_v3.csv"),
    index=False
)

summary_baseline.to_csv(
    os.path.join(results_dir, "summary_naive_bayes_v3.csv")
)

per_seed_summary_baseline.to_csv(
    os.path.join(results_dir, "per_seed_naive_bayes_v3.csv")
)

