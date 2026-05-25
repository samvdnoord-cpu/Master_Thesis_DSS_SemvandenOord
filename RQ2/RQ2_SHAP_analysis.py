# --- Imports ---
import joblib
import shap
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# --- Load RQ1 final model and data splits ---

base_path = "final_model_rq1/"

final_xgb_pipeline = joblib.load(base_path + "final_xgb_pipeline.pkl")
X_train_final = joblib.load(base_path + "X_train_final.pkl")
X_test_final = joblib.load(base_path + "X_test_final.pkl")
y_train_final = joblib.load(base_path + "y_train_final.pkl")
y_test_final = joblib.load(base_path + "y_test_final.pkl")

# --- Apply preprocessing separately to obtain processed feature matrices ---
X_train_processed = final_xgb_pipeline.named_steps["preprocessor"].transform(X_train_final)
X_test_processed = final_xgb_pipeline.named_steps["preprocessor"].transform(X_test_final)


# --- Extract the trained XGBoost model from the pipeline ---
xgb_model = final_xgb_pipeline.named_steps["model"]

# --- Retrieve processed feature names from the preprocessor ---
processed_feature_names = final_xgb_pipeline.named_steps["preprocessor"].get_feature_names_out()

# Map raw column names to human-readable labels for plots
feature_name_map = {
    "ap_hi":   "Systolic BP",
    "age":     "Age",
    "cholesterol": "Cholesterol",
    "weight":  "Weight",
    "ap_lo":   "Diastolic BP",
    "active":  "Physical Activity",
    "height":  "Height",
    "gluc":    "Glucose",
    "smoke":   "Smoking",
    "gender":  "Gender",
    "alco":    "Alcohol consumption",
}


def get_original_feature_name(processed_name, original_columns):
    clean_name = processed_name.split("__")[-1]
    for col in sorted(original_columns, key=len, reverse=True):
        if clean_name == col or clean_name.startswith(col + "_"):
            return col
    return clean_name


readable_feature_names = [
    feature_name_map.get(get_original_feature_name(n, X_train_final.columns), n)
    for n in processed_feature_names
]

X_train_full_shap = X_train_processed
X_test_full_shap = X_test_processed

# --- Create SHAP TreeExplainer on the trained XGBoost model ---
# TreeExplainer works directly on tree-based models without requiring background data
explainer = shap.TreeExplainer(xgb_model)

# --- Compute SHAP values for train and test sets ---
shap_values_train = explainer.shap_values(X_train_full_shap)
shap_values_test = explainer.shap_values(X_test_full_shap)


# --- Step 1: Compute global feature importance (mean |SHAP|) for train and test ---
# Aggregates per-instance SHAP values into a single importance score per feature
# and computes the train/test difference to assess stability across unseen data

importance_train = np.abs(shap_values_train).mean(axis=0)
importance_test = np.abs(shap_values_test).mean(axis=0)

feature_importance_df = pd.DataFrame({
    "feature": readable_feature_names,
    "train_importance": importance_train,
    "test_importance": importance_test
})

feature_importance_df["difference"] = (
    feature_importance_df["train_importance"]  - feature_importance_df["test_importance"]
)

feature_importance_df["abs_difference"] = feature_importance_df["difference"].abs()

feature_importance_df = feature_importance_df.sort_values(by="train_importance", ascending=False)

print(feature_importance_df.head(10))


# --- Step 2: Save feature importance table to CSV for use in the results section ---
feature_importance_df.to_csv("RQ2_feature_importance_xgb.csv", index=False)


output_dir = "RQ2"
os.makedirs(output_dir, exist_ok=True)

# --- Step 3: SHAP beeswarm summary plots (train and test) ---
# Reveals which features matter most, whether the pattern holds on unseen data,
# and whether high/low feature values increase or decrease the predicted risk

shap.summary_plot(shap_values_train, X_train_full_shap, feature_names=readable_feature_names, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "RQ2_SHAP_beeswarm_train.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(output_dir, "RQ2_SHAP_beeswarm_train.pdf"), bbox_inches="tight")
plt.show()

shap.summary_plot(shap_values_test, X_test_full_shap, feature_names=readable_feature_names, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "RQ2_SHAP_beeswarm_test.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(output_dir, "RQ2_SHAP_beeswarm_test.pdf"), bbox_inches="tight")
plt.show()

# --- Step 4: SHAP bar plots of mean absolute feature importance (train and test) ---
shap.summary_plot(shap_values_train, X_train_full_shap, feature_names=readable_feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "RQ2_SHAP_bar_train.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(output_dir, "RQ2_SHAP_bar_train.pdf"), bbox_inches="tight")
plt.show()

shap.summary_plot(shap_values_test, X_test_full_shap, feature_names=readable_feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "RQ2_SHAP_bar_test.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(output_dir, "RQ2_SHAP_bar_test.pdf"), bbox_inches="tight")
plt.show()
