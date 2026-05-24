# ============================================================
# RQ3 - Fairness and subgroup error analysis
# Final XGBoost model on fixed held-out test set
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    count,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    false_positive_rate,
    false_negative_rate,
    true_positive_rate,
    true_negative_rate
)

from scipy.stats import fisher_exact

# ============================================================
# 1. Paths
# ============================================================

final_model_dir = "final_model_rq1"
output_dir = "rq3_results"
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(final_model_dir, "final_xgb_pipeline.pkl")
model_info_path = os.path.join(final_model_dir, "final_model_info.pkl")

X_train_path = os.path.join(final_model_dir, "X_train_final.pkl")
X_test_path = os.path.join(final_model_dir, "X_test_final.pkl")
y_train_path = os.path.join(final_model_dir, "y_train_final.pkl")
y_test_path = os.path.join(final_model_dir, "y_test_final.pkl")

y_pred_path = os.path.join(final_model_dir, "y_pred_final.pkl")
y_proba_path = os.path.join(final_model_dir, "y_proba_final.pkl")

# ============================================================
# 2. Load saved objects
# ============================================================

final_model = joblib.load(model_path)
final_model_info = joblib.load(model_info_path)

X_train_final = joblib.load(X_train_path)
X_test_final = joblib.load(X_test_path)
y_train_final = joblib.load(y_train_path)
y_test_final = joblib.load(y_test_path)

if os.path.exists(y_pred_path):
    y_pred_final = joblib.load(y_pred_path)
else:
    y_pred_final = final_model.predict(X_test_final)

if os.path.exists(y_proba_path):
    y_proba_final = joblib.load(y_proba_path)
else:
    y_proba_final = final_model.predict_proba(X_test_final)[:, 1]

X_train_final = pd.DataFrame(X_train_final).reset_index(drop=True)
X_test_final = pd.DataFrame(X_test_final).reset_index(drop=True)
y_test_final = pd.Series(y_test_final).reset_index(drop=True)
y_pred_final = pd.Series(y_pred_final).reset_index(drop=True)
y_proba_final = pd.Series(y_proba_final).reset_index(drop=True)

print("Loaded final model info:")
print(final_model_info)

# ============================================================
# 3. Helper functions
# ============================================================

def safe_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)

def safe_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=0)

def safe_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=0)

def get_confusion_elements(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp

def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = get_confusion_elements(y_true, y_pred)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else np.nan,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }

def label_gender(series):
    # Your coding: 1 = women, 0 = men
    return series.map({1: "women", 0: "men"})

def make_age_group(age_series, threshold):
    return pd.Series(
        np.where(age_series <= threshold, "younger", "older"),
        index=age_series.index
    )

def classify_error_type(row):
    if row["y_true"] == 1 and row["y_pred"] == 1:
        return "TP"
    elif row["y_true"] == 0 and row["y_pred"] == 0:
        return "TN"
    elif row["y_true"] == 0 and row["y_pred"] == 1:
        return "FP"
    elif row["y_true"] == 1 and row["y_pred"] == 0:
        return "FN"
    else:
        return np.nan

def subgroup_metrics(df, group_col):
    rows = []

    for group_value in sorted(df[group_col].dropna().unique()):
        subset = df[df[group_col] == group_value]
        metrics = compute_metrics(subset["y_true"], subset["y_pred"])

        rows.append({
            "group_variable": group_col,
            "group_value": group_value,
            "n": len(subset),
            **metrics
        })

    return pd.DataFrame(rows)

def plot_combined_confusion_matrices(subgroups, output_dir):
    """
    Create a combined 2x2 publication-ready figure of confusion matrices.
    subgroups: list of (title, y_true, y_pred) tuples ordered as:
               (a) Gender: Men, (b) Gender: Women, (c) Age: Older, (d) Age: Younger
    """
    cms = [confusion_matrix(y_true, y_pred, labels=[0, 1])
           for _, y_true, y_pred in subgroups]

    vmin = 0
    vmax = max(cm.max() for cm in cms)

    class_labels = ["No CVD", "CVD"]
    corner_labels = [["TN", "FP"], ["FN", "TP"]]
    subplot_letters = ["(a)", "(b)", "(c)", "(d)"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.38, wspace=0.32, right=0.84)

    cmap = plt.cm.Blues
    im_ref = None

    for idx, ((title, y_true, y_pred), cm) in enumerate(zip(subgroups, cms)):
        ax = axes[idx // 2, idx % 2]

        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = np.where(row_sums > 0, cm / row_sums * 100, 0.0)

        fn, tp = cm[1, 0], cm[1, 1]
        fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")

        im_ref = ax.imshow(cm, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

        # Cell separators
        ax.axhline(0.5, color="white", linewidth=1.5)
        ax.axvline(0.5, color="white", linewidth=1.5)

        # Cell annotations
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                pct = cm_pct[i, j]
                label = corner_labels[i][j]

                if i == 1 and j == 0:
                    text = f"{label}\n{count}\n({pct:.1f}%)\nFNR = {fnr:.3f}"
                else:
                    text = f"{label}\n{count}\n({pct:.1f}%)"

                text_color = "white" if cm[i, j] > vmax * 0.65 else "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold",
                        linespacing=1.4)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(class_labels, fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(class_labels, fontsize=9, rotation=90, va="center")
        ax.set_xlabel("Predicted class", fontsize=10, labelpad=6)
        ax.set_ylabel("Actual class", fontsize=10, labelpad=6)
        ax.set_title(f"{subplot_letters[idx]} {title}", fontsize=11,
                     fontweight="bold", pad=8)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.68])
    cbar = fig.colorbar(im_ref, cax=cbar_ax)
    cbar.set_label("Count", fontsize=10, labelpad=8)
    cbar.ax.tick_params(labelsize=9)

    save_path = os.path.join(output_dir, "subgroup_confusion_matrices.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"\nSaved combined confusion matrices to: {save_path}")
    plt.show()

def metricframe_by_group(df, sensitive_col):
    mf = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "precision": safe_precision,
            "recall": safe_recall,
            "f1": safe_f1,
            "selection_rate": selection_rate,
            "fpr": false_positive_rate,
            "fnr": false_negative_rate,
            "tpr": true_positive_rate,
            "tnr": true_negative_rate,
            "count": count
        },
        y_true=df["y_true"],
        y_pred=df["y_pred"],
        sensitive_features=df[sensitive_col]
    )

    return mf.by_group.reset_index()

def fairness_summary(df, sensitive_col):
    return pd.DataFrame([{
        "sensitive_feature": sensitive_col,
        "demographic_parity_difference": demographic_parity_difference(
            y_true=df["y_true"],
            y_pred=df["y_pred"],
            sensitive_features=df[sensitive_col]
        ),
        "demographic_parity_ratio": demographic_parity_ratio(
            y_true=df["y_true"],
            y_pred=df["y_pred"],
            sensitive_features=df[sensitive_col]
        ),
        "equalized_odds_difference": equalized_odds_difference(
            y_true=df["y_true"],
            y_pred=df["y_pred"],
            sensitive_features=df[sensitive_col]
        ),
        "equalized_odds_ratio": equalized_odds_ratio(
            y_true=df["y_true"],
            y_pred=df["y_pred"],
            sensitive_features=df[sensitive_col]
        )
    }])

def fisher_test_error_rate(df, group_col, group_a, group_b):
    temp = df[df[group_col].isin([group_a, group_b])].copy()

    a = temp[temp[group_col] == group_a]
    b = temp[temp[group_col] == group_b]

    table = np.array([
        [a["is_error"].sum(), (1 - a["is_error"]).sum()],
        [b["is_error"].sum(), (1 - b["is_error"]).sum()]
    ])

    oddsratio, p_value = fisher_exact(table)

    return {
        "test": "fisher_error_rate",
        "group_variable": group_col,
        "group_a": group_a,
        "group_b": group_b,
        "odds_ratio": oddsratio,
        "p_value": p_value
    }

def fisher_test_false_negatives(df, group_col, group_a, group_b):
    temp = df[(df[group_col].isin([group_a, group_b])) & (df["y_true"] == 1)].copy()

    a = temp[temp[group_col] == group_a]
    b = temp[temp[group_col] == group_b]

    table = np.array([
        [(a["error_type"] == "FN").sum(), (a["error_type"] == "TP").sum()],
        [(b["error_type"] == "FN").sum(), (b["error_type"] == "TP").sum()]
    ])

    oddsratio, p_value = fisher_exact(table)

    return {
        "test": "fisher_false_negatives",
        "group_variable": group_col,
        "group_a": group_a,
        "group_b": group_b,
        "odds_ratio": oddsratio,
        "p_value": p_value
    }

# ============================================================
# 4. Create RQ3 dataframe
# ============================================================

rq3_df = X_test_final.copy()
rq3_df["y_true"] = y_test_final
rq3_df["y_pred"] = y_pred_final
rq3_df["y_proba"] = y_proba_final

rq3_df["gender_group"] = label_gender(rq3_df["gender"])

age_threshold = X_train_final["age"].median()
rq3_df["age_group"] = make_age_group(rq3_df["age"], threshold=age_threshold)

rq3_df["error_type"] = rq3_df.apply(classify_error_type, axis=1)
rq3_df["is_error"] = (rq3_df["y_true"] != rq3_df["y_pred"]).astype(int)

print(f"\nAge median split threshold from training set: {age_threshold:.2f}")

# ============================================================
# 5. Overall performance
# ============================================================

overall_metrics_df = pd.DataFrame([
    compute_metrics(rq3_df["y_true"], rq3_df["y_pred"])
])

print("\nOverall test performance")
print(overall_metrics_df.round(4))

# ============================================================
# 6. Subgroup performance metrics
# ============================================================

gender_metrics_df = subgroup_metrics(rq3_df, "gender_group")
age_metrics_df = subgroup_metrics(rq3_df, "age_group")

print("\nSubgroup metrics - gender")
print(gender_metrics_df.round(4))

print("\nSubgroup metrics - age")
print(age_metrics_df.round(4))

# ============================================================
# 7. Error distribution tables
# ============================================================

gender_error_distribution_df = (
    rq3_df
    .groupby(["gender_group", "error_type"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

age_error_distribution_df = (
    rq3_df
    .groupby(["age_group", "error_type"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

print("\nError distribution - gender")
print(gender_error_distribution_df)

print("\nError distribution - age")
print(age_error_distribution_df)

# ============================================================
# 8. Probability summaries by error type
# ============================================================

gender_probability_by_error_df = (
    rq3_df
    .groupby(["gender_group", "error_type"])["y_proba"]
    .agg(["count", "mean", "std", "median", "min", "max"])
    .reset_index()
)

age_probability_by_error_df = (
    rq3_df
    .groupby(["age_group", "error_type"])["y_proba"]
    .agg(["count", "mean", "std", "median", "min", "max"])
    .reset_index()
)

print("\nProbability summary by error type - gender")
print(gender_probability_by_error_df.round(4))

print("\nProbability summary by error type - age")
print(age_probability_by_error_df.round(4))

# ============================================================
# 9. Fairlearn subgroup metrics
# ============================================================

gender_metricframe_df = metricframe_by_group(rq3_df, "gender_group")
age_metricframe_df = metricframe_by_group(rq3_df, "age_group")

print("\nFairlearn MetricFrame - gender")
print(gender_metricframe_df.round(4))

print("\nFairlearn MetricFrame - age")
print(age_metricframe_df.round(4))

# ============================================================
# 10. Fairness summary metrics
# ============================================================

gender_fairness_df = fairness_summary(rq3_df, "gender_group")
age_fairness_df = fairness_summary(rq3_df, "age_group")

print("\nFairness summary - gender")
print(gender_fairness_df.round(4))

print("\nFairness summary - age")
print(age_fairness_df.round(4))

# ============================================================
# 11. Statistical tests
# ============================================================

stat_tests_df = pd.DataFrame([
    fisher_test_error_rate(rq3_df, "gender_group", "women", "men"),
    fisher_test_false_negatives(rq3_df, "gender_group", "women", "men"),
    fisher_test_error_rate(rq3_df, "age_group", "younger", "older"),
    fisher_test_false_negatives(rq3_df, "age_group", "younger", "older")
])

print("\nStatistical tests")
print(stat_tests_df.round(6))

# ============================================================
# 12. Confusion matrices per subgroup - combined figure
# ============================================================

# Print per-subgroup confusion matrix metrics
subgroups_for_metrics = [
    ("Gender: Men",    rq3_df[rq3_df["gender_group"] == "men"]),
    ("Gender: Women",  rq3_df[rq3_df["gender_group"] == "women"]),
    ("Age: Older",     rq3_df[rq3_df["age_group"] == "older"]),
    ("Age: Younger",   rq3_df[rq3_df["age_group"] == "younger"]),
]

header = f"{'Subgroup':<20} {'TN':>6} {'FP':>6} {'FN':>6} {'TP':>6} {'FNR':>8} {'FPR':>8} {'Recall':>8} {'Specificity':>12}"
print("\nPer-subgroup confusion matrix metrics:")
print(header)
print("-" * len(header))
for label, subset in subgroups_for_metrics:
    m = compute_metrics(subset["y_true"], subset["y_pred"])
    print(
        f"{label:<20} {m['tn']:>6} {m['fp']:>6} {m['fn']:>6} {m['tp']:>6} "
        f"{m['fnr']:>8.4f} {m['fpr']:>8.4f} {m['recall']:>8.4f} {m['specificity']:>12.4f}"
    )

# Combined publication-ready figure
subgroup_plot_data = [
    ("Gender: Men",
     rq3_df[rq3_df["gender_group"] == "men"]["y_true"],
     rq3_df[rq3_df["gender_group"] == "men"]["y_pred"]),
    ("Gender: Women",
     rq3_df[rq3_df["gender_group"] == "women"]["y_true"],
     rq3_df[rq3_df["gender_group"] == "women"]["y_pred"]),
    ("Age: Older",
     rq3_df[rq3_df["age_group"] == "older"]["y_true"],
     rq3_df[rq3_df["age_group"] == "older"]["y_pred"]),
    ("Age: Younger",
     rq3_df[rq3_df["age_group"] == "younger"]["y_true"],
     rq3_df[rq3_df["age_group"] == "younger"]["y_pred"]),
]

plot_combined_confusion_matrices(subgroup_plot_data, output_dir)

# ============================================================
# 13. Optional plots: probability by error type
# ============================================================

plt.figure(figsize=(8, 5))
sns.boxplot(data=rq3_df, x="age_group", y="y_proba", hue="error_type")
plt.title("Predicted Probability by Error Type and Age Group")
plt.xlabel("Age group")
plt.ylabel("Predicted probability of CVD")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "probability_by_error_type_age.png"), dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=rq3_df, x="gender_group", y="y_proba", hue="error_type")
plt.title("Predicted Probability by Error Type and Gender Group")
plt.xlabel("Gender group")
plt.ylabel("Predicted probability of CVD")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "probability_by_error_type_gender.png"), dpi=300)
plt.show()

# ============================================================
# 14. Save all outputs
# ============================================================

rq3_df.to_csv(os.path.join(output_dir, "rq3_predictions_with_groups.csv"), index=False)

overall_metrics_df.to_csv(os.path.join(output_dir, "rq3_overall_metrics.csv"), index=False)

gender_metrics_df.to_csv(os.path.join(output_dir, "rq3_gender_metrics.csv"), index=False)
age_metrics_df.to_csv(os.path.join(output_dir, "rq3_age_metrics.csv"), index=False)

gender_error_distribution_df.to_csv(
    os.path.join(output_dir, "rq3_gender_error_distribution.csv"),
    index=False
)
age_error_distribution_df.to_csv(
    os.path.join(output_dir, "rq3_age_error_distribution.csv"),
    index=False
)

gender_probability_by_error_df.to_csv(
    os.path.join(output_dir, "rq3_gender_probability_by_error_type.csv"),
    index=False
)
age_probability_by_error_df.to_csv(
    os.path.join(output_dir, "rq3_age_probability_by_error_type.csv"),
    index=False
)

gender_metricframe_df.to_csv(os.path.join(output_dir, "rq3_gender_metricframe.csv"), index=False)
age_metricframe_df.to_csv(os.path.join(output_dir, "rq3_age_metricframe.csv"), index=False)

gender_fairness_df.to_csv(os.path.join(output_dir, "rq3_gender_fairness_summary.csv"), index=False)
age_fairness_df.to_csv(os.path.join(output_dir, "rq3_age_fairness_summary.csv"), index=False)

stat_tests_df.to_csv(os.path.join(output_dir, "rq3_statistical_tests.csv"), index=False)

print(f"\nAll RQ3 outputs saved in: {output_dir}")