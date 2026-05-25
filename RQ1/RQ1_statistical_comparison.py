# --- RQ1: Statistical Comparison of Models ---

# --- Imports ---
import pandas as pd
import numpy as np
from scipy import stats

# --- Step 1: Load nested CV results for each model ---
nb_path = "results_naive_bayes/results_naive_bayes_v3.csv"
log_path = "resultaten_logistische_regressie/results_logistic_regression.csv"
rf_path = "results_random_forest/results_randomforest_v2.csv"
xgb_path = "results_XGboost/results_xgboost_v2.csv"

naive_bayes_results = pd.read_csv(nb_path)
logistic_regression_results = pd.read_csv(log_path)
random_forest_results = pd.read_csv(rf_path)
xgboost_results = pd.read_csv(xgb_path)

# --- Step 2: Verify alignment across result files ---
# All models must have been evaluated on identical seed/fold splits for paired comparisons to be valid
print(random_forest_results[["seed", "fold"]].equals(naive_bayes_results[["seed", "fold"]]))
print(random_forest_results[["seed", "fold"]].equals(logistic_regression_results[["seed", "fold"]]))
print(random_forest_results[["seed", "fold"]].equals(xgboost_results[["seed", "fold"]]))


# --- Step 3: Compute paired F1 differences per fold ---
# Pairing on the same folds and seeds ensures differences reflect model performance, not data variation
diff_rf_nb = random_forest_results["f1"] - naive_bayes_results["f1"]
diff_rf_xgb = random_forest_results["f1"] - xgboost_results["f1"]
diff_rf_log = random_forest_results["f1"] - logistic_regression_results["f1"]
diff_xgb_log = xgboost_results["f1"] - logistic_regression_results["f1"]
diff_log_nb = logistic_regression_results["f1"] - naive_bayes_results["f1"]
diff_xgb_nb = xgboost_results["f1"] - naive_bayes_results["f1"]


# --- Step 4: Corrected resampled t-test  ---
# Applies a variance correction to account for the dependence between CV folds
def corrected_t_test(differences, k=5):
    n = len(differences)
    mean_diff = np.mean(differences)
    var_diff = np.var(differences, ddof=1)

    correction = (1/ k) + (1 / (k - 1))

    t_stat = mean_diff / np.sqrt(correction * var_diff)
    df = n -1
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    return mean_diff, t_stat, p_value

# --- Step 5: Run pairwise t-tests ---
print("Random Forest vs XGBoost")
print(corrected_t_test(diff_rf_xgb))

print("random forest vs NB")
print(corrected_t_test(diff_rf_nb))

print("random forest vs logistic regression")
print(corrected_t_test(diff_rf_log))

print("XGBoost vs logistic regression ")
print(corrected_t_test(diff_xgb_log))

print("XGBoost vs NB")
print(corrected_t_test(diff_xgb_nb))

print("logistic regression vs NB")
print(corrected_t_test(diff_log_nb))


# --- Step 6: Bonferroni correction for multiple comparisons ---
n_tests = 6
alpha = 0.05
alpha_bonf = alpha / n_tests

print("\nBonferroni corrected alpha:")
print(alpha_bonf)

# --- Step 7: Collect all pairwise results into a summary table ---

results_list = []

comparisons = {
    "RF vs XGB": diff_rf_xgb,
    "RF vs NB": diff_rf_nb,
    "RF vs LOG": diff_rf_log,
    "XGB vs LOG": diff_xgb_log,
    "XGB vs NB": diff_xgb_nb,
    "LOG vs NB": diff_log_nb
}

for comparison_name, diff_values in comparisons.items():
    mean_diff, t_stat, p_value = corrected_t_test(diff_values)

    if mean_diff > 0:
        better_model = comparison_name.split(" vs ")[0]
    elif mean_diff < 0:
        better_model = comparison_name.split(" vs ")[1]
    else:
        better_model = "No difference"

    results_list.append({
        "comparison": comparison_name,
        "mean_diff": mean_diff,
        "t_stat": t_stat,
        "p_value": p_value,
        "alpha_bonf": alpha_bonf,
        "significant": p_value < alpha_bonf,
        "better_model": better_model
    })

results_table = pd.DataFrame(results_list)

print("\nStatistical comparison table:")
print(results_table)

# --- Save results table to CSV ---
results_table.to_csv("RQ1/statistical_model_comparison_f1_v2.csv", index=False)
