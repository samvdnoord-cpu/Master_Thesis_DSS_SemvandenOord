import pandas as pd 
import numpy as np
from scipy import stats 

## step 1: loading in the results of the models (CSV documents per model)
nb_path = "results_naive_bayes/results_naive_bayes_v3.csv"
log_path = "resultaten_logistische_regressie/results_logistic_regression.csv"
rf_path = "results_random_forest/results_randomforest_v2.csv"
xgb_path = "results_XGboost/results_xgboost_v2.csv"

naive_bayes_results = pd.read_csv(nb_path)
logistic_regression_results = pd.read_csv(log_path)
random_forest_results = pd.read_csv(rf_path)
xgboost_results = pd.read_csv(xgb_path)

### Step 2: check alignment -> must be to see if they all have identical splits (everything needs to be true)
print(random_forest_results[["seed", "fold"]].equals(naive_bayes_results[["seed", "fold"]]))
print(random_forest_results[["seed", "fold"]].equals(logistic_regression_results[["seed", "fold"]]))
print(random_forest_results[["seed", "fold"]].equals(xgboost_results[["seed", "fold"]]))

# corresponds to TRUE -> step 2 check 


## Step 3: calculate paired difference 
## why you must compare models on the exact same splits so this way the models will be compared for every fold and every seed 
## here you have difference between performance on the same exact same folds and seeds so you can comapre them fairly
diff_rf_nb = random_forest_results["f1"] - naive_bayes_results["f1"]
diff_rf_xgb = random_forest_results["f1"] - xgboost_results["f1"]
diff_rf_log = random_forest_results["f1"] - logistic_regression_results["f1"]
diff_xgb_log = xgboost_results["f1"] - logistic_regression_results["f1"]
diff_log_nb = logistic_regression_results["f1"] - naive_bayes_results["f1"]
diff_xgb_nb = xgboost_results["f1"] - naive_bayes_results["f1"]


## step 4: t-tes function -> need resampled t-tes (nadeau & bengio) because there is dependence between folds. 
## function writing that gets out mean_diff, t-tes, p_value 
def corrected_t_test(differences, k=5):
    n = len(differences)
    mean_diff = np.mean(differences)
    var_diff = np.var(differences, ddof=1)

    correction = (1/ k) + (1 / (k - 1))

    t_stat = mean_diff / np.sqrt(correction * var_diff)
    df = n -1 
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    return mean_diff, t_stat, p_value

## step 5: T-tes implemnetation
print("Radom forest vs XGBoost")
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


## step 6: correcting for multiple copmparison (vanaf hier chat)
n_tests = 6
alpha = 0.05
alpha_bonf = alpha / n_tests

print("\nBonferroni corrected alpha:")
print(alpha_bonf)

## step 7: store all the results in a table: 

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

## optional: save results table
results_table.to_csv("RQ1/statistical_model_comparison_f1_v2.csv", index=False)
