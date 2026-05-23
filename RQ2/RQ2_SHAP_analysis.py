### Model van RQ1 inladen 
import joblib
import shap
import pandas as pd
import numpy as np 
import os 
import matplotlib.pyplot as plt



### loading in the RQ1 final model and data splits

base_path = "final_model_rq1/"

final_xgb_pipeline = joblib.load(base_path + "final_xgb_pipeline.pkl")
X_train_final = joblib.load(base_path + "X_train_final.pkl")
X_test_final = joblib.load(base_path + "X_test_final.pkl")
y_train_final = joblib.load(base_path + "y_train_final.pkl")
y_test_final = joblib.load(base_path + "y_test_final.pkl")

### apply preprocessing separatly
X_train_processed = final_xgb_pipeline.named_steps["preprocessor"].transform(X_train_final)
X_test_processed = final_xgb_pipeline.named_steps["preprocessor"].transform(X_test_final)


## extract trained v1 XGboost model from pipeine 
xgb_model = final_xgb_pipeline.named_steps["model"]

## keep feature names
processed_feature_names = final_xgb_pipeline.named_steps["preprocessor"].get_feature_names_out()

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

#### optional: smaller sample first for speed
X_train_full_shap = X_train_processed
X_test_full_shap = X_test_processed

## create SHAP explainer 
explainer = shap.TreeExplainer(xgb_model) ### if you want to use kernerl explainer there need to need split, but for tree explainer guess no need train test split.
### check of ik het op hele data gefit kan worden en dat je dat model dan gebruikt. even opzoeken. 
## depends on reseacrh question -> global dont need train test split als je voor local explanation test set wel nodog 


## compute SHAP values 
shap_values_train = explainer.shap_values(X_train_full_shap)
shap_values_test = explainer.shap_values(X_test_full_shap)


### till now I have all the infromation that is needed to further do the analysis that will give answers t RQ2 -> which features are important and is the pattern consistent over out of sample data. 
### so now I can go further with the analyses for RQ 2: 1. which features importance , 2. is pattern stabale across unseen data. 


## step 1: gloabl feature importance for train and test set. 
# you do not want shap values for each individual intances but you want a global overvieuw of wich features are important. 
# what this code under does is: calculate per feature the importance, look at the difference between train and test, Let seen where re big differences 

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


## step 2: Tabel that was made in step 1 saving to use later in result section 
feature_importance_df.to_csv("RQ2_feature_importance_xgb.csv", index=False)


output_dir = "RQ2"
os.makedirs(output_dir, exist_ok=True)

## Step 3: make SHAP beeswarm summary plot (result section)
## what willl be seen in these plots which features most importance
## if pattren is same across unseen data
## if high/ low feature values the prediction increase or decrease

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

## step 4: bar plot of feature importance (result section)
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