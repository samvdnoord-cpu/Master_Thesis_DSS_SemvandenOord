import openml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data loading in 
dataset = openml.datasets.get_dataset(45547)

df, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe"
)


# Step 1 data sanity check 
print("dataframe shape: rows and columns")
print(df.shape) 
print("dataframe info")
print(df.info()) 
print("dataframe first 5 rows")
print(df.head()) 
print("missing values")
print(df.isnull().sum())
print("Showing datatype")
print(df.dtypes.value_counts())

#Step 2: Univariate analyses 
# Here the variables will be analysed individually.
#This is important to do because 
# 1. this gives insights in whats normal in the data. So what are normal values, But also are there outliers or is the variable skuwed 
# 2. If the variables have enough variation, and if classes are balanced or not. 
# 3. For fairness this analysis is also very important because it tells you how much of the data belongs to a specific group. And maybe if a group is underperformed in the data. If the target frequencu difference across groups. 

#Univariate analysis numerical columns 
#stap 2.1: Numeric columns describtives 
print("The descriptives of numeric variables")
print(df[["age", "height", "weight", "ap_hi", "ap_lo"]].describe())

#step 2.2: historgram + KDE distribution (KDe is alternative what more smooth is than a histogram)
numerical_columns = ["age", "height", "weight", "ap_hi", "ap_lo"]

for col in numerical_columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

#Step 2.3: Boxplot 
for col in numerical_columns:
    plt.figure(figsize=(4,2))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

#Univariate analyse-> categorical variables 
#frequency tabel: here you can see class imbalamce and rare categories 
categorical_variables = ["cholesterol", "gluc", "smoke", "alco", "active", "cardio", "gender"]

for col in categorical_variables:
    print(df[col].value_counts())


##barplot for categorical variables 
for col in categorical_variables:
    plt.figure(figsize=(5,3))
    sns.countplot(x=df[col])
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()


## univariate analysis target variable 
print(df['cardio'].value_counts())
print(df['cardio'].value_counts(normalize=True) * 100)

#Step 3: Target analysis 
#Gender
print("cardio distribution for wimon")
df_female = df[df["gender"] == 1]
print(df_female["cardio"].value_counts())

print("cardio distribution for men")
df_male = df[df["gender"] == 2]
print(df["cardio"].value_counts())

# catgeorical variables cardio distribution
import matplotlib.pyplot as plt
import seaborn as sns

categorical_cols = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]

for col in categorical_cols:
    plt.figure(figsize=(5,3))
    sns.countplot(data=df, x=col, hue="cardio")
    plt.title(f"Target distribution by {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend(title="cardio")
    plt.show()


# numerical variables target distribution
numeric_cols = ["age", "weight", "ap_hi", "ap_lo"]

for col in numeric_cols:
    plt.figure(figsize=(5,3))
    sns.histplot(data=df, x=col, hue="cardio", bins=30, kde=True, stat="density", common_norm=False)
    plt.title(f"{col} distribution by target (cardio)")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.show()



#Step 4: Demographic subgroup analysis 
# A gender analyses 
#distribution 
print("distribution gender")
print(df["gender"].value_counts())
print(df["gender"].value_counts(normalize=True) * 100)


#Blood preassure per gender 
print("blood pressure per gender")

for col in ["ap_hi", "ap_lo"]:
    plt.figure(figsize=(5,3))
    sns.boxplot(data=df, x="gender", y=col)
    plt.title(f"{col} by gender")
    plt.show()

#Cholesterol and glucose per gender 
print("cholestorol and glucose per gender")
for col in ["cholesterol", "gluc"]:
    plt.figure(figsize=(5,3))
    sns.countplot(data=df, x=col, hue="gender")
    plt.title(f"{col} distribution by gender")
    plt.show()

# Lifestyle features per gender
print("lifestyle factors per gender")
for col in ["smoke", "alco", "active"]:
    plt.figure(figsize=(5,3))
    sns.countplot(data=df, x=col, hue="gender")
    plt.title(f"{col} by gender")
    plt.show()

# B age groups 
# define jong and old 
print("defining young and old")
age_median = df["age"].median()

df["age_group"] = df["age"].apply(
    lambda x: "younger" if x < age_median else "older"
)



# how big the age groups are
print("how big are the age groups")
print(df["age_group"].value_counts())
print(df["age_group"].value_counts(normalize=True) * 100)


# feature shifts for each age group: 
# bloodpressure 
print("bloodpressure shift between age")
for col in ["ap_hi", "ap_lo"]:
    plt.figure(figsize=(5,3))
    sns.boxplot(data=df, x="age_group", y=col)
    plt.title(f"{col} by age group")
    plt.show()

#cholesterol and glucose
print("cholesterol and glucose shift between age")
for col in ["cholesterol", "gluc"]:
    plt.figure(figsize=(5,3))
    sns.countplot(data=df, x=col, hue="age_group")
    plt.title(f"{col} by age group")
    plt.show()

#lifestyle 
print("lifestyle shift in age")
for col in ["smoke", "alco", "active"]:
    plt.figure(figsize=(5,3))
    sns.countplot(data=df, x=col, hue="age_group")
    plt.title(f"{col} by age group")
    plt.show()

# step 5: Bivariate analyses
# Numerical features against target variable 
numeric_cols_2 = ["age", "weight", "ap_hi", "ap_lo"]

for col in numeric_cols_2:
    plt.figure(figsize=(5,3))
    sns.boxplot(data=df, x="cardio", y=col)
    plt.title(f"{col} by target (cardio)")
    plt.xlabel("cardio")
    plt.ylabel(col)
    plt.show()

# categorical features against target variable 
cat_cols_2 = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]

for col in cat_cols_2:
    plt.figure(figsize=(5,3))
    sns.countplot(data=df, x=col, hue="cardio")
    plt.title(f"Target distribution by {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend(title="cardio")
    plt.show()

#Correlation bewteen numerical features
corr_numerical = df[["age", "height", "weight", "ap_hi", "ap_lo"]].corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr_numerical, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix of numerical features")
plt.show()