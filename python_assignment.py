import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv("heart_attack_south_africa.csv")
print("Dataset loaded. Shape:", df.shape)
print(df.head())

# Standardize column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Detect gender column
gender_col = None
for col in df.columns:
    if 'gender' in col.lower():
        gender_col = col
        break

# Detect smoking column
smoker_col = None
for col in df.columns:
    if 'smok' in col.lower():
        smoker_col = col
        break

# KDE Plot: Age vs Heart Attack Outcome
if 'Age' in df.columns and 'Heart_Attack_Outcome' in df.columns:
    sns.kdeplot(
        data=df, x="Age", hue="Heart_Attack_Outcome",
        fill=True, bw_adjust=1.2, palette="Set1"
    )
    median_age = df.loc[df['Heart_Attack_Outcome']==1, 'Age'].median()
    plt.axvline(median_age, color='black', linestyle='--', label=f"Median Age (HA=1): {median_age}")
    plt.title("Age Distribution by Heart Attack Outcome")
    plt.xlabel("Age (Years)")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

# Heatmap: Correlation of numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", mask=mask)
plt.title("Correlation of Heart Attack Risk Factors")
plt.show()

# Boxplot: Cholesterol levels by Heart Attack Outcome and Gender
chol_col = None
for col in df.columns:
    if 'cholesterol' in col.lower():
        chol_col = col
        break

if chol_col and gender_col:
    sns.boxplot(
        data=df, x="Heart_Attack_Outcome", y=chol_col,
        hue=gender_col, palette="Set2", showfliers=False
    )
    plt.title("Cholesterol Levels Distribution by Heart Attack Outcome and Gender")
    plt.xlabel("Heart Attack Outcome (0 = No, 1 = Yes)")
    plt.ylabel("Cholesterol Level (mg/dL)")
    plt.show()

# Pairplot for selected risk factors
risk_factors = ['Cholesterol_Level', 'Obesity_Index', 'LDL_Level', 'Heart_Attack_Outcome']
risk_factors = [col for col in risk_factors if col in df.columns]
if risk_factors:
    sns.pairplot(df[risk_factors], diag_kind="kde", height=2.5)
    plt.suptitle("Pairwise Comparison of Key Risk Factors", y=1.02)
    plt.show()

# Violin plot: Blood Pressure by Heart Attack Outcome
bp_cols = [col for col in df.columns if 'blood_pressure' in col.lower()]
if bp_cols:
    sns.violinplot(
    data=df.melt(id_vars="Heart_Attack_Outcome", value_vars=bp_cols, 
                 var_name='BP_Type', value_name='BP_Value'),
    x="Heart_Attack_Outcome", y="BP_Value", hue="BP_Type", split=True, palette="Set3"
)

    plt.title("Blood Pressure Distribution by Heart Attack Outcome")
    plt.xlabel("Heart Attack Outcome (0 = No, 1 = Yes)")
    plt.ylabel("Blood Pressure (mmHg)")
    plt.show()

# Countplot: Smoking and Heart Attack Outcome
if smoker_col and gender_col:
    sns.countplot(data=df, x=smoker_col, hue=gender_col, palette="Set1")
    plt.title("Smoking Status and Heart Attack Outcome by Gender")
    plt.xlabel("Smoking Status (Yes/No)")
    plt.ylabel("Count")
    plt.show()
