"""
FRAMEWORK FOR DATA AND VISUAL ANALYTICS - MINI PROJECT
Healthcare Data Analytics: Exploratory Data Analysis (EDA) of the Pima Indians Diabetes Dataset
Dataset : Pima Indians Diabetes Database (768 patients, 8 medical measurements + diabetes result)
          Originally from the US National Institute of Diabetes and Digestive and Kidney Diseases,
          available on UCI / Kaggle. This script downloads a public GitHub copy of the same CSV.
Libraries: Pandas, NumPy, Matplotlib

====================================================================
1. AIM OF THE PROJECT
====================================================================
This project performs Exploratory Data Analysis (EDA) on a public healthcare dataset of 768
female patients to understand which medical measurements are linked to diabetes. It cleans the
data (hidden missing values, duplicates, outliers), computes statistics and correlations, and
uses Matplotlib visualizations to present the key findings.

====================================================================
2. PROCEDURE
====================================================================
1. Import the required libraries.
2. Load the dataset (from a public URL, or a local diabetes.csv if you have one).
3. Inspect the dataset (shape, data types, first rows, summary statistics).
4. Handle missing values: find the hidden missing values (impossible zeros) and fill them
   with the median.
5. Remove duplicate records.
6. Perform data cleaning (consistent data types, final check of the cleaned data).
7. Detect and analyze outliers using the IQR method and box plots.
8. Perform descriptive analysis (mean, median, skewness, comparison by diabetes outcome).
9. Analyze relationships between variables (scatter plots, grouped bar and line charts).
10. Calculate correlations and draw a correlation heatmap.
11. Create visualizations (bar chart, histograms, box plots, scatter plots, heatmap, line charts).
12. Interpret the results and print the findings.

====================================================================
3. CODE
====================================================================
"""

# ---------------------------------------------------------------
# Step 1: Import required libraries
# ---------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

# Two colours used in every chart: blue = no diabetes, orange = diabetes
COLOR_NO, COLOR_YES = "#2a6fbb", "#e07b00"


def heading(text):
    """Print a clear section heading in the console output."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ---------------------------------------------------------------
# Step 2: Load the dataset
# ---------------------------------------------------------------
# Public copies of the same dataset (no login needed). A local 'diabetes.csv' is used first
# if it exists next to this script, which is handy when you are offline.
DATA_URLS = [
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv",
]


def load_data():
    if os.path.exists("diabetes.csv"):
        print("Loading local file: diabetes.csv")
        return pd.read_csv("diabetes.csv")
    for url in DATA_URLS:
        try:
            print("Downloading:", url)
            return pd.read_csv(url)
        except Exception as err:
            print("  failed:", type(err).__name__)
    raise SystemExit("Could not download the dataset. Download 'diabetes.csv' (Pima Indians "
                     "Diabetes) from Kaggle and save it next to this script, then run again.")


df = load_data()

# ---------------------------------------------------------------
# Step 3: Inspect the dataset
# ---------------------------------------------------------------
heading("STEP 3: DATASET INSPECTION")
print("Shape (rows, columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn data types and non-null counts:")
df.info()
print("\nSummary statistics of the raw data:")
print(df.describe().T)

# Meaning of the columns (for reference):
#   Pregnancies = number of pregnancies, Glucose = plasma glucose (mg/dL),
#   BloodPressure = diastolic blood pressure (mm Hg), SkinThickness = triceps skin fold (mm),
#   Insulin = 2-hour serum insulin (mu U/ml), BMI = body mass index,
#   DiabetesPedigreeFunction = family history score, Age = age in years,
#   Outcome = 1 if the patient has diabetes, otherwise 0

print("\nOutcome counts (0 = no diabetes, 1 = diabetes):")
print(df["Outcome"].value_counts())

# ---------------------------------------------------------------
# Step 4: Handle missing values
# ---------------------------------------------------------------
heading("STEP 4: MISSING VALUES")
print("Standard missing values (NaN) in the raw data:", int(df.isnull().sum().sum()))

# This dataset has no NaN cells, BUT a zero is physically impossible for these columns
# (nobody has 0 blood pressure or 0 BMI). The zeros are really missing values.
# Pregnancies = 0 is valid, so it is not included here.
zero_not_allowed = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
hidden_missing = (df[zero_not_allowed] == 0).sum()
hidden_missing_pct = (hidden_missing / len(df) * 100).round(1)
print("\nHidden missing values (impossible zeros):")
print(pd.DataFrame({"count": hidden_missing, "percent": hidden_missing_pct}))

# Replace those zeros with NaN so Pandas treats them as missing
df[zero_not_allowed] = df[zero_not_allowed].replace(0, np.nan)

# Keep a copy that still has the missing values (NaN). Filled-in values are not real
# measurements: if 49% of Insulin were all set to the same median, the distribution would be
# distorted. So outliers, histograms, group statistics and correlations below use only the
# REAL measured values (Pandas skips NaN automatically), while 'df' becomes the fully
# cleaned, filled dataset.
df_observed = df.copy()

# Chart 1: hidden missing values per column + class balance
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
bars = ax[0].bar(hidden_missing_pct.index, hidden_missing_pct.values, color="#7a7a7a")
ax[0].bar_label(bars, fmt="%.1f%%")
ax[0].set_title("Hidden Missing Values (zeros) per Column")
ax[0].set_ylabel("Percent of rows")
ax[0].set_ylim(0, 60)
ax[0].tick_params(axis="x", rotation=25)

counts = df["Outcome"].value_counts().sort_index()
bars = ax[1].bar(["No diabetes (0)", "Diabetes (1)"], counts.values, color=[COLOR_NO, COLOR_YES])
ax[1].bar_label(bars)
ax[1].set_title("Class Distribution of Outcome")
ax[1].set_ylabel("Number of patients")
plt.tight_layout()
plt.show()

# Fill missing values with the MEDIAN of each column. The median is used (not the mean)
# because these columns are skewed and have outliers, and the median is not affected by them.
medians = df[zero_not_allowed].median()
print("\nMedian values used to fill the missing entries:")
print(medians.round(2))
df[zero_not_allowed] = df[zero_not_allowed].fillna(medians)
print("\nMissing values after filling:", int(df.isnull().sum().sum()))

# ---------------------------------------------------------------
# Step 5: Remove duplicate records
# ---------------------------------------------------------------
heading("STEP 5: DUPLICATE RECORDS")
n_dup = int(df.duplicated().sum())
print("Duplicate rows found:", n_dup)
df = df.drop_duplicates().reset_index(drop=True)
print("Rows after the duplicate check:", len(df))

# ---------------------------------------------------------------
# Step 6: Data cleaning (data types and final check)
# ---------------------------------------------------------------
heading("STEP 6: DATA CLEANING SUMMARY")
# Columns that are whole numbers by nature are stored as integers
# (the filled medical columns stay as decimals)
df["Outcome"] = df["Outcome"].astype(int)
df["Age"] = df["Age"].astype(int)
df["Pregnancies"] = df["Pregnancies"].astype(int)
print("Final shape:", df.shape)
print("Missing values left:", int(df.isnull().sum().sum()), "| Duplicates left:", int(df.duplicated().sum()))
print("Smallest value in each cleaned medical column (should now be > 0):")
print(df[zero_not_allowed].min())

feature_cols = [c for c in df.columns if c != "Outcome"]   # the 8 measurements

# ---------------------------------------------------------------
# Step 7: Detect and analyze outliers (IQR method)
# ---------------------------------------------------------------
heading("STEP 7: OUTLIERS (IQR METHOD)")
# A value is an outlier if it is below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
# Only real measured values are used (missing values are excluded).
rows = []
for col in feature_cols:
    values = df_observed[col].dropna()
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((values < low) | (values > high)).sum())
    rows.append([col, round(low, 2), round(high, 2), n_out, round(n_out / len(values) * 100, 1)])
outlier_table = pd.DataFrame(rows, columns=["Column", "Lower limit", "Upper limit", "Outliers", "Percent"])
print(outlier_table.to_string(index=False))

# Chart 2: box plots of every feature (dots outside the whiskers are outliers)
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, col in zip(axes.ravel(), feature_cols):
    bp = ax.boxplot(df_observed[col].dropna(), patch_artist=True, widths=0.5,
                    flierprops=dict(marker="o", markersize=3, markerfacecolor=COLOR_YES,
                                    markeredgecolor=COLOR_YES))
    bp["boxes"][0].set_facecolor("#a9c7e8")
    bp["medians"][0].set_color("black")
    ax.set_title(col, fontsize=10)
    ax.set_xticks([])
fig.suptitle("Box Plots of All Features (orange dots = outliers)", fontsize=13)
plt.tight_layout()
plt.show()

# Handling outliers: these are real medical readings (e.g. very high insulin, many
# pregnancies), not typing errors, so they are KEPT in the analysis. To show what they do,
# we compare the data with a "capped" copy where extreme values are clipped to the IQR limits.
df_capped = df_observed.copy()
for _, r in outlier_table.iterrows():
    df_capped[r["Column"]] = df_capped[r["Column"]].clip(r["Lower limit"], r["Upper limit"])
comparison = pd.DataFrame({
    "Mean (original)": df_observed[feature_cols].mean(),
    "Mean (capped)": df_capped[feature_cols].mean(),
    "Std (original)": df_observed[feature_cols].std(),
    "Std (capped)": df_capped[feature_cols].std(),
}).round(2)
print("\nEffect of capping outliers on mean and standard deviation:")
print(comparison)

# ---------------------------------------------------------------
# Step 8: Descriptive analysis
# ---------------------------------------------------------------
heading("STEP 8: DESCRIPTIVE STATISTICS")
print("Summary statistics of the real measured values ('count' = values that were not missing):")
print(df_observed.describe().T.round(2))

print("\nSkewness (0 = symmetric, > 1 = strongly right-skewed):")
print(df_observed[feature_cols].skew().round(2))

print("\nMean of each measurement by Outcome:")
print(df_observed.groupby("Outcome")[feature_cols].mean().round(2).T)
print("\nMedian of each measurement by Outcome:")
print(df_observed.groupby("Outcome")[feature_cols].median().round(2).T)

# Chart 3: histograms (distribution of every feature, real measured values only)
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, col in zip(axes.ravel(), feature_cols):
    values = df_observed[col].dropna()
    ax.hist(values, bins=20, color="#6a9fd4", edgecolor="white")
    ax.axvline(values.mean(), color="black", linestyle="--", linewidth=1.2,
               label="mean = %.1f" % values.mean())
    ax.set_title(col, fontsize=10)
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
fig.suptitle("Histograms of All Features (real measured values, missing excluded)", fontsize=13)
plt.tight_layout()
plt.show()

# Chart 4: box plots split by Outcome for four important features
key_features = ["Glucose", "BMI", "Age", "Insulin"]
fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
for ax, col in zip(axes, key_features):
    data = [df_observed.loc[df_observed["Outcome"] == 0, col].dropna(),
            df_observed.loc[df_observed["Outcome"] == 1, col].dropna()]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    flierprops=dict(marker="o", markersize=3, markerfacecolor="grey",
                                    markeredgecolor="grey"))
    for patch, colour in zip(bp["boxes"], [COLOR_NO, COLOR_YES]):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["No diabetes", "Diabetes"])
    ax.set_title(col)
fig.suptitle("Key Features: Diabetic vs Non-Diabetic Patients", fontsize=13)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# Step 9 & 10: Relationships and correlations
# ---------------------------------------------------------------
heading("STEP 10: CORRELATION ANALYSIS")
# Pearson correlation: +1 = move together, -1 = move opposite, 0 = no linear relationship.
# Calculated on real measured values (each pair uses the rows where both values exist).
corr = df_observed.corr()
print(corr.round(2))

# Chart 5: correlation heatmap (drawn with Matplotlib imshow)
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        value = corr.values[i, j]
        ax.text(j, i, "%.2f" % value, ha="center", va="center", fontsize=9,
                color="white" if abs(value) > 0.6 else "black")
fig.colorbar(im, ax=ax, label="Correlation coefficient")
ax.set_title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Chart 6: which feature is most related to diabetes?
corr_outcome = corr["Outcome"].drop("Outcome").sort_values()
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh(corr_outcome.index, corr_outcome.values,
               color=[COLOR_YES if v > 0 else COLOR_NO for v in corr_outcome.values])
ax.bar_label(bars, fmt="%.2f", padding=3)
ax.set_xlabel("Correlation with Outcome")
ax.set_title("Correlation of Each Feature with Diabetes")
ax.set_xlim(0, corr_outcome.max() + 0.1)
plt.tight_layout()
plt.show()

# Chart 7: scatter plots coloured by outcome
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for outcome, colour, label in [(0, COLOR_NO, "No diabetes"), (1, COLOR_YES, "Diabetes")]:
    part = df[df["Outcome"] == outcome]
    ax[0].scatter(part["BMI"], part["Glucose"], c=colour, alpha=0.55, s=22, label=label)
    ax[1].scatter(part["Age"], part["Glucose"], c=colour, alpha=0.55, s=22, label=label)
ax[0].set_xlabel("BMI"); ax[0].set_ylabel("Glucose"); ax[0].set_title("Glucose vs BMI")
ax[1].set_xlabel("Age"); ax[1].set_ylabel("Glucose"); ax[1].set_title("Glucose vs Age")
ax[0].legend(); ax[1].legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# Step 11: Group-wise analysis with bar charts and line charts
# ---------------------------------------------------------------
heading("STEP 11: DIABETES RATE BY GROUP")
# Group the continuous columns into categories so patterns are easy to read
df["AgeGroup"] = pd.cut(df["Age"], bins=[20, 30, 40, 50, 60, 100],
                        labels=["21-30", "31-40", "41-50", "51-60", "60+"])
df["BMICategory"] = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, 100],
                           labels=["Underweight", "Normal", "Overweight", "Obese"])
df["PregnancyGroup"] = pd.cut(df["Pregnancies"], bins=[-1, 1, 3, 5, 7, 9, 20],
                              labels=["0-1", "2-3", "4-5", "6-7", "8-9", "10+"])


def rate_table(group_col):
    """Percentage of diabetic patients and number of patients in each group."""
    g = df.groupby(group_col, observed=True)["Outcome"]
    return pd.DataFrame({"patients": g.count(), "diabetes_rate_%": (g.mean() * 100).round(1)})


age_rate = rate_table("AgeGroup")
bmi_rate = rate_table("BMICategory")
preg_rate = rate_table("PregnancyGroup")
for name, table in [("Age group", age_rate), ("BMI category", bmi_rate), ("Pregnancies", preg_rate)]:
    print("\nDiabetes rate by", name)
    print(table)

# Chart 8: bar charts - diabetes rate by BMI category and by age group
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for a, table, title in [(ax[0], bmi_rate, "Diabetes Rate by BMI Category"),
                        (ax[1], age_rate, "Diabetes Rate by Age Group")]:
    labels = ["%s\n(n=%d)" % (idx, n) for idx, n in zip(table.index, table["patients"])]
    bars = a.bar(labels, table["diabetes_rate_%"], color="#6a9fd4", edgecolor="white")
    a.bar_label(bars, fmt="%.1f%%")
    a.set_title(title)
    a.set_ylabel("Patients with diabetes (%)")
    a.set_ylim(0, max(table["diabetes_rate_%"]) + 15)
plt.tight_layout()
plt.show()

# Chart 9: line charts - trend over groups
mean_glucose = df.groupby(["AgeGroup", "Outcome"], observed=True)["Glucose"].mean().unstack()
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
labels = ["%s\n(n=%d)" % (idx, n) for idx, n in zip(preg_rate.index, preg_rate["patients"])]
ax[0].plot(labels, preg_rate["diabetes_rate_%"], marker="o", color=COLOR_YES, linewidth=2)
ax[0].set_title("Diabetes Rate vs Number of Pregnancies")
ax[0].set_ylabel("Patients with diabetes (%)")
ax[0].set_xlabel("Number of pregnancies")
ax[0].grid(alpha=0.3)

ax[1].plot(mean_glucose.index.astype(str), mean_glucose[0], marker="o", color=COLOR_NO,
           linewidth=2, label="No diabetes")
ax[1].plot(mean_glucose.index.astype(str), mean_glucose[1], marker="o", color=COLOR_YES,
           linewidth=2, label="Diabetes")
ax[1].set_title("Average Glucose by Age Group")
ax[1].set_ylabel("Mean glucose")
ax[1].set_xlabel("Age group")
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# Step 12: Interpret the results (all numbers below are computed from the data)
# ---------------------------------------------------------------
heading("STEP 12: KEY INSIGHTS FROM THE EDA")

diabetic_pct = df["Outcome"].mean() * 100
top_corr = corr_outcome.sort_values(ascending=False).head(3)
med_glucose = df_observed.groupby("Outcome")["Glucose"].median()
med_bmi = df_observed.groupby("Outcome")["BMI"].median()
med_age = df_observed.groupby("Outcome")["Age"].median()

# strongest relationship between two features (ignoring Outcome)
feat_corr = corr.loc[feature_cols, feature_cols]
pairs = feat_corr.where(np.triu(np.ones(feat_corr.shape, dtype=bool), k=1)).stack()
best_pair = pairs.abs().idxmax()

worst_outlier_col = outlier_table.sort_values("Outliers", ascending=False).iloc[0]
top_missing = hidden_missing_pct.sort_values(ascending=False)

print("1. Dataset: %d patients, %d features. %.1f%% have diabetes, so the classes are "
      "imbalanced (about 1 in 3)." % (len(df), len(feature_cols), diabetic_pct))
print("2. Data quality: no NaN cells and %d duplicates, but impossible zeros were hidden "
      "missing values: %s (%.1f%%) and %s (%.1f%%) were affected the most. They were "
      "filled with the median." % (n_dup, top_missing.index[0], top_missing.iloc[0],
                                   top_missing.index[1], top_missing.iloc[1]))
print("3. Outliers: %s has the most (%d values, %.1f%%). They are real readings, so they "
      "were kept; skewed columns like Insulin are the reason the median was used."
      % (worst_outlier_col["Column"], worst_outlier_col["Outliers"], worst_outlier_col["Percent"]))
print("4. Strongest correlations with diabetes: %s (r=%.2f), %s (r=%.2f), %s (r=%.2f)."
      % (top_corr.index[0], top_corr.iloc[0], top_corr.index[1], top_corr.iloc[1],
         top_corr.index[2], top_corr.iloc[2]))
print("5. Median glucose is %.0f for diabetic vs %.0f for non-diabetic patients; median BMI "
      "is %.1f vs %.1f; median age is %.0f vs %.0f." % (
          med_glucose[1], med_glucose[0], med_bmi[1], med_bmi[0], med_age[1], med_age[0]))
# (very small groups, e.g. only 4 underweight patients, are left out of the best/worst comparison)
bmi_big = bmi_rate[bmi_rate["patients"] >= 20]["diabetes_rate_%"]
age_big = age_rate[age_rate["patients"] >= 20]["diabetes_rate_%"]
print("6. By BMI category (groups with 20+ patients): highest diabetes rate is %s (%.1f%%), "
      "lowest is %s (%.1f%%)." % (bmi_big.idxmax(), bmi_big.max(), bmi_big.idxmin(), bmi_big.min()))
print("7. By age group (groups with 20+ patients): highest diabetes rate is %s (%.1f%%), "
      "lowest is %s (%.1f%%)." % (age_big.idxmax(), age_big.max(), age_big.idxmin(), age_big.min()))
print("8. Strongest relationship between two features: %s and %s (r=%.2f), so some "
      "features carry overlapping information." % (best_pair[0], best_pair[1], pairs[best_pair]))
print("\nLimitations: correlation does not prove causation; Insulin and SkinThickness had many "
      "missing values filled by the median, which reduces their variation; and the data only "
      "covers female patients (aged 21+) of Pima Indian heritage, so the findings may not "
      "apply to other groups.")
