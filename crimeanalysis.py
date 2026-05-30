# =============================================================================
# Indian Crime Dataset Analysis
# Dataset: crime_dataset_india.csv (Kaggle)
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# =============================================================================
# 1. LOAD & CLEAN
# =============================================================================

df = pd.read_csv("crime_dataset_india.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape         : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Columns       : {list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Parse dates
for col in ["Date Reported", "Date of Occurrence", "Date Case Closed"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# Extract year and month from Date of Occurrence
df["Year"]  = df["Date of Occurrence"].dt.year
df["Month"] = df["Date of Occurrence"].dt.month

# Binary target for ML
df["Case Closed Binary"] = (df["Case Closed"] == "Yes").astype(int)

print("\nCleaning done. Proceeding to analysis...\n")

# =============================================================================
# 2. CRIME DISTRIBUTION BY CITY  (Top 10)
# =============================================================================

top_cities = df["City"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=top_cities.values, y=top_cities.index, ax=ax, palette="Blues_r")
ax.set_title("Top 10 Cities by Total Reported Crimes", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Crimes")
ax.set_ylabel("City")
for bar, val in zip(ax.patches, top_cities.values):
    ax.text(val + 30, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("01_crimes_by_city.png")
plt.show()
print("Saved: 01_crimes_by_city.png")

# =============================================================================
# 3. CRIME DOMAIN BREAKDOWN
# =============================================================================

domain_counts = df["Crime Domain"].value_counts()

fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(domain_counts.values, labels=domain_counts.index, autopct="%1.1f%%",
       startangle=140, colors=sns.color_palette("pastel"))
ax.set_title("Crime Domain Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("02_crime_domain_pie.png")
plt.show()
print("Saved: 02_crime_domain_pie.png")

# =============================================================================
# 4. TOP 10 CRIME TYPES
# =============================================================================

top_crimes = df["Crime Description"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=top_crimes.values, y=top_crimes.index, ax=ax, palette="Reds_r")
ax.set_title("Top 10 Crime Types", fontsize=14, fontweight="bold")
ax.set_xlabel("Count")
ax.set_ylabel("Crime Type")
plt.tight_layout()
plt.savefig("03_top_crime_types.png")
plt.show()
print("Saved: 03_top_crime_types.png")

# =============================================================================
# 5. CRIME TREND OVER TIME (by Year)
# =============================================================================

yearly = df.groupby("Year").size().reset_index(name="Count")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(yearly["Year"], yearly["Count"], marker="o", linewidth=2, color="#e74c3c")
ax.fill_between(yearly["Year"], yearly["Count"], alpha=0.15, color="#e74c3c")
ax.set_title("Yearly Crime Trend", fontsize=14, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Crimes")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("04_yearly_trend.png")
plt.show()
print("Saved: 04_yearly_trend.png")

# =============================================================================
# 6. VICTIM AGE DISTRIBUTION
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df["Victim Age"].dropna(), bins=30, kde=True, ax=ax, color="#3498db")
ax.set_title("Victim Age Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Age")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("05_victim_age.png")
plt.show()
print("Saved: 05_victim_age.png")

# =============================================================================
# 7. VICTIM GENDER BREAKDOWN
# =============================================================================

gender_counts = df["Victim Gender"].value_counts()

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=ax,
            palette=["#3498db", "#e91e8c", "#95a5a6"])
ax.set_title("Crimes by Victim Gender", fontsize=14, fontweight="bold")
ax.set_xlabel("Gender")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("06_victim_gender.png")
plt.show()
print("Saved: 06_victim_gender.png")

# =============================================================================
# 8. WEAPON USAGE
# =============================================================================

weapon_counts = df["Weapon Used"].value_counts().dropna()

fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(x=weapon_counts.values, y=weapon_counts.index, ax=ax, palette="Oranges_r")
ax.set_title("Weapons Used in Crimes", fontsize=14, fontweight="bold")
ax.set_xlabel("Count")
ax.set_ylabel("Weapon")
plt.tight_layout()
plt.savefig("07_weapons.png")
plt.show()
print("Saved: 07_weapons.png")

# =============================================================================
# 9. CASE RESOLUTION RATE BY CITY  (Top 10 cities)
# =============================================================================

top10 = df["City"].value_counts().head(10).index
df_top = df[df["City"].isin(top10)]
resolution = df_top.groupby("City")["Case Closed Binary"].mean().sort_values() * 100

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=resolution.values, y=resolution.index, ax=ax, palette="Greens_r")
ax.set_title("Case Resolution Rate by City (Top 10)", fontsize=14, fontweight="bold")
ax.set_xlabel("Resolution Rate (%)")
ax.set_ylabel("City")
ax.axvline(50, color="red", linestyle="--", linewidth=1, label="50% mark")
ax.legend()
plt.tight_layout()
plt.savefig("08_resolution_rate.png")
plt.show()
print("Saved: 08_resolution_rate.png")

# =============================================================================
# 10. HEATMAP — Crime Type vs City  (Top 10 each)
# =============================================================================

top_crime_types = df["Crime Description"].value_counts().head(10).index
heatmap_data = (
    df[df["City"].isin(top10) & df["Crime Description"].isin(top_crime_types)]
    .groupby(["City", "Crime Description"])
    .size()
    .unstack(fill_value=0)
)

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.5)
ax.set_title("Crime Type vs City Heatmap (Top 10 each)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("09_heatmap_city_crime.png")
plt.show()
print("Saved: 09_heatmap_city_crime.png")

# =============================================================================
# 11. ML MODEL — Predict Case Closure
# =============================================================================

print("\n" + "=" * 60)
print("ML MODEL: Predicting Case Closure")
print("=" * 60)

# Feature engineering
ml_df = df[["City", "Crime Description", "Victim Age",
            "Victim Gender", "Weapon Used", "Crime Domain",
            "Police Deployed", "Case Closed Binary"]].dropna()

le = LabelEncoder()
cat_cols = ["City", "Crime Description", "Victim Gender", "Weapon Used", "Crime Domain"]
for col in cat_cols:
    ml_df[col] = le.fit_transform(ml_df[col].astype(str))

X = ml_df.drop("Case Closed Binary", axis=1)
y = ml_df["Case Closed Binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Closed", "Closed"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Not Closed", "Closed"],
            yticklabels=["Not Closed", "Closed"])
ax.set_title("Confusion Matrix — Case Closure Prediction", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("10_confusion_matrix.png")
plt.show()
print("Saved: 10_confusion_matrix.png")

# Feature importance
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind="barh", ax=ax, color="#2ecc71")
ax.set_title("Feature Importance — Random Forest", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("11_feature_importance.png")
plt.show()
print("Saved: 11_feature_importance.png")

# =============================================================================
# DONE
# =============================================================================

print("\n" + "=" * 60)
print("All analyses complete! Charts saved as PNG files.")
print("=" * 60)
