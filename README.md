# Indian Crime Data Analysis

An exploratory data analysis and machine learning project on crime patterns across major Indian cities, using a dataset of 40,000+ reported cases.
This project analyzes crime trends, victim demographics, weapon usage, and case resolution rates across 29 Indian cities. It also builds a Random Forest classifier to predict whether a reported case will be closed.

---

## Analyses

| # | Analysis |
|---|----------|
| 1 | Top 10 cities by total reported crimes |
| 2 | Crime domain breakdown (Violent, Traffic, Fire, Other) |
| 3 | Top 10 crime types |
| 4 | Yearly crime trend |
| 5 | Victim age distribution |
| 6 | Victim gender breakdown |
| 7 | Weapons used in crimes |
| 8 | Case resolution rate by city |
| 9 | Crime type vs city heatmap |
| 10 | ML model — predicting case closure (Random Forest) |
| 11 | Feature importance analysis |

---

## Dataset

- **Source:** [Kaggle — Indian Crime Dataset](https://www.kaggle.com/)
- **Size:** 40,160 records × 14 features
- **Cities covered:** 29 major Indian cities including Delhi, Mumbai, Bangalore, Chennai, Hyderabad
- **Features:** City, Crime Type, Crime Domain, Victim Age, Victim Gender, Weapon Used, Police Deployed, Case Closed Status, Dates

---

## Tech Stack

- **Python 3**
- **pandas** — data loading and cleaning
- **matplotlib & seaborn** — visualizations
- **scikit-learn** — Random Forest classifier, label encoding, train/test split


## Sample Findings

- **Delhi** has the highest number of reported crimes among all cities
- **Burglary** and **Vandalism** are the most frequently reported crime types
- Case resolution rates vary significantly across cities
- The Random Forest model predicts case closure with ~50% balanced accuracy, with **Police Deployed** and **Victim Age** being the most important features

---

## Project Structure

```
india-crime-analysis/
│
├── crime_analysis.py          # Main analysis script
├── crime_dataset_india.csv    # Dataset (download from Kaggle)
├── README.md                  # Project documentation
├── 01_crimes_by_city.png
├── 02_crime_domain_pie.png
├── 03_top_crime_types.png
├── 04_yearly_trend.png
├── 05_victim_age.png
├── 06_victim_gender.png
├── 07_weapons.png
├── 08_resolution_rate.png
├── 09_heatmap_city_crime.png
├── 10_confusion_matrix.png
└── 11_feature_importance.png
```

---
Feel free to fork, star ⭐, or raise issues!
