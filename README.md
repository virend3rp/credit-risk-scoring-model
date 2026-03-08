# Credit Risk Scoring Model

A machine learning project that predicts loan default risk using the German Credit Dataset.

## Problem Statement
Banks need to assess the creditworthiness of loan applicants to minimize default risk. This model predicts whether an applicant is a **good or bad credit risk**, explains key risk drivers, and surfaces insights through an interactive dashboard.

## Dataset
- **Source:** [UCI German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **Size:** 1,000 applicants, 20 features
- **Target:** Binary — Good Credit (1) / Bad Credit (2)

## Tech Stack
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)

## Project Structure
```
credit-risk-model/
├── data/
│   ├── raw/          # Original dataset — never edited
│   └── processed/    # Cleaned, feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_insights.ipynb
├── sql/              # BigQuery/SQLite analytical queries
├── app/              # Streamlit application
├── dashboard/        # Power BI / Tableau files
├── requirements.txt
└── README.md
```

## Key Findings
*(To be updated after modeling)*

## How to Run
```bash
pip install -r requirements.txt
streamlit run app/app.py
```
