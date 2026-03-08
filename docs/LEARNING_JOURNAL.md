# Credit Risk Scoring Model — Learning Journal

A step-by-step record of everything done in this project, with explanations of **why** each decision was made.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Environment Setup](#2-environment-setup)
3. [Dataset](#3-dataset)
4. [Phase 1 — EDA](#4-phase-1--exploratory-data-analysis-eda)
5. [Phase 2 — Feature Engineering](#5-phase-2--feature-engineering)
6. [Phase 3 — Modeling](#6-phase-3--modeling)
7. [Phase 4 — Explainability](#7-phase-4--explainability-shap)
8. [Phase 5 — Dashboard](#8-phase-5--dashboard)
9. [Phase 6 — Streamlit App](#9-phase-6--streamlit-app)

---

## 1. Project Overview

### Business Problem
Banks need to decide who to give loans to. Giving a loan to someone who defaults = financial loss. Rejecting a good borrower = lost revenue opportunity. A credit scoring model helps automate and standardize this decision.

### Goal
Build an ML model that:
- Predicts whether an applicant is **Good Credit (1)** or **Bad Credit (2)**
- Explains *why* it made a prediction (not a black box)
- Surfaces insights through a dashboard
- Allows real-time prediction via a Streamlit web app

### The Cost Matrix — Why This Project Is Different From Standard Classification
From the dataset documentation:

```
        Predicted Good   Predicted Bad
Actual Good      0              1
Actual Bad       5              0
```

Missing a bad borrower (false negative) costs **5x more** than wrongly rejecting a good one (false positive). This means:
- We should NOT just optimize for accuracy
- We need to tune our decision threshold to reduce false negatives
- This is a real-world constraint that separates a portfolio project from a production system

---

## 2. Environment Setup

### Why a Virtual Environment?
- Packages installed inside `venv/` are isolated from the system Python
- Prevents version conflicts between projects
- Anyone can recreate the exact environment using `requirements.txt`
- Industry standard practice

### Steps Performed
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Why These Libraries?

| Library | Purpose |
|---|---|
| `pandas` | Load, inspect, and manipulate tabular data |
| `numpy` | Numerical operations, array math |
| `scikit-learn` | Logistic Regression, Random Forest, evaluation metrics |
| `imbalanced-learn` | SMOTE — synthetic oversampling to fix class imbalance |
| `shap` | Explain model predictions (feature contribution per prediction) |
| `matplotlib / seaborn` | Plotting for EDA and model evaluation |
| `streamlit` | Turn the model into an interactive web app with minimal code |
| `xgboost` | Gradient boosting — powerful tree-based model for comparison |
| `jupyter` | Interactive notebooks for exploration and analysis |

### Extra Step: Install `libomp`
XGBoost uses OpenMP for parallel CPU computation (trains multiple trees simultaneously). macOS doesn't ship with it.
```bash
brew install libomp
```

### Project Folder Structure
```
credit-risk-model/
├── data/
│   ├── raw/          ← Original dataset. NEVER edit this.
│   └── processed/    ← Cleaned, feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb         ← Exploratory Data Analysis
│   ├── 02_modeling.ipynb    ← Model training & evaluation
│   └── 03_insights.ipynb   ← Business insights & simulations
├── sql/              ← Analytical SQL queries
├── app/              ← Streamlit web app
├── dashboard/        ← Power BI / Tableau files
├── docs/             ← This learning journal + any reference docs
├── requirements.txt  ← Pinned dependencies
└── README.md         ← GitHub-facing project summary
```

**Why separate raw and processed data?**
Raw data is sacred. If a cleaning step goes wrong, you can always re-derive processed data from raw. You can never recreate raw data once overwritten.

---

## 3. Dataset

### Source
- **Name:** Statlog (German Credit Data)
- **Origin:** UCI Machine Learning Repository
- **Creator:** Prof. Dr. Hans Hofmann, Universität Hamburg
- **File downloaded:** `data/raw/german.data`
- **Documentation:** `data/raw/german.doc`

### Dataset Facts
- 1,000 rows (loan applicants)
- 20 features (mix of categorical and numerical)
- Binary target: `1` = Good Credit, `2` = Bad Credit
- No column headers in the raw file — codes map to meanings in the `.doc` file

### Column Reference

| # | Column Name | Type | Description |
|---|---|---|---|
| 1 | `checking_status` | Categorical | Status of existing checking account (balance range or none) |
| 2 | `duration` | Numerical | Loan duration in months |
| 3 | `credit_history` | Categorical | Past credit behavior (paid duly, delays, critical) |
| 4 | `purpose` | Categorical | What the loan is for (car, furniture, education, etc.) |
| 5 | `credit_amount` | Numerical | Loan amount in DM (Deutschmarks) |
| 6 | `savings_status` | Categorical | Savings account balance range or unknown |
| 7 | `employment` | Categorical | Years at current employer |
| 8 | `installment_commitment` | Numerical | Installment rate as % of disposable income |
| 9 | `personal_status` | Categorical | Sex and marital status combined |
| 10 | `other_parties` | Categorical | Co-applicant or guarantor present |
| 11 | `residence_since` | Numerical | Years at current residence |
| 12 | `property_magnitude` | Categorical | Most valuable owned property |
| 13 | `age` | Numerical | Applicant age in years |
| 14 | `other_payment_plans` | Categorical | Other installment plans (bank, stores, none) |
| 15 | `housing` | Categorical | Rent / own / free |
| 16 | `existing_credits` | Numerical | Number of existing credits at this bank |
| 17 | `job` | Categorical | Employment skill level |
| 18 | `num_dependents` | Numerical | Number of people financially dependent on applicant |
| 19 | `own_telephone` | Categorical | Has registered telephone or not |
| 20 | `foreign_worker` | Categorical | Is a foreign worker |
| 21 | `target` | Target | 1 = Good Credit, 2 = Bad Credit |

---

## 4. Phase 1 — Exploratory Data Analysis (EDA)

> *(In progress — steps will be added as we work through the notebook)*

### What is EDA and Why Do We Do It?
EDA = getting to know your data before modeling. You would never build a house without inspecting the land first. Similarly, you never train a model without understanding the data.

EDA answers:
- Are there missing values?
- What does the distribution of each feature look like?
- Is the target class balanced?
- Which features seem most related to the target?
- Are there outliers that could skew the model?

### Steps Completed
- [x] Load raw data with proper column names
- [x] Check shape, dtypes, nulls
- [x] Class balance — 700 Good (70%) vs 300 Bad (30%)
- [x] Distribution plots for numerical features
- [x] Box plots — numerical features split by target class
- [x] Count plots for categorical features
- [x] Correlation heatmap
- [x] Bad credit rate by checking status, credit history, purpose, savings, employment, job
- [x] Age group risk analysis + KDE density plot

### Key Findings from EDA

| Finding | Implication |
|---|---|
| 70% Good / 30% Bad | Moderate class imbalance — use SMOTE in Phase 2 |
| `checking_status = < 0 DM` has highest default rate | Strong predictor |
| `credit_history = delay in past / critical account` has high default rate | Strong predictor |
| Applicants aged 18–25 default more than older groups | Age is a useful feature |
| `credit_amount` and `duration` are right-skewed | Consider log transform in Phase 2 |
| No missing values | No imputation needed |
| `duration` and `credit_amount` are moderately correlated | Worth noting but not severe |

### Outputs Generated
- `data/processed/german_named.csv` — base dataset for Phase 2
- `data/processed/class_balance.png`
- `data/processed/numerical_distributions.png`
- `data/processed/boxplots_vs_target.png`
- `data/processed/categorical_vs_target.png`
- `data/processed/bad_rate_by_feature.png`
- `data/processed/age_vs_risk.png`
- `data/processed/correlation_heatmap.png`

### Concepts Introduced
- **Class imbalance** — when one class has far fewer samples than another. Accuracy is misleading here.
- **SMOTE** — Synthetic Minority Oversampling Technique. Creates synthetic samples of the minority class (Bad Credit) to balance the dataset.
- **Box plot** — shows median, spread, and outliers. Split by target = instant visual signal of feature importance.
- **Bad rate** — % of applicants in a group who defaulted. Normalized for group size — fairer than raw counts.
- **KDE plot** — Kernel Density Estimate. A smooth version of a histogram. Great for comparing distributions between two groups.

---

## 5. Phase 2 — Feature Engineering

### What is Feature Engineering and Why Do We Do It?
Raw data is never model-ready. Feature engineering transforms it into a form that algorithms can actually learn from.

### Steps Completed
- [x] Recode target: `1 (Good) → 0`, `2 (Bad) → 1`
- [x] Ordinal encoding for ordered categoricals
- [x] One-hot encoding for nominal categoricals
- [x] Created 3 new interaction features
- [x] Log transform for skewed features
- [x] Train / test split (80/20, stratified)
- [x] StandardScaler — fit on train, transform both
- [x] SMOTE on training set only
- [x] Saved all processed files + scaler

### Key Decisions & Why

**Ordinal Encoding (not one-hot) for ordered columns**
- `checking_status`, `savings_status`, `employment`, `job` have a natural rank
- Ordinal integers preserve that order so the model can learn from it
- One-hot would treat `unemployed` and `>=7yrs employed` as unrelated

**One-Hot Encoding for nominal columns**
- `purpose`, `housing`, `credit_history` etc. have no order
- `drop_first=True` prevents multicollinearity (the "dummy variable trap")

**New Features Created**
| Feature | Formula | Purpose |
|---|---|---|
| `monthly_rate` | `credit_amount / duration` | Monthly repayment burden |
| `age_employment_ratio` | `age / (employment + 1)` | Proportion of life employed |
| `credit_to_age_ratio` | `credit_amount / age` | Loan size relative to life stage |

**Log Transform (`log1p`)**
- `credit_amount`, `duration`, `monthly_rate`, `credit_to_age_ratio` were right-skewed
- Logistic Regression assumes linear relationship with log-odds — skew violates this
- `log1p` (not `log`) handles zero values safely

**Why Split BEFORE Scaling and SMOTE**
- Scaling after split prevents data leakage (test stats must never influence training)
- SMOTE after split prevents synthetic test samples appearing in training
- Correct order: `Split → Scale train → SMOTE train → Train → Evaluate on raw test`

**SMOTE (Synthetic Minority Oversampling Technique)**
- Creates new synthetic Bad Credit samples by interpolating between existing ones
- Better than simple duplication (model learns pattern, not specific cases)
- Better than undersampling Good Credit (we'd lose real data)
- Applied only to training set — test set stays at real-world 70/30 ratio

### Concepts Introduced
- **Data leakage** — when test data influences training, making evaluation dishonest
- **Multicollinearity** — when two features carry the same information, confusing linear models
- **Dummy variable trap** — keeping all one-hot columns creates perfect multicollinearity
- **Stratified split** — ensures both train and test maintain the same class ratio
- **StandardScaler** — subtracts mean, divides by std. Result: mean=0, std=1 per feature

### Outputs Generated
- `data/processed/X_train.csv` — 1,120 rows (SMOTE balanced), all features scaled
- `data/processed/X_test.csv` — 200 rows, scaled
- `data/processed/y_train.csv` — 1,120 labels
- `data/processed/y_test.csv` — 200 labels (real distribution)
- `data/processed/scaler.pkl` — fitted scaler for Streamlit app
- `data/processed/log_transform.png`
- `data/processed/smote_balance.png`

---

## 6. Phase 3 — Modeling

### Steps Completed
- [x] Trained Logistic Regression (C=1.0, max_iter=1000)
- [x] Trained Random Forest (300 trees, min_samples_leaf=2)
- [x] 5-fold cross-validation ROC-AUC on training set
- [x] ROC curve comparison
- [x] Precision-Recall curve comparison
- [x] Confusion matrices at default threshold (0.5)
- [x] Cost-sensitive threshold tuning using 5:1 cost matrix
- [x] Final evaluation at optimal threshold
- [x] Saved both models + metadata

### Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| ROC-AUC | 0.7919 | **0.8035** |
| Optimal Threshold | 0.23 | **0.22** |

**Winner: Random Forest** with AUC = 0.8035

### Key Concepts

**ROC-AUC**
- Measures how well the model ranks bad borrowers above good ones
- Threshold-independent — fair comparison between models
- 0.5 = random guessing, 1.0 = perfect
- Better than accuracy on imbalanced data

**Precision vs Recall trade-off**
- Precision: when we flag someone as risky, how often are we right?
- Recall: of all actual bad borrowers, how many did we catch?
- Lowering the threshold increases recall, decreases precision
- The cost matrix (FN costs 5×) means we should prioritize recall

**Cost-Sensitive Threshold Tuning**
- Default threshold of 0.5 treats all errors equally — wrong for this problem
- We scan every threshold from 0.01–0.99 and compute: `5×FN + 1×FP`
- Both models converge to a threshold around 0.22–0.23 (much lower than 0.5)
- This catches more bad borrowers at the cost of more false alarms — correct trade-off

**Cross-Validation**
- Train on 4 folds, evaluate on 1, rotate 5 times
- Gives a more reliable estimate of performance than a single train/test split
- Prevents lucky/unlucky splits from distorting results

### Outputs Generated
- `data/processed/roc_curve.png`
- `data/processed/precision_recall_curve.png`
- `data/processed/confusion_matrices.png`
- `data/processed/threshold_tuning.png`
- `data/processed/lr_model.pkl`
- `data/processed/rf_model.pkl`
- `data/processed/model_meta.json`

---

## 7. Phase 4 — Explainability (SHAP)

### Steps Completed
- [x] Computed SHAP values using TreeExplainer on test set
- [x] Global feature importance bar chart (mean |SHAP|)
- [x] Beeswarm summary plot — direction + magnitude for all features
- [x] Dependence plots for top 2 features
- [x] Waterfall plots for high-risk and low-risk applicants
- [x] Business insights summary — top drivers, direction, group comparisons
- [x] Saved explainer + SHAP values for Streamlit app

### Key Concepts

**SHAP (SHapley Additive exPlanations)**
- Based on Shapley values from cooperative game theory
- Each feature gets a fair share of credit/blame for a prediction
- Positive SHAP → pushes prediction toward Bad Credit
- Negative SHAP → pushes prediction toward Good Credit
- Sum of all SHAP values = model output (log-odds) − baseline

**TreeExplainer vs Generic Explainer**
- Generic SHAP estimates contributions via random permutations — slow
- TreeExplainer uses tree structure directly → exact values, much faster
- Always prefer TreeExplainer for Random Forest, XGBoost, LightGBM

**Beeswarm Plot**
- Y-axis: features ranked by importance
- X-axis: SHAP value (direction of impact)
- Color: actual feature value (red=high, blue=low)
- Each dot = one applicant
- Most information-dense SHAP chart — shows direction + magnitude + distribution

**Waterfall Plot**
- Shows one individual prediction broken down feature-by-feature
- Starts from baseline (average prediction across all applicants)
- Each bar = one feature's contribution
- Final bar = model's predicted probability
- Critical for explainable AI in regulated industries like banking

### Outputs Generated
- `data/processed/shap_global_importance.png`
- `data/processed/shap_beeswarm.png`
- `data/processed/shap_dependence.png`
- `data/processed/shap_waterfall_highrisk.png`
- `data/processed/shap_waterfall_lowrisk.png`
- `data/processed/shap_explainer.pkl`
- `data/processed/shap_values_test.csv`

---

## 8. Phase 5 — Business Insights

### Steps Completed
- [x] Risk score distribution — histogram by actual class + default rate by risk bucket
- [x] Risk concentration (Lorenz-style) — cumulative default capture curve + decile analysis
- [x] Threshold simulation — approval/rejection trade-off across all thresholds
- [x] Segment default rates — checking, savings, employment, purpose, age, credit amount
- [x] High-risk profile analysis — top combinations of features
- [x] Final business summary — quantified, actionable findings

### Key Numbers
| Insight | Value |
|---|---|
| Model ROC-AUC | 0.8035 |
| Top 20% riskiest → % defaults captured | 46.7% |
| Defaults caught at threshold 0.50 | see notebook |
| Extra defaults caught at optimal threshold | see notebook |

### Key Concepts

**Risk Concentration (Lorenz Curve)**
- Sort applicants from riskiest to safest
- Plot: "what % of defaults do we catch if we reject the top X% of applicants?"
- A perfect model: rejecting 30% captures 100% of defaults
- Our model: top 20% captures ~47% of defaults — far better than random (20%)

**Decile Analysis**
- Split applicants into 10 equal-sized groups (deciles) by risk score
- D10 (highest risk) should have much higher default rate than D1 (lowest risk)
- Clean separation across deciles = model is well-calibrated and useful

**Threshold Simulation**
- Every threshold is a business policy decision, not just a model parameter
- Lower threshold → catch more defaults → reject more good borrowers
- The simulation makes this trade-off visible and quantifiable
- Business stakeholders can then choose based on their risk appetite

### Outputs Generated
- `data/processed/risk_score_distribution.png`
- `data/processed/risk_concentration.png`
- `data/processed/threshold_simulation.png`
- `data/processed/segment_default_rates.png`
- `data/processed/threshold_simulation.csv`

---

## 9. Phase 6 — Streamlit App

### Steps Completed
- [x] Built `app/app.py` — full Streamlit web application
- [x] Input form with all 20 applicant features (human-readable dropdowns)
- [x] Preprocessing pipeline replicates Phase 2 exactly (ordinal, one-hot, log, scale)
- [x] Column alignment — handles missing one-hot columns via `reindex(fill_value=0)`
- [x] Real-time prediction with probability + approve/reject decision
- [x] Visual risk gauge bar
- [x] SHAP waterfall plot per prediction (live computation)
- [x] Expandable raw input summary table

### How to Run
```bash
cd credit-risk-model/app
source ../venv/bin/activate
streamlit run app.py
# Opens at http://localhost:8501
```

### Key Engineering Decisions

**`@st.cache_resource` for model loading**
- Models are loaded once on startup, not on every user interaction
- Without this, every prediction would reload 300 trees from disk — very slow

**Preprocessing must exactly mirror training**
- Same ordinal maps, same nominal columns, same log transforms, same scaler
- Any deviation = the model sees a different distribution than it was trained on → bad predictions
- This is why we saved the scaler as `scaler.pkl` — reuse it, never refit

**Column alignment with `reindex(fill_value=0)`**
- A single row won't have all one-hot categories (e.g., if purpose=A43, all other purpose_* cols are absent)
- `reindex(columns=FEATURE_NAMES, fill_value=0)` adds all missing columns as 0
- Without this the model receives the wrong number of features → error

**SHAP on live input**
- The explainer computes SHAP values for the user's specific applicant in real time
- Shows exactly which features drove that individual prediction
- This is what makes the app useful to a loan officer, not just a data scientist
