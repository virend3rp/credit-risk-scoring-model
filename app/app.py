import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="🏦",
    layout="wide"
)

# ─────────────────────────────────────────────
# Load model artefacts (cached so they load once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    with open('../data/processed/rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('../data/processed/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('../data/processed/shap_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    with open('../data/processed/model_meta.json') as f:
        meta = json.load(f)
    return model, scaler, explainer, meta

model, scaler, explainer, meta = load_artefacts()
THRESHOLD      = meta['rf_threshold']
FEATURE_NAMES  = meta['feature_names']

# ─────────────────────────────────────────────
# Preprocessing — mirrors Phase 2 exactly
# ─────────────────────────────────────────────
ORDINAL_MAPS = {
    'checking_status': {'A11': 0, 'A12': 1, 'A13': 2, 'A14': 3},
    'savings_status':  {'A65': 0, 'A61': 1, 'A62': 2, 'A63': 3, 'A64': 4},
    'employment':      {'A71': 0, 'A72': 1, 'A73': 2, 'A74': 3, 'A75': 4},
    'job':             {'A171': 0, 'A172': 1, 'A173': 2, 'A174': 3},
}

NOMINAL_COLS = [
    'credit_history', 'purpose', 'personal_status',
    'other_parties', 'property_magnitude', 'other_payment_plans',
    'housing', 'own_telephone', 'foreign_worker'
]

SKEWED_COLS = ['credit_amount', 'duration', 'monthly_rate', 'credit_to_age_ratio']

def preprocess(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    # Ordinal encoding
    for col, mapping in ORDINAL_MAPS.items():
        df[col] = df[col].map(mapping)

    # One-hot encoding
    df = pd.get_dummies(df, columns=NOMINAL_COLS, drop_first=True)

    # Engineered features
    df['monthly_rate']         = df['credit_amount'] / df['duration']
    df['age_employment_ratio'] = df['age'] / (df['employment'] + 1)
    df['credit_to_age_ratio']  = df['credit_amount'] / df['age']

    # Log transform
    for col in SKEWED_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # Align columns to training feature set (fill missing one-hot cols with 0)
    df = df.reindex(columns=FEATURE_NAMES, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)
    return pd.DataFrame(df_scaled, columns=FEATURE_NAMES)

# ─────────────────────────────────────────────
# SHAP waterfall — returns matplotlib figure
# ─────────────────────────────────────────────
def shap_waterfall(processed_df: pd.DataFrame) -> plt.Figure:
    sv = explainer.shap_values(processed_df)
    explanation = shap.Explanation(
        values        = sv[0, :, 1],
        base_values   = explainer.expected_value[1],
        data          = processed_df.values[0],
        feature_names = FEATURE_NAMES
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.tight_layout()
    return plt.gcf()

# ─────────────────────────────────────────────
# UI — Header
# ─────────────────────────────────────────────
st.title("🏦 Credit Risk Scoring Model")
st.markdown(
    "Enter the applicant's details below. The model will predict their **default risk** "
    "and explain which factors drove the decision."
)
st.markdown(f"**Model:** Random Forest  |  **ROC-AUC:** {meta['rf_auc']:.3f}  |  "
            f"**Decision Threshold:** {THRESHOLD:.2f} (cost-optimised)")
st.divider()

# ─────────────────────────────────────────────
# UI — Input Form (two columns)
# ─────────────────────────────────────────────
with st.form("applicant_form"):
    st.subheader("Applicant Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Financial**")
        credit_amount = st.number_input("Loan Amount (DM)", min_value=100, max_value=20000, value=3000, step=100)
        duration      = st.number_input("Loan Duration (months)", min_value=1, max_value=72, value=24)
        installment_commitment = st.slider("Installment as % of Income", 1, 4, 2)
        existing_credits = st.slider("Existing Credits at This Bank", 1, 4, 1)

        checking_status = st.selectbox("Checking Account Balance", options=[
            ("< 0 DM (overdrawn)", "A11"),
            ("0 – 200 DM",         "A12"),
            (">= 200 DM",          "A13"),
            ("No Checking Account","A14"),
        ], format_func=lambda x: x[0])

        savings_status = st.selectbox("Savings Account Balance", options=[
            ("Unknown / None",    "A65"),
            ("< 100 DM",          "A61"),
            ("100 – 500 DM",      "A62"),
            ("500 – 1000 DM",     "A63"),
            (">= 1000 DM",        "A64"),
        ], format_func=lambda x: x[0])

    with col2:
        st.markdown("**Employment & Personal**")
        age              = st.number_input("Age (years)", min_value=18, max_value=80, value=35)
        num_dependents   = st.slider("Number of Dependents", 1, 2, 1)
        residence_since  = st.slider("Years at Current Residence", 1, 4, 2)

        employment = st.selectbox("Employment Duration", options=[
            ("Unemployed",     "A71"),
            ("< 1 year",       "A72"),
            ("1 – 4 years",    "A73"),
            ("4 – 7 years",    "A74"),
            (">= 7 years",     "A75"),
        ], format_func=lambda x: x[0])

        job = st.selectbox("Job Type", options=[
            ("Unskilled – Non-Resident", "A171"),
            ("Unskilled – Resident",     "A172"),
            ("Skilled / Official",       "A173"),
            ("Management / Self-Employed","A174"),
        ], format_func=lambda x: x[0])

        personal_status = st.selectbox("Personal Status & Sex", options=[
            ("Male – Divorced/Separated",     "A91"),
            ("Female – Divorced/Separated/Married", "A92"),
            ("Male – Single",                 "A93"),
            ("Male – Married/Widowed",        "A94"),
        ], format_func=lambda x: x[0])

    with col3:
        st.markdown("**Loan & Property**")
        purpose = st.selectbox("Loan Purpose", options=[
            ("New Car",         "A40"),
            ("Used Car",        "A41"),
            ("Furniture",       "A42"),
            ("Radio / TV",      "A43"),
            ("Education",       "A46"),
            ("Business",        "A49"),
            ("Repairs",         "A45"),
            ("Retraining",      "A48"),
            ("Others",          "A410"),
        ], format_func=lambda x: x[0])

        credit_history = st.selectbox("Credit History", options=[
            ("No credits / All paid duly",   "A30"),
            ("All credits at bank paid",     "A31"),
            ("Existing credits paid duly",   "A32"),
            ("Delay in paying in the past",  "A33"),
            ("Critical / Other credits",     "A34"),
        ], format_func=lambda x: x[0])

        property_magnitude = st.selectbox("Most Valuable Property", options=[
            ("Real Estate",             "A121"),
            ("Building Society / Life Insurance", "A122"),
            ("Car or Other",            "A123"),
            ("Unknown / No Property",   "A124"),
        ], format_func=lambda x: x[0])

        housing = st.selectbox("Housing", options=[
            ("Own",      "A152"),
            ("Rent",     "A151"),
            ("For Free", "A153"),
        ], format_func=lambda x: x[0])

        other_payment_plans = st.selectbox("Other Installment Plans", options=[
            ("None",  "A143"),
            ("Bank",  "A141"),
            ("Stores","A142"),
        ], format_func=lambda x: x[0])

        other_parties = st.selectbox("Other Parties (Guarantor / Co-applicant)", options=[
            ("None",          "A101"),
            ("Co-applicant",  "A102"),
            ("Guarantor",     "A103"),
        ], format_func=lambda x: x[0])

        own_telephone = st.selectbox("Telephone Registered", options=[
            ("No",  "A191"),
            ("Yes", "A192"),
        ], format_func=lambda x: x[0])

        foreign_worker = st.selectbox("Foreign Worker", options=[
            ("Yes", "A201"),
            ("No",  "A202"),
        ], format_func=lambda x: x[0])

    submitted = st.form_submit_button("🔍 Predict Credit Risk", use_container_width=True)

# ─────────────────────────────────────────────
# Prediction & Results
# ─────────────────────────────────────────────
if submitted:
    raw_input = {
        'checking_status':        checking_status[1],
        'duration':               duration,
        'credit_history':         credit_history[1],
        'purpose':                purpose[1],
        'credit_amount':          credit_amount,
        'savings_status':         savings_status[1],
        'employment':             employment[1],
        'installment_commitment': installment_commitment,
        'personal_status':        personal_status[1],
        'other_parties':          other_parties[1],
        'residence_since':        residence_since,
        'property_magnitude':     property_magnitude[1],
        'age':                    age,
        'other_payment_plans':    other_payment_plans[1],
        'housing':                housing[1],
        'existing_credits':       existing_credits,
        'job':                    job[1],
        'num_dependents':         num_dependents,
        'own_telephone':          own_telephone[1],
        'foreign_worker':         foreign_worker[1],
    }

    processed = preprocess(raw_input)
    prob_default = model.predict_proba(processed)[0, 1]
    decision     = "REJECT" if prob_default >= THRESHOLD else "APPROVE"

    st.divider()
    st.subheader("📊 Prediction Result")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.metric("Default Probability", f"{prob_default:.1%}")

    with res_col2:
        st.metric("Decision Threshold", f"{THRESHOLD:.2f}")

    with res_col3:
        if decision == "APPROVE":
            st.success(f"✅ Decision: APPROVE", icon="✅")
        else:
            st.error(f"❌ Decision: REJECT", icon="❌")

    # Risk gauge bar
    st.markdown("**Risk Score**")
    bar_color = "#4CAF50" if prob_default < 0.4 else "#FF9800" if prob_default < 0.6 else "#F44336"
    st.markdown(
        f"""
        <div style="background:#eee;border-radius:8px;height:20px;width:100%">
          <div style="background:{bar_color};width:{prob_default*100:.1f}%;height:20px;
                      border-radius:8px;text-align:right;padding-right:6px;
                      line-height:20px;color:white;font-size:12px;font-weight:bold">
            {prob_default:.1%}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # SHAP explanation
    st.divider()
    st.subheader("🔍 Why This Decision? (SHAP Explanation)")
    st.markdown(
        "Each bar shows how much a feature **pushed the risk score up (red)** "
        "or **down (blue)**. The final score is the sum of all contributions plus the baseline."
    )

    with st.spinner("Computing explanation..."):
        fig = shap_waterfall(processed)
        st.pyplot(fig)
        plt.close()

    # Raw input summary
    with st.expander("📋 View Submitted Applicant Data"):
        display = {
            "Loan Amount (DM)":       credit_amount,
            "Duration (months)":      duration,
            "Age":                    age,
            "Checking Account":       checking_status[0],
            "Savings Account":        savings_status[0],
            "Employment":             employment[0],
            "Credit History":         credit_history[0],
            "Loan Purpose":           purpose[0],
            "Housing":                housing[0],
            "Job":                    job[0],
            "Installment % of Income":installment_commitment,
            "Existing Credits":       existing_credits,
        }
        st.table(pd.DataFrame(display.items(), columns=["Field", "Value"]))
