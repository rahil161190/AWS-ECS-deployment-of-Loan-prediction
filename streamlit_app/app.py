"""
LoanTap Loan Eligibility Predictor — Streamlit POC
----------------------------------------------------
Loads the real trained model (classifier.pkl) reproduced from
Rahil's LoanTap_LogisticRegression_Rahil_.ipynb — Pipeline #1
(SMOTE-balanced Logistic Regression, the best of the notebook's
three attempts: test F1 = 0.43 on the "Charged Off" class,
recall = 0.64).

User fills in the same raw fields as the original dataset,
the app replicates the notebook's exact preprocessing chain
(label encode -> emp_length map -> one-hot encode -> date
features -> target encode -> scale -> drop loan_amnt), then
calls the pickled LogisticRegression for a prediction.

loan_status: 0 = Fully Paid (ELIGIBLE), 1 = Charged Off (NOT ELIGIBLE)
"""

import pickle
import datetime
import numpy as np
import pandas as pd
import streamlit as st

# category_encoders must be installed (see requirements.txt) — pickle needs
# the TargetEncoder class importable at load time, even without an explicit
# import here, since pickle resolves it internally.

st.set_page_config(page_title="LoanTap Eligibility Predictor", page_icon="💰", layout="centered")

# ----------------------------
# LOAD MODEL BUNDLE
# ----------------------------
@st.cache_resource
def load_bundle():
    with open("classifier.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_bundle()
model = bundle["model"]
target_encoder = bundle["target_encoder"]
scaler = bundle["scaler"]
one_hot_encoder = bundle["one_hot_encoder"]
emp_length_map = bundle["emp_length_map"]
term_map = bundle["term_map"]
initial_list_status_map = bundle["initial_list_status_map"]

# NOTE: bundle["feature_order"] was found to be stale (missing 'int_rate'
# and 'grade' — captured from a different pipeline state than what the
# model actually trained on). The model's own feature_names_in_, recorded
# automatically by sklearn at .fit() time, is the reliable source of truth.
feature_order = list(model.feature_names_in_)  # 22 features, final model input

# The scaler was fit BEFORE loan_amnt was dropped (23 columns), so its
# expected input order differs from the model's. sklearn stores the
# fit-time column order automatically on any transformer fit with a
# DataFrame — no need for us to have saved this separately.
scaler_feature_order = list(scaler.feature_names_in_)  # 23 features, incl. loan_amnt

st.title("💰 LoanTap Loan Eligibility Predictor")
st.caption("Proof of Concept — powered by the actual Logistic Regression model trained on the LoanTap case study data.")

st.divider()

# ----------------------------
# INPUT FORM
# ----------------------------
with st.form("loan_form"):
    st.subheader("Applicant & Loan Details")

    col1, col2 = st.columns(2)

    with col1:
        loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=40000, value=10000, step=500)
        st.caption("The total amount the applicant wants to borrow.")

        term = st.selectbox("Loan Term", ["36 months", "60 months"])
        st.caption("How long the applicant has to repay the loan.")

        int_rate = st.slider("Interest Rate (%)", 5.0, 31.0, 13.0, step=0.1)
        st.caption("The annual interest rate offered on this loan.")

        installment = st.number_input("Monthly Installment ($)", min_value=15.0, max_value=1600.0, value=350.0, step=10.0)
        st.caption("The fixed amount the applicant would pay every month toward this loan.")

        grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        st.caption("LoanTap's internal risk rating — A is lowest risk, G is highest risk. Usually tied to the interest rate offered.")

        emp_length = st.selectbox("Employment Length", list(emp_length_map.keys()))
        st.caption("How many years the applicant has been continuously employed.")

        home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER", "NONE", "ANY"])
        st.caption("Whether the applicant rents, owns outright, or is still paying a mortgage on their home.")

        annual_inc = st.number_input("Annual Income ($)", min_value=0, value=65000, step=1000)
        st.caption("The applicant's self-reported yearly income before tax.")

        verification_status = st.selectbox("Income Verification", ["Not Verified", "Source Verified", "Verified"])
        st.caption("Whether LoanTap independently confirmed the applicant's stated income, and how thoroughly.")

        initial_list_status = st.selectbox("Initial Listing Status", ["f", "w"])
        st.caption("How the loan was first listed for funding: 'f' = fractional (many small investors), 'w' = whole (one investor). A backend platform detail, not tied to the applicant's own finances.")

    with col2:
        purpose = st.selectbox(
            "Loan Purpose",
            ["debt_consolidation", "credit_card", "home_improvement", "small_business",
             "major_purchase", "medical", "car", "vacation", "moving", "house",
             "renewable_energy", "educational", "wedding", "other"]
        )
        st.caption("The stated reason the applicant is borrowing this money.")

        dti = st.slider("Debt-to-Income Ratio (DTI)", 0.0, 45.0, 17.0, step=0.5)
        st.caption("Monthly debt payments as a % of monthly income. Lower is better — shows how much income is already committed elsewhere.")

        open_acc = st.number_input("Number of Open Credit Lines", min_value=0, max_value=90, value=11)
        st.caption("How many active credit accounts (credit cards, loans, etc.) the applicant currently has open.")

        pub_rec = st.selectbox("Any Public Derogatory Record?", ["No", "Yes"])
        st.caption("Whether the applicant has any negative public records on file, such as tax liens or civil judgments.")

        revol_bal = st.number_input("Revolving Balance ($)", min_value=0, value=16000, step=500)
        st.caption("The total outstanding balance across all the applicant's revolving credit (mainly credit cards).")

        revol_util = st.slider("Revolving Line Utilization (%)", 0.0, 150.0, 54.0, step=1.0)
        st.caption("How much of the applicant's available credit card limit is currently being used. Higher = more financial strain.")

        total_acc = st.number_input("Total Credit Accounts", min_value=1, max_value=160, value=25)
        st.caption("The total number of credit accounts the applicant has ever opened (open and closed).")

        mort_acc = st.number_input("Number of Mortgage Accounts", min_value=0, max_value=35, value=2)
        st.caption("How many mortgage accounts the applicant currently has.")

        pub_rec_bankruptcies = st.selectbox("Any Bankruptcy Record?", ["No", "Yes"])
        st.caption("Whether the applicant has ever filed for bankruptcy.")

        earliest_cr_year = st.number_input("Earliest Credit Line — Year", min_value=1950, max_value=2025, value=2005)
        st.caption("The year the applicant's oldest credit account was opened. Used to estimate credit history length.")

        issue_year = st.number_input("Loan Issue Year", min_value=2007, max_value=2030, value=datetime.date.today().year)
        st.caption("The year this loan would be issued.")

        issue_month = st.slider("Loan Issue Month", 1, 12, datetime.date.today().month)
        st.caption("The month this loan would be issued.")

    submitted = st.form_submit_button("🔍 Predict Eligibility", use_container_width=True)


# ----------------------------
# PREPROCESS + PREDICT
# (replicates the notebook's exact pipeline, in order)
# ----------------------------
def predict_eligibility(raw: dict):
    row = {}

    # 1. label-encoded / mapped fields
    row["term"] = term_map[raw["term"]]
    row["initial_list_status"] = initial_list_status_map[raw["initial_list_status"]]
    row["emp_length"] = emp_length_map[raw["emp_length"]]

    # 2. binarized public-record fields (0/1, matching notebook)
    row["pub_rec"] = 1 if raw["pub_rec"] == "Yes" else 0
    row["pub_rec_bankruptcies"] = 1 if raw["pub_rec_bankruptcies"] == "Yes" else 0

    # 3. passthrough numeric fields
    for k in ["loan_amnt", "int_rate", "installment", "annual_inc", "dti",
              "open_acc", "revol_bal", "revol_util", "total_acc", "mort_acc"]:
        row[k] = raw[k]

    # 4. categorical fields to be target-encoded later
    row["grade"] = raw["grade"]
    row["home_ownership"] = raw["home_ownership"]
    row["purpose"] = raw["purpose"]

    # 5. one-hot encode verification_status using the fitted encoder
    ohe_input = pd.DataFrame({"verification_status": [raw["verification_status"]]})
    ohe_out = one_hot_encoder.transform(ohe_input)
    ohe_cols = one_hot_encoder.get_feature_names_out(["verification_status"])
    for c, v in zip(ohe_cols, ohe_out[0]):
        row[c] = v

    # 6. engineered date features
    row["issue_month"] = raw["issue_month"]
    row["issue_year"] = raw["issue_year"]
    row["credit_age_years"] = raw["issue_year"] - raw["earliest_cr_year"]

    # Build single-row DataFrame in the order the target encoder / scaler expect
    df_row = pd.DataFrame([row])
    df_row = target_encoder.transform(df_row)
    df_row = df_row[scaler_feature_order]  # column order matching scaler fit (incl. loan_amnt)

    scaled = scaler.transform(df_row)
    scaled_df = pd.DataFrame(scaled, columns=scaler_feature_order)
    scaled_df = scaled_df[feature_order]  # drop loan_amnt, reorder to model's expected input

    pred = model.predict(scaled_df)[0]
    proba = model.predict_proba(scaled_df)[0]  # [P(class 0=eligible), P(class 1=not eligible)]

    return pred, proba


if submitted:
    raw_inputs = dict(
        loan_amnt=loan_amnt, term=term, int_rate=int_rate, installment=installment,
        grade=grade, emp_length=emp_length, home_ownership=home_ownership,
        annual_inc=annual_inc, verification_status=verification_status,
        initial_list_status=initial_list_status, purpose=purpose, dti=dti,
        open_acc=open_acc, pub_rec=pub_rec, revol_bal=revol_bal, revol_util=revol_util,
        total_acc=total_acc, mort_acc=mort_acc, pub_rec_bankruptcies=pub_rec_bankruptcies,
        earliest_cr_year=earliest_cr_year, issue_year=issue_year, issue_month=issue_month,
    )

    pred, proba = predict_eligibility(raw_inputs)
    p_not_eligible = proba[1] * 100

    st.divider()
    st.subheader("Prediction Result")

    if pred == 0:
        st.success(f"✅ ELIGIBLE — Predicted risk of default: {p_not_eligible:.1f}%")
    else:
        st.error(f"❌ NOT ELIGIBLE — Predicted risk of default: {p_not_eligible:.1f}%")

    st.progress(min(int(p_not_eligible), 100))
    st.caption(f"Model confidence — Fully Paid: {proba[0]*100:.1f}% | Charged Off: {proba[1]*100:.1f}%")

    with st.expander("View submitted inputs"):
        st.dataframe(pd.DataFrame([raw_inputs]).T.rename(columns={0: "Value"}))

    st.caption("Model: Logistic Regression trained on SMOTE-balanced LoanTap data "
               "(396,030 records) — test set precision 0.32 / recall 0.64 / F1 0.43 "
               "on the 'Charged Off' class.")

st.divider()
st.caption("This model prioritizes catching risky borrowers (high recall) over precision, "
           "per the case study's own risk-appetite recommendation for lenders.")