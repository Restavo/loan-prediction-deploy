import streamlit as st
import pickle
import pandas as pd

# ===== Load model and encoders =====
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# ===== Threshold for approval =====
optimal_threshold = 0.01

# ===== Preprocessing input =====
def preprocess_input(user_input: dict):
    df = pd.DataFrame([user_input])
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])
    return df

# ===== Predict function =====
def predict(input_dict):
    data = preprocess_input(input_dict)
    proba = model.predict_proba(data)[0][1]
    return proba, int(proba >= optimal_threshold)

# ===== Streamlit UI =====
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")
st.title("💳 Loan Approval Prediction App")
st.markdown("Predict whether a loan application will be **approved or rejected** based on applicant details.")

st.markdown("---")

# ===== Test Case Buttons =====
st.subheader("📋 Quick Test Cases")

col1, col2 = st.columns(2)

if col1.button("✅ Test Case: Approved"):
    st.session_state.update({
        'person_age': 40,
        'person_gender': 'male',
        'person_education': 'Master',
        'person_income': 120000,
        'person_emp_exp': 15,
        'person_home_ownership': 'MORTGAGE',
        'loan_amnt': 3000,
        'loan_intent': 'MEDICAL',
        'loan_int_rate': 8.5,
        'loan_percent_income': 0.025,
        'cb_person_cred_hist_length': 10,
        'credit_score': 780,
        'previous_loan_defaults_on_file': 'No'
    })

if col2.button("❌ Test Case: Rejected"):
    st.session_state.update({
        'person_age': 22,
        'person_gender': 'female',
        'person_education': 'High School',
        'person_income': 15000,
        'person_emp_exp': 1,
        'person_home_ownership': 'RENT',
        'loan_amnt': 30000,
        'loan_intent': 'DEBTCONSOLIDATION',
        'loan_int_rate': 19.5,
        'loan_percent_income': 1.5,
        'cb_person_cred_hist_length': 1,
        'credit_score': 470,
        'previous_loan_defaults_on_file': 'Yes'
    })

# ===== Form =====
st.markdown("---")
st.subheader("📄 Applicant Information")

with st.form("loan_form"):
    person_age = st.number_input("Applicant Age", min_value=18, max_value=100, value=st.session_state.get("person_age", 30))
    person_gender = st.selectbox("Gender", ["male", "female"], index=0 if st.session_state.get("person_gender") == "male" else 1)
    person_education = st.selectbox("Highest Education", ["High School", "Bachelor", "Master", "PhD"], index=["High School", "Bachelor", "Master", "PhD"].index(st.session_state.get("person_education", "Bachelor")))
    person_income = st.number_input("Annual Income ($)", min_value=0, value=st.session_state.get("person_income", 50000))
    person_emp_exp = st.slider("Years of Work Experience", 0, 40, st.session_state.get("person_emp_exp", 5))
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"], index=["RENT", "OWN", "MORTGAGE"].index(st.session_state.get("person_home_ownership", "RENT")))
    loan_amnt = st.number_input("Loan Amount Requested ($)", min_value=1000, value=st.session_state.get("loan_amnt", 10000))
    loan_intent = st.selectbox("Purpose of Loan", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"], index=0)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=st.session_state.get("loan_int_rate", 10.5))
    loan_percent_income = st.number_input("Loan-to-Income Ratio", min_value=0.0, value=st.session_state.get("loan_percent_income", loan_amnt / max(person_income, 1)))
    cb_person_cred_hist_length = st.slider("Credit History Length (years)", 0, 30, st.session_state.get("cb_person_cred_hist_length", 5))
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=st.session_state.get("credit_score", 600))
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["Yes", "No"], index=0 if st.session_state.get("previous_loan_defaults_on_file") == "Yes" else 1)

    submitted = st.form_submit_button("🔍 Predict")

# ===== Prediction Result =====
if submitted:
    input_data = {
        "person_age": person_age,
        "person_gender": person_gender,
        "person_education": person_education,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "person_home_ownership": person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file
    }

    proba, result = predict(input_data)
    st.markdown(f"### 🧮 Approval Probability: **{proba:.3f}**")

    if result == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
