import streamlit as st
import pandas as pd
import pickle
import time

# ===== Load model and encoders =====
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

optimal_threshold = 0.01

def preprocess_input(user_input: dict):
    df = pd.DataFrame([user_input])
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])
    return df

def predict(input_dict):
    data = preprocess_input(input_dict)
    proba = model.predict_proba(data)[0][1]
    return proba, int(proba >= optimal_threshold)

# ===== INITIAL STATE =====
if "tab" not in st.session_state:
    st.session_state.tab = "form"

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("💳 Loan Approval Prediction App")
st.markdown("Predict whether a loan will be **approved or rejected** based on applicant details.")

st.markdown("---")

# ===== TABS MANUAL BASED ON STATE =====
tab_choice = st.session_state.tab

tabs = {"form": "📝 Application Form", "result": "📊 Prediction Result"}
st.markdown(f"### {tabs[tab_choice]}")

if tab_choice == "form":
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

    st.subheader("📄 Applicant Information")
    with st.form("loan_form"):
        person_age = st.number_input("Age", 18, 100, st.session_state.get("person_age", 30))
        person_gender = st.selectbox("Gender", ["male", "female"], index=0 if st.session_state.get("person_gender") == "male" else 1)
        person_education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"], index=["High School", "Bachelor", "Master", "PhD"].index(st.session_state.get("person_education", "Bachelor")))
        person_income = st.number_input("Annual Income ($)", 0, 1_000_000, st.session_state.get("person_income", 50000))
        person_emp_exp = st.slider("Work Experience (Years)", 0, 40, st.session_state.get("person_emp_exp", 5))
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"], index=["RENT", "OWN", "MORTGAGE"].index(st.session_state.get("person_home_ownership", "RENT")))
        loan_amnt = st.number_input("Loan Amount ($)", 1000, 500000, st.session_state.get("loan_amnt", 10000))
        loan_intent = st.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
        loan_int_rate = st.number_input("Interest Rate (%)", 0.0, 40.0, st.session_state.get("loan_int_rate", 10.5))
        loan_percent_income = st.number_input("Loan-to-Income Ratio", 0.0, 2.0, st.session_state.get("loan_percent_income", loan_amnt / max(person_income, 1)))
        cb_person_cred_hist_length = st.slider("Credit History Length (Years)", 0, 30, st.session_state.get("cb_person_cred_hist_length", 5))
        credit_score = st.number_input("Credit Score", 300, 900, st.session_state.get("credit_score", 600))
        previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults?", ["Yes", "No"], index=0 if st.session_state.get("previous_loan_defaults_on_file") == "Yes" else 1)

        submit = st.form_submit_button("🔍 Predict")

        if submit:
            st.session_state.input_data = {
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
            st.session_state.tab = "result"
            st.experimental_rerun()

# ===== SHOW PREDICTION RESULT TAB =====
elif tab_choice == "result":
    st.subheader("📊 Prediction Result")

    with st.spinner("Analyzing loan application..."):
        time.sleep(1.2)
        proba, result = predict(st.session_state.input_data)

    st.metric("📈 Approval Probability", f"{proba:.2%}")
    st.progress(min(int(proba * 100), 100))

    if result == 1:
        st.success("✅ Approved")
        st.balloons()
    else:
        st.error("❌ Rejected")

    if st.button("🔙 Back to Form"):
        st.session_state.tab = "form"
        st.experimental_rerun()
