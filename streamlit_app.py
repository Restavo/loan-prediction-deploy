import streamlit as st
import pickle
import pandas as pd

# ===== Load model dan encoder =====
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# ===== Threshold final =====
optimal_threshold = 0.01

# ===== Preprocess user input =====
def preprocess_input(user_input: dict):
    df = pd.DataFrame([user_input])
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])
    return df

# ===== Predict =====
def predict(input_dict):
    data = preprocess_input(input_dict)
    proba = model.predict_proba(data)[0][1]
    return proba, int(proba >= optimal_threshold)

# ===== STREAMLIT UI =====
st.title("Prediksi Persetujuan Pinjaman")

with st.form("form_pinjaman"):
    person_age = st.number_input("Usia", min_value=18, max_value=100, value=30)
    person_gender = st.selectbox("Gender", ["male", "female"])
    person_education = st.selectbox("Pendidikan", ["High School", "Bachelor", "Master", "PhD"])
    person_income = st.number_input("Pendapatan Tahunan", min_value=0, value=50000)
    person_emp_exp = st.slider("Lama Pengalaman Kerja (tahun)", 0, 40, 5)
    person_home_ownership = st.selectbox("Status Tempat Tinggal", ["RENT", "OWN", "MORTGAGE"])
    loan_amnt = st.number_input("Jumlah Pinjaman", min_value=1000, value=10000)
    loan_intent = st.selectbox("Tujuan Pinjaman", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    loan_int_rate = st.number_input("Suku Bunga (%)", min_value=0.0, value=10.5)
    loan_percent_income = st.number_input("Rasio Cicilan terhadap Pendapatan", min_value=0.0, value=loan_amnt / person_income)
    cb_person_cred_hist_length = st.slider("Lama Riwayat Kredit (tahun)", 0, 30, 5)
    credit_score = st.number_input("Skor Kredit", min_value=300, max_value=900, value=600)
    previous_loan_defaults_on_file = st.selectbox("Pernah Default Sebelumnya?", ["Yes", "No"])

    submitted = st.form_submit_button("Prediksi")

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

    prob, hasil = predict(input_data)
    st.write(f"Probabilitas disetujui: **{prob:.3f}**")
    if hasil == 1:
        st.success("✅ Pinjaman Disetujui")
    else:
        st.error("❌ Pinjaman Ditolak")