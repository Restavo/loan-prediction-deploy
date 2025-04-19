import pickle
import pandas as pd

# ===== LOAD MODEL DAN ENCODER =====
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# ===== PREPROCESSING UNTUK INPUT USER =====
def preprocess_input(user_input: dict):
    df = pd.DataFrame([user_input])
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])
    return df

# ===== THRESHOLD FINAL (PAKAI ROC TADI) =====
optimal_threshold = 0.01

def predict(input_dict):
    data = preprocess_input(input_dict)
    proba = model.predict_proba(data)[0][1]
    print(f"Probabilitas disetujui: {proba:.3f}")
    return int(proba >= optimal_threshold)
