import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

class LoanModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.encoders = {}
        self.model = None

    def preprocess(self):
        # Imputasi missing value
        self.df["person_income"] = self.df["person_income"].fillna(self.df["person_income"].mean())

        # Label encoding untuk semua kolom kategorik
        for col in self.df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.encoders[col] = le

    def split_data(self):
        X = self.df.drop("loan_status", axis=1)
        y = self.df["loan_status"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

    def train_model(self):
        # Hitung scale_pos_weight untuk XGBoost
        ratio = (self.y_train == 0).sum() / (self.y_train == 1).sum()

        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=ratio,
            random_state=42
        )

        params = {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.1, 0.01]
        }

        grid = GridSearchCV(model, params, cv=3, scoring='roc_auc', n_jobs=-1)
        grid.fit(self.X_train, self.y_train)

        self.model = grid.best_estimator_

    def save_artifacts(self, model_path="best_model.pkl", encoder_path="encoders.pkl"):
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(encoder_path, "wb") as f:
            pickle.dump(self.encoders, f)

    def run_all(self):
        self.preprocess()
        self.split_data()
        self.train_model()
        self.save_artifacts()
