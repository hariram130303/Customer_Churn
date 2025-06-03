import pandas as pd
import shap
import io
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Load Dataset
def load_data():
    url = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    return df

# Preprocess the data
def preprocess(df):
    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

# Train both models
def train_models(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=500)
    logreg.fit(X_train_scaled, y_train)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb.fit(X_train, y_train)

    return logreg, xgb, scaler, X_train, X_test, y_train, y_test

# Predict churn and probability
def predict_churn(model, input_df, scaler=None):
    if scaler:
        input_scaled = scaler.transform(input_df)
        pred_prob = model.predict_proba(input_scaled)[0][1]
        pred = model.predict(input_scaled)[0]
    else:
        pred_prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]
    return pred, pred_prob

# Generate SHAP values
def explain_shap(model, background_data, input_df):
    explainer = shap.Explainer(model, background_data)
    shap_values = explainer(input_df)
    return shap_values

# Render SHAP plot in Streamlit
def st_shap_plot(shap_values, width=500):
    shap.plots.waterfall(shap_values[0], show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close()
    buf.seek(0)
    st.image(buf, width=width)
