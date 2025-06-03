import streamlit as st
import pandas as pd
from churn_utils import load_data, preprocess, train_models
from churn_utils import predict_churn, explain_shap, st_shap_plot

@st.cache_data(show_spinner=False)
def load_and_train():
    """
    Loads the dataset, preprocesses it, and trains the Logistic Regression
    and XGBoost models. This function is cached to avoid re-computation.
    Returns the cleaned dataframe and the trained models along with necessary
    scalers and data splits.
    """
    df = load_data()
    df = preprocess(df)
    models = train_models(df)
    return df, models

def user_input_features(df):
    """
    Creates input fields in the sidebar for each feature in the dataset
    (except the target). Defaults to the median value of each feature.
    User inputs are captured and returned as a single-row DataFrame.
    """
    inputs = {}
    for col in df.drop("Churn", axis=1).columns:
        default = df[col].median()
        val = st.sidebar.text_input(f"{col}", str(default))
        try:
            val = float(val)  # convert input to float if possible
        except:
            pass  # keep as string if not convertible
        inputs[col] = val
    return pd.DataFrame([inputs])

def main():
    st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
    st.title("📞 Telecom Customer Churn Prediction")
    
    st.markdown(
        """
        This app predicts whether a telecom customer will **churn** (stop using the service)
        based on their demographic and service usage data.

        **Instructions:**
        - Use the sidebar to input customer features such as tenure, service subscriptions,
          payment method, and monthly charges.
        - Select a machine learning model for prediction:
          - **Logistic Regression:** Simple, interpretable, baseline model.
          - **XGBoost:** Advanced model that often yields higher accuracy.
        - Click **Predict** to see:
          - The probability the customer will churn.
          - The predicted churn status.
          - An explanation plot showing which features influenced the prediction using SHAP values.

        This tool can help telecom providers identify customers at risk of leaving and tailor retention strategies accordingly.
        """
    )

    # Load data and models (cached)
    df, (logreg, xgb, scaler, X_train, X_test, y_train, y_test) = load_and_train()

    st.sidebar.header("Input Customer Data")
    st.sidebar.markdown(
        "Fill in the customer attributes below. Default values are medians from the dataset."
    )
    input_df = user_input_features(df)

    model_choice = st.sidebar.selectbox(
        "Choose Model",
        ["Logistic Regression", "XGBoost"],
        help="Select which machine learning model to use for prediction."
    )

    if st.sidebar.button("Predict"):
        # Perform prediction using selected model
        if model_choice == "Logistic Regression":
            pred, pred_prob = predict_churn(logreg, input_df, scaler)
            background = scaler.transform(X_train)
        else:
            pred, pred_prob = predict_churn(xgb, input_df)
            background = X_train

        # Display results
        st.subheader("Prediction Results")
        st.write(f"**Churn Probability:** {pred_prob:.2%}")
        st.write("**Prediction:**", "🔴 Churn" if pred == 1 else "🟢 No Churn")

        # Add user tip about probability threshold
        st.info(
            "Note: Probabilities closer to 1 indicate a higher risk of churn. "
            "Consider customers with probability above 0.5 as likely to churn."
        )

        # Explain prediction using SHAP values
        st.subheader("Model Explanation with SHAP")
        shap_values = explain_shap(xgb if model_choice == "XGBoost" else logreg, background, input_df)
        st_shap_plot(shap_values)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Developed by Hari Ram | "
        "[GitHub Repository](https://github.com/hariram130303/Customer_Churn) | "
        "[Dataset source](https://www.kaggle.com/blastchar/telco-customer-churn)"
    )

if __name__ == "__main__":
    main()
