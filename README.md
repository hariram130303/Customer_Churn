# Customer Churn Prediction App

This Streamlit app predicts telecom customer churn using two trained models: Logistic Regression and XGBoost. It also provides model interpretability with SHAP.

## Features

- Predicts customer churn based on input features
- Allows choosing between Logistic Regression and XGBoost models
- Visualizes model explanation with SHAP plots

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/hariram130303/Customer_Churn
   cd Customer_Churn
   ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the app:
    ```bash
    streamlit run app.py
    ```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

- Enter customer feature values in the input form

- Select a model (Logistic Regression or XGBoost)

- Click Predict to see the churn prediction

- View SHAP explanation plot for model interpretability


## Deploymenet

**Streamlit URL:** [Customer_Churn](https://customer-churn-prediction1237.streamlit.app/)

## 📄 License

This project is licensed under the [MIT License](https://github.com/hariram130303/Customer_Churn/blob/main/LICENSE).
