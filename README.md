# Customer Churn Prediction App

This Streamlit app predicts telecom customer churn using two trained models: Logistic Regression and XGBoost. It also provides model interpretability with SHAP.

## Features

- Predicts customer churn based on input features
- Allows choosing between Logistic Regression and XGBoost models
- Visualizes model explanation with SHAP plots

## Installation

**1. Clone the repo:**
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

**2. Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

**3. Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the app:
    ```bash
    streamlit run app.py
    ```
### Then open your browser at http://localhost:8501

## Usage

- Enter customer feature values in the input form

- Select a model (Logistic Regression or XGBoost)

- Click Predict to see the churn prediction

- View SHAP explanation plot for model interpretability


## Deploymenet
**Streamlit URL:**

## 📄 License

This project is licensed under the [MIT License](https://github.com/hariram130303/Customer_Churn/blob/main/LICENSE).
