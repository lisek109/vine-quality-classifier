# 🍷 Wine Quality Classifier

This project uses machine learning to classify wine as **good** or **not good**, based on physicochemical properties like acidity, alcohol content, and more.

## 🚀 How to Run

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   
2. Install dependencies:
   pip install -r requirements.txt
   
3.Run the notebook:
    Open notebooks/model_training.ipynb and run all cells to train and save the model.

📊 Dataset Description

Source: UCI Machine Learning Repository
Wine Quality Data Set

Type: Red wine only

Rows: 1,599 samples

Columns: 11 numerical features + 1 target column

Features: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol

Target:

Original: Integer score from 0 to 10

Transformed: Binary label

1 → good (quality ≥ 7)

0 → not good (quality < 7)


🧠 Modeling Approach

EDA: Data types and missing values, Target distribution, Correlation heatmap

Preprocessing: Feature scaling (StandardScaler), Train-test split

Modeling: Logistic Regression (baseline), Random Forest Classifier, Hyperparameter tuning with GridSearchCV

Model Selection: Cross-validation, ROC-AUC, Accuracy, F1-score


Run the Streamlit app:
streamlit run app/streamlit_app.py


