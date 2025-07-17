import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

# Load model (you can save and load trained XGBoost model using joblib)
@st.cache_resource
def load_model():
    model = XGBClassifier(learning_rate=0.4, max_depth=7)
    df = pd.read_csv('dataset2.csv')
    X = df.drop(columns=['phishing'])
    y = df['phishing']
    model.fit(X, y)
    return model

model = load_model()

# UI
st.title("🔐 Phishing Website Detector")

uploaded_file = st.file_uploader("Upload a CSV with features", type=['csv'])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    prediction = model.predict(data)
    st.write("### 🔍 Prediction Results")
    st.write(prediction)
