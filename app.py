from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from Study_predictor import plot_feature_importance

import joblib


df = pd.read_csv("studypredictor.csv", encoding="utf-8-sig")

lr_model = joblib.load("model.pkl")
tree_model = joblib.load("tree_model.pkl")
rf_model = joblib.load("rf_model.pkl")

st.set_page_config(page_title='Study Scores Predictor🌸', layout="wide")
st.title("🌸 Your Study Score Predictor 🌸")
st.write("Predict your study effectiveness based on your given resources, mood, confidence, and time!")
st.sidebar.header("Input your study session")

duration = st.sidebar.slider("Duration ( in minutes)", 0, 300, 60)
mood = st.sidebar.slider("Mood (1=bad😢, 3=good🎉)", 1, 3, 2)
confidence = st.sidebar.slider("Confidence (1=low😔, 3=high😄)", 1, 3, 2)
resource = st.sidebar.selectbox("Resource Type", [
                                'YouTube', 'Practice', 'Blog', 'Book', 'ChatGPT', 'Online Course'])
hour = st.sidebar.slider("Hour of day (0-23)", 0, 23, 18)
is_weekend = st.sidebar.selectbox("Is it weekend?", ["No", "Yes"])
is_weekend_val = 1 if is_weekend == "Yes" else 0

duration_mood = duration * mood
confidence_hour = confidence * hour

input_df = pd.DataFrame({
    'Duration_Minutes': [duration],
    'Mood': [mood],
    'Confidence': [confidence],
    'Hour': [hour],
    'Confidence_Hour': [confidence_hour],
    'Duration_Mood': [duration_mood],
    'is_weekend': [is_weekend_val]
})
for col in lr_model.feature_names_in_:
    if col.startswith('Resource_Type_'):
        val = 1 if col.replace('Resource_Type_', '') == resource else 0
        input_df[col] = val
input_df = input_df[lr_model.feature_names_in_]

st.subheader("Predicted Scores")
st.write(f'Final Predicted Score: {rf_model.predict(input_df)[0]:.2f}')
st.write("Prediction From all models.")
st.write(f"Linear Regression: {lr_model.predict(input_df)[0]:.2f}")
st.write(f"Decision Tree: {tree_model.predict(input_df)[0]:.2f}")
st.write(f"Random Forest: {rf_model.predict(input_df)[0]:.2f}")

st.write(f'Graph of Important Features')
fig = plot_feature_importance(rf_model, lr_model.feature_names_in_)
st.pyplot(fig)


mae_values = joblib.load("mae_values.pkl")


mae_df = pd.DataFrame(list(mae_values.items()), columns=["Model", "MAE"])

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="Model", y="MAE", data=mae_df, palette="Pastel1", ax=ax)
ax.set_title("MAE of Different Models")
ax.set_ylim(0, max(mae_values.values()) + 5)
ax.bar_label(ax.containers[0], fmt="%.2f")

st.pyplot(fig)

st.markdown("""
    <style>
    body {
        background-color: #ffe6f0;  /* light pink */
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* Change background */
    .stApp {
        background: linear-gradient(135deg, #ffe6f0 0%, #e6f7ff 100%);
        font-family: "Trebuchet MS", sans-serif;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #fff0f5;
        border-right: 2px solid #ffb6c1;
    }

    /* Titles */
    h1, h2, h3 {
        color: #d63384;
        text-align: center;
    }

    /* Metrics cards */
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
        color: #ff4081;
    }

    /* Button styling */
    .stButton button {
        background: linear-gradient(to right, #ff99cc, #ff66b2);
        color: white;
        border-radius: 8px;
        font-weight: bold;
        border: none;
    }
    .stButton button:hover {
        background: linear-gradient(to right, #ff66b2, #ff3385);
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    "<h2 style='text-align: center;'>✨ Your Personalized Study Prediction ✨</h2>",
    unsafe_allow_html=True
)
col1, col2, col3 = st.columns(3)
col1.metric("Linear Regression", f"{lr_model.predict(input_df)[0]:.2f}")
col2.metric("Decision Tree", f"{tree_model.predict(input_df)[0]:.2f}")
col3.metric("Random Forest", f"{rf_model.predict(input_df)[0]:.2f}")

st.markdown(
    f"<h3 style='text-align: center; color:#ff4081;'> 🎯 Final Predicted Score: {rf_model.predict(input_df)[0]:.2f} </h3>",
    unsafe_allow_html=True
)

st.markdown("---")

