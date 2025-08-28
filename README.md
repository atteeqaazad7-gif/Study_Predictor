# Study_Predictor
This project is a Streamlit web application that predicts study effectiveness based on user inputs such as study duration, mood, confidence, time of day, and type of resource used. It leverages machine learning models trained on study session data and presents predictions with interactive UI elements and visualizations.
**Data Input ( Streamlit sidebar)**
Duration (minutes studied)
Mood (1 = bad, 3 = good)
Confidence (1 = low, 3 = high)
Resource Type (YouTube, Practice, Blog, Book, ChatGPT, Online Course)
Hour of the day (0–23)
Weekend indicator (Yes/No)

**Feature Engineering**

Creates additional features such as:
Duration_Mood = duration × mood
Confidence_Hour = confidence × hour
Encodes the categorical variable Resource_Type using one-hot encoding.


**Three pre-trained models** are loaded using joblib:
Linear Regression
Decision Tree Regressor
Random Forest Regressor

Predictions from all three models are displayed.
The Random Forest model is used as the final prediction.

**Visualization**
Feature importance plot (from Random Forest).
Model MAE comparison using seaborn bar charts.

**Custom Styling**
Light pastel theme applied with custom CSS.
Sidebar styling and formatted prediction display.

**Tech Stack**
Python: pandas, numpy, scikit-learn, seaborn, matplotlib

**Streamlit**: for web app development and deployment

Joblib: for saving and loading ML models
