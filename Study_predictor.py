import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import streamlit as st


print("sklearn")
print(sklearn.__version__)
print("sklearn")

df = pd.read_csv(r"C:\Users\ELECTRO LINKS\Downloads\studypredictor.csv",
                 encoding='utf-8-sig')
print(df.head(10))

print(df.info())

print(df.isnull().sum())

# study distrib histogram
sns.histplot(df['Duration_Minutes'], bins=50, kde=True,
             color='skyblue',
             stat='density'
             )
sns.kdeplot(df['Duration_Minutes'], color='blue', linewidth=2)
plt.title("Study Duration Distribution")
plt.xlabel("Duration in Minutes")
plt.ylabel("Probability Denisty")
plt.show()

# avg score by resource type
resources = df.groupby('Resource_Type')['Score'].agg(
    ['mean', 'std']).reset_index()

df['Time'] = pd.to_datetime(df['Time'], format='%I:%M %p')
df['Hour'] = df['Time'].dt.hour
sns.barplot(x='Resource_Type', y='Score', data=df,
            palette="Set2", ci=None)
plt.xlabel("Resources")
plt.ylabel("Scores")
plt.xticks(rotation=30)
plt.title("Resource Effectiveness with Variability")
plt.show()

pivot = df.pivot_table(index='Hour', columns='Resource_Type',
                       values='Score', aggfunc='mean')
sns.heatmap(pivot, annot=True, cmap="YlGnBu")
plt.title("Average Score by Resource and Hour")
plt.show()

df['Duration_Mood'] = df['Duration_Minutes'] * df['Mood']
df['Confidence_Hour'] = df['Confidence'] * df['Hour']


df['Date'] = pd.to_datetime(df['Date'])
df['WeekDay'] = df['Date'].dt.dayofweek
df['is_weekend'] = df['WeekDay'].apply(lambda x: 1 if x >= 5 else 0)
features = ['Duration_Minutes', 'Mood', 'Confidence',
            'Resource_Type', 'Hour', 'Confidence_Hour', 'Duration_Mood', 'is_weekend']
target = 'Score'

df_encoded = pd.get_dummies(df[features], drop_first=True)
X = df_encoded
y = df[target]

X_train, X_test, y_train, y_test,  = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, "model.pkl")
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

# decision treee
tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
joblib.dump(tree_model, "tree_model.pkl")
y_tree_pred = tree_model.predict(X_test)

tree_mse = mean_squared_error(y_test, y_tree_pred)
tree_r2 = r2_score(y_test, y_tree_pred)
print("Mean Squared Error of tree :", tree_mse)
print("R² Score of tree:", tree_r2)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "rf_model.pkl")
rf_tree_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_tree_pred)
rf_r2 = r2_score(y_test, rf_tree_pred)
print("Mean Squared Error of rf tree :", rf_mse)
print("R² Score of  rf tree:", rf_r2)


mae = mean_absolute_error(y_test, y_pred)
mae_tr = mean_absolute_error(y_test, y_tree_pred)
mae_rf = mean_absolute_error(y_test, rf_tree_pred)
mae_values = {
    "Linear Regression": mae,
    "Decision Tree": mae_tr,
    "Random Forest": mae_rf
}
joblib.dump(mae_values, "mae_values.pkl")

print("Linear Regression MAE:", mae)
print("Decision Tree MAE:", mae_tr)
print("Random Forest MAE:", mae_rf)

# scatter plot of these models
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].scatter(y_test, y_pred, alpha=0.6, color='pink')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_title(f'Linear Regression \n    Mae={mae:.2f}')
axes[0].set_xlabel("Actual Score")
axes[0].set_ylabel("Predicted Score")

axes[1].scatter(y_test, y_tree_pred, alpha=0.6, color='blue')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1].set_title(f'Decision Tree  \n  Mae={mae_tr:.2f}')
axes[1].set_xlabel("Actual Score")
axes[1].set_ylabel("Predicted Score")

axes[2].scatter(y_test, rf_tree_pred, alpha=0.6, color='green')
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[2].set_title(f'Random Forest   \n Mae={mae_rf:.2f}')
axes[2].set_xlabel("Actual Score")
axes[2].set_ylabel("Predicted Score")

plt.tight_layout()
plt.show()


# 3feature importance
def plot_feature_importance(rf_model, feature_names):
    importances = rf_model.feature_importances_
    feat_imp = pd.Series(
        importances, index=feature_names).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(18, 6))
    sns.barplot(x=feat_imp, y=feat_imp.index, palette="Pastel1", ax=ax)
    ax.set_title("Feature Importance - Random Forest")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")

    return fig
