import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

# Load the dataset
data = load_wine()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names

# Train the models
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_scaled, y)

# Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X, y)

# Streamlit App
st.title("Wine Classification Predictor")
st.write("Predict the wine type based on chemical features using Logistic Regression and Decision Tree models.")

# Input sliders for features
st.sidebar.header("Input Features")
input_features = []
for i, feature in enumerate(feature_names):
    value = st.sidebar.slider(feature, float(X[:, i].min()), float(X[:, i].max()), float(X[:, i].mean()))
    input_features.append(value)

# Convert input to numpy array
input_array = np.array(input_features).reshape(1, -1)

# Logistic Regression Prediction
scaled_input = scaler.transform(input_array)
log_reg_pred = log_reg.predict(scaled_input)[0]
log_reg_prob = log_reg.predict_proba(scaled_input)

# Decision Tree Prediction
tree_pred = decision_tree.predict(input_array)[0]
tree_prob = decision_tree.predict_proba(input_array)

# Display predictions
st.subheader("Predictions")
st.write("### Logistic Regression Prediction")
st.write(f"Predicted Class: {target_names[log_reg_pred]}")
st.write(f"Prediction Probabilities: {log_reg_prob[0]}")

st.write("### Decision Tree Prediction")
st.write(f"Predicted Class: {target_names[tree_pred]}")
st.write(f"Prediction Probabilities: {tree_prob[0]}")
