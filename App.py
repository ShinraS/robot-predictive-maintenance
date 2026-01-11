import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- Load models ---
dt_model = joblib.load("dt_best.pkl")
rf_model = joblib.load("rf_best.pkl")
xgb_model = joblib.load("xgb_best.pkl")

# --- Label encoder for XGBoost ---
le = LabelEncoder()
le.classes_ = np.array(['Heat Dissipation Failure', 'No Failure', 'Overstrain Failure', 'Power Failure', 'Tool Wear Failure'])

# --- Title ---
st.title("Robot Predictive Maintenance - Model Tester")
st.write("Test your Decision Tree, Random Forest, and XGBoost models on new sensor data.")

# --- Sidebar inputs ---
st.sidebar.header("Enter sensor data")

# Noms exacts utilisés pour DT et RF (avec crochets)
feature_names_dt_rf = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'Type_L',
    'Type_M'
]

# Noms exacts utilisés pour XGBoost (sans crochets d'après votre log)
feature_names_xgb = ['Air temperature K', 'Process temperature K', 'Rotational speed rpm', 'Torque Nm', 'Tool wear min', 'Type_L', 'Type_M']

# Valeurs par défaut
default_values = {
    'Air temperature [K]': 298.0,
    'Process temperature [K]': 310.0,
    'Rotational speed [rpm]': 1500,
    'Torque [Nm]': 40.0,
    'Tool wear [min]': 50.0,
    'Type_L': 0,
    'Type_M': 1
}

# 1. Collecte des entrées utilisateur
input_data = {}
for feature in feature_names_dt_rf:
    input_data[feature] = st.sidebar.number_input(feature, value=default_values[feature])

# 2. Préparation du DataFrame pour DT & RF
input_df_dt_rf = pd.DataFrame([input_data])

# 3. Préparation du DataFrame pour XGBoost (Mapping des noms)
# On crée un dictionnaire qui fait correspondre les noms DT_RF -> XGB
mapping = dict(zip(feature_names_dt_rf, feature_names_xgb))
input_data_xgb = {mapping[k]: v for k, v in input_data.items()}
input_df_xgb = pd.DataFrame([input_data_xgb])

# --- Predictions ---
st.header("Predictions")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Decision Tree")
    try:
        pred_dt = dt_model.predict(input_df_dt_rf)[0]
        st.success(f"**Result:** {pred_dt}")
    except Exception as e:
        st.error(f"Error: {e}")

with col2:
    st.subheader("Random Forest")
    try:
        pred_rf = rf_model.predict(input_df_dt_rf)[0]
        st.success(f"**Result:** {pred_rf}")
    except Exception as e:
        st.error(f"Error: {e}")

with col3:
    st.subheader("XGBoost")
    try:
        # On s'assure que XGBoost reçoit un DataFrame avec les bons noms de colonnes
        pred_xgb_num = xgb_model.predict(input_df_xgb)[0]
        # Si le modèle renvoie un index, on décode, sinon on affiche direct
        if isinstance(pred_xgb_num, (int, np.integer)):
            pred_xgb = le.inverse_transform([pred_xgb_num])[0]
        else:
            pred_xgb = pred_xgb_num
        st.success(f"**Result:** {pred_xgb}")
    except Exception as e:
        st.error(f"Error: {e}")

# --- Show input data ---
st.divider()
st.header("Input Data Summary")
st.table(input_df_dt_rf)