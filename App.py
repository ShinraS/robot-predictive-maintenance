import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Robot Predictive Maintenance", page_icon="🤖", layout="wide")

# --- CHARGEMENT DES MODÈLES ---
@st.cache_resource
def load_models():
    dt = joblib.load("dt_best.pkl")
    rf = joblib.load("rf_best.pkl")
    xgb = joblib.load("xgb_best.pkl")
    return dt, rf, xgb

try:
    dt_model, rf_model, xgb_model = load_models()
    
    # Configuration de l'encodeur pour XGBoost
    le = LabelEncoder()
    le.classes_ = np.array(['Heat Dissipation Failure', 'No Failure', 'Overstrain Failure', 'Power Failure', 'Tool Wear Failure'])
    
except Exception as e:
    st.error(f"Erreur de chargement des fichiers : {e}")

# --- INTERFACE UTILISATEUR (TITRE) ---
st.title("🤖 Robot Predictive Maintenance - Model Tester")
st.markdown("""
Cette application compare trois modèles (**Arbre de décision, Random Forest, XGBoost**) pour prédire les types de pannes machines.
""")

# --- BARRE LATÉRALE (INPUTS) ---
st.sidebar.header("📥 Paramètres des Capteurs")

# Noms pour DT et RF (avec crochets)
feature_names_dt_rf = [
    'Air temperature [K]', 'Process temperature [K]', 
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
    'Type_L', 'Type_M'
]

# Noms pour XGBoost (sans crochets)
feature_names_xgb = [
    'Air temperature K', 'Process temperature K', 
    'Rotational speed rpm', 'Torque Nm', 'Tool wear min', 
    'Type_L', 'Type_M'
]

# Collecte des entrées
input_data = {}
input_data['Air temperature [K]'] = st.sidebar.slider('Température Air [K]', 295.0, 305.0, 298.0)
input_data['Process temperature [K]'] = st.sidebar.slider('Température Process [K]', 305.0, 315.0, 310.0)
input_data['Rotational speed [rpm]'] = st.sidebar.number_input('Vitesse de rotation [rpm]', 1300, 2800, 1500)
input_data['Torque [Nm]'] = st.sidebar.number_input('Couple [Nm]', 0.0, 80.0, 40.0)
input_data['Tool wear [min]'] = st.sidebar.number_input('Usure outil [min]', 0, 250, 50)
input_data['Type_L'] = st.sidebar.selectbox('Type L (Low Quality)', [0, 1], index=0)
input_data['Type_M'] = st.sidebar.selectbox('Type M (Medium Quality)', [0, 1], index=1)

# Préparation des DataFrames
input_df_dt_rf = pd.DataFrame([input_data])

mapping = dict(zip(feature_names_dt_rf, feature_names_xgb))
input_data_xgb = {mapping[k]: v for k, v in input_data.items()}
input_df_xgb = pd.DataFrame([input_data_xgb])

# --- PRÉDICTIONS ET AFFICHAGE ---
st.header("📊 Résultats du Diagnostic")

col1, col2, col3 = st.columns(3)

def display_diag(name, prediction):
    is_failure = prediction != "No Failure"
    if is_failure:
        st.error(f"### {name}")
        st.metric(label="STATUS", value="FAILURE", delta="Anomalie Détectée", delta_color="inverse")
        st.markdown(f"**Type:** `{prediction}`")
    else:
        st.success(f"### {name}")
        st.metric(label="STATUS", value="NORMAL", delta="RAS")
        st.markdown("**Type:** Aucun")

with col1:
    pred_dt = dt_model.predict(input_df_dt_rf)[0]
    display_diag("Decision Tree", pred_dt)

with col2:
    pred_rf = rf_model.predict(input_df_dt_rf)[0]
    display_diag("Random Forest", pred_rf)

with col3:
    pred_xgb_num = xgb_model.predict(input_df_xgb)[0]
    # Décodage XGBoost
    if isinstance(pred_xgb_num, (int, np.integer)):
        pred_xgb = le.inverse_transform([pred_xgb_num])[0]
    else:
        pred_xgb = pred_xgb_num
    display_diag("XGBoost", pred_xgb)


# --- TABLE RÉCAPITULATIVE ---
with st.expander("Voir les données envoyées aux modèles"):
    st.table(input_df_dt_rf)