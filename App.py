import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
st.set_page_config(page_title="Multi-Class Maintenance", page_icon="🤖", layout="wide")

@st.cache_resource
def load_models():
    dt = joblib.load("dt_best.pkl")
    rf = joblib.load("rf_best.pkl")
    xgb = joblib.load("xgb_best.pkl")
    return dt, rf, xgb

try:
    dt_model, rf_model, xgb_model = load_models()
    le = LabelEncoder()
    # On s'assure que l'ordre des classes correspond à ton entraînement
    le.classes_ = np.array(['Heat Dissipation Failure', 'No Failure', 'Overstrain Failure', 'Power Failure', 'Tool Wear Failure'])
except Exception as e:
    st.error(f"Erreur de chargement des modèles : {e}")
    st.stop()

# --- SCÉNARIOS DE TEST (Sidebar) ---
st.sidebar.header("🚀 Scénarios de Test Rapide")

# Initialisation des valeurs par défaut dans le session_state
if 'm_type' not in st.session_state:
    st.session_state.m_type, st.session_state.m_air, st.session_state.m_proc, \
    st.session_state.m_speed, st.session_state.m_torque, st.session_state.m_wear = 'L', 301.0, 310.6, 1493, 37.8, 206

# BOUTON 1 : CAS NORMAL
if st.sidebar.button("✅ Charger Cas Normal"):
    st.session_state.m_type, st.session_state.m_air, st.session_state.m_proc, \
    st.session_state.m_speed, st.session_state.m_torque, st.session_state.m_wear = 'L', 298.1, 308.6, 1551, 42.8, 0

# BOUTON 2 : TON CAS REEL (Tool Wear Failure)
if st.sidebar.button("⚠️ Charger Cas Panne"):
    st.session_state.m_type, st.session_state.m_air, st.session_state.m_proc, \
    st.session_state.m_speed, st.session_state.m_torque, st.session_state.m_wear = 'L', 301.0, 310.6, 1493, 37.8, 206

st.sidebar.divider()

# --- SAISIE MANUELLE ---
st.sidebar.header("📡 Ajustements Manuels")
m_type = st.sidebar.selectbox("Type", ["L", "M", "H"], index=["L", "M", "H"].index(st.session_state.m_type))
air = st.sidebar.number_input("Air temperature [K]", value=st.session_state.m_air)
proc = st.sidebar.number_input("Process temperature [K]", value=st.session_state.m_proc)
speed = st.sidebar.number_input("Rotational speed [rpm]", value=int(st.session_state.m_speed))
torque = st.sidebar.number_input("Torque [Nm]", value=st.session_state.m_torque)
wear = st.sidebar.number_input("Tool wear [min]", value=int(st.session_state.m_wear))

# Encodage manuel du Type pour DT et RF
type_l = 1 if m_type == "L" else 0
type_m = 1 if m_type == "M" else 0
input_data = [air, proc, speed, torque, wear, type_l, type_m]

# Création des DataFrames avec les noms de colonnes spécifiques
df_dt_rf = pd.DataFrame([input_data], columns=['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Type_L', 'Type_M'])
df_xgb = pd.DataFrame([input_data], columns=['Air temperature K', 'Process temperature K', 'Rotational speed rpm', 'Torque Nm', 'Tool wear min', 'Type_L', 'Type_M'])

# --- AFFICHAGE ---
st.title("📊 Multi-Class Maintenance Diagnostic")
st.markdown("### Analyse comparative des modèles de classification")

col1, col2, col3 = st.columns(3)

def display_multiclass(name, model, df, is_xgb=False):
    try:
        # Prédiction
        pred_val = model.predict(df)[0]
        # Décodage du label si XGBoost (qui renvoie souvent des entiers)
        prediction = le.inverse_transform([int(pred_val)])[0] if is_xgb else pred_val
        
        # Probabilité (Certitude)
        proba = np.max(model.predict_proba(df)[0])
        
        if prediction != "No Failure":
            st.error(f"### {name}")
            st.metric("STATUS", "FAILURE")
            st.warning(f"**Type :** {prediction}")
        else:
            st.success(f"### {name}")
            st.metric("STATUS", "NORMAL")
            st.write("**Type :** Aucun")
        
        # Barre de progression (Conversion float obligatoire pour Streamlit)
        st.progress(float(proba), text=f"Certitude : {proba:.1%}")
        
    except Exception as e:
        st.warning(f"Erreur avec {name}: {e}")

with col1: display_multiclass("Decision Tree", dt_model, df_dt_rf)
with col2: display_multiclass("Random Forest", rf_model, df_dt_rf)
with col3: display_multiclass("XGBoost", xgb_model, df_xgb, is_xgb=True)

