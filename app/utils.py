import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Chemins absolus depuis la racine du repo ──────────────────────
# utils.py est dans .../app/
# PROJECT_ROOT est .../SmartCityTraffic-StressIndex-Prediction/
APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model_tuned_xgboost.pkl"
BOOST_PATH = PROJECT_ROOT / "data" / "processed" / "smart-city-traffic-stress-index-dataset_clean_boost.csv"
RAW_PATH   = PROJECT_ROOT / "data" / "raw"       / "smart_city_traffic_stress_dataset.csv"

# Colonnes exactes de data_boost (02_preprocessing.ipynb cell 4)
# ['avg_speed', 'road_quality_score', 'stress_index',
#  'driver_experience_encoded', 'weather_Foggy', 'weather_Hot',
#  'weather_Rainy', 'congestion_score', 'horn_density']
FEATURE_COLS = [
    "avg_speed",
    "road_quality_score",
    "driver_experience_encoded",
    "weather_Foggy",
    "weather_Hot",
    "weather_Rainy",
    "congestion_score",
    "horn_density",
]

EXP_MAP = {"Beginner": 0, "Intermediate": 1, "Expert": 2}


@st.cache_resource
def load_model():
    """
    Charge best_model_tuned_xgboost.pkl
    Entraîné sur data_boost SANS scaling (XGBoost n'en a pas besoin).
    """
    if not MODEL_PATH.exists():
        files = [p.name for p in MODEL_PATH.parent.iterdir()] \
            if MODEL_PATH.parent.exists() else ["dossier absent"]
        st.error(
            f"Modèle introuvable : {MODEL_PATH}\n"
            f"Fichiers présents : {files}"
        )
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_raw_data():
    """Charge le dataset brut pour la page Exploration."""
    if not RAW_PATH.exists():
        st.error(f"Dataset brut introuvable : {RAW_PATH}")
        return pd.DataFrame()
    return pd.read_csv(RAW_PATH)


@st.cache_data
def load_boost_data():
    """Charge data_boost pour le calcul des résidus (page Performance)."""
    if not BOOST_PATH.exists():
        files = [p.name for p in BOOST_PATH.parent.iterdir()] \
            if BOOST_PATH.parent.exists() else ["dossier absent"]
        st.error(
            f"Dataset boost introuvable : {BOOST_PATH}\n"
            f"Fichiers présents : {files}"
        )
        return pd.DataFrame()
    return pd.read_csv(BOOST_PATH)


def preprocess_input(traffic_density, signal_wait_time, avg_speed,
                     road_quality, experience, weather,
                     horn_events=8.82):
    """
    Reproduit exactement le preprocessing de 02_preprocessing.ipynb
    pour une observation unique à soumettre au modèle XGBoost.

    IMPORTANT : PAS de StandardScaler ici.
    Le scaler s'applique uniquement à data_lin (régression linéaire).
    XGBoost a été entraîné sur data_boost brut.
    """
    congestion_score = (traffic_density * signal_wait_time) / 100
    horn_density     = horn_events / (traffic_density + 1)

    return pd.DataFrame([{
        "avg_speed"                : avg_speed,
        "road_quality_score"       : road_quality,
        "driver_experience_encoded": EXP_MAP.get(experience, 0),
        "weather_Foggy"            : 1 if weather == "Foggy"  else 0,
        "weather_Hot"              : 1 if weather == "Hot"    else 0,
        "weather_Rainy"            : 1 if weather == "Rainy"  else 0,
        "congestion_score"         : congestion_score,
        "horn_density"             : horn_density,
    }])[FEATURE_COLS]


def stress_level(score):
    """Retourne le label et la couleur selon le score."""
    if score >= 70:
        return "Élevé", "#dc3545"
    elif score >= 40:
        return "Modéré", "#fd7e14"
    else:
        return "Faible", "#198754"
    
