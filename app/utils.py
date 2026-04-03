import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# Résolution dynamique des chemins
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

DATA_FILE = "smart_city_traffic_stress_dataset.csv"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / DATA_FILE

MODEL_FILE = "best_model_tuned_xgboost.pkl"
MODEL_PATH = PROJECT_ROOT / "models" / MODEL_FILE


@st.cache_resource
def load_model():
    """Charge le modèle XGBoost tuné."""
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Fichier modèle introuvable au chemin : {MODEL_PATH}")
        return None


@st.cache_data
def load_data():
    """Charge le dataset brut pour la page d'exploration."""
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        st.error(f"Dataset introuvable au chemin : {DATA_PATH}")
        return pd.DataFrame()


def preprocess_input(input_data):
    """
    Applique l'encodage et le feature engineering sur les données saisies,
    pour les aligner sur la structure de 'data_boost' (50000x9).
    """
    # 1. Encodage Ordinal
    exp_mapping = {"Beginner": 0, "Intermediate": 1, "Expert": 2}
    exp_encoded = exp_mapping.get(input_data["driver_experience_level"], 0)

    # 2. Encodage One-Hot
    weather = input_data["weather_condition"]
    w_foggy = 1 if weather == "Foggy" else 0
    w_hot = 1 if weather == "Hot" else 0
    w_rainy = 1 if weather == "Rainy" else 0

    # 3. Feature Engineering
    avg_speed = input_data["avg_speed"]
    traffic_density = input_data["traffic_density"]
    horn_events = input_data["horn_events_per_min"]
    signal_wait_time = input_data["signal_wait_time"]

    congestion_score = (traffic_density * signal_wait_time) / 100
    horn_density = horn_events / (traffic_density + 1)

    # 4. Construction du DataFrame final
    expected_columns = [
        'avg_speed',
        'road_quality_score',
        'driver_experience_encoded',
        'weather_Foggy',
        'weather_Hot',
        'weather_Rainy',
        'congestion_score',
        'horn_density'
    ]

    df_processed = pd.DataFrame([{
        'avg_speed': avg_speed,
        'road_quality_score': input_data["road_quality_score"],
        'driver_experience_encoded': exp_encoded,
        'weather_Foggy': w_foggy,
        'weather_Hot': w_hot,
        'weather_Rainy': w_rainy,
        'congestion_score': congestion_score,
        'horn_density': horn_density
    }])

    return df_processed[expected_columns]
