import pickle
import pandas as pd
import streamlit as st
import os
from pathlib import Path

# Chemin absolu vers la racine du repo
# utils.py est dans app/ donc on remonte d'un niveau
APP_DIR = Path(__file__).resolve().parent      # .../app/
ROOT_DIR = APP_DIR.parent                       # .../SmartCityTraffic-StressIndex-Prediction/

MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

EXPERIENCE_MAP = {"Beginner": 0, "Intermediate": 1, "Expert": 2}

FEATURE_COLS_BOOST = [
    "avg_speed", "road_quality_score", "driver_experience_encoded",
    "weather_Foggy", "weather_Hot", "weather_Rainy",
    "congestion_score", "horn_density"
]


@st.cache_resource
def load_model():
    path = MODELS_DIR / "best_model_tuned_xgboost.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {path}\n"
            f"Contenu de {MODELS_DIR} : "
            f"{list(MODELS_DIR.iterdir()) if MODELS_DIR.exists() else 'dossier absent'}"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    path = MODELS_DIR / "scaler.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    processed_dir = DATA_DIR / "processed"
    candidates = [
        processed_dir / "smart-city-traffic-stress-index-dataset_clean_2.csv",
        processed_dir / "data_boost.csv",
        processed_dir / "smart-city-traffic-stress-index-dataset_clean_boost.csv",
    ]
    df = None
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            break

    if df is None:
        existing = (
            [str(p.name) for p in processed_dir.iterdir()]
            if processed_dir.exists()
            else ["dossier absent"]
        )
        raise FileNotFoundError(
            f"Dataset introuvable dans {processed_dir}\n"
            f"Fichiers présents : {existing}"
        )

    raw_path = DATA_DIR / "raw" / "smart_city_traffic_stress_dataset.csv"
    raw = pd.read_csv(raw_path)
    return df, raw


def build_input(traffic_density, signal_wait_time, avg_speed,
                road_quality, experience, weather, horn_events=8.82):
    congestion_score = (traffic_density * signal_wait_time) / 100
    horn_density = horn_events / (traffic_density + 1)
    return pd.DataFrame({
        "avg_speed": [avg_speed],
        "road_quality_score": [road_quality],
        "driver_experience_encoded": [EXPERIENCE_MAP[experience]],
        "weather_Foggy": [1 if weather == "Foggy" else 0],
        "weather_Hot": [1 if weather == "Hot" else 0],
        "weather_Rainy": [1 if weather == "Rainy" else 0],
        "congestion_score": [congestion_score],
        "horn_density": [horn_density],
    })[FEATURE_COLS_BOOST]


def stress_level(score):
    if score >= 70:
        return "Élevé", "#dc3545"
    elif score >= 40:
        return "Modéré", "#fd7e14"
    else:
        return "Faible", "#198754"