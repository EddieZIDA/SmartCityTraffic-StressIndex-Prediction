import pickle
import pandas as pd
import streamlit as st

EXPERIENCE_MAP = {"Beginner": 0, "Intermediate": 1, "Expert": 2}

FEATURE_COLS_BOOST = [
    "avg_speed", "road_quality_score", "driver_experience_encoded",
    "weather_Foggy", "weather_Hot", "weather_Rainy",
    "congestion_score", "horn_density"
]


@st.cache_resource
def load_model():
    with open("../models/best_model_tuned_xgboost.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    with open("../models/scaler.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    # Essaie les deux noms de fichiers possibles
    import os
    candidates = [
        "../data/processed/smart-city-traffic-stress-index-dataset_clean_2.csv",
        "../data/processed/data_boost.csv",
        "../data/processed/smart-city-traffic-stress-index-dataset_clean_boost.csv",
    ]
    df = None
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    if df is None:
        raise FileNotFoundError(
            "Dataset introuvable. Vérifiez ../data/processed/"
        )
    raw = pd.read_csv("../data/raw/smart_city_traffic_stress_dataset.csv")
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