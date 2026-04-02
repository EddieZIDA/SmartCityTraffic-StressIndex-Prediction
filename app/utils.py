import pickle
import pandas as pd
import streamlit as st
from pathlib import Path

# --- GESTION DES CHEMINS DYNAMIQUE ---
# On définit ROOT_DIR de manière à ce qu'il trouve les dossiers models/ et data/ 
# peu importe si on lance depuis la racine ou depuis le dossier app/
CURRENT_DIR = Path(__file__).resolve().parent
if (CURRENT_DIR / "models").exists():
    ROOT_DIR = CURRENT_DIR
elif (CURRENT_DIR.parent / "models").exists():
    ROOT_DIR = CURRENT_DIR.parent
else:
    ROOT_DIR = Path.cwd()

MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

EXPERIENCE_MAP = {"Beginner": 0, "Intermediate": 1, "Expert": 2}

# L'ordre exact des colonnes attendu par votre modèle XGBoost
FEATURE_COLS_BOOST = [
    "avg_speed", "road_quality_score", "driver_experience_encoded",
    "weather_Foggy", "weather_Hot", "weather_Rainy",
    "congestion_score", "horn_density"
]

@st.cache_resource
def load_model():
    """Charge le modèle XGBoost tuné."""
    path = MODELS_DIR / "best_model_tuned_xgboost.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable à l'emplacement : {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    """Charge le scaler utilisé lors de l'entraînement."""
    path = MODELS_DIR / "scaler.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Scaler introuvable à l'emplacement : {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    """Charge les datasets pour l'exploration."""
    processed_dir = DATA_DIR / "processed"
    # Liste des noms de fichiers possibles
    candidates = [
        "smart-city-traffic-stress-index-dataset_clean_2.csv",
        "data_boost.csv",
        "smart-city-traffic-stress-index-dataset_clean_boost.csv",
    ]
    
    df = None
    for filename in candidates:
        path = processed_dir / filename
        if path.exists():
            df = pd.read_csv(path)
            break

    if df is None:
        raise FileNotFoundError(f"Aucun dataset valide trouvé dans {processed_dir}")

    raw_path = DATA_DIR / "raw" / "smart_city_traffic_stress_dataset.csv"
    raw = pd.read_csv(raw_path)
    return df, raw

def build_input(traffic_density, signal_wait_time, avg_speed,
                road_quality, experience, weather, horn_events=8.82):
    """Prépare les données d'entrée pour la prédiction."""
    # Recréation du Feature Engineering identique à l'entraînement
    congestion_score = (traffic_density * signal_wait_time) / 100
    horn_density = horn_events / (traffic_density + 1)
    
    input_data = pd.DataFrame({
        "avg_speed": [float(avg_speed)],
        "road_quality_score": [float(road_quality)],
        "driver_experience_encoded": [EXPERIENCE_MAP[experience]],
        "weather_Foggy": [1 if weather == "Foggy" else 0],
        "weather_Hot": [1 if weather == "Hot" else 0],
        "weather_Rainy": [1 if weather == "Rainy" else 0],
        "congestion_score": [float(congestion_score)],
        "horn_density": [float(horn_density)],
    })
    
    # Réorganisation des colonnes dans l'ordre exact
    return input_data[FEATURE_COLS_BOOST]

def stress_level(score):
    """Retourne le libellé et la couleur selon le score prédit."""
    if score >= 70:
        return "Élevé", "#dc3545"
    elif score >= 40:
        return "Modéré", "#fd7e14"
    else:
        return "Faible", "#198754"