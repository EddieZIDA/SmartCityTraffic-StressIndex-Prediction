# Smart City — Prédiction de l'Indice de Stress Urbain

> Projet Data Science complet · EDA → Preprocessing → Modélisation → Déploiement Streamlit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smartcitytraffic-stressindex-prediction.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Contexte et objectif

Dans le contexte des **Smart Cities**, la gestion du trafic urbain est un enjeu majeur pour la qualité de vie des habitants. La congestion routière, les longues attentes aux feux et les comportements agressifs au volant génèrent un stress croissant chez les conducteurs, avec des impacts directs sur la sécurité routière et la santé publique.

**Objectif :** Construire un modèle de Machine Learning capable de **prédire l'indice de stress des conducteurs** (`stress_index`, échelle 0-100) à partir de variables mesurables en temps réel.

**Questions clés adressées :**
- Quelles variables influencent le plus le stress des conducteurs ?
- La relation est-elle linéaire ou non-linéaire ?
- Comment gérer la forte multicolinéarité entre les variables de trafic ?
- Quel modèle offre le meilleur compromis performance / généralisation ?

---

## Dataset

| Attribut | Valeur |
|----------|--------|
| **Source** | [Smart City Traffic Stress Index Dataset - Kaggle](https://www.kaggle.com/datasets/sonalshinde123/smart-city-traffic-stress-index-dataset/data) |
| **Observations** | 50 000 |
| **Variables** | 8 (6 numériques + 2 catégorielles) |
| **Valeurs manquantes** | 0 |
| **Doublons** | 0 |

### Variables originales

| Variable | Type | Description | Plage |
|----------|------|-------------|-------|
| traffic_density | int64 | Nombre de véhicules en circulation | 10 – 119 |
| horn_events_per_min | float64 | Klaxons par minute | 0 – 24.86 |
| avg_speed | float64 | Vitesse moyenne (km/h) | 13.86 – 90 |
| signal_wait_time | float64 | Attente aux feux (secondes) | 5 – 74.5 |
| weather_condition | object | Météo (Clear/Foggy/Hot/Rainy) | 4 modalités |
| road_quality_score | float64 | Qualité de la route (0-10) | 1 – 10 |
| driver_experience_level | object | Expérience du conducteur | 3 modalités |
| stress_index | float64 | **Variable cible** | 0 – 100 |

---

## Structure du projet

```
SmartCityTraffic-StressIndex-Prediction/
│
├── app/                          # Application Streamlit
│   ├── app.py                    # Page d'accueil
│   ├── utils.py                  # Chargement modèle, données, preprocessing
│   └── pages/
│       ├── 01_prediction.py      # Prédiction temps réel + simulation
│       ├── 02_exploration.py     # Exploration interactive du dataset
│       └── 03_performance.py     # Comparaison des modèles + résidus
│
├── data/
│   ├── raw/                      # Données brutes (CSV original)
│   └── processed/                # Datasets prétraités (lin + boost)
│
├── models/                       # Modèles et artefacts sauvegardés
│   ├── best_model_tuned_xgboost.pkl
│   ├── scaler.pkl
│   ├── ordinal_encoder.pkl
│   └── best_params.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb              # Analyse exploratoire
│   ├── 02_preprocessing.ipynb   # Preprocessing + feature engineering
│   └── 03_modeling.ipynb        # Modélisation + tuning
│
├── results/
│   └── figures/                  # Graphiques exportés
│
├── requirements.txt
└── README.md
```

---


### 2. Preprocessing

**Encodage :**
- `driver_experience_level` → **OrdinalEncoder** (Beginner=0, Intermediate=1, Expert=2)
- `weather_condition` → **One-Hot Encoding** avec `drop_first=True` → 3 colonnes binaires

**Feature Engineering :**

| Feature créée | Formule | Corrélation avec target | Décision |
|---------------|---------|------------------------|----------|
| `congestion_score` | `traffic_density × signal_wait_time / 100` | **+0.835** | Conservée |
| `horn_density` | `horn_events_per_min / (traffic_density + 1)` | -0.169 | Conservée (data_boost) |
| `speed_efficiency` | `avg_speed / (traffic_density + 1)` | -0.705 | Supprimée (redondante) |

**Traitement de la multicolinéarité (VIF) :**
- Variables supprimées : `traffic_density` (VIF=269), `signal_wait_time` (VIF=239), `horn_events_per_min` (VIF=39)
- VIF final `data_lin` : tous < 3.5 

**Deux datasets distincts selon le modèle :**

| Dataset | Features | Usage |
|---------|----------|-------|
| `data_lin` | `congestion_score`, `driver_experience_encoded`, `road_quality_score` | Régression linéaire (VIF strict) |
| `data_boost` | + `avg_speed`, `horn_density`, `weather_*` | RandomForest, XGBoost, LightGBM |

**Normalisation :** `StandardScaler` appliqué uniquement sur `data_lin`, après le split (pas de data leakage).

### 3. Modélisation

**Split :** 80% train / 20% test - `random_state=42`

**Résultats avant tuning :**

| Modèle | R² test | RMSE | MAE | Gap overfit |
|--------|---------|------|-----|-------------|
| LinearRegression | 0.8587 | 6.107 | 4.871 | 0.0049 |
| RandomForest | 0.9019 | 5.088 | 4.078 | **0.0844** |
| XGBoost | 0.9045 | 5.022 | 4.017 | 0.0256 |
| LightGBM | 0.9086 | 4.913 | 3.930 | 0.0084 |

**Tuning** - `RandomizedSearchCV` (30 itérations × 5 folds KFold) :

**Résultats après tuning :**

| Modèle | R² avant | R² après | RMSE après | Gain | Gap |
|--------|----------|----------|------------|------|-----|
| RandomForest | 0.9019 | 0.9077 | 4.937 | +0.0058 | 0.019 |
| XGBoost | 0.9045 | **0.9090** | **4.902** | +0.0045 | 0.011 |
| LightGBM | 0.9086 | 0.9085 | 4.915 | -0.0001 | 0.004|

---

## Résultats

### Meilleur modèle : XGBoost tuné

```
R²   = 0.9090  →  explique 90.9% de la variance du stress_index
RMSE = 4.902   →  erreur quadratique moyenne sur échelle 0-100
MAE  = 3.927   →  erreur absolue moyenne de ±3.9 points
Gap  = 0.011   →  pas d'overfitting significatif
```

**Hyperparamètres optimaux :**
```python
XGBRegressor(
    n_estimators     = 563,
    learning_rate    = 0.019,
    max_depth        = 6,
    subsample        = 0.693,
    colsample_bytree = 0.760,
    random_state     = 42
)
```

> **Insight clé :** `congestion_score` (feature engineered) est de loin la plus prédictive, validant l'approche de feature engineering. La relation est majoritairement linéaire (R²=0.859 en régression) - les arbres capturent les 5% d'interactions non-linéaires restants.

---

## Application Streamlit

**Application déployée :** [smartcitytraffic-stressindex-prediction.streamlit.app](https://smartcitytraffic-stressindex-prediction.streamlit.app/)

### Pages disponibles

**Prédiction** - Prédiction en temps réel      
**Exploration** - Analyse interactive du dataset     
**Performance** - Comparaison des modèles      

---

## Installation

### Prérequis

- Python 3.11+
- Git

### Cloner et installer

```bash
git clone https://github.com/ton-username/SmartCityTraffic-StressIndex-Prediction.git
cd SmartCityTraffic-StressIndex-Prediction

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### Lancer l'application

```bash
cd app
streamlit run app.py
```

L'app s'ouvre sur `http://localhost:8501`.

### Reproduire les notebooks

```bash
cd notebooks
jupyter notebook
```

Exécuter dans l'ordre : `01_eda.ipynb` → `02_preprocessing.ipynb` → `03_modeling.ipynb`

---

## Technologies utilisées

| Catégorie | Outil | Version |
|-----------|-------|---------|
| Langage | Python | 3.11 |
| Manipulation données | pandas | 2.2.2 |
| Calcul numérique | numpy | 1.26.4 |
| Visualisation EDA | matplotlib, seaborn | 3.9.0 / 0.13.2 |
| Machine Learning | scikit-learn | 1.5.1 |
| Gradient Boosting | XGBoost | 2.1.1 |
| Gradient Boosting | LightGBM | 4.5.0 |
| Statistiques | scipy, statsmodels | 1.13.1 / 0.14.2 |
| Application web | Streamlit | 1.38.0 |
| Visualisation interactive | Plotly | 5.22.0 |

---

## Références

- [Smart City Traffic Stress Index Dataset](https://www.kaggle.com/datasets/sonalshinde123/smart-city-traffic-stress-index-dataset/data)
- [Traffic Stress Index EDA & Prediction (XGBoost)](https://www.kaggle.com/code/pialghosh/traffic-stress-index-eda-prediction-xgboost)
- [Smart City Traffic Stress Insights](https://www.kaggle.com/code/sumedh1507/smart-city-traffic-stress-insights)

---

## Auteur

**ZIDA Wend Kouni Eddie Eliel**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/wend-kouni-eddie-eliel-zida-501815260/?skipRedirect=true)

---

*Projet réalisé dans le cadre d'une formation certifiante en Data Science de Africa Techup Tour · Avril 2026*
