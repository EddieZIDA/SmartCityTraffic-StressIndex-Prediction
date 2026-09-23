# Smart City — Prédiction de l'Indice de Stress Urbain

> Projet Data Science complet · EDA → Preprocessing → Modélisation → Déploiement Streamlit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smartcitytraffic-stressindex-prediction.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://github.com/EddieZIDA/SmartCityTraffic-StressIndex-Prediction/actions/workflows/ci.yml/badge.svg)

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
│   ├── best_model_tuned_xgboost.json   # format natif XGBoost (chargé en priorité)
│   ├── best_model_tuned_xgboost.pkl
│   ├── best_model_lightgbm.pkl
│   ├── best_model_linear_regression.pkl
│   ├── scaler.pkl                      # data_lin / LinearRegression uniquement
│   ├── ordinal_encoder.pkl
│   └── best_params.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb              # Analyse exploratoire
│   ├── 02_preprocessing.ipynb   # Preprocessing + feature engineering
│   └── 03_modeling.ipynb        # Modélisation + tuning
│
├── src/
│   └── train_models.py           # Entraînement des 4 modèles + tuning tracé (MLflow)
│
├── results/
│   └── figures/                  # Graphiques exportés
│
├── tests/                        # Tests unitaires (pytest)
│   ├── conftest.py
│   ├── test_preprocessing.py
│   └── test_model_contract.py    # Contrat dataset ↔ modèle ↔ app
│
├── .github/workflows/ci.yml      # CI : pytest + flake8 sur Python 3.11 et 3.12
│
├── .flake8                       # Config lint (racine : couvre app/, src/, tests/)
├── pytest.ini
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Méthodologie

### 1. Analyse exploratoire (EDA)

Notebook `01_eda.ipynb` — contrôle de la qualité des données (0 valeur manquante,
0 doublon sur 50 000 observations), distributions univariées, détection des
valeurs extrêmes et premières corrélations avec `stress_index`.
Figures exportées dans `results/figures/` : `distributions.png`,
`boxplots.png`, `correlation_matrix.png`, `pairplot.png`,
`stress_vs_categorical.png`.

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

La suppression de `speed_efficiency` a été re-testée : l'ajouter donne un R² en
validation croisée de 0.9104 contre 0.9106 sans elle. Elle n'apporte donc rien
une fois `avg_speed` et `traffic_density` toutes deux présentes.

**Traitement de la multicolinéarité (VIF) :**
- Variables à fort VIF : `traffic_density` (269), `signal_wait_time` (239), `horn_events_per_min` (39)
- **Écartées de `data_lin` seulement.** Un VIF élevé déstabilise les coefficients
  d'une régression linéaire ; les modèles à base d'arbres n'estiment pas de
  coefficients et n'en souffrent pas. Les retirer de `data_boost` coûtait
  **0.0014 de R²**, mesuré en validation croisée appariée (gagnant sur 25 folds
  sur 25, p = 3×10⁻¹⁹) puis confirmé sur le jeu de test.
- VIF final `data_lin` : tous < 3.5

**Deux datasets distincts selon le modèle :**

| Dataset | Features | Usage |
|---------|----------|-------|
| `data_lin` (3) | `congestion_score`, `driver_experience_encoded`, `road_quality_score` | Régression linéaire (VIF strict) |
| `data_boost` (11) | + `traffic_density`, `signal_wait_time`, `horn_events_per_min`, `avg_speed`, `horn_density`, `weather_*` | RandomForest, XGBoost, LightGBM |

**Normalisation :** `StandardScaler` appliqué uniquement sur `data_lin`, après le split (pas de data leakage).

### 3. Modélisation

**Split :** 80% train / 20% test - `random_state=42`

**Résultats avant tuning :**

| Modèle | R² test | RMSE | MAE | Gap overfit |
|--------|---------|------|-----|-------------|
| LinearRegression | 0.8587 | 6.107 | 4.871 | 0.0049 |
| RandomForest | 0.9039 | 5.037 | 4.031 | **0.0828** |
| XGBoost | 0.9056 | 4.992 | 3.998 | 0.0261 |
| LightGBM | 0.9089 | 4.903 | 3.928 | 0.0093 |

**Tuning** - `RandomizedSearchCV` (60 itérations × 5 folds KFold) :

> La grille initiale plafonnait à `n_estimators=600`. L'optimum réel se situe
> vers 930 arbres à faible learning rate : il était **hors de portée de la
> recherche**, qui ne pouvait donc pas le trouver. L'élargissement de la grille
> a suffi à le faire apparaître.

**Résultats après tuning :**

| Modèle | R² avant | R² après | RMSE après | Gain | Gap |
|--------|----------|----------|------------|------|-----|
| RandomForest | 0.9039 | 0.9083 | 4.919 | +0.0044 | 0.0254 |
| XGBoost | 0.9056 | **0.9100** | **4.873** | +0.0044 | 0.0046 |
| LightGBM | 0.9089 | 0.9096 | 4.886 | +0.0006 | 0.0076 |

Le `Gap` est ici `R² train − R² test`. Les versions antérieures de ce tableau
comparaient le R² d'entraînement au R² de validation croisée, deux estimateurs
différents, ce qui sous-estimait l'écart réel.

> **Reproductibilité du tuning :** `RandomizedSearchCV` tire ses candidats via
> `ParameterSampler`, dont l'implémentation dépend de la version de
> scikit-learn. À `random_state=42` constant, un changement de version modifie
> l'ensemble des candidats évalués et donc l'optimum retenu — constaté ici en
> passant d'une version à l'autre. Les hyperparamètres ci-dessous ont été
> obtenus avec **scikit-learn 1.4.2**, la version épinglée dans
> `requirements.txt` ; c'est elle qu'il faut installer pour les retrouver.

---

## Résultats

### Meilleur modèle : XGBoost tuné

```
R²   = 0.9100  →  explique 91.0% de la variance du stress_index
RMSE = 4.873   →  erreur quadratique moyenne sur échelle 0-100
MAE  = 3.901   →  erreur absolue moyenne de ±3.9 points
Gap  = 0.0046  →  pas d'overfitting significatif
```

**Hyperparamètres optimaux :**
```python
XGBRegressor(
    n_estimators     = 933,
    learning_rate    = 0.0208,
    max_depth        = 3,
    min_child_weight = 4,
    subsample        = 0.8418,
    colsample_bytree = 0.6102,
    reg_lambda       = 0.6967,
    random_state     = 42
)
```

`src/train_models.py` reprend ces valeurs en pleine précision : les arrondir
produit un modèle légèrement différent de celui livré.

> **Insight clé :** le trio de la congestion — `traffic_density`,
> `congestion_score` et `signal_wait_time` — concentre 72 % à 76 % du gain selon
> le modèle. L'importance se répartit entre la feature composite et les deux
> variables dont elle est le produit : c'est l'effet attendu de la colinéarité,
> deux features redondantes se partageant le crédit. Ce classement ne se lit
> donc pas comme une hiérarchie causale.
>
> La relation est majoritairement linéaire (R²=0.859 en régression) — les arbres
> capturent les ~5 % d'interactions non-linéaires restants.

> **Où est le plafond ?** Un XGBoost à très forte capacité (2 000 arbres,
> profondeur 8) obtient un R² en validation croisée de **0.905**, soit *moins*
> que le modèle retenu : ajouter de la capacité ne fait plus que sur-apprendre.
> Moyenner XGBoost, LightGBM et RandomForest donne 0.911, également en deçà du
> meilleur modèle seul. Le signal exploitable de ce dataset est donc épuisé
> autour de R² ≈ 0.91, le reste étant du bruit irréductible.

---

## Application Streamlit

**Application déployée :** [smartcitytraffic-stressindex-prediction.streamlit.app](https://smartcitytraffic-stressindex-prediction.streamlit.app/)

### Pages disponibles

| Page | Contenu |
|------|---------|
| **Prédiction** | Saisie des conditions de circulation, jauge du stress prédit, contribution estimée des variables et simulation de l'effet d'un paramètre sur toute sa plage |
| **Exploration** | Distributions, corrélations et relations du dataset brut, avec filtres météo / expérience / plage de stress |
| **Performance** | Comparaison des 4 modèles avant et après tuning, analyse de l'overfitting, importances des features et résidus calculés en direct sur `data_boost` |

Le modèle est chargé depuis `models/best_model_tuned_xgboost.json` (format natif
XGBoost), avec repli sur le `.pkl` si le premier est absent.

---

## Installation

### Prérequis

- Python 3.11 ou 3.12 (les deux sont validées en CI)
- Git

> Sur Streamlit Community Cloud, la version de Python ne se déclare pas dans un
> fichier du dépôt : elle se choisit dans les *Advanced settings* au moment du
> déploiement. Elle doit correspondre à celle utilisée en développement.

### Cloner et installer

```bash
git clone https://github.com/EddieZIDA/SmartCityTraffic-StressIndex-Prediction.git
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

### Ré-entraîner les modèles (tracking MLflow)

```bash
python src/train_models.py
```

Le script crée un experiment MLflow nommé `smartcity-stress-prediction` et lance
un run par modèle : `LinearRegression_baseline`, `RandomForest_final`,
`XGBoost_final` et `LightGBM_final`. Chaque run enregistre :

- les hyperparamètres et le `random_state` ;
- les métriques **train et test** (`R2`, `RMSE`, `MAE`) ainsi que `gap_R2`,
  l'écart de généralisation — ne logger que le test masquerait l'overfitting,
  or c'est lui qui a départagé les modèles de ce projet ;
- le modèle avec sa **signature** et un `input_example`, pour que MLflow valide
  le schéma d'entrée au chargement ;
- des tags identifiant le dataset et son empreinte SHA-256, sans quoi deux runs
  aux scores différents seraient indiscernables d'un changement de données.

Il écrit `models/best_model_tuned_xgboost.json` (format natif XGBoost, chargé en
priorité par l'application), le `.pkl` équivalent et
`models/best_model_linear_regression.pkl`. Il retourne un code de sortie non nul
si une étape échoue.

Pour rejouer la recherche d'hyperparamètres — chaque candidat évalué devient un
run imbriqué avec ses paramètres et son score de validation croisée :

```bash
python src/train_models.py --tune          # long : 60 itérations x 5 folds x 3 modèles
python src/train_models.py --tune --n-iter 5   # version courte, pour vérifier la mécanique
```

Les optima trouvés ne sont pas repris automatiquement dans `BEST_PARAMS` : le
script le signale, à reporter manuellement après vérification.

Consulter les runs :

```bash
mlflow ui
```

### Lancer les tests

```bash
pytest
flake8 app src tests
```

Les tests couvrent le preprocessing d'une observation isolée (`congestion_score`,
`horn_density`, encodage ordinal, one-hot météo, seuils de niveau de stress) et
le **contrat entre le dataset, le modèle et l'application** :

- les colonnes de `data_boost` contiennent bien les features attendues ;
- l'ordre des features du booster correspond exactement à celui de l'app — un
  simple réordonnancement ne lève pas toujours d'erreur mais fausse les
  prédictions ;
- les hyperparamètres de `BEST_PARAMS` correspondent à l'optimum sauvegardé dans
  `best_params.pkl`, et le modèle livré les porte réellement.

Cette dernière vérification verrouille une dérive constatée sur ce projet : un
script entraînant avec des hyperparamètres arrondis, donc un modèle livré qui
n'était plus celui que la recherche avait sélectionné, sans qu'aucune erreur ne
le signale.

La CI GitHub Actions exécute `pytest` et `flake8` sur Python 3.11 et 3.12 à
chaque push et pull request, en installant `requirements.txt` en entier — ce qui
valide aussi que le jeu de versions épinglé s'installe réellement.

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
| Langage | Python | 3.11 / 3.12 |
| Manipulation données | pandas | 2.2.2 |
| Calcul numérique | numpy | 1.26.4 |
| Visualisation EDA | matplotlib, seaborn | 3.10.8 / 0.13.2 |
| Machine Learning | scikit-learn | 1.4.2 |
| Gradient Boosting | XGBoost | 2.1.1 |
| Gradient Boosting | LightGBM | 4.3.0 |
| Statistiques | scipy, statsmodels | 1.13.1 / 0.14.6 |
| Application web | Streamlit | 1.60.0 |
| Visualisation interactive | Plotly | 5.22.0 |
| Tracking d'expériences | MLflow | 2.16.2 |
| Tests / lint | pytest, flake8 | 8.3.2 / 7.1.1 |

---

## Références

- [Smart City Traffic Stress Index Dataset](https://www.kaggle.com/datasets/sonalshinde123/smart-city-traffic-stress-index-dataset/data)
- [Traffic Stress Index EDA & Prediction (XGBoost)](https://www.kaggle.com/code/pialghosh/traffic-stress-index-eda-prediction-xgboost)
- [Smart City Traffic Stress Insights](https://www.kaggle.com/code/sumedh1507/smart-city-traffic-stress-insights)

---

## Licence

Ce projet est distribué sous licence MIT — voir [LICENSE](LICENSE).

---

## Auteur

**ZIDA Wend Kouni Eddie Eliel**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/wend-kouni-eddie-eliel-zida-501815260/?skipRedirect=true)

---

*Projet réalisé dans le cadre d'une formation certifiante en Data Science de Africa Techup Tour · Avril 2026*
