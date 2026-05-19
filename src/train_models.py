from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "figures"
EXPERIMENT_NAME = "smartcity-stress-prediction"

BOOST_DATA_PATH = DATA_DIR / "smart-city-traffic-stress-index-dataset_clean_boost.csv"
LIN_DATA_PATH = DATA_DIR / "smart-city-traffic-stress-index-dataset_clean_lin.csv"

BOOST_FEATURES = [
    "avg_speed",
    "road_quality_score",
    "driver_experience_encoded",
    "weather_Foggy",
    "weather_Hot",
    "weather_Rainy",
    "congestion_score",
    "horn_density",
]
LIN_FEATURES = [
    "road_quality_score",
    "driver_experience_encoded",
    "congestion_score",
]
TARGET_COLUMN = "stress_index"


def ensure_directories():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {path}")
    return pd.read_csv(path)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def log_plot(fig: plt.Figure, filename: str) -> None:
    path = RESULTS_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    mlflow.log_artifact(str(path))


def train_linear_regression(random_state: int = 42) -> LinearRegression:
    ensure_directories()

    df = load_csv(LIN_DATA_PATH)
    X = df[LIN_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, shuffle=True
    )

    model = LinearRegression()
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="LinearRegression_baseline"):
        mlflow.log_params({
            "model_type": "LinearRegression",
            "fit_intercept": model.fit_intercept,
            "normalize": False,
            "random_state": random_state,
        })

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)
        mlflow.log_metrics(metrics)

        fig, ax = plt.subplots(figsize=(8, 5))
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.4, color="steelblue", s=20)
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Prédictions")
        ax.set_ylabel("Résidus")
        ax.set_title("Régression Linéaire — Résidus")
        ax.grid(alpha=0.3)
        log_plot(fig, "residuals_linear_regression.png")
        plt.close(fig)

        mlflow.sklearn.log_model(model, artifact_path="linear-regression-model")

    model_path = MODEL_DIR / "best_model_linear_regression.pkl"
    with open(model_path, "wb") as f:
        pd.to_pickle(model, f)

    return model


def train_xgboost(random_state: int = 42) -> XGBRegressor:
    ensure_directories()

    df = load_csv(BOOST_DATA_PATH)
    X = df[BOOST_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, shuffle=True
    )

    model = XGBRegressor(
        random_state=random_state,
        verbosity=0,
        n_estimators=563,
        learning_rate=0.019,
        max_depth=6,
        subsample=0.693,
        colsample_bytree=0.76,
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="XGBoost_final"):
        mlflow.log_params({
            "model_type": "XGBoost",
            "n_estimators": model.n_estimators,
            "learning_rate": model.learning_rate,
            "max_depth": model.max_depth,
            "subsample": model.subsample,
            "colsample_bytree": model.colsample_bytree,
            "random_state": random_state,
        })

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)
        mlflow.log_metrics(metrics)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        residuals = y_test - y_pred
        axes[0].scatter(y_pred, residuals, alpha=0.4, color="steelblue", s=20)
        axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
        axes[0].set_xlabel("Prédictions")
        axes[0].set_ylabel("Résidus")
        axes[0].set_title("XGBoost — Résidus")
        axes[0].grid(alpha=0.3)

        sns.histplot(residuals, kde=True, ax=axes[1], color="steelblue", bins=40)
        axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
        axes[1].set_title("XGBoost — Distribution des résidus")
        axes[1].grid(alpha=0.3)

        log_plot(fig, "residuals_xgboost.png")
        plt.close(fig)

        mlflow.xgboost.log_model(model, artifact_path="xgboost-model")

    model_path = MODEL_DIR / "best_model_tuned_xgboost.pkl"
    with open(model_path, "wb") as f:
        pd.to_pickle(model, f)

    return model


def main() -> None:
    ensure_directories()
    print("Démarrage du suivi MLflow pour les modèles XGBoost et LinearRegression...")

    try:
        train_linear_regression()
        print("Modèle LinearRegression entraîné et tracé dans MLflow.")
    except Exception as exc:
        print(f"Erreur lors de l'entraînement du modèle Linéaire : {exc}")

    try:
        train_xgboost()
        print("Modèle XGBoost entraîné et tracé dans MLflow.")
    except Exception as exc:
        print(f"Erreur lors de l'entraînement XGBoost : {exc}")


if __name__ == "__main__":
    main()
