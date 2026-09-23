"""Entraînement et suivi MLflow des modèles de prédiction du stress index.

Usage :
    python src/train_models.py              # entraîne les 4 modèles finaux
    python src/train_models.py --tune       # rejoue aussi la recherche d'hyperparamètres

Chaque run logge les hyperparamètres, les métriques train ET test, l'écart de
généralisation, une empreinte du dataset et le modèle avec sa signature.
"""
import argparse
import hashlib
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "figures"
EXPERIMENT_NAME = "smartcity-stress-prediction"

BOOST_DATA_PATH = DATA_DIR / "smart-city-traffic-stress-index-dataset_clean_boost.csv"
LIN_DATA_PATH = DATA_DIR / "smart-city-traffic-stress-index-dataset_clean_lin.csv"

# Ordre exact des colonnes de data_boost (stress_index exclu). Les features
# brutes à fort VIF sont conservées : la multicolinéarité ne pénalise que les
# modèles à coefficients, et les retirer coûtait 0.0014 de R² en CV appariée.
BOOST_FEATURES = [
    "traffic_density",
    "horn_events_per_min",
    "avg_speed",
    "signal_wait_time",
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

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Optima du RandomizedSearchCV de 03_modeling.ipynb, en pleine précision :
# arrondir ces valeurs produit des modèles différents de ceux livrés.
BEST_PARAMS = {
    "XGBoost": {
        "n_estimators": 933,
        "learning_rate": 0.020789142699330444,
        "max_depth": 3,
        "min_child_weight": 4,
        "subsample": 0.8417669517111269,
        "colsample_bytree": 0.610167650697638,
        "reg_lambda": 0.6966572720293784,
    },
    "RandomForest": {
        "n_estimators": 443,
        "max_depth": 13,
        "min_samples_leaf": 10,
        "max_features": 0.5,
    },
    "LightGBM": {
        "n_estimators": 755,
        "learning_rate": 0.01313132924555586,
        "max_depth": 5,
        "num_leaves": 53,
        "min_child_samples": 35,
        "subsample": 0.7580600944007257,
        "colsample_bytree": 0.8270801311279966,
    },
}

# Grilles de 03_modeling.ipynb, rejouées telles quelles par --tune.
PARAM_GRIDS = {
    "RandomForest": {
        "n_estimators": randint(100, 500),
        "max_depth": randint(5, 20),
        "min_samples_leaf": randint(2, 20),
        "max_features": ["sqrt", "log2", 0.5],
    },
    # Plages élargies : l'ancienne grille plafonnait à 600 arbres alors que
    # l'optimum se situe vers 900 à faible learning_rate — il était donc hors
    # de portée de la recherche.
    "XGBoost": {
        "n_estimators": randint(200, 1200),
        "learning_rate": uniform(0.01, 0.1),
        "max_depth": randint(3, 9),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 20),
        "reg_lambda": uniform(0, 5),
    },
    "LightGBM": {
        "n_estimators": randint(200, 1200),
        "learning_rate": uniform(0.01, 0.1),
        "max_depth": randint(4, 12),
        "num_leaves": randint(20, 100),
        "min_child_samples": randint(10, 50),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
    },
}


def ensure_directories():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {path}")
    return pd.read_csv(path)


def dataset_fingerprint(path: Path) -> str:
    """Empreinte SHA-256 du CSV source.

    Une métrique n'est interprétable que si l'on sait sur quelles données elle a
    été obtenue : sans cette empreinte, deux runs aux scores différents sont
    indiscernables d'un changement de données silencieux.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def log_split_metrics(model, X_train, y_train, X_test, y_test) -> dict:
    """Logge les métriques train et test, plus l'écart de généralisation.

    Ne logger que le test masque l'overfitting : c'est précisément l'écart
    train/test qui a départagé les modèles de ce projet.
    """
    train_metrics = evaluate_predictions(y_train, model.predict(X_train))
    test_metrics = evaluate_predictions(y_test, model.predict(X_test))

    payload = {f"train_{k}": v for k, v in train_metrics.items()}
    payload.update({f"test_{k}": v for k, v in test_metrics.items()})
    payload["gap_R2"] = train_metrics["R2"] - test_metrics["R2"]
    mlflow.log_metrics(payload)
    return payload


def log_plot(fig: plt.Figure, filename: str) -> None:
    path = RESULTS_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    mlflow.log_artifact(str(path))


def log_residual_plots(y_test, y_pred, title: str, filename: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    residuals = y_test - y_pred

    axes[0].scatter(y_pred, residuals, alpha=0.4, color="steelblue", s=20)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Prédictions")
    axes[0].set_ylabel("Résidus")
    axes[0].set_title(f"{title} — Résidus")
    axes[0].grid(alpha=0.3)

    sns.histplot(residuals, kde=True, ax=axes[1], color="steelblue", bins=40)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title(f"{title} — Distribution des résidus")
    axes[1].grid(alpha=0.3)

    log_plot(fig, filename)
    plt.close(fig)


def tag_run(dataset_path: Path, feature_set: str, features: list) -> None:
    mlflow.set_tags({
        "dataset": dataset_path.name,
        "dataset_sha256": dataset_fingerprint(dataset_path),
        "feature_set": feature_set,
        "n_features": len(features),
        "test_size": TEST_SIZE,
    })


def split(df: pd.DataFrame, features: list):
    return train_test_split(
        df[features], df[TARGET_COLUMN],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True,
    )


def log_model_with_signature(flavour, model, artifact_path: str, X_sample: pd.DataFrame) -> None:
    """Logge le modèle avec sa signature d'entrée/sortie.

    Sans signature, MLflow ne peut pas valider le schéma au chargement : une
    colonne renommée ou réordonnée passerait sans erreur jusqu'à produire des
    prédictions fausses.
    """
    example = X_sample.head(5)
    signature = infer_signature(example, model.predict(example))
    flavour.log_model(
        model,
        artifact_path=artifact_path,
        signature=signature,
        input_example=example,
    )


def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def train_linear_regression() -> LinearRegression:
    df = load_csv(LIN_DATA_PATH)
    X_train, X_test, y_train, y_test = split(df, LIN_FEATURES)

    model = LinearRegression()
    with mlflow.start_run(run_name="LinearRegression_baseline"):
        tag_run(LIN_DATA_PATH, "data_lin", LIN_FEATURES)
        mlflow.log_params({
            "model_type": "LinearRegression",
            "fit_intercept": model.fit_intercept,
            "random_state": RANDOM_STATE,
        })

        model.fit(X_train, y_train)
        metrics = log_split_metrics(model, X_train, y_train, X_test, y_test)

        fig, ax = plt.subplots(figsize=(8, 5))
        y_pred = model.predict(X_test)
        ax.scatter(y_pred, y_test - y_pred, alpha=0.4, color="steelblue", s=20)
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Prédictions")
        ax.set_ylabel("Résidus")
        ax.set_title("Régression Linéaire — Résidus")
        ax.grid(alpha=0.3)
        log_plot(fig, "residuals_linear_regression.png")
        plt.close(fig)

        log_model_with_signature(
            mlflow.sklearn, model, "linear-regression-model", X_train
        )

    save_pickle(model, MODEL_DIR / "best_model_linear_regression.pkl")
    print(f"  LinearRegression  test R2={metrics['test_R2']:.4f} gap={metrics['gap_R2']:.4f}")
    return model


def train_boost_model(name: str, estimator, flavour, artifact_path: str):
    df = load_csv(BOOST_DATA_PATH)
    X_train, X_test, y_train, y_test = split(df, BOOST_FEATURES)

    with mlflow.start_run(run_name=f"{name}_final"):
        tag_run(BOOST_DATA_PATH, "data_boost", BOOST_FEATURES)
        mlflow.log_params({"model_type": name, "random_state": RANDOM_STATE,
                           **BEST_PARAMS[name]})

        estimator.fit(X_train, y_train)
        metrics = log_split_metrics(estimator, X_train, y_train, X_test, y_test)
        log_residual_plots(y_test, estimator.predict(X_test), name,
                           f"residuals_{name.lower()}.png")
        log_model_with_signature(flavour, estimator, artifact_path, X_train)

    print(f"  {name:16s}  test R2={metrics['test_R2']:.4f} gap={metrics['gap_R2']:.4f}")
    return estimator


def train_xgboost() -> XGBRegressor:
    model = XGBRegressor(random_state=RANDOM_STATE, verbosity=0, **BEST_PARAMS["XGBoost"])
    train_boost_model("XGBoost", model, mlflow.xgboost, "xgboost-model")

    # Format natif : portable entre versions et sans désérialisation de code
    # arbitraire. C'est ce fichier que l'application charge en priorité.
    model.save_model(MODEL_DIR / "best_model_tuned_xgboost.json")
    save_pickle(model, MODEL_DIR / "best_model_tuned_xgboost.pkl")
    return model


def train_random_forest() -> RandomForestRegressor:
    model = RandomForestRegressor(
        random_state=RANDOM_STATE, n_jobs=-1, **BEST_PARAMS["RandomForest"]
    )
    return train_boost_model("RandomForest", model, mlflow.sklearn, "random-forest-model")


def train_lightgbm() -> LGBMRegressor:
    model = LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, **BEST_PARAMS["LightGBM"])
    return train_boost_model("LightGBM", model, mlflow.lightgbm, "lightgbm-model")


def tune_model(name: str, estimator, n_iter: int = 30):
    """Rejoue la recherche d'hyperparamètres en traçant chaque candidat.

    Chaque combinaison évaluée devient un run imbriqué. Sans cela, seul
    l'optimum survit et rien ne permet de voir que la recherche a exploré un
    espace différent — ce qui arrive dès que la version de scikit-learn change,
    `ParameterSampler` ne tirant pas les mêmes candidats à graine constante.
    """
    df = load_csv(BOOST_DATA_PATH)
    X_train, X_test, y_train, y_test = split(df, BOOST_FEATURES)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=PARAM_GRIDS[name],
        n_iter=n_iter,
        scoring="r2",
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    with mlflow.start_run(run_name=f"{name}_tuning"):
        tag_run(BOOST_DATA_PATH, "data_boost", BOOST_FEATURES)
        mlflow.log_params({"model_type": name, "search": "RandomizedSearchCV",
                           "n_iter": n_iter, "cv_folds": 5, "scoring": "r2",
                           "random_state": RANDOM_STATE})

        search.fit(X_train, y_train)

        cv = search.cv_results_
        for i in range(len(cv["params"])):
            with mlflow.start_run(run_name=f"{name}_candidate_{i:02d}", nested=True):
                mlflow.log_params(cv["params"][i])
                mlflow.log_metrics({
                    "cv_r2_mean": float(cv["mean_test_score"][i]),
                    "cv_r2_std": float(cv["std_test_score"][i]),
                    "rank": int(cv["rank_test_score"][i]),
                })

        best = search.best_estimator_
        mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
        mlflow.log_metric("best_cv_r2", float(search.best_score_))
        log_split_metrics(best, X_train, y_train, X_test, y_test)

        print(f"  {name:16s}  best CV R2={search.best_score_:.4f}  "
              f"params={search.best_params_}")

    return search


def run_tuning(n_iter: int = 30) -> dict:
    estimators = {
        "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
        "LightGBM": LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
    }
    results = {}
    for name, estimator in estimators.items():
        search = tune_model(name, estimator, n_iter=n_iter)
        results[name] = {"best_params": search.best_params_,
                         "cv_r2": float(search.best_score_)}
    save_pickle(results, MODEL_DIR / "best_params.pkl")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune", action="store_true",
                        help="rejoue la recherche d'hyperparamètres (long)")
    parser.add_argument("--n-iter", type=int, default=60,
                        help="itérations de RandomizedSearchCV (défaut : 60, "
                             "comme 03_modeling.ipynb)")
    args = parser.parse_args()

    ensure_directories()
    mlflow.set_experiment(EXPERIMENT_NAME)

    failures = []

    if args.tune:
        print(f"Recherche d'hyperparamètres ({args.n_iter} itérations x 5 folds)...")
        try:
            run_tuning(n_iter=args.n_iter)
        except Exception as exc:
            failures.append(f"tuning : {exc}")
            print(f"Erreur pendant le tuning : {exc}", file=sys.stderr)
        print("Attention : les optima trouvés ne sont pas repris automatiquement "
              "par BEST_PARAMS, à reporter manuellement après vérification.")

    print("Entraînement des modèles finaux...")
    for label, fn in (("LinearRegression", train_linear_regression),
                      ("RandomForest", train_random_forest),
                      ("XGBoost", train_xgboost),
                      ("LightGBM", train_lightgbm)):
        try:
            fn()
        except Exception as exc:
            failures.append(f"{label} : {exc}")
            print(f"Erreur lors de l'entraînement de {label} : {exc}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} étape(s) en échec.", file=sys.stderr)
        return 1

    print(f"\nTerminé. Runs consultables via : mlflow ui "
          f"--backend-store-uri {PROJECT_ROOT / 'mlruns'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
