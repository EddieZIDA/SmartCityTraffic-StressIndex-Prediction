"""Contrat entre le dataset data_boost, le modèle sauvegardé et l'app.

Ces vérifications attrapent la dérive de schéma : une colonne renommée ou
réordonnée dans le preprocessing casse silencieusement les prédictions,
XGBoost n'échouant pas toujours sur un simple changement d'ordre.
"""
import pickle

import pandas as pd
import pytest

from app import utils

BOOST_PATH = utils.BOOST_PATH


def _load_model():
    if utils.MODEL_JSON_PATH.exists():
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.load_model(utils.MODEL_JSON_PATH)
        return model
    if utils.MODEL_PATH.exists():
        with open(utils.MODEL_PATH, "rb") as f:
            return pickle.load(f)
    pytest.skip("Aucun modèle entraîné disponible")


def test_boost_dataset_contains_feature_columns():
    if not BOOST_PATH.exists():
        pytest.skip("data_boost absent")
    cols = pd.read_csv(BOOST_PATH, nrows=1).columns.tolist()
    missing = [c for c in utils.FEATURE_COLS if c not in cols]
    assert missing == [], f"Colonnes absentes de data_boost : {missing}"
    assert "stress_index" in cols


def test_model_features_match_app_feature_cols():
    model = _load_model()
    names = model.get_booster().feature_names
    assert names == utils.FEATURE_COLS, (
        "Le modèle attend un autre ordre/jeu de features que l'application"
    )


def test_prediction_is_finite_and_in_range():
    model = _load_model()
    X = utils.preprocess_input(
        traffic_density=64, signal_wait_time=37, avg_speed=54,
        road_quality=7.0, experience="Intermediate", weather="Rainy",
        horn_events=9,
    )
    assert list(X.columns) == utils.FEATURE_COLS
    score = float(model.predict(X)[0])
    assert score == score, "prédiction NaN"
    assert 0.0 <= score <= 100.0


def test_best_params_match_training_script():
    """BEST_PARAMS doit refléter l'optimum sauvegardé par le tuning.

    L'audit a trouvé un script qui entraînait avec des hyperparamètres arrondis,
    puis divergents de ceux du notebook : le modèle livré n'était plus celui que
    la recherche avait sélectionné, sans qu'aucune erreur ne le signale.
    """
    params_path = utils.PROJECT_ROOT / "models" / "best_params.pkl"
    if not params_path.exists():
        pytest.skip("best_params.pkl absent")

    from src.train_models import BEST_PARAMS

    with open(params_path, "rb") as f:
        tuned = pickle.load(f)

    for name, entry in tuned.items():
        expected = entry["best_params"]
        actual = BEST_PARAMS[name]
        for key, value in expected.items():
            assert actual[key] == pytest.approx(value), (
                f"{name}.{key} : le script utilise {actual[key]}, "
                f"le tuning a retenu {value}"
            )


def test_shipped_model_matches_best_params():
    """Le modèle .pkl livré doit porter les hyperparamètres de BEST_PARAMS."""
    if not utils.MODEL_PATH.exists():
        pytest.skip("modèle absent")

    from src.train_models import BEST_PARAMS

    with open(utils.MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    actual = model.get_params()
    for key, value in BEST_PARAMS["XGBoost"].items():
        assert actual[key] == pytest.approx(value), (
            f"XGBoost.{key} : modèle livré={actual[key]}, attendu={value}"
        )


@pytest.mark.parametrize("weather", ["Clear", "Foggy", "Hot", "Rainy"])
def test_weather_one_hot_is_mutually_exclusive(weather):
    """Clear est la modalité de référence (drop_first=True) : tout à 0."""
    X = utils.preprocess_input(
        traffic_density=50, signal_wait_time=30, avg_speed=40,
        road_quality=7.5, experience="Expert", weather=weather,
        horn_events=5,
    )
    dummies = X[["weather_Foggy", "weather_Hot", "weather_Rainy"]].iloc[0]
    expected = 0 if weather == "Clear" else 1
    assert dummies.sum() == expected
