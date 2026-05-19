import math
import pytest
import pandas as pd

from app import utils


def test_preprocess_input_nominal(sample_observation):
    df = utils.preprocess_input(**sample_observation)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, len(utils.FEATURE_COLS))
    assert list(df.columns) == utils.FEATURE_COLS

    row = df.iloc[0]
    assert row["avg_speed"] == sample_observation["avg_speed"]
    assert row["driver_experience_encoded"] == utils.EXP_MAP[sample_observation["experience"]]
    assert row["weather_Rainy"] == 1

    expected_congestion = (sample_observation["traffic_density"] * sample_observation["signal_wait_time"]) / 100
    assert math.isclose(row["congestion_score"], expected_congestion, rel_tol=1e-9)

    expected_horn_density = sample_observation["horn_events"] / (sample_observation["traffic_density"] + 1)
    assert math.isclose(row["horn_density"], expected_horn_density, rel_tol=1e-9)


def test_preprocess_input_edge_cases_and_invalid_types():
    # Missing numeric -> should raise TypeError when arithmetic attempted
    with pytest.raises(TypeError):
        utils.preprocess_input(traffic_density=None, signal_wait_time=30, avg_speed=40, road_quality=7.5, experience="Beginner", weather="Hot")

    # Incorrect types for numeric field -> should raise
    with pytest.raises(TypeError):
        utils.preprocess_input(traffic_density="heavy", signal_wait_time=30, avg_speed=40, road_quality=7.5, experience="Beginner", weather="Hot")

    # Unknown experience maps to default 0
    df_unknown = utils.preprocess_input(traffic_density=10, signal_wait_time=5, avg_speed=30, road_quality=6.0, experience="Unknown", weather="Hot", horn_events=0)
    assert df_unknown.iloc[0]["driver_experience_encoded"] == 0

    # Unknown weather results in all weather_* == 0
    df_weather = utils.preprocess_input(traffic_density=10, signal_wait_time=5, avg_speed=30, road_quality=6.0, experience="Expert", weather="Sunny", horn_events=0)
    assert df_weather.iloc[0]["weather_Foggy"] == 0
    assert df_weather.iloc[0]["weather_Hot"] == 0
    assert df_weather.iloc[0]["weather_Rainy"] == 0


def test_output_coherence_bounds(sample_observation):
    df = utils.preprocess_input(**sample_observation)
    row = df.iloc[0]

    # congestion_score non-négatif et cohérent
    assert row["congestion_score"] >= 0
    assert row["congestion_score"] == pytest.approx((sample_observation["traffic_density"] * sample_observation["signal_wait_time"]) / 100)

    # horn_density non-négatif
    assert row["horn_density"] >= 0


def test_stress_level_boundaries_and_types():
    assert utils.stress_level(80) == ("Élevé", "#dc3545")
    assert utils.stress_level(70) == ("Élevé", "#dc3545")
    assert utils.stress_level(69.9999) == ("Modéré", "#fd7e14")
    assert utils.stress_level(40) == ("Modéré", "#fd7e14")
    assert utils.stress_level(39.9999) == ("Faible", "#198754")
    assert utils.stress_level(0) == ("Faible", "#198754")

    label, color = utils.stress_level(50)
    assert isinstance(label, str)
    assert isinstance(color, str) and color.startswith("#") and len(color) == 7
