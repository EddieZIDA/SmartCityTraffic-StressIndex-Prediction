import pytest
import pandas as pd


@pytest.fixture
def sample_observation():
    """Un dictionnaire représentant une observation typique trafic/météo."""
    return {
        "traffic_density": 50,
        "signal_wait_time": 30,
        "avg_speed": 40.0,
        "road_quality": 7.5,
        "experience": "Intermediate",
        "weather": "Rainy",
        "horn_events": 10,
    }


@pytest.fixture
def sample_dataframe():
    """Un DataFrame fictif représentatif de quelques observations."""
    return pd.DataFrame([
        {
            "traffic_density": 20,
            "signal_wait_time": 10,
            "avg_speed": 60.0,
            "road_quality": 8.0,
            "experience": "Expert",
            "weather": "Hot",
            "horn_events": 5,
        },
        {
            "traffic_density": 80,
            "signal_wait_time": 45,
            "avg_speed": 25.0,
            "road_quality": 5.0,
            "experience": "Beginner",
            "weather": "Foggy",
            "horn_events": 12,
        },
        {
            "traffic_density": 50,
            "signal_wait_time": 30,
            "avg_speed": 40.0,
            "road_quality": 7.5,
            "experience": "Intermediate",
            "weather": "Rainy",
            "horn_events": 10,
        },
    ])
