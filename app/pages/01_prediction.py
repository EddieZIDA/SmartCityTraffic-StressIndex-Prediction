import sys
from pathlib import Path

import streamlit as st

# Import depuis utils.py
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import load_model, preprocess_input  # noqa: E402

st.title("🔮 Prédiction du Stress Index")
st.write(
    "Saisissez les paramètres de circulation pour estimer "
    "le niveau de stress du conducteur."
)

model = load_model()

if model is not None:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            driver_experience = st.selectbox(
                "Expérience du conducteur",
                ["Beginner", "Intermediate", "Expert"]
            )
            weather_condition = st.selectbox(
                "Conditions météo",
                ["Clear", "Rainy", "Foggy", "Hot"]
            )
            road_quality_score = st.slider(
                "Qualité de la route (Score)", 0.0, 10.0, 5.0
            )

        with col2:
            avg_speed = st.number_input(
                "Vitesse moyenne (km/h)", min_value=0.0, value=30.0
            )
            traffic_density = st.slider(
                "Densité du trafic", 0.0, 200.0, 100.0
            )
            signal_wait_time = st.slider(
                "Temps d'attente aux feux (s)", 0.0, 120.0, 30.0
            )
            horn_events = st.number_input(
                "Coups de klaxon (par min)", min_value=0.0, value=2.0
            )

        submit_button = st.form_submit_button(label="Prédire le stress")

    if submit_button:
        input_data = {
            "driver_experience_level": driver_experience,
            "weather_condition": weather_condition,
            "road_quality_score": road_quality_score,
            "avg_speed": avg_speed,
            "traffic_density": traffic_density,
            "signal_wait_time": signal_wait_time,
            "horn_events_per_min": horn_events
        }

        processed_data = preprocess_input(input_data)

        try:
            prediction = model.predict(processed_data)[0]
            st.success(f"🎯 **Stress Index estimé : {prediction:.2f}**")

            with st.expander("Voir les features envoyées au modèle"):
                st.dataframe(processed_data)

        except Exception as e:
            st.error(f"Erreur lors de la prédiction. Détail : {e}")
            

