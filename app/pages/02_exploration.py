import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

# Import dynamique depuis utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import load_data  # noqa: E402

st.set_page_config(page_title="Exploration des données", page_icon="📊")

st.title("📊 Exploration du Dataset")
st.markdown(
    "Cette page présente les données brutes issues de "
    "**smart_city_traffic_stress_dataset.csv** avant le Feature Engineering."
)

df = load_data()

if not df.empty:
    col1, col2 = st.columns(2)
    col1.metric("Nombre de lignes", df.shape[0])
    col2.metric("Nombre de colonnes", df.shape[1])

    st.subheader("Aperçu des données")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Distributions et Relations clés")
    col3, col4 = st.columns(2)

    with col3:
        if 'stress_index' in df.columns:
            fig1 = px.histogram(
                df, x="stress_index",
                title="Distribution du Stress Index",
                marginal="box",
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig1, use_container_width=True)

    with col4:
        if 'avg_speed' in df.columns and 'stress_index' in df.columns:
            sample_df = df.sample(min(2000, len(df)), random_state=42)
            fig2 = px.scatter(
                sample_df, x="avg_speed", y="stress_index",
                color="weather_condition",
                title="Stress vs Vitesse moyenne (Échantillon)",
                opacity=0.6
            )
            st.plotly_chart(fig2, use_container_width=True)

    if 'driver_experience_level' in df.columns:
        st.subheader("Expérience des conducteurs")
        fig3 = px.pie(
            df, names='driver_experience_level',
            title="Répartition de l'expérience"
        )
        st.plotly_chart(fig3, use_container_width=True)

else:
    st.warning(
        "⚠️ Les données n'ont pas pu être chargées. "
        "Vérifiez le chemin dans `utils.py`."
    )

