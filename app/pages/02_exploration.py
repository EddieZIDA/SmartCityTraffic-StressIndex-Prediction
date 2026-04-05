import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.express as px
import pandas as pd
from utils import load_raw_data

st.set_page_config(page_title="Exploration des données", page_icon="📊",
                   layout="wide")

st.title("📊 Exploration du Dataset")
st.markdown(
    "Cette page présente les données brutes issues de "
    "**smart_city_traffic_stress_dataset.csv** avant le Feature Engineering."
)

df = load_raw_data()

if df.empty:
    st.warning("⚠️ Les données n'ont pas pu être chargées.")
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lignes",            f"{df.shape[0]:,}")
c2.metric("Colonnes",          df.shape[1])
c3.metric("Valeurs manquantes", df.isnull().sum().sum())
c4.metric("Doublons",          df.duplicated().sum())

st.divider()

# ── Sidebar filtres ───────────────────────────────────────────────
st.sidebar.header("Filtres")
weather_filter = st.sidebar.multiselect(
    "Météo", df["weather_condition"].unique().tolist(),
    default=df["weather_condition"].unique().tolist()
)
exp_filter = st.sidebar.multiselect(
    "Expérience conducteur",
    ["Beginner", "Intermediate", "Expert"],
    default=["Beginner", "Intermediate", "Expert"]
)
stress_range = st.sidebar.slider("Plage stress_index", 0, 100, (0, 100))

filtered = df[
    df["weather_condition"].isin(weather_filter) &
    df["driver_experience_level"].isin(exp_filter) &
    df["stress_index"].between(*stress_range)
]

if filtered.empty:
    st.warning("Aucune observation avec ces filtres.")
    st.stop()

st.sidebar.metric("Observations filtrées", f"{len(filtered):,}",
                  f"{len(filtered)/len(df)*100:.1f}%")

# ── Aperçu ────────────────────────────────────────────────────────
st.subheader("Aperçu des données")
st.dataframe(filtered.head(100), use_container_width=True)
st.divider()

# ── Distributions et relations ────────────────────────────────────
st.subheader("Distributions et Relations clés")
col3, col4 = st.columns(2)

with col3:
    # marginal="box" fonctionne avec plotly 5.22 (pas "kde")
    fig1 = px.histogram(
        filtered, x="stress_index",
        title="Distribution du Stress Index",
        marginal="box",
        color_discrete_sequence=["#1F4E79"]
    )
    fig1.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig1, use_container_width=True)

with col4:
    sample_df = filtered.sample(min(2000, len(filtered)), random_state=42)
    fig2 = px.scatter(
        sample_df, x="avg_speed", y="stress_index",
        color="weather_condition",
        title="Stress vs Vitesse moyenne (échantillon)",
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig2.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Catégorielles ─────────────────────────────────────────────────
st.subheader("Variables catégorielles")
c1, c2 = st.columns(2)

with c1:
    fig3 = px.pie(
        filtered, names="driver_experience_level",
        title="Répartition de l'expérience conducteur",
        color_discrete_sequence=["#fd7e14", "#2E75B6", "#198754"],
        hole=0.4
    )
    st.plotly_chart(fig3, use_container_width=True)

with c2:
    fig4 = px.pie(
        filtered, names="weather_condition",
        title="Répartition des conditions météo",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.4
    )
    st.plotly_chart(fig4, use_container_width=True)

# Boxplot stress par catégorie
st.subheader("Stress Index par catégorie")
cat = st.radio("Variable :",
               ["weather_condition", "driver_experience_level"],
               horizontal=True)
fig5 = px.box(
    filtered, x=cat, y="stress_index", color=cat,
    color_discrete_sequence=px.colors.qualitative.Set2,
    labels={cat: "", "stress_index": "Stress Index"},
    height=380
)
fig5.update_layout(showlegend=False, plot_bgcolor="white")
st.plotly_chart(fig5, use_container_width=True)

st.divider()

# ── Matrice de corrélation ────────────────────────────────────────
st.subheader("Matrice de corrélation")
num_cols = ["traffic_density", "horn_events_per_min", "avg_speed",
            "signal_wait_time", "road_quality_score", "stress_index"]
corr = filtered[num_cols].corr().round(2)
fig6 = px.imshow(
    corr, text_auto=True, color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1, aspect="auto", height=460
)
fig6.update_layout(margin=dict(t=30, b=20))
st.plotly_chart(fig6, use_container_width=True)

