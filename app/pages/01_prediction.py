import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from utils import load_model, preprocess_input, stress_level

st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Prédiction du Stress Index")
st.write(
    "Saisissez les paramètres de circulation pour estimer "
    "le niveau de stress du conducteur."
)

model = load_model()
if model is None:
    st.stop()

# ── Sidebar inputs ────────────────────────────────────────────────
st.sidebar.header("Paramètres de conduite")
traffic_density  = st.sidebar.slider("🚗 Densité du trafic",       10,   119,  64)
signal_wait_time = st.sidebar.slider("🚦 Attente aux feux (s)",      5,    75,  37)
avg_speed        = st.sidebar.slider("💨 Vitesse moyenne (km/h)",   14,    90,  54)
horn_events      = st.sidebar.slider("📯 Klaxons / minute",          0,    25,   9)
road_quality     = st.sidebar.slider(
    "🛣️ Qualité de route (0-10)", 1.0, 10.0, 7.0, 0.1
)
st.sidebar.divider()
experience = st.sidebar.selectbox(
    "👤 Expérience conducteur",
    ["Beginner", "Intermediate", "Expert"], index=1
)
weather = st.sidebar.selectbox(
    "🌤️ Conditions météo", ["Clear", "Foggy", "Hot", "Rainy"]
)

# ── Prédiction ────────────────────────────────────────────────────
X = preprocess_input(
    traffic_density, signal_wait_time, avg_speed,
    road_quality, experience, weather, horn_events
)
score = float(model.predict(X)[0])
score = round(max(0.0, min(100.0, score)), 1)
level, color = stress_level(score)
congestion = round((traffic_density * signal_wait_time) / 100, 1)

# ── Jauge + métriques ─────────────────────────────────────────────
col_g, col_m = st.columns([1, 1])

with col_g:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 44.4, "valueformat": ".1f"},
        title={
            "text": (
                f"Indice de stress<br>"
                f"<span style='font-size:0.8em;color:{color}'>"
                f"{level}</span>"
            )
        },
        number={"valueformat": ".1f", "font": {"size": 56}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [0,  40], "color": "#d4edda"},
                {"range": [40, 70], "color": "#fff3cd"},
                {"range": [70, 100], "color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": "#333", "width": 3},
                "thickness": 0.8,
                "value": 44.4
            }
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=60, b=20, l=30, r=30))
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.caption("▲ Le trait noir indique la moyenne du dataset (44.4)")

with col_m:
    st.subheader("Détail des inputs")
    m1, m2 = st.columns(2)
    m1.metric("Densité trafic",   traffic_density,
              help="Véhicules en circulation")
    m2.metric("Attente feux",     f"{signal_wait_time}s")
    m1.metric("Vitesse moyenne",  f"{avg_speed} km/h")
    m2.metric("Klaxons/min",      horn_events)
    m1.metric("Qualité route",    f"{road_quality}/10")
    m2.metric("Congestion score", f"{congestion}",
              help="traffic × attente / 100")
    badge_exp = {"Beginner": "🟡", "Intermediate": "🔵", "Expert": "🟢"}
    badge_wea = {"Clear": "☀️", "Foggy": "🌫️", "Hot": "🌡️", "Rainy": "🌧️"}
    st.info(
        f"{badge_exp[experience]} **{experience}**  ·  "
        f"{badge_wea[weather]} **{weather}**"
    )
    with st.expander("Voir les features envoyées au modèle"):
        st.dataframe(X)

st.divider()

# ── Contribution estimée ──────────────────────────────────────────
st.subheader("📌 Contribution estimée des variables")

exp_map   = {"Beginner": 0, "Intermediate": 1, "Expert": 2}
horn_d    = horn_events / (traffic_density + 1)

contrib_df = pd.DataFrame({
    "Feature": [
        "congestion_score", "avg_speed", "experience",
        "road_quality", "horn_density", "weather"
    ],
    "Valeur": [
        str(congestion), str(avg_speed), experience,
        str(road_quality), f"{horn_d:.3f}", weather
    ],
    "Corrélation": [0.835, -0.810, -0.330, -0.250, -0.169, 0.010],
    "Impact": [
        min(congestion / 88.7, 1.0),
        (90 - avg_speed) / 90,
        (2 - exp_map[experience]) / 2,
        (10 - road_quality) / 10,
        min(horn_d / 0.25, 1.0),
        0.02,
    ]
})

fig_bar = px.bar(
    contrib_df, x="Impact", y="Feature", orientation="h",
    color="Corrélation",
    color_continuous_scale=["#2E75B6", "#f0f0f0", "#dc3545"],
    color_continuous_midpoint=0,
    text="Valeur",
    labels={"Impact": "Contribution relative", "Feature": ""},
    height=320
)
fig_bar.update_traces(textposition="outside")
fig_bar.update_layout(
    margin=dict(l=10, r=80, t=20, b=20),
    coloraxis_colorbar=dict(title="Corrélation<br>avec stress"),
    plot_bgcolor="white"
)
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Simulation ────────────────────────────────────────────────────
st.subheader("🔭 Simulation — impact d'un paramètre")

param = st.selectbox("Faire varier :",
    ["Densité trafic", "Vitesse moyenne",
     "Attente aux feux", "Qualité de route"])

ranges = {
    "Densité trafic":   range(10, 120),
    "Vitesse moyenne":  range(14, 91),
    "Attente aux feux": range(5, 75),
    "Qualité de route": [x / 10 for x in range(10, 101)],
}

sim_scores = []
for v in ranges[param]:
    td  = v if param == "Densité trafic"   else traffic_density
    spd = v if param == "Vitesse moyenne"  else avg_speed
    swt = v if param == "Attente aux feux" else signal_wait_time
    rq  = v if param == "Qualité de route" else road_quality
    sim_scores.append(float(model.predict(
        preprocess_input(td, swt, spd, rq, experience, weather, horn_events)
    )[0]))

sim_df = pd.DataFrame({"x": list(ranges[param]), "stress": sim_scores})

fig_sim = px.line(
    sim_df, x="x", y="stress",
    labels={"x": param, "stress": "Stress prédit"}, height=300
)
fig_sim.add_hline(y=score, line_dash="dot", line_color="orange",
                  annotation_text=f"Valeur actuelle : {score}")
fig_sim.add_hline(y=44.4, line_dash="dash", line_color="#999",
                  annotation_text="Moyenne dataset (44.4)")
fig_sim.update_layout(plot_bgcolor="white", margin=dict(t=30, b=20))
fig_sim.update_traces(line_color="#2E75B6", line_width=2.5)
st.plotly_chart(fig_sim, use_container_width=True)

