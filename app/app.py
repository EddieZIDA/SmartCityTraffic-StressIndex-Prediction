import streamlit as st

st.set_page_config(
    page_title="Traffic Stress Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1F4E79;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .nav-card {
        background-color: #FFFFFF !important;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        min-height: 160px;
    }
    .nav-card h3 { color: #1F4E79 !important; margin-top: 0; }
    .nav-card p  { color: #444 !important; }
    [data-testid="stSidebar"]      { background-color: #f0f4f8 !important; }
    [data-testid="stMetricValue"]  { color: #1F4E79; }
    h1 { color: #1F4E79; }
    h2 { color: #2E75B6; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="main-title">🚦 Smart City Traffic Stress Predictor</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-title">Analyse prédictive de l\'indice de stress '
    'des conducteurs en ville intelligente</p>',
    unsafe_allow_html=True
)

st.markdown(
    "Cette application exploite un modèle de Machine Learning "
    "**(XGBoost tuné)** pour évaluer et prédire le **Stress Index** "
    "des conducteurs en fonction des conditions de circulation, "
    "de la météo et de leur expérience."
)

st.divider()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset",         "50 000 obs.",   "8 variables originales")
c2.metric("Meilleur modèle", "XGBoost tuné",  "R² = 0.909")
c3.metric("RMSE",            "4.902",         "sur échelle 0-100")
c4.metric("MAE",             "3.927",         "erreur absolue moyenne")

st.divider()
st.subheader("👈 Utilisez le menu latéral pour naviguer")

n1, n2, n3 = st.columns(3)
with n1:
    st.markdown("""<div class="nav-card">
        <h3>🔮 Prediction</h3>
        <p>Estimez le stress en entrant de nouvelles données de conduite.
        Jauge interactive + simulation de l'impact de chaque paramètre.</p>
    </div>""", unsafe_allow_html=True)
with n2:
    st.markdown("""<div class="nav-card">
        <h3>📊 Exploration</h3>
        <p>Visualisez les données brutes de l'étude : distributions,
        corrélations et relations entre variables.</p>
    </div>""", unsafe_allow_html=True)
with n3:
    st.markdown("""<div class="nav-card">
        <h3>🏆 Performance</h3>
        <p>Consultez les métriques du modèle final, l'analyse de l'overfitting
        et les feature importances.</p>
    </div>""", unsafe_allow_html=True)

st.divider()
st.info(
    "Développé dans le cadre d'un projet d'analyse de données complexes "
    "et de modélisation prédictive."
)
st.caption(
    "Smart City Traffic Stress · XGBoost · Streamlit · "
    "ZIDA Wend Kouni Eddie Eliel"
)

