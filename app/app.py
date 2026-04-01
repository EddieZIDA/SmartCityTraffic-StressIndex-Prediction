import streamlit as st

st.set_page_config(
    page_title="Smart City — Stress Prédictif",
    page_icon="🏙️",
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
        color: #666;
        margin-bottom: 2rem;
    }
    .nav-card {
        background-color: #FFFFFF !important;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        min-height: 180px;
    }
    .nav-card h3 {
        color: #1F4E79 !important;
        margin-top: 0;
    }
    .nav-card p {
        color: #262730 !important;
    }

    /* --- FIX NAVIGATION SIDEBAR --- */
    [data-testid="stSidebar"] {
        background-color: #f0f4f8 !important;
    }
    
    /* Force la couleur du texte des liens (app, prediction, etc.) */
    [data-testid="stSidebarNav"] span {
        color: #1F4E79 !important;
        font-weight: 600 !important;
    }

    /* Force la couleur des icônes à côté des noms de pages */
    [data-testid="stSidebarNav"] svg {
        fill: #1F4E79 !important;
    }
    /* ------------------------------ */

    [data-testid="stMetricValue"] {
        color: #1F4E79;
    }
    h1 { color: #1F4E79; }
    h2 { color: #2E75B6; }
    hr { border-color: #2E75B6; opacity: 0.3; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="main-title">🏙️ Smart City — Prédiction du Stress Urbain</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-title">Analyse prédictive de l\'indice de stress '
    'des conducteurs en ville intelligente</p>',
    unsafe_allow_html=True
)

st.divider()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Dataset", "50 000 obs.", "8 variables originales")
with col2:
    st.metric("Meilleur modèle", "XGBoost tuné", "R² = 0.909")
with col3:
    st.metric("RMSE", "4.902", "sur échelle 0-100")
with col4:
    st.metric("MAE", "3.927", "erreur absolue moyenne")

st.divider()

st.subheader("Navigation")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="nav-card">
        <h3>🎯 Prédiction</h3>
        <p>Entrez les paramètres de conduite et obtenez une
        prédiction du stress en temps réel.</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="nav-card">
        <h3>📊 Exploration</h3>
        <p>Visualisez les distributions, corrélations et
        patterns du dataset interactivement.</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="nav-card">
        <h3>🏆 Performance</h3>
        <p>Comparez les métriques des 4 modèles entraînés et
        analysez les feature importances.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("Projet Data Science · Smart City Traffic Stress Dataset · XGBoost · Streamlit")