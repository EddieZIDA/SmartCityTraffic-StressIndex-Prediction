import streamlit as st

st.set_page_config(
    page_title="Traffic Stress Predictor",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Smart City Traffic Stress Predictor")

st.markdown(
    "### Bienvenue sur l'interface d'analyse du stress lié "
    "au trafic urbain !\n\n"
    "Cette application exploite un modèle de Machine Learning (XGBoost) "
    "pour évaluer et prédire le **Stress Index** des conducteurs en fonction "
    "des conditions de circulation, de la météo et de leur expérience.\n\n"
    "👈 **Utilisez le menu latéral pour naviguer :**\n"
    "* **Prediction :** Estimez le stress en entrant de nouvelles données.\n"
    "* **Exploration :** Visualisez les données brutes de l'étude.\n"
    "* **Performance :** Consultez les métriques du modèle final."
)

st.info(
    "Développé dans le cadre d'un projet d'analyse de données complexes "
    "et de modélisation prédictive."
)
