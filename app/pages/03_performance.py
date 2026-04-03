import streamlit as st

st.set_page_config(page_title="Performances", page_icon="🏆")

st.title("🏆 Performances du Modèle")

st.markdown(
    "Cette section détaille les performances des modèles évalués durant la "
    "phase de développement, ainsi que les caractéristiques du modèle final."
)

st.header("1. Évaluation des modèles de base")
st.markdown("Quatre modèles ont été évalués initialement :")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Linear Regression", "R² = 0.8587")
col2.metric("Random Forest", "R² = 0.9019", "Overfitting")
col3.metric("LightGBM", "R² = 0.9086")
col4.metric("XGBoost", "R² = 0.9045")

st.divider()

st.header("2. Phase d'optimisation (Tuning)")
st.markdown(
    "Une recherche d'hyperparamètres via `RandomizedSearchCV` avec "
    "validation croisée a été menée pour corriger l'overfitting."
)

st.subheader("Comparaison Avant / Après Tuning")

comparison_data = {
    "Modèle": ["RandomForest", "XGBoost", "LightGBM"],
    "R² Avant": ["0.9019", "0.9045", "0.9086"],
    "R² Après": ["0.9077", "0.9090", "0.9085"],
    "RMSE Avant": ["5.088", "5.022", "4.913"],
    "RMSE Après": ["4.937", "4.902", "4.915"],
    "Gain R²": ["+0.0058", "+0.0045", "-0.0001"]
}
st.table(comparison_data)

st.success(
    "**Le modèle XGBoost Tuné a été retenu comme modèle final.**\n"
    "Il offre le meilleur score de généralisation et corrige l'écart "
    "d'apprentissage (gap minime de 0.0106 entre train et test)."
)

with st.expander("Voir les hyperparamètres finaux de XGBoost"):
    st.code(
        "{\n"
        "    'colsample_bytree': 0.7599,\n"
        "    'learning_rate': 0.0193,\n"
        "    'max_depth': 6,\n"
        "    'n_estimators': 563,\n"
        "    'subsample': 0.6931\n"
        "}",
        language="python"
    )


