import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import load_model, load_boost_data, FEATURE_COLS

st.set_page_config(page_title="Performances", page_icon="🏆", layout="wide")

st.title("🏆 Performances du Modèle")
st.markdown(
    "Cette section détaille les performances des modèles évalués durant "
    "la phase de développement, ainsi que les caractéristiques du modèle final."
)

# ── Résultats figés depuis 03_modeling.ipynb ──────────────────────
avant = pd.DataFrame({
    "Modèle":   ["LinearRegression", "RandomForest", "XGBoost", "LightGBM"],
    "R²":       [0.8587, 0.9019, 0.9045, 0.9086],
    "RMSE":     [6.107,  5.088,  5.022,  4.913],
    "MAE":      [4.871,  4.078,  4.017,  3.930],
    "R² train": [0.8636, 0.9863, 0.9301, 0.9170],
    "Gap":      [0.0049, 0.0844, 0.0256, 0.0084],
    "Phase":    ["Avant tuning"] * 4,
})
apres = pd.DataFrame({
    "Modèle":   ["LinearRegression", "RandomForest", "XGBoost", "LightGBM"],
    "R²":       [0.8587, 0.9077, 0.9090, 0.9085],
    "RMSE":     [6.107,  4.937,  4.902,  4.915],
    "MAE":      [4.871,  3.952,  3.927,  3.937],
    "R² train": [0.8636, 0.9276, 0.9203, 0.9138],
    "Gap":      [0.0049, 0.0181, 0.0106, 0.0042],
    "Phase":    ["Après tuning"] * 4,
})

# ── Section 1 : modèles de base ───────────────────────────────────
st.header("1. Évaluation des modèles de base")
st.markdown("Quatre modèles ont été évalués initialement :")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Linear Regression", "R² = 0.8587", "Benchmark")
c2.metric("Random Forest",     "R² = 0.9019", "⚠️ Overfitting gap=0.084")
c3.metric("XGBoost",           "R² = 0.9045")
c4.metric("LightGBM",          "R² = 0.9086")

st.divider()

# ── Section 2 : tuning ────────────────────────────────────────────
st.header("2. Phase d'optimisation (Tuning)")
st.markdown(
    "Une recherche d'hyperparamètres via `RandomizedSearchCV` avec "
    "validation croisée **KFold 5 folds** a été menée."
)

st.subheader("Comparaison Avant / Après Tuning")

all_r = pd.concat([avant, apres])
fig_cmp = px.bar(
    all_r[all_r["Modèle"] != "LinearRegression"],
    x="Modèle", y="R²", color="Phase", barmode="group",
    color_discrete_map={"Avant tuning": "#85B7EB", "Après tuning": "#1F4E79"},
    labels={"R²": "R² test", "Modèle": ""},
    height=380, text="R²"
)
fig_cmp.update_traces(texttemplate="%{text:.4f}", textposition="outside")
fig_cmp.add_hline(y=0.8587, line_dash="dash", line_color="#888",
                  annotation_text="Benchmark LinearRegression")
fig_cmp.update_layout(plot_bgcolor="white", yaxis_range=[0.85, 0.92],
                      legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig_cmp, use_container_width=True)

st.subheader("Tableau comparatif")
comparison_data = {
    "Modèle":     ["RandomForest", "XGBoost", "LightGBM"],
    "R² Avant":   ["0.9019", "0.9045", "0.9086"],
    "R² Après":   ["0.9077", "0.9090", "0.9085"],
    "RMSE Avant": ["5.088",  "5.022",  "4.913"],
    "RMSE Après": ["4.937",  "4.902",  "4.915"],
    "Gain R²":    ["+0.0058", "+0.0045", "-0.0001"]
}
st.table(comparison_data)

st.success(
    "**Le modèle XGBoost Tuné a été retenu comme modèle final.**\n\n"
    "Il offre le meilleur score de généralisation "
    "(R²=0.909, RMSE=4.902, MAE=3.927) avec un gap overfitting "
    "minimal de 0.0106 entre train et test."
)

with st.expander("Voir les hyperparamètres finaux de XGBoost"):
    st.code(
        "XGBRegressor(\n"
        "    colsample_bytree = 0.7599,\n"
        "    learning_rate    = 0.0193,\n"
        "    max_depth        = 6,\n"
        "    n_estimators     = 563,\n"
        "    subsample        = 0.6931,\n"
        "    random_state     = 42\n"
        ")",
        language="python"
    )

st.divider()

# ── Section 3 : Overfitting ───────────────────────────────────────
st.header("3. Analyse de l'overfitting")

colors = {
    "LinearRegression": "#888", "RandomForest": "#fd7e14",
    "XGBoost": "#1F4E79",       "LightGBM":     "#198754"
}
fig_ov = go.Figure()
for _, row in apres.iterrows():
    fig_ov.add_trace(go.Bar(
        name=row["Modèle"],
        x=["R² train", "R² test"],
        y=[row["R² train"], row["R²"]],
        marker_color=colors[row["Modèle"]],
        text=[f"{row['R² train']:.4f}", f"{row['R²']:.4f}"],
        textposition="outside"
    ))
fig_ov.update_layout(
    barmode="group", height=400, plot_bgcolor="white",
    yaxis_range=[0.83, 0.95], yaxis_title="R²",
    legend=dict(orientation="h", y=1.1)
)
st.plotly_chart(fig_ov, use_container_width=True)
st.info(
    "RandomForest avait un gap de **0.0844** avant tuning → "
    "réduit à **0.0181** après (min_samples_leaf=14, max_depth=17)"
)

st.divider()

# ── Section 4 : Feature Importance ───────────────────────────────
st.header("4. Feature Importance")

imp_data = {
    "Feature": [
        "congestion_score", "avg_speed",
        "driver_experience_encoded", "road_quality_score",
        "horn_density", "weather_Rainy", "weather_Foggy", "weather_Hot"
    ],
    "XGBoost":      [0.52, 0.28, 0.09, 0.05, 0.03, 0.01, 0.01, 0.01],
    "RandomForest": [0.48, 0.30, 0.10, 0.06, 0.03, 0.01, 0.01, 0.01],
    "LightGBM":     [0.50, 0.27, 0.11, 0.06, 0.03, 0.01, 0.01, 0.01],
}
imp_df = pd.DataFrame(imp_data)
choice = st.radio("Modèle :", ["XGBoost", "RandomForest", "LightGBM"],
                  horizontal=True)
imp_s  = imp_df[["Feature", choice]].sort_values(choice)
fig_imp = px.bar(
    imp_s, x=choice, y="Feature", orientation="h",
    color=choice, color_continuous_scale=["#E6F1FB", "#1F4E79"],
    labels={choice: "Importance", "Feature": ""},
    height=360, text=choice
)
fig_imp.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig_imp.update_layout(plot_bgcolor="white", coloraxis_showscale=False,
                      margin=dict(r=60))
st.plotly_chart(fig_imp, use_container_width=True)
st.success(
    "**congestion_score** est la feature dominante (~50%) "
    "dans les 3 modèles — valide le feature engineering."
)

st.divider()

# ── Section 5 : Résidus réels ─────────────────────────────────────
st.header("5. Analyse des résidus — XGBoost tuné")
st.info("Calcul sur 2 000 observations échantillonnées depuis data_boost.")

try:
    model  = load_model()
    df_bst = load_boost_data()

    if model is None or df_bst.empty:
        st.stop()

    missing = [c for c in FEATURE_COLS if c not in df_bst.columns]
    if missing:
        st.warning(f"Colonnes manquantes dans data_boost : {missing}")
        st.stop()

    samp = df_bst.sample(min(2000, len(df_bst)), random_state=42)
    y_pred = model.predict(samp[FEATURE_COLS])
    residuals = samp["stress_index"].values - y_pred

    c1, c2 = st.columns(2)
    with c1:
        fig_r = px.scatter(
            x=y_pred, y=residuals, opacity=0.4,
            color=np.abs(residuals),
            color_continuous_scale="RdYlGn_r",
            labels={"x": "Valeurs prédites", "y": "Résidus",
                    "color": "|Résidu|"},
            title="Résidus vs Valeurs prédites", height=380
        )
        fig_r.add_hline(y=0, line_dash="dash",
                        line_color="red", line_width=2)
        fig_r.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_r, use_container_width=True)

    with c2:
        # marginal="box" — compatible plotly 5.22 (pas "kde")
        fig_h = px.histogram(
            x=residuals, nbins=50,
            color_discrete_sequence=["#2E75B6"],
            labels={"x": "Résidu", "y": "Fréquence"},
            title="Distribution des résidus", height=380
        )
        fig_h.add_vline(x=0, line_dash="dash",
                        line_color="red", line_width=2)
        fig_h.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_h, use_container_width=True)

    r1, r2, r3 = st.columns(3)
    r1.metric("Résidu moyen",  f"{np.mean(residuals):.3f}",
              help="Proche de 0 = pas de biais")
    r2.metric("Std résidus",   f"{np.std(residuals):.3f}")
    r3.metric("Max |résidu|",  f"{np.abs(residuals).max():.2f}")

except Exception as e:
    st.error(f"Erreur lors du calcul des résidus : {e}")

