import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import load_model, load_boost_data, FEATURE_COLS

st.set_page_config(page_title="Performances", page_icon="", layout="wide")

st.title("Performances du Modèle")
st.markdown(
    "Cette section détaille les performances des modèles évalués durant "
    "la phase de développement, ainsi que les caractéristiques du modèle final."
)

# ── Résultats figés depuis 03_modeling.ipynb ──────────────────────
avant = pd.DataFrame({
    "Modèle":   ["LinearRegression", "RandomForest", "XGBoost", "LightGBM"],
    "R²":       [0.8587, 0.9039, 0.9056, 0.9089],
    "RMSE":     [6.107,  5.037,  4.992,  4.903],
    "MAE":      [4.871,  4.031,  3.998,  3.928],
    "R² train": [0.8636, 0.9866, 0.9317, 0.9182],
    "Gap":      [0.0049, 0.0828, 0.0261, 0.0093],
    "Phase":    ["Avant tuning"] * 4,
})
# Gap = R² train − R² test (mesure d'overfitting sur le jeu tenu à l'écart).
# La version précédente comparait R² train au R² de validation croisée, deux
# estimateurs différents : les écarts affichés n'étaient pas comparables.
apres = pd.DataFrame({
    "Modèle":   ["LinearRegression", "RandomForest", "XGBoost", "LightGBM"],
    "R²":       [0.8587, 0.9083, 0.9100, 0.9096],
    "RMSE":     [6.107,  4.919,  4.873,  4.886],
    "MAE":      [4.871,  3.938,  3.901,  3.913],
    "R² train": [0.8636, 0.9337, 0.9147, 0.9172],
    "Gap":      [0.0049, 0.0254, 0.0046, 0.0076],
    "Phase":    ["Après tuning"] * 4,
})

# ── Section 1 : modèles de base ───────────────────────────────────
st.header("1. Évaluation des modèles de base")
st.markdown("Quatre modèles ont été évalués initialement :")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Linear Regression", "R² = 0.8587", "Benchmark")
c2.metric("Random Forest",     "R² = 0.9039", "Overfitting gap=0.083")
c3.metric("XGBoost",           "R² = 0.9056")
c4.metric("LightGBM",          "R² = 0.9089")

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
    "R² Avant":   ["0.9039", "0.9056", "0.9089"],
    "R² Après":   ["0.9083", "0.9100", "0.9096"],
    "RMSE Avant": ["5.037",  "4.992",  "4.903"],
    "RMSE Après": ["4.919",  "4.873",  "4.886"],
    "Gain R²":    ["+0.0044", "+0.0044", "+0.0006"]
}
st.table(comparison_data)

st.success(
    "**Le modèle XGBoost Tuné a été retenu comme modèle final.**\n\n"
    "Il offre le meilleur score de généralisation "
    "(R²=0.910, RMSE=4.873, MAE=3.901) avec un gap overfitting "
    "minimal de 0.0046 entre train et test."
)

with st.expander("Voir les hyperparamètres finaux de XGBoost"):
    st.code(
        "XGBRegressor(\n"
        "    colsample_bytree = 0.6102,\n"
        "    learning_rate    = 0.0208,\n"
        "    max_depth        = 3,\n"
        "    min_child_weight = 4,\n"
        "    n_estimators     = 933,\n"
        "    reg_lambda       = 0.6967,\n"
        "    subsample        = 0.8418,\n"
        "    random_state     = 42\n"
        ")",
        language="python"
    )
    st.caption(
        "La grille de recherche initiale plafonnait à 600 arbres : l'optimum, "
        "situé vers 930 arbres à faible learning rate, était hors de sa portée. "
        "L'élargir a suffi à le faire apparaître."
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
    "RandomForest avait un gap de **0.0828** avant tuning → "
    "réduit à **0.0254** après (min_samples_leaf=10, max_depth=13)"
)

st.divider()

# ── Section 4 : Feature Importance ───────────────────────────────
st.header("4. Feature Importance")

st.caption(
    "Importances mesurées sur les modèles tunés, ramenées à une base "
    "comparable (*gain* pour XGBoost et LightGBM, réduction d'impureté pour "
    "RandomForest) et normalisées à 1. Comparer le *gain* de XGBoost au "
    "nombre de splits de LightGBM — son défaut — donne un classement "
    "trompeur."
)

imp_data = {
    "Feature": [
        "traffic_density", "congestion_score", "signal_wait_time",
        "driver_experience_encoded", "horn_events_per_min", "avg_speed",
        "road_quality_score", "horn_density",
        "weather_Foggy", "weather_Hot", "weather_Rainy"
    ],
    "XGBoost":      [0.3254, 0.2227, 0.1867, 0.1437, 0.0422, 0.0420,
                     0.0352, 0.0011, 0.0004, 0.0003, 0.0003],
    "RandomForest": [0.2502, 0.3524, 0.1167, 0.1210, 0.0332, 0.0574,
                     0.0622, 0.0062, 0.0002, 0.0001, 0.0003],
    "LightGBM":     [0.3399, 0.3922, 0.0241, 0.1259, 0.0207, 0.0330,
                     0.0634, 0.0007, 0.0000, 0.0000, 0.0000],
}
imp_df = pd.DataFrame(imp_data)
choice = st.radio("Modèle :", ["XGBoost", "RandomForest", "LightGBM"],
                  horizontal=True)
imp_s  = imp_df[["Feature", choice]].sort_values(choice)
fig_imp = px.bar(
    imp_s, x=choice, y="Feature", orientation="h",
    color=choice, color_continuous_scale=["#E6F1FB", "#1F4E79"],
    labels={choice: "Importance", "Feature": ""},
    height=420, text=choice
)
fig_imp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig_imp.update_layout(plot_bgcolor="white", coloraxis_showscale=False,
                      margin=dict(r=60))
st.plotly_chart(fig_imp, use_container_width=True)
st.success(
    "**Le trio congestion : `traffic_density`, `congestion_score` et "
    "`signal_wait_time` concentre 72 % à 76 % du gain** selon le modèle.\n\n"
    "L'importance se répartit entre `congestion_score` et les deux variables "
    "dont il est le produit : c'est l'effet attendu de la colinéarité, "
    "deux features redondantes se partageant le crédit. Cela ne dégrade pas "
    "les prédictions — conserver les trois fait gagner 0.0014 de R² — mais "
    "interdit de lire ce classement comme une hiérarchie causale."
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
