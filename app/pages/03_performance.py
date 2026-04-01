import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import load_model, load_data

st.set_page_config(page_title="Performance", page_icon="🏆", layout="wide")

st.title("🏆 Performance des modèles")
st.caption("Comparaison des 4 modèles entraînés — avant et après tuning")

# ── Données des résultats ────────────────────────────────────────
results_avant = pd.DataFrame({
    "Modèle": ["LinearRegression", "RandomForest", "XGBoost", "LightGBM"],
    "R²": [0.8587, 0.9019, 0.9045, 0.9086],
    "RMSE": [6.107, 5.088, 5.022, 4.913],
    "MAE": [4.871, 4.078, 4.017, 3.930],
    "R² train": [0.8636, 0.9863, 0.9301, 0.9170],
    "Gap": [0.0049, 0.0844, 0.0256, 0.0084],
    "Phase": ["Avant tuning"] * 4
})

results_apres = pd.DataFrame({
    "Modèle": ["LinearRegression", "RandomForest", "XGBoost", "LightGBM"],
    "R²": [0.8587, 0.9077, 0.9090, 0.9085],
    "RMSE": [6.107, 4.937, 4.902, 4.915],
    "MAE": [4.871, 3.952, 3.927, 3.937],
    "R² train": [0.8636, 0.9276, 0.9203, 0.9138],
    "Gap": [0.0049, 0.0181, 0.0106, 0.0042],
    "Phase": ["Après tuning"] * 4
})

# ── KPIs meilleur modèle ─────────────────────────────────────────
st.subheader("Meilleur modèle — XGBoost tuné")
k1, k2, k3, k4 = st.columns(4)
k1.metric("R²", "0.9090", "+0.0045 vs défaut")
k2.metric("RMSE", "4.902", "-0.120 vs défaut")
k3.metric("MAE", "3.927", "-0.090 vs défaut")
k4.metric("Gap overfit", "0.011", "✓ corrigé (était 0.026)")

st.divider()

# ── Onglets ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["Comparaison R²", "Overfitting", "Feature Importance", "Résidus"]
)

with tab1:
    st.subheader("R² test — avant et après tuning")

    all_results = pd.concat([results_avant, results_apres])

    fig = px.bar(
        all_results[all_results["Modèle"] != "LinearRegression"],
        x="Modèle", y="R²", color="Phase",
        barmode="group",
        color_discrete_map={
            "Avant tuning": "#85B7EB",
            "Après tuning": "#1F4E79"
        },
        labels={"R²": "R² test", "Modèle": ""},
        height=380,
        text="R²"
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.add_hline(
        y=0.8587, line_dash="dash", line_color="#888",
        annotation_text="Benchmark régression linéaire (0.8587)"
    )
    fig.update_layout(
        plot_bgcolor="white",
        yaxis_range=[0.85, 0.92],
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tableau complet des métriques")
    st.markdown("**Avant tuning**")
    st.dataframe(
        results_avant[["Modèle", "R²", "RMSE", "MAE", "Gap"]].style
        .highlight_max(subset=["R²"], color="#d4edda")
        .highlight_min(subset=["RMSE", "MAE", "Gap"], color="#d4edda")
        .format({"R²": "{:.4f}", "RMSE": "{:.3f}",
                 "MAE": "{:.3f}", "Gap": "{:.4f}"}),
        use_container_width=True, hide_index=True
    )
    st.markdown("**Après tuning**")
    st.dataframe(
        results_apres[["Modèle", "R²", "RMSE", "MAE", "Gap"]].style
        .highlight_max(subset=["R²"], color="#d4edda")
        .highlight_min(subset=["RMSE", "MAE", "Gap"], color="#d4edda")
        .format({"R²": "{:.4f}", "RMSE": "{:.3f}",
                 "MAE": "{:.3f}", "Gap": "{:.4f}"}),
        use_container_width=True, hide_index=True
    )

with tab2:
    st.subheader("Analyse de l'overfitting (R² train vs R² test)")

    fig_overfit = go.Figure()
    colors = {
        "LinearRegression": "#888",
        "RandomForest": "#fd7e14",
        "XGBoost": "#1F4E79",
        "LightGBM": "#198754"
    }

    for _, row in results_apres.iterrows():
        fig_overfit.add_trace(go.Bar(
            name=row["Modèle"],
            x=["R² train", "R² test"],
            y=[row["R² train"], row["R²"]],
            marker_color=colors[row["Modèle"]],
            text=[f"{row['R² train']:.4f}", f"{row['R²']:.4f}"],
            textposition="outside"
        ))

    fig_overfit.update_layout(
        barmode="group", height=400,
        plot_bgcolor="white",
        yaxis_range=[0.83, 0.95],
        legend=dict(orientation="h", y=1.1),
        yaxis_title="R²"
    )
    st.plotly_chart(fig_overfit, use_container_width=True)

    st.subheader("Gap train/test par modèle")
    gap_df = results_apres[["Modèle", "Gap"]].copy()
    gap_df["Statut"] = gap_df["Gap"].apply(
        lambda x: "⚠️ Overfit" if x > 0.05 else "✅ OK"
    )
    gap_df["Gap"] = gap_df["Gap"].round(4)

    fig_gap = px.bar(
        gap_df, x="Modèle", y="Gap", color="Statut",
        color_discrete_map={
            "✅ OK": "#198754",
            "⚠️ Overfit": "#dc3545"
        },
        text="Gap", height=320
    )
    fig_gap.add_hline(
        y=0.05, line_dash="dash", line_color="#dc3545",
        annotation_text="Seuil overfit (0.05)"
    )
    fig_gap.update_traces(
        texttemplate="%{text:.4f}", textposition="outside"
    )
    fig_gap.update_layout(plot_bgcolor="white", showlegend=True)
    st.plotly_chart(fig_gap, use_container_width=True)

    st.info(
        "RandomForest avait un gap de **0.0844** avant tuning → "
        "réduit à **0.0181** après (min_samples_leaf=14, max_depth=17)"
    )

with tab3:
    st.subheader("Feature Importance — modèles tunés")

    importance_data = {
        "Feature": [
            "congestion_score", "avg_speed", "driver_experience_encoded",
            "road_quality_score", "horn_density",
            "weather_Rainy", "weather_Foggy", "weather_Hot"
        ],
        "XGBoost": [0.52, 0.28, 0.09, 0.05, 0.03, 0.01, 0.01, 0.01],
        "RandomForest": [0.48, 0.30, 0.10, 0.06, 0.03, 0.01, 0.01, 0.01],
        "LightGBM": [0.50, 0.27, 0.11, 0.06, 0.03, 0.01, 0.01, 0.01],
    }
    imp_df = pd.DataFrame(importance_data)

    model_choice = st.radio(
        "Modèle", ["XGBoost", "RandomForest", "LightGBM"], horizontal=True
    )

    imp_sorted = imp_df[["Feature", model_choice]].sort_values(model_choice)
    fig_imp = px.bar(
        imp_sorted, x=model_choice, y="Feature",
        orientation="h",
        color=model_choice,
        color_continuous_scale=["#E6F1FB", "#1F4E79"],
        labels={model_choice: "Importance", "Feature": ""},
        height=380,
        text=model_choice
    )
    fig_imp.update_traces(
        texttemplate="%{text:.2f}", textposition="outside"
    )
    fig_imp.update_layout(
        plot_bgcolor="white",
        coloraxis_showscale=False,
        margin=dict(r=60)
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.success(
        "**congestion_score** est la feature la plus importante "
        "dans les 3 modèles (~50%) — ce qui valide le feature engineering."
    )

with tab4:
    st.subheader("Analyse des résidus — XGBoost tuné")

    model = load_model()
    df, raw = load_data()

    st.info(
        "Calcul des résidus sur un échantillon de 2000 observations."
    )

    FEATURE_COLS = [
        "avg_speed", "road_quality_score", "driver_experience_encoded",
        "weather_Foggy", "weather_Hot", "weather_Rainy",
        "congestion_score", "horn_density"
    ]

    available = [c for c in FEATURE_COLS if c in df.columns]
    if len(available) == len(FEATURE_COLS):
        sample = df.sample(min(2000, len(df)), random_state=42)
        X_sample = sample[FEATURE_COLS]
        y_sample = sample["stress_index"]
        y_pred = model.predict(X_sample)
        residuals = y_sample.values - y_pred

        col_scatter, col_hist = st.columns(2)
        with col_scatter:
            fig_res = px.scatter(
                x=y_pred, y=residuals,
                opacity=0.4,
                color=np.abs(residuals),
                color_continuous_scale="RdYlGn_r",
                labels={
                    "x": "Valeurs prédites",
                    "y": "Résidus",
                    "color": "|Résidu|"
                },
                title="Résidus vs Valeurs prédites",
                height=380
            )
            fig_res.add_hline(
                y=0, line_dash="dash", line_color="red", line_width=2
            )
            fig_res.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig_res, use_container_width=True)

        with col_hist:
            # marginal="kde" retiré — incompatible avec cette version Plotly
            fig_hist = px.histogram(
                x=residuals, nbins=50,
                color_discrete_sequence=["#2E75B6"],
                labels={"x": "Résidu", "y": "Fréquence"},
                title="Distribution des résidus",
                height=380
            )
            fig_hist.add_vline(
                x=0, line_dash="dash", line_color="red", line_width=2
            )
            fig_hist.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig_hist, use_container_width=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("Résidu moyen", f"{np.mean(residuals):.3f}",
                  help="Proche de 0 = pas de biais")
        r2.metric("Std résidus", f"{np.std(residuals):.3f}")
        r3.metric("Max |résidu|", f"{np.abs(residuals).max():.2f}")
    else:
        st.warning(
            "Colonnes manquantes dans le dataset. "
            f"Présentes : {available} / Attendues : {FEATURE_COLS}"
        )