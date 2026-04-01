import streamlit as st
import plotly.express as px
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import load_data

st.set_page_config(page_title="Exploration", page_icon="📊", layout="wide")

st.title("📊 Exploration du dataset")
st.caption("50 000 observations · Smart City Traffic Stress Dataset")

df, raw = load_data()

# ── Filtres sidebar ──────────────────────────────────────────────
st.sidebar.header("Filtres")
weather_filter = st.sidebar.multiselect(
    "Météo", ["Clear", "Foggy", "Hot", "Rainy"],
    default=["Clear", "Foggy", "Hot", "Rainy"]
)
exp_filter = st.sidebar.multiselect(
    "Expérience conducteur", ["Beginner", "Intermediate", "Expert"],
    default=["Beginner", "Intermediate", "Expert"]
)
stress_range = st.sidebar.slider("Plage stress_index", 0, 100, (0, 100))

filtered = raw[
    raw["weather_condition"].isin(weather_filter) &
    raw["driver_experience_level"].isin(exp_filter) &
    raw["stress_index"].between(*stress_range)
]

st.sidebar.metric(
    "Observations filtrées",
    f"{len(filtered):,}",
    f"{len(filtered)/len(raw)*100:.1f}%"
)

# ── KPIs ─────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Stress moyen", f"{filtered['stress_index'].mean():.1f}")
k2.metric("Vitesse moyenne", f"{filtered['avg_speed'].mean():.1f} km/h")
k3.metric("Densité moyenne", f"{filtered['traffic_density'].mean():.0f}")
k4.metric("Attente moyenne", f"{filtered['signal_wait_time'].mean():.1f}s")

st.divider()

# ── Onglets ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["Distributions", "Corrélations", "Catégorielles", "Bivariée"]
)

with tab1:
    st.subheader("Distributions des variables numériques")
    num_var = st.selectbox("Variable",
        ["stress_index", "traffic_density", "avg_speed",
         "signal_wait_time", "horn_events_per_min", "road_quality_score"])

    col_hist, col_box = st.columns([2, 1])
    with col_hist:
        # marginal="kde" retiré — incompatible avec cette version de Plotly
        fig = px.histogram(
            filtered, x=num_var, nbins=40,
            color_discrete_sequence=["#2E75B6"],
            labels={num_var: num_var}
        )
        fig.update_layout(plot_bgcolor="white", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col_box:
        stats = filtered[num_var].describe()
        st.markdown("**Statistiques**")
        for stat, val in stats.items():
            st.metric(stat, f"{val:.2f}")

with tab2:
    st.subheader("Matrice de corrélation")
    num_cols = ["traffic_density", "horn_events_per_min", "avg_speed",
                "signal_wait_time", "road_quality_score", "stress_index"]

    corr = filtered[num_cols].corr().round(2)
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        height=480
    )
    fig_corr.update_layout(margin=dict(t=30, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Corrélations avec stress_index")
    corr_target = (
        filtered[num_cols].corr()["stress_index"]
        .drop("stress_index")
        .sort_values()
    )
    fig_corr_bar = px.bar(
        x=corr_target.values, y=corr_target.index,
        orientation="h",
        color=corr_target.values,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        labels={"x": "Corrélation de Pearson", "y": ""},
        height=300
    )
    fig_corr_bar.update_layout(
        plot_bgcolor="white",
        showlegend=False,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_corr_bar, use_container_width=True)

with tab3:
    st.subheader("Distribution des variables catégorielles")
    c1, c2 = st.columns(2)

    with c1:
        weather_counts = (
            filtered["weather_condition"].value_counts().reset_index()
        )
        fig_w = px.pie(
            weather_counts, values="count", names="weather_condition",
            title="Météo",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        fig_w.update_layout(height=320)
        st.plotly_chart(fig_w, use_container_width=True)

    with c2:
        exp_counts = (
            filtered["driver_experience_level"].value_counts().reset_index()
        )
        order = ["Beginner", "Intermediate", "Expert"]
        exp_counts["driver_experience_level"] = pd.Categorical(
            exp_counts["driver_experience_level"],
            categories=order, ordered=True
        )
        exp_counts = exp_counts.sort_values("driver_experience_level")
        fig_e = px.bar(
            exp_counts, x="driver_experience_level", y="count",
            title="Expérience conducteur",
            color="driver_experience_level",
            color_discrete_sequence=["#fd7e14", "#2E75B6", "#198754"],
            labels={"driver_experience_level": "", "count": "Observations"}
        )
        fig_e.update_layout(
            showlegend=False, plot_bgcolor="white", height=320
        )
        st.plotly_chart(fig_e, use_container_width=True)

    st.subheader("Stress index par catégorie")
    cat_choice = st.radio(
        "Variable",
        ["weather_condition", "driver_experience_level"],
        horizontal=True
    )
    fig_box = px.box(
        filtered, x=cat_choice, y="stress_index",
        color=cat_choice,
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={cat_choice: "", "stress_index": "Stress index"},
        height=380
    )
    fig_box.update_layout(showlegend=False, plot_bgcolor="white")
    st.plotly_chart(fig_box, use_container_width=True)

with tab4:
    st.subheader("Relation entre deux variables")
    c1, c2 = st.columns(2)
    with c1:
        x_var = st.selectbox("Axe X",
            ["traffic_density", "avg_speed", "signal_wait_time",
             "horn_events_per_min", "road_quality_score"], key="xvar")
    with c2:
        y_var = st.selectbox("Axe Y",
            ["stress_index", "traffic_density", "avg_speed",
             "signal_wait_time", "horn_events_per_min"], key="yvar")

    color_by = st.selectbox("Colorer par",
        ["weather_condition", "driver_experience_level", "Aucun"])

    sample = filtered.sample(min(3000, len(filtered)), random_state=42)

    fig_scatter = px.scatter(
        sample, x=x_var, y=y_var,
        color=None if color_by == "Aucun" else color_by,
        opacity=0.4,
        trendline="ols",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={x_var: x_var, y_var: y_var},
        height=420
    )
    fig_scatter.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig_scatter, use_container_width=True)

    corr_val = sample[x_var].corr(sample[y_var])
    st.info(
        f"Corrélation de Pearson entre **{x_var}** et **{y_var}** : "
        f"**r = {corr_val:.3f}**"
    )