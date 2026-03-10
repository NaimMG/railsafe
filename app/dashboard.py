"""
RailSafe — Dashboard Streamlit
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

API_URL  = "http://localhost:8000"
DATA_DIR = Path(__file__).parent.parent / "data"

st.set_page_config(
    page_title = "🚂 RailSafe",
    page_icon  = "🚂",
    layout     = "wide",
)

# ── Header ───────────────────────────────────────────────────────
st.title("🚂 RailSafe — Prédiction de retards SNCF")
st.markdown("*XGBoost + SHAP sur données SNCF Open Data (TGV, TER, Intercités)*")
st.divider()

# ── Chargement liaisons/régions depuis API ────────────────────────
@st.cache_data
def get_liaisons():
    try:
        r = requests.get(f"{API_URL}/liaisons", timeout=5)
        return r.json()["liaisons"]
    except:
        return ["PARIS MONTPARNASSE -> BORDEAUX ST JEAN",
                "PARIS LYON -> MARSEILLE ST CHARLES"]

@st.cache_data
def get_regions():
    try:
        r = requests.get(f"{API_URL}/regions", timeout=5)
        return r.json()["regions"]
    except:
        return ["National", "Ile-de-France"]

liaisons = get_liaisons()
regions  = get_regions()

# ── Layout 2 colonnes ─────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("🎯 Paramètres de prédiction")

    type_service = st.selectbox(
        "Type de service",
        ["TGV", "TER", "IC"],
        help="TGV = grande vitesse, TER = régional, IC = Intercités"
    )

    # Filtre liaisons selon type
    if type_service == "TER":
        liaisons_filtered = [l for l in liaisons if l.startswith("TER_")]
    elif type_service == "IC":
        liaisons_filtered = [l for l in liaisons if "->" in l and not l.startswith("TER_")]
    else:
        liaisons_filtered = [l for l in liaisons if "->" in l and not l.startswith("TER_")]

    if not liaisons_filtered:
        liaisons_filtered = liaisons

    liaison = st.selectbox("Liaison", liaisons_filtered)

    col_m, col_a = st.columns(2)
    with col_m:
        mois  = st.slider("Mois", 1, 12, 7,
                          format="%d",
                          help="1=Janvier … 12=Décembre")
    with col_a:
        annee = st.slider("Année", 2018, 2026, 2024)

    region = st.selectbox("Région", regions,
                          index=regions.index("National") if "National" in regions else 0)

    st.divider()
    st.caption("⚙️ Paramètres avancés (optionnels)")
    with st.expander("Causes de retards estimées (%)"):
        prct_externe  = st.slider("Causes externes",   0, 100, 20)
        prct_infra    = st.slider("Infrastructure",    0, 100, 22)
        prct_trafic   = st.slider("Gestion trafic",    0, 100, 20)
        prct_materiel = st.slider("Matériel roulant",  0, 100, 19)
        taux_annul    = st.slider("Taux annulation (%)", 0, 20, 2) / 100

    predict_btn = st.button("🔮 Prédire", type="primary", use_container_width=True)

# ── Résultats ─────────────────────────────────────────────────────
with col_result:
    st.subheader("📊 Résultat")

    if predict_btn:
        payload = {
            "type_service"      : type_service,
            "region"            : region,
            "liaison"           : liaison,
            "mois"              : mois,
            "annee"             : annee,
            "taux_annulation"   : taux_annul,
            "prct_cause_externe": prct_externe,
            "prct_cause_infra"  : prct_infra,
            "prct_cause_trafic" : prct_trafic,
            "prct_cause_materiel": prct_materiel,
        }

        with st.spinner("Prédiction en cours..."):
            try:
                r = requests.post(f"{API_URL}/predict",
                                  json=payload, timeout=10)
                result = r.json()

                proba  = result["probabilite"]
                risque = result["niveau_risque"]

                # Jauge
                fig = go.Figure(go.Indicator(
                    mode  = "gauge+number+delta",
                    value = proba * 100,
                    title = {"text": "Probabilité de retard élevé (%)"},
                    delta = {"reference": 25},
                    gauge = {
                        "axis"  : {"range": [0, 100]},
                        "bar"   : {"color": "red" if proba >= 0.5 else "green"},
                        "steps" : [
                            {"range": [0,  25], "color": "#d4edda"},
                            {"range": [25, 50], "color": "#fff3cd"},
                            {"range": [50, 100],"color": "#f8d7da"},
                        ],
                        "threshold": {
                            "line" : {"color": "black", "width": 3},
                            "thickness": 0.75,
                            "value": 50
                        }
                    }
                ))
                fig.update_layout(height=300, margin=dict(t=50, b=0))
                st.plotly_chart(fig, use_container_width=True)

                # Badge résultat
                if result["retard_eleve"]:
                    st.error(f"## {risque}\nProbabilité : **{proba*100:.1f}%**")
                else:
                    st.success(f"## {risque}\nProbabilité : **{proba*100:.1f}%**")

                # Détails
                st.markdown(f"""
                | Paramètre | Valeur |
                |-----------|--------|
                | Service   | {type_service} |
                | Liaison   | {liaison} |
                | Mois      | {mois} |
                | Année     | {annee} |
                | Région    | {region} |
                """)

            except Exception as e:
                st.error(f"❌ Erreur API : {e}")
    else:
        st.info("👈 Configure les paramètres et clique sur **Prédire**")

# ── Section visualisations ────────────────────────────────────────
st.divider()
st.subheader("📈 Analyse exploratoire")

tab1, tab2, tab3 = st.tabs(["Évolution TGV", "Saisonnalité", "SHAP"])

with tab1:
    img = DATA_DIR / "viz_taux_retard_tgv.png"
    if img.exists():
        st.image(str(img), use_container_width=True)

with tab2:
    img = DATA_DIR / "viz_saisonnalite_tgv.png"
    if img.exists():
        st.image(str(img), use_container_width=True)

with tab3:
    img = DATA_DIR / "viz_shap_beeswarm.png"
    if img.exists():
        st.image(str(img), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption("RailSafe · XGBoost · SHAP · FastAPI · Streamlit | "
           "Data : SNCF Open Data | "
           "[GitHub](https://github.com/NaimMG/railsafe)")