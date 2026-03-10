# 🚂 RailSafe — Prédiction de retards ferroviaires SNCF

> Système de prédiction de retards ferroviaires combinant **SNCF Open Data** et **données météo Open-Meteo** pour anticiper les retards sur le réseau TGV français.

## 🏗️ Architecture
```
SNCF Open Data (TGV/TER/Intercités) + Open-Meteo API
        ↓
Feature Engineering (temporel + causes + météo)
        ↓
XGBoost + SHAP (explicabilité)
        ↓
FastAPI ──▶ Dashboard Streamlit
```

## 🧠 Key Insights

> **La météo n'améliore pas le modèle (+0.0002 AUC)** — le mois capture déjà la saisonnalité météo. Les retards TGV sont structurels (infrastructure, trafic) plus que météorologiques.

> **Juillet = mois le plus retardé (17.6%)**, Mai le moins retardé (11.3%). La liaison Lyon Part Dieu → Lille atteint 94% de probabilité de retard en juillet 2023.

## 🧰 Stack

| Layer | Tools |
|-------|-------|
| Data | SNCF Open Data, Open-Meteo API |
| Feature Engineering | Pandas, NumPy |
| ML | XGBoost, Scikit-learn |
| Explicabilité | SHAP |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Météo | openmeteo-requests |

## 📁 Project Structure
```
railsafe/
├── data/
│   ├── raw/                    # Données brutes (gitignored)
│   └── viz_*.png               # Visualisations EDA
├── notebooks/
│   ├── 01_eda.ipynb            # Exploration données SNCF
│   ├── 02_training.ipynb       # XGBoost baseline
│   └── 03_training_meteo.ipynb # Comparaison avec météo
├── src/
│   └── model.py                # Classe d'inférence
├── api/
│   └── main.py                 # FastAPI endpoints
├── app/
│   └── dashboard.py            # Streamlit dashboard
└── scripts/
    ├── download_data.py        # Téléchargement SNCF
    └── download_weather.py     # Téléchargement météo
```

## 📊 Résultats

| Expérience | ROC-AUC | F1 | Notes |
|------------|---------|-----|-------|
| Baseline (TGV+TER+IC) | 0.867 | 0.579 | Dataset unifié |
| TGV seul | 0.871 | 0.606 | Meilleurs résultats |
| TGV + météo | 0.870 | 0.605 | Gain négligeable |

**Top features SHAP :** Liaison · Mois · Année · Gestion trafic · Infrastructure

## 🚀 Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/NaimMG/railsafe.git
cd railsafe
python -m venv RailSafe
source RailSafe/bin/activate
pip install -r requirements.txt
```

### 2. Télécharger les données
```bash
python scripts/download_data.py
python scripts/download_weather.py
```

### 3. Entraîner le modèle
```bash
jupyter notebook notebooks/03_training_meteo.ipynb
```

### 4. Lancer l'API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 5. Lancer le dashboard
```bash
streamlit run app/dashboard.py
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Info API + métriques |
| GET | `/health` | Status modèle |
| POST | `/predict` | Prédiction retard |
| GET | `/liaisons` | Liste des 130 liaisons |
| GET | `/regions` | Liste des régions |

### Exemple prédiction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"liaison": "LYON PART DIEU -> LILLE", "mois": 7, "annee": 2023}'

# → {"retard_eleve": true, "probabilite": 0.9404, "niveau_risque": "🔴 ÉLEVÉ"}
```

## 👤 Author

**NaimMG** · [GitHub](https://github.com/NaimMG) · [HuggingFace](https://huggingface.co/Chasston)