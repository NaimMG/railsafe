# 🚂 RailSafe — Prédiction d'incidents ferroviaires SNCF

> Système de prédiction d'incidents ferroviaires combinant les données **SNCF Open Data** et les données **météorologiques** pour anticiper les défaillances sur le réseau ferroviaire français.

## 🏗️ Architecture
```
SNCF Open Data + Open-Meteo API
        ↓
Feature Engineering (temporel + météo)
        ↓
XGBoost + SHAP (explicabilité)
        ↓
FastAPI ──▶ Dashboard Streamlit
```

## 🧰 Stack

| Layer | Tools |
|-------|-------|
| Data | SNCF Open Data, Open-Meteo |
| Feature Engineering | Pandas, NumPy |
| ML | XGBoost, Scikit-learn, SHAP |
| Imbalanced Data | Imbalanced-learn (SMOTE) |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |

## 📁 Project Structure
```
railsafe/
├── data/               # Scripts de collecte
├── notebooks/          # Exploration et entraînement
├── src/                # Code source
├── api/                # FastAPI
├── app/                # Streamlit dashboard
└── scripts/            # Utilitaires
```

## 📊 Results

*To be updated after training.*

## 🚀 Quickstart
```bash
# Coming soon
```

## 👤 Author

**NaimMG** · [GitHub](https://github.com/NaimMG) · [HuggingFace](https://huggingface.co/Chasston)