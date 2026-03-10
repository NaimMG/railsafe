"""
Téléchargement des données météo historiques via Open-Meteo.
Villes principales du réseau TGV français.
Usage : python scripts/download_weather.py
"""
import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Villes principales réseau TGV ────────────────────────────────
VILLES = {
    "Paris"    : {"lat": 48.8566, "lon": 2.3522},
    "Lyon"     : {"lat": 45.7640, "lon": 4.8357},
    "Marseille": {"lat": 43.2965, "lon": 5.3698},
    "Bordeaux" : {"lat": 44.8378, "lon": -0.5792},
    "Lille"    : {"lat": 50.6292, "lon": 3.0573},
    "Toulouse" : {"lat": 43.6047, "lon": 1.4442},
    "Nantes"   : {"lat": 47.2184, "lon": -1.5536},
    "Rennes"   : {"lat": 48.1173, "lon": -1.6778},
}

# ── Client Open-Meteo avec cache ─────────────────────────────────
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
om_client     = openmeteo_requests.Client(session=retry_session)

def fetch_weather(ville: str, lat: float, lon: float) -> pd.DataFrame:
    """Télécharge météo mensuelle agrégée pour une ville."""
    print(f"  📦 {ville} ({lat}, {lon})")

    url    = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude"   : lat,
        "longitude"  : lon,
        "start_date" : "2013-01-01",
        "end_date"   : "2025-12-31",
        "daily"      : [
            "temperature_2m_mean",
            "precipitation_sum",
            "windspeed_10m_max",
            "snowfall_sum",
        ],
        "timezone"   : "Europe/Paris",
    }

    responses = om_client.weather_api(url, params=params)
    response  = responses[0]
    daily     = response.Daily()

    df = pd.DataFrame({
        "date"        : pd.date_range(
            start = pd.to_datetime(daily.Time(), unit="s", utc=True),
            end   = pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq  = pd.tseries.frequencies.to_offset(pd.Timedelta(seconds=daily.Interval())),
            inclusive = "left"
        ),
        "temp_mean"   : daily.Variables(0).ValuesAsNumpy(),
        "precip_sum"  : daily.Variables(1).ValuesAsNumpy(),
        "wind_max"    : daily.Variables(2).ValuesAsNumpy(),
        "snow_sum"    : daily.Variables(3).ValuesAsNumpy(),
    })

    # Agrégation mensuelle
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["mois_annee"] = df["date"].dt.to_period("M")

    monthly = df.groupby("mois_annee").agg(
        temp_mean_mois   = ("temp_mean",  "mean"),
        precip_sum_mois  = ("precip_sum", "sum"),
        wind_max_mois    = ("wind_max",   "max"),
        snow_sum_mois    = ("snow_sum",   "sum"),
        jours_pluie      = ("precip_sum", lambda x: (x > 1).sum()),
        jours_neige      = ("snow_sum",   lambda x: (x > 0).sum()),
    ).reset_index()

    monthly["ville"] = ville
    monthly["annee"] = monthly["mois_annee"].dt.year
    monthly["mois"]  = monthly["mois_annee"].dt.month
    monthly = monthly.drop(columns=["mois_annee"])

    return monthly


if __name__ == "__main__":
    print("🌦️  RailSafe — Téléchargement météo Open-Meteo\n")

    dfs = []
    for ville, coords in VILLES.items():
        try:
            df = fetch_weather(ville, coords["lat"], coords["lon"])
            dfs.append(df)
            print(f"  ✅ {ville} : {len(df)} mois")
        except Exception as e:
            print(f"  ❌ {ville} : {e}")

    if dfs:
        meteo = pd.concat(dfs, ignore_index=True)
        dest  = RAW_DIR / "meteo_villes.csv"
        meteo.to_csv(dest, index=False)
        print(f"\n✅ Sauvegardé : {dest}")
        print(f"Shape : {meteo.shape}")
        print(meteo.head())