"""
Téléchargement des données SNCF Open Data.
Usage : python scripts/download_data.py
"""
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Dossiers ────────────────────────────────────────────────────
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Datasets SNCF Open Data ──────────────────────────────────────
DATASETS = [
    {
        "name"    : "regularite_tgv",
        "url"     : "https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets/regularite-mensuelle-tgv-aqst/exports/csv?lang=fr&timezone=Europe%2FParis&use_labels=true&delimiter=%3B",
        "filename": "regularite_tgv.csv",
        "desc"    : "Régularité mensuelle TGV (2015-2024)",
    },
    {
        "name"    : "regularite_ter",
        "url"     : "https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets/regularite-mensuelle-ter/exports/csv?lang=fr&timezone=Europe%2FParis&use_labels=true&delimiter=%3B",
        "filename": "regularite_ter.csv",
        "desc"    : "Régularité mensuelle TER (2013-2024)",
    },
    {
        "name"    : "regularite_intercites",
        "url"     : "https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets/regularite-mensuelle-intercites/exports/csv?lang=fr&timezone=Europe%2FParis&use_labels=true&delimiter=%3B",
        "filename": "regularite_intercites.csv",
        "desc"    : "Régularité mensuelle Intercités (2014-2024)",
    },
]


def download_file(url: str, dest: Path, desc: str) -> bool:
    """Télécharge un fichier CSV depuis une URL."""
    if dest.exists():
        print(f"  ✅ Déjà présent : {dest.name}")
        return True

    print(f"  📦 Téléchargement : {desc}")
    try:
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

        print(f"  ✅ Sauvegardé : {dest}")
        return True

    except Exception as e:
        print(f"  ❌ Erreur : {e}")
        return False


def check_dataset(filepath: Path) -> None:
    """Affiche un aperçu rapide du dataset téléchargé."""
    try:
        df = pd.read_csv(filepath, sep=";", encoding="utf-8", nrows=5)
        print(f"  Colonnes ({len(df.columns)}) : {list(df.columns[:6])}...")
        print(f"  Shape preview       : {df.shape}")
    except Exception as e:
        print(f"  ⚠️  Aperçu impossible : {e}")


if __name__ == "__main__":
    print("🚂 RailSafe — Téléchargement des données SNCF\n")

    for ds in DATASETS:
        print(f"\n{'='*55}")
        print(f"Dataset : {ds['desc']}")
        dest = RAW_DIR / ds["filename"]
        ok   = download_file(ds["url"], dest, ds["desc"])
        if ok:
            check_dataset(dest)

    print(f"\n{'='*55}")
    print(f"✅ Données sauvegardées dans : {RAW_DIR}")
    print("\nFichiers disponibles :")
    for f in sorted(RAW_DIR.glob("*.csv")):
        size = f.stat().st_size / 1e6
        print(f"  {f.name:35s} : {size:.2f} MB")