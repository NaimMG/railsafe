"""
RailSafe — Classe d'inférence XGBoost.
Compatible avec le bundle TGV-only (03_training_meteo.ipynb).
"""
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "railsafe_model.joblib"

MOIS_LABELS = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
}


class RailSafeModel:

    def __init__(self, model_path: str = None):
        path   = Path(model_path) if model_path else MODEL_PATH
        bundle = joblib.load(path)

        self.model        = bundle["model"]
        self.feature_cols = bundle["feature_cols"]
        self.le_type      = bundle.get("le_type")
        self.le_region    = bundle.get("le_region")
        self.le_liaison   = bundle["le_liaison"]
        self.with_meteo   = bundle.get("with_meteo", False)
        self.metrics      = bundle["metrics"]

        # Liaisons valides (sans anomalies "0")
        self.liaisons_valides = [
            l for l in self.le_liaison.classes_
            if "->" in str(l) and not str(l).startswith("0")
        ]

        print(f"✅ RailSafeModel chargé")
        print(f"   ROC-AUC  : {self.metrics['roc_auc']}")
        print(f"   Météo    : {self.with_meteo}")
        print(f"   Liaisons : {len(self.liaisons_valides)}")

    def _encode_safe(self, le, value: str) -> int:
        try:
            return int(le.transform([value])[0])
        except ValueError:
            return -1

    def predict(
        self,
        liaison             : str,
        mois                : int,
        annee               : int,
        type_service        : str   = "TGV",
        region              : str   = "National",
        taux_annulation     : float = 0.02,
        prct_cause_externe  : float = 20.0,
        prct_cause_infra    : float = 22.0,
        prct_cause_trafic   : float = 20.0,
        prct_cause_materiel : float = 19.0,
        # Features météo (optionnelles)
        temp_mean_mois      : float = None,
        precip_sum_mois     : float = None,
        wind_max_mois       : float = None,
        snow_sum_mois       : float = None,
        jours_pluie         : int   = None,
        jours_neige         : int   = None,
    ) -> dict:

        trimestre = (mois - 1) // 3 + 1

        features = {
            "annee"                                    : annee,
            "mois"                                     : mois,
            "trimestre"                                : trimestre,
            "taux_annulation"                          : taux_annulation,
            "Prct retard pour causes externes"         : prct_cause_externe,
            "Prct retard pour cause infrastructure"    : prct_cause_infra,
            "Prct retard pour cause gestion trafic"    : prct_cause_trafic,
            "Prct retard pour cause matériel roulant"  : prct_cause_materiel,
            "liaison_enc" : self._encode_safe(self.le_liaison, liaison),
        }

        # Ajout météo si le modèle l'utilise
        if self.with_meteo:
            features.update({
                "temp_mean_mois" : temp_mean_mois,
                "precip_sum_mois": precip_sum_mois,
                "wind_max_mois"  : wind_max_mois,
                "snow_sum_mois"  : snow_sum_mois,
                "jours_pluie"    : jours_pluie,
                "jours_neige"    : jours_neige,
            })

        X     = pd.DataFrame([features])[self.feature_cols]
        proba = float(self.model.predict_proba(X)[0][1])

        # Seuil abaissé à 0.35 pour meilleur recall
        retard_eleve = proba >= 0.35

        return {
            "retard_eleve"  : retard_eleve,
            "probabilite"   : round(proba, 4),
            "niveau_risque" : "🔴 ÉLEVÉ" if retard_eleve else "🟢 NORMAL",
            "type_service"  : type_service,
            "region"        : region,
            "liaison"       : liaison,
            "mois"          : mois,
            "mois_label"    : MOIS_LABELS[mois],
            "annee"         : annee,
        }