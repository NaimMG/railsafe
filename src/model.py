"""
RailSafe — Classe d'inférence XGBoost.
Chargement du modèle et prédiction.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "railsafe_model.joblib"


class RailSafeModel:

    def __init__(self, model_path: str = None):
        path = Path(model_path) if model_path else MODEL_PATH
        bundle = joblib.load(path)

        self.model        = bundle["model"]
        self.feature_cols = bundle["feature_cols"]
        self.le_type      = bundle["le_type"]
        self.le_region    = bundle["le_region"]
        self.le_liaison   = bundle["le_liaison"]
        self.metrics      = bundle["metrics"]
        print(f"✅ RailSafeModel chargé — ROC-AUC={self.metrics['roc_auc']}")

    def _encode_safe(self, le, value: str) -> int:
        """Encode une valeur, retourne -1 si inconnue."""
        try:
            return int(le.transform([value])[0])
        except ValueError:
            return -1

    def predict(
        self,
        type_service : str,
        region       : str,
        liaison      : str,
        mois         : int,
        annee        : int,
        taux_annulation          : float = 0.0,
        prct_cause_externe       : float = 0.0,
        prct_cause_infra         : float = 0.0,
        prct_cause_trafic        : float = 0.0,
        prct_cause_materiel      : float = 0.0,
    ) -> dict:
        """
        Prédit si le retard sera élevé pour une liaison donnée.

        Returns:
            dict avec retard_eleve (bool), probabilite (float), features
        """
        trimestre = (mois - 1) // 3 + 1

        features = {
            "annee"                                         : annee,
            "mois"                                          : mois,
            "trimestre"                                     : trimestre,
            "taux_annulation"                               : taux_annulation,
            "Prct retard pour causes externes"              : prct_cause_externe,
            "Prct retard pour cause infrastructure"         : prct_cause_infra,
            "Prct retard pour cause gestion trafic"         : prct_cause_trafic,
            "Prct retard pour cause matériel roulant"       : prct_cause_materiel,
            "type_service_enc" : self._encode_safe(self.le_type,    type_service),
            "region_enc"       : self._encode_safe(self.le_region,  region),
            "liaison_enc"      : self._encode_safe(self.le_liaison, liaison),
        }

        X = pd.DataFrame([features])[self.feature_cols]
        proba         = float(self.model.predict_proba(X)[0][1])
        retard_eleve  = proba >= 0.35

        return {
            "retard_eleve"  : retard_eleve,
            "probabilite"   : round(proba, 4),
            "niveau_risque" : "🔴 ÉLEVÉ" if retard_eleve else "🟢 NORMAL",
            "type_service"  : type_service,
            "region"        : region,
            "liaison"       : liaison,
            "mois"          : mois,
            "annee"         : annee,
        }