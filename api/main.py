"""
RailSafe — API FastAPI
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.model import RailSafeModel

# ── Modèle global ────────────────────────────────────────────────
model: RailSafeModel = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = RailSafeModel()
        print("✅ Modèle chargé")
    except Exception as e:
        print(f"❌ Erreur chargement modèle : {e}")
    yield

# ── App ──────────────────────────────────────────────────────────
app = FastAPI(
    title       = "RailSafe API",
    description = "Prédiction de retards ferroviaires SNCF",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── Schémas ──────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    type_service        : str   = Field("TGV", description="TGV, TER ou IC")
    region              : str   = Field("National", description="Région SNCF")
    liaison             : str   = Field("PARIS MONTPARNASSE -> BORDEAUX ST JEAN")
    mois                : int   = Field(7, ge=1, le=12)
    annee               : int   = Field(2024, ge=2013, le=2030)
    taux_annulation     : float = Field(0.02, ge=0.0, le=1.0)
    prct_cause_externe  : float = Field(20.0, ge=0.0, le=100.0)
    prct_cause_infra    : float = Field(22.0, ge=0.0, le=100.0)
    prct_cause_trafic   : float = Field(20.0, ge=0.0, le=100.0)
    prct_cause_materiel : float = Field(19.0, ge=0.0, le=100.0)

class PredictResponse(BaseModel):
    retard_eleve  : bool
    probabilite   : float
    niveau_risque : str
    type_service  : str
    region        : str
    liaison       : str
    mois          : int
    annee         : int

# ── Endpoints ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name"      : "RailSafe API",
        "version"   : "1.0.0",
        "model"     : model is not None,
        "metrics"   : model.metrics if model else None,
        "endpoints" : [
            "POST /predict",
            "GET  /health",
            "GET  /liaisons",
            "GET  /regions",
        ]
    }

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {
        "status"  : "ok",
        "model"   : "XGBoost",
        "roc_auc" : model.metrics["roc_auc"],
        "f1"      : model.metrics["f1"],
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    try:
        result = model.predict(
            type_service        = request.type_service,
            region              = request.region,
            liaison             = request.liaison,
            mois                = request.mois,
            annee               = request.annee,
            taux_annulation     = request.taux_annulation,
            prct_cause_externe  = request.prct_cause_externe,
            prct_cause_infra    = request.prct_cause_infra,
            prct_cause_trafic   = request.prct_cause_trafic,
            prct_cause_materiel = request.prct_cause_materiel,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/liaisons")
def get_liaisons():
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {
        "liaisons": list(model.le_liaison.classes_),
        "count"   : len(model.le_liaison.classes_)
    }

@app.get("/regions")
def get_regions():
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {
        "regions": list(model.le_region.classes_),
        "count"  : len(model.le_region.classes_)
    }