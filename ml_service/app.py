"""FastAPI service that exposes the trained KNN model as an HTTP API."""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional
import sys
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure custom transformers are registered before loading the joblib bundle
from custom_transformers import SecondaryChoiceImputer  # noqa: F401

# Algunos notebooks serializan la clase apuntando a "__main__", así que la registramos.
sys.modules.setdefault("__main__", sys.modules[__name__])
setattr(sys.modules["__main__"], "SecondaryChoiceImputer", SecondaryChoiceImputer)

DEFAULT_ORIGINS = "http://localhost:4173,https://voterintentions.vercel.app"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",")
    if origin.strip()
]

MODEL_BUNDLE_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "knn_voter_intentions.joblib"

try:
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    PIPELINE = bundle["model"]
    LABEL_ENCODER = bundle["label_encoder"]
except FileNotFoundError as exc:
    raise RuntimeError("Model artifact not found. Train the model before starting the service.") from exc


class VoterFeatures(BaseModel):
    age: float = Field(..., ge=18, le=120)
    gender: float = Field(..., ge=0, le=1)
    education: float = Field(..., ge=0)
    employment_status: float
    employment_sector: float
    income_bracket: float
    marital_status: float
    household_size: float
    has_children: float
    urbanicity: float
    region: float
    voted_last: float
    party_id_strength: float
    union_member: float
    public_sector: float
    home_owner: float
    small_biz_owner: float
    owns_car: float
    wa_groups: float
    refused_count: float
    attention_check: float
    will_turnout: float
    undecided: float
    preference_strength: float
    survey_confidence: float
    tv_news_hours: float
    social_media_hours: float
    trust_media: float
    civic_participation: float
    job_tenure_years: float
    primary_choice: Literal[
        "CAND_Azon",
        "CAND_Boreal",
        "CAND_Civico",
        "CAND_Demetra",
        "CAND_Electra",
        "CAND_Frontera",
        "CAND_Gaia",
        "CAND_Halley",
        "CAND_Icaro",
        "CAND_Jade",
    ]
    secondary_choice: Literal[
        "CAND_Azon",
        "CAND_Boreal",
        "CAND_Civico",
        "CAND_Demetra",
        "CAND_Electra",
        "CAND_Frontera",
        "CAND_Gaia",
        "CAND_Halley",
        "CAND_Icaro",
        "CAND_Jade",
        "Unknown",
    ]


class CandidateProbability(BaseModel):
    candidate: str
    probability: float


class PredictionResponse(BaseModel):
    intended_vote: str
    confidence: float
    runner_up: Optional[CandidateProbability]
    top_candidates: List[CandidateProbability]
    confidence_note: str


app = FastAPI(
    title="Voter Intention KNN API",
    description="Servicio de inferencia para el modelo KNN entrenado con 3,000 votantes.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_intention(features: VoterFeatures):
    try:
        payload = pd.DataFrame([features.model_dump()])
        probabilities = PIPELINE.predict_proba(payload)[0]
        classes = LABEL_ENCODER.classes_
        ordering = np.argsort(probabilities)[::-1]
        top_candidates = [
            CandidateProbability(candidate=classes[idx], probability=float(probabilities[idx]))
            for idx in ordering[:5]
        ]
        best_idx = ordering[0]
        prediction = classes[best_idx]
        runner_up = top_candidates[1] if len(top_candidates) > 1 else None
    except Exception as exc:  # pragma: no cover - FastAPI surfaces the error
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    msg = "Recuerda complementar este resultado con trabajo de campo y encuestas cualitativas."
    return PredictionResponse(
        intended_vote=prediction,
        confidence=float(probabilities[best_idx]),
        runner_up=runner_up,
        top_candidates=top_candidates,
        confidence_note=msg,
    )
