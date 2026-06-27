"""
app.py
======
FastAPI service that loads a trained XGBoost pipeline (model.pkl) and
serves Extrovert/Introvert predictions over a public POST endpoint.

Architecture:
    Client -> POST /predict -> FastAPI -> loads model.pkl -> predicts -> JSON

The pipeline saved in model.pkl already contains preprocessing
(StandardScaler + OneHotEncoder), so this service only needs to:
  1. Validate the incoming JSON body.
  2. Re-create the engineered features (same formulas as the training
     notebook).
  3. Feed the row into the pipeline's .predict() / .predict_proba().
  4. Return the result as JSON.
"""

from contextlib import asynccontextmanager
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "best_xgboost.pkl"
LABEL_MAP = {0: "Extrovert", 1: "Introvert"}

# Column order the pipeline's ColumnTransformer was fit on.
MODEL_INPUT_COLUMNS = [
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency",
    "social_activity_score",
    "isolation_score",
    "social_fatigue_index",
    "social_confidence",
    "online_vs_offline_ratio",
    "behavior_variance",
    "alone_vs_friends",
    "social_outing_density",
]

ml_model = {}  # populated at startup, avoids reloading the pickle per-request


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup, kept warm in memory for every request.
    ml_model["pipeline"] = joblib.load(MODEL_PATH)
    yield
    ml_model.clear()


app = FastAPI(
    title="Personality Prediction API",
    description="Predicts Extrovert/Introvert personality from behavioral survey inputs.",
    version="1.0.0",
    lifespan=lifespan,
)


class PersonalityInput(BaseModel):
    """Raw survey inputs, matching the original dataset's columns."""

    Time_spent_Alone: float = Field(..., ge=0, description="Hours spent alone per day")
    Stage_fear: Literal["Yes", "No"]
    Social_event_attendance: float = Field(..., ge=0, description="Events attended per month")
    Going_outside: float = Field(..., ge=0, description="Days going outside per week")
    Drained_after_socializing: Literal["Yes", "No"]
    Friends_circle_size: float = Field(..., ge=0)
    Post_frequency: float = Field(..., ge=0, description="Social media posts per week")

    class Config:
        json_schema_extra = {
            "example": {
                "Time_spent_Alone": 4.0,
                "Stage_fear": "No",
                "Social_event_attendance": 4.0,
                "Going_outside": 6.0,
                "Drained_after_socializing": "No",
                "Friends_circle_size": 13.0,
                "Post_frequency": 5.0,
            }
        }


class PersonalityOutput(BaseModel):
    prediction: Literal["Extrovert", "Introvert"]
    probability_extrovert: float
    probability_introvert: float


def _engineer_features(payload: PersonalityInput) -> dict:
    """Re-create the same engineered features used during training."""
    row = payload.model_dump()
    yes_no_map = {"Yes": 1, "No": 0}

    row["social_activity_score"] = (
        row["Social_event_attendance"] + row["Going_outside"] + row["Post_frequency"]
    )
    row["isolation_score"] = row["Time_spent_Alone"] - row["Social_event_attendance"]
    row["social_fatigue_index"] = (
        yes_no_map[row["Drained_after_socializing"]] * yes_no_map[row["Stage_fear"]]
    )
    row["social_confidence"] = row["Friends_circle_size"] * row["Social_event_attendance"]
    row["online_vs_offline_ratio"] = row["Post_frequency"] / (row["Going_outside"] + 1)

    behavior_cols = [
        "Time_spent_Alone",
        "Social_event_attendance",
        "Going_outside",
        "Friends_circle_size",
        "Post_frequency",
    ]
    behavior_values = [row[c] for c in behavior_cols]
    row["behavior_variance"] = float(np.var(behavior_values, ddof=1))

    row["alone_vs_friends"] = row["Time_spent_Alone"] / (row["Friends_circle_size"] + 1)
    row["social_outing_density"] = row["Social_event_attendance"] * row["Going_outside"]

    return row


@app.get("/")
def root():
    return {"status": "ok", "message": "Personality Prediction API is running."}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": "pipeline" in ml_model}


@app.post("/predict", response_model=PersonalityOutput)
def predict(payload: PersonalityInput):
    pipeline = ml_model.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        engineered_row = _engineer_features(payload)
        X_new = pd.DataFrame([engineered_row], columns=MODEL_INPUT_COLUMNS)

        pred = pipeline.predict(X_new)[0]
        proba = pipeline.predict_proba(X_new)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return PersonalityOutput(
        prediction=LABEL_MAP[int(pred)],
        probability_extrovert=float(proba[0]),
        probability_introvert=float(proba[1]),
    )
