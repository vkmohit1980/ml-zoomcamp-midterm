"""Serve a training-produced QuietCount model with FastAPI."""
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import ConfigDict, Field, create_model

from data_source import ROOT

bundle = joblib.load(ROOT / "artifacts" / "model.joblib")
model = bundle["model"]
features = bundle["features"]
SensorSnapshot = create_model(
    "SensorSnapshot",
    __config__=ConfigDict(extra="forbid"),
    **{name: (float, Field(strict=True, allow_inf_nan=False,
                           ge=0 if name.endswith(("_Light", "_CO2")) else None))
       for name in features},
)

app = FastAPI(title="QuietCount", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "model": bundle["id"], "features": features}


@app.post("/predict")
def predict(snapshot: SensorSnapshot):
    frame = pd.DataFrame([snapshot.model_dump()], columns=features)
    return {"occupancy_count": int(model.predict(frame)[0]), "model": bundle["id"]}
