"""PathSnap FastAPI backend.

Run:  uvicorn main:app --host 0.0.0.0 --port 8000
All map coordinates in responses are [latitude, longitude].
"""
from __future__ import annotations

import logging
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import available_configs, get_config
from src.data_loader import TripDataError, get_data_dir, list_trips, load_trip
from src.evaluation import load_reference_route, route_metrics
from src.pipeline import predict_trip, to_latlon

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# Default decoder configuration: "ml_v3" = HMM with Random-Forest emissions,
# validated on held-out train trips. Rollback / alternatives (also selectable
# per request via {"config": ...}): "tuned_v2" (geometric emissions),
# "baseline_v1" (original hand-set HMM). If the ML model cannot be loaded the
# pipeline falls back to geometric emissions and reports it in details.
DEFAULT_CONFIG = "ml_v3"


@app.on_event("startup")
def _warm_model() -> None:
    """Load the default classifier once so the first request is not slow."""
    try:
        cfg = get_config(DEFAULT_CONFIG)
        if cfg.emission != "geometric" and cfg.ml_model:
            from src.ml_emission import load_model_bundle
            b = load_model_bundle(cfg.ml_model)
            log.info("Loaded classifier %s (%s, trained on %d trips)", cfg.ml_model,
                     b.get("model_name"), len(b.get("trained_on_trips", [])))
    except Exception as e:  # noqa: BLE001
        log.warning("Could not preload model for %s: %s (geometric fallback will be used)",
                    DEFAULT_CONFIG, e)


class PredictRequest(BaseModel):
    trip_id: str | int = Field(..., examples=["00000038"])
    method: Literal["hmm", "baseline"] = "hmm"
    config: str | None = Field(None, description="HMM config name; default DEFAULT_CONFIG")


@app.get("/health")
def health():
    return {"status": "ok", "data_dir": str(get_data_dir()),
            "default_config": DEFAULT_CONFIG, "configs": available_configs()}


@app.get("/trips")
def trips():
    items = list_trips()
    return {"count": len(items), "trips": items}


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    try:
        trip = load_trip(req.trip_id)
    except TripDataError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        cfg = get_config(req.config or DEFAULT_CONFIG)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Prediction: track + nodes + arcs only.
    result, net, timings = predict_trip(trip, method=req.method, config=cfg)
    runtime_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Evaluation happens strictly after prediction, in isolated code.
    ref = load_reference_route(trip.trip_id)
    metrics = (route_metrics(result.route_arc_ids, ref, trip.arcs, net.arc_len)
               if ref is not None else None)

    log.info("predict trip=%s split=%s method=%s config=%s points=%d route_arcs=%d runtime_ms=%.1f metrics=%s",
             trip.trip_id, trip.split, result.method, cfg.name, len(trip.track),
             len(result.route_arc_ids), runtime_ms,
             None if metrics is None else {k: metrics[k] for k in ("arc_precision", "arc_recall", "arc_f1")})

    return {
        "trip_id": trip.trip_id,
        "gps_coordinates": [[float(lat), float(lon)] for lon, lat in trip.track_lonlat],
        "predicted_coordinates": to_latlon(net, result.point_xy),
        "predicted_arc_ids": [int(a) for a in result.route_arc_ids],
        "point_arc_ids": [int(a) for a in result.point_arc_ids],
        "metrics": metrics,
        "method": result.method,
        "runtime_ms": runtime_ms,
        "details": {"split": trip.split, "timings_ms": timings,
                    **{k: v for k, v in result.info.items() if not k.startswith("_")}},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
