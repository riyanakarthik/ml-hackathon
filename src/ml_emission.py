"""Classifier-based HMM emissions (inference side).

A trained candidate classifier gives p_tj = P(candidate j is the true road of
GPS point t | features). It is turned into an HMM emission score as follows:

  1. clip:      p = clip(p, prob_floor, 1)      (no log(0) = -inf; floor 1e-4)
  2. normalise: q_tj = p_tj / sum_j p_tj         per point, so scores of one
                point form a distribution over its candidates
  3. log:       e_tj = log q_tj
  4. combine:   mode "ml":     emission = ml_weight * e_tj
                mode "hybrid": emission = geometric Gaussian log-lik + ml_weight * e_tj

Note: Viterbi's argmax path is unchanged by adding a constant per point, so
step 2 does not change the decoded path relative to log(p); it is kept so
emissions are interpretable and comparable across points. ml_weight sets the
balance against the transition scores (tuned for geometric emissions).

Inference uses only features from src.features (no reference-route data).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from .features import FEATURE_NAMES, SCHEMA_VERSION, candidate_features
from .matcher import Candidates, RoadNetwork

log = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class ModelSchemaError(RuntimeError):
    pass


@lru_cache(maxsize=8)
def load_model_bundle(name_or_path: str) -> dict:
    """Load a saved bundle {model, feature_names, schema_version, ...}."""
    p = Path(name_or_path)
    if not p.suffix:
        p = MODEL_DIR / f"{name_or_path}.joblib"
    bundle = joblib.load(p)
    if bundle.get("schema_version") != SCHEMA_VERSION or \
            list(bundle.get("feature_names", [])) != FEATURE_NAMES:
        raise ModelSchemaError(f"{p.name}: feature schema mismatch with src.features")
    return bundle


def predict_candidate_probs(bundle: dict, net: RoadNetwork, track_xy: np.ndarray,
                            times: np.ndarray, cands: Sequence[Candidates],
                            radius_m: float) -> list[np.ndarray]:
    """Per-point arrays of P(correct) aligned with each point's candidates."""
    X, _, _ = candidate_features(net, track_xy, times, cands, radius_m=radius_m)
    counts = [len(c) for c in cands]
    if X.shape[0] == 0:
        return [np.empty(0) for _ in cands]
    p = bundle["model"].predict_proba(X)[:, 1]
    return np.split(p, np.cumsum(counts)[:-1])


def combine_emissions(probs: Sequence[np.ndarray], geometric: Sequence[np.ndarray],
                      mode: str, weight: float, prob_floor: float = 1e-4) -> list[np.ndarray]:
    if mode not in ("ml", "hybrid"):
        raise ValueError(f"unknown emission mode {mode!r}")
    out = []
    for p, g in zip(probs, geometric):
        if len(p) == 0:
            out.append(np.empty(0))
            continue
        q = np.clip(np.asarray(p, dtype=float), prob_floor, 1.0)
        e = weight * np.log(q / q.sum())
        out.append(e if mode == "ml" else np.asarray(g) + e)
    return out
