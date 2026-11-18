"""Equal-data ablation: nearest-road vs classifier-only vs geometric HMM vs ML+HMM.

Held fixed across every variant:
  * candidate generation (k, radius, fallback radius from configs/tuned_v2.json)
  * decoder hyperparameters (tuned_v2: sigma scale, beta, backtrack tol, U-turn
    penalty, thinning) and ML emission weight (4.0, `ml` mode)
  * route assembly: per-point arcs -> collapse -> shortest-path gap filling
  * classifier training data: models/<name>__tune64.joblib (the same 64 tune trips);
    on the tune trips themselves, 5-fold out-of-fold probabilities are used.
Evaluation reads .route only after prediction (via src.experiment assets).
"""
from __future__ import annotations

import time

import numpy as np

from .config import get_config
from .decoder import NetworkDistance, _positions, assemble_route
from .evaluation import route_diagnostics, route_metrics, sequence_metrics
from .experiment import _trip_assets
from .features import candidate_features
from .matcher import generate_candidates
from .ml_emission import load_model_bundle
from .pipeline import predict_trip

MODELS = ("logreg", "decision_tree", "random_forest", "extra_trees", "hist_gb")
_FEAT: dict = {}
_OOF: dict = {}


def _cands_and_features(trip_id, cfg):
    if trip_id not in _FEAT:
        trip, net, xy, *_ = _trip_assets(trip_id)
        cands = generate_candidates(net, xy, k=cfg.k, radius_m=cfg.radius_m,
                                    fallback_radius_m=cfg.fallback_radius_m)
        X, _, _ = candidate_features(net, xy, trip.timestamps, cands, radius_m=cfg.radius_m)
        _FEAT[trip_id] = (cands, X)
    return _FEAT[trip_id]


def _probs(trip_id, model, cfg, oof_file):
    cands, X = _cands_and_features(trip_id, cfg)
    t0 = time.perf_counter()
    if oof_file:
        if oof_file not in _OOF:
            import joblib
            _OOF[oof_file] = joblib.load(oof_file)
        p = _OOF[oof_file][trip_id]
    else:
        p = load_model_bundle(f"{model}__tune64")["model"].predict_proba(X)[:, 1]
    ms = (time.perf_counter() - t0) * 1000
    return np.split(p, np.cumsum([len(c) for c in cands])[:-1]), ms


def run_variant(args) -> dict:
    """args = (trip_id, variant, model, oof_file). variant in
    {nearest_road, classifier_only, geometric_hmm, ml_hmm}."""
    trip_id, variant, model, oof_file = args
    cfg = get_config("tuned_v2")
    trip, net, xy, ref, _, al2 = _trip_assets(trip_id)
    cands, _ = _cands_and_features(trip_id, cfg)
    t0 = time.perf_counter()
    extra_ms = 0.0
    if variant in ("nearest_road", "classifier_only"):
        if variant == "nearest_road":
            choice = np.array([0 if len(c) else -1 for c in cands])
        else:
            probs, extra_ms = _probs(trip_id, model, cfg, oof_file)
            choice = np.array([int(np.argmax(p)) if len(p) else -1 for p in probs])
        arcs, _ = _positions(cands, choice)
        nd = NetworkDistance(net, cfg.dijkstra_limit_m)
        route, _, fill = assemble_route(arcs, net, nd, cfg.fill_max_gap_m, return_fill_mask=True)
        point_arcs = arcs
    else:
        probs = None
        c = cfg
        if variant == "ml_hmm":
            probs, extra_ms = _probs(trip_id, model, cfg, oof_file)
            c = cfg.with_(emission="ml", ml_weight=4.0)
        res, _, _ = predict_trip(trip, "hmm", config=c, network=(net, xy), ml_probs=probs)
        route, fill, point_arcs = res.route_arc_ids, res.route_is_fill, res.point_arc_ids
    ms = (time.perf_counter() - t0) * 1000 + (extra_ms if oof_file is None else 0.0)
    row = {"trip_id": trip_id, "variant": variant, "model": model or "-",
           "runtime_ms_excl_features": round(ms, 1)}
    m = route_metrics(route, ref, trip.arcs, net.arc_len)
    row.update({k: v for k, v in m.items() if k != "metric_type"})
    row.update(sequence_metrics(route, ref))
    d = route_diagnostics(route, fill, point_arcs, ref, trip.arcs, net.arc_len)
    row.update({k: d[k] for k in ("fp_reverse_direction", "fp_gap_fill_only", "fp_matched",
                                  "repeated_traversals", "n_fill_arcs")})
    v = al2.valid
    row["pseudo_top1_accuracy"] = round(float(np.mean(point_arcs[v] == al2.labels[v])), 4) \
        if v.any() else None
    return row
