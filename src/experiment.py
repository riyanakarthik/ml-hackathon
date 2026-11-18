"""Experiment harness: evaluate a MatcherConfig on a list of trips (parallel).

Evaluation-side code: reads .route via src.evaluation only AFTER prediction.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .config import MatcherConfig
from .data_loader import list_trips, load_trip
from .evaluation import (align_points_to_route, load_reference_route, observation_metrics,
                         route_diagnostics, route_metrics)
from .labels import align_track_to_route
from .pipeline import build_network, predict_trip

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "experiments"
PARTITION_FILE = EXPERIMENT_DIR / "train_partition.json"

_CACHE: dict = {}


def make_partition(seed: int = 42, n_holdout: int = 16, n_folds: int = 4) -> dict:
    """Fixed, seeded split of the 80 TRAIN trips (student_test never involved):
    `holdout` trips are never seen by the optimiser; `tune` trips are split
    into `folds` (trip-level groups) for fold-wise stability reporting."""
    ids = sorted(t["trip_id"] for t in list_trips() if t["split"] == "train")
    rng = np.random.default_rng(seed)
    perm = [ids[i] for i in rng.permutation(len(ids))]
    hold, tune = sorted(perm[:n_holdout]), perm[n_holdout:]
    folds = [sorted(tune[i::n_folds]) for i in range(n_folds)]
    return {"seed": seed, "holdout": hold, "tune": sorted(tune), "folds": folds}


def load_partition() -> dict:
    if PARTITION_FILE.exists():
        return json.loads(PARTITION_FILE.read_text())
    EXPERIMENT_DIR.mkdir(exist_ok=True)
    part = make_partition()
    PARTITION_FILE.write_text(json.dumps(part, indent=2) + "\n")
    return part


def _trip_assets(trip_id: str):
    if trip_id not in _CACHE:
        trip = load_trip(trip_id)
        net, track_xy = build_network(trip)
        ref = load_reference_route(trip_id)  # evaluation only
        labels, _ = align_points_to_route(track_xy, ref, net.nodes_xy, trip.arcs)
        al2 = align_track_to_route(track_xy, trip.timestamps, ref, net)
        _CACHE[trip_id] = (trip, net, track_xy, ref, labels, al2)
    return _CACHE[trip_id]


_PROBS: dict = {}


def _load_probs(probs_file: str, trip_id: str, cands) -> list[np.ndarray]:
    """Per-point OOF probabilities for a trip from a joblib {trip_id: (N,) array}."""
    if probs_file not in _PROBS:
        import joblib
        _PROBS.clear()
        _PROBS[probs_file] = joblib.load(probs_file)
    p = _PROBS[probs_file][trip_id]
    counts = [len(c) for c in cands]
    if len(p) != sum(counts):
        raise ValueError(f"{trip_id}: OOF probs ({len(p)}) do not match candidates ({sum(counts)})")
    return np.split(p, np.cumsum(counts)[:-1])


def evaluate_trip(trip_id: str, config: MatcherConfig, method: str = "hmm",
                  probs_file: str | None = None) -> dict:
    trip, net, track_xy, ref, labels, al2 = _trip_assets(trip_id)
    ml_probs = None
    if probs_file is not None:
        from .matcher import generate_candidates
        c0 = generate_candidates(net, track_xy, k=config.k, radius_m=config.radius_m,
                                 fallback_radius_m=config.fallback_radius_m)
        ml_probs = _load_probs(probs_file, trip_id, c0)
    t0 = time.perf_counter()
    res, _, _ = predict_trip(trip, method, config=config, network=(net, track_xy),
                             ml_probs=ml_probs)
    runtime_ms = (time.perf_counter() - t0) * 1000  # excludes cached network build
    cands = res.info["_candidates"]
    row = {"trip_id": trip_id, "runtime_ms": round(runtime_ms, 1)}
    m = route_metrics(res.route_arc_ids, ref, trip.arcs, net.arc_len)
    row.update({k: v for k, v in m.items() if k != "metric_type"})
    row.update(observation_metrics(res.point_arc_ids, labels, cands))
    row.update(route_diagnostics(res.route_arc_ids, res.route_is_fill, res.point_arc_ids,
                                 ref, trip.arcs, net.arc_len, cands))
    row["hmm_breaks"] = res.info.get("hmm_breaks")
    row["emission"] = res.info.get("emission")
    # pseudo-label v2 (src.labels, heading-aware), valid-label points only
    v = al2.valid
    row["pl2_valid_points"] = int(v.sum())
    row["pl2_top1_accuracy"] = round(float(np.mean(res.point_arc_ids[v] == al2.labels[v])), 4) \
        if v.any() else None
    return row


def _worker(args):
    trip_id, cfg_dict, method, probs_file = args
    return evaluate_trip(trip_id, MatcherConfig.from_dict(cfg_dict), method, probs_file)


_POOL: ProcessPoolExecutor | None = None


def get_pool(n_jobs: int = 8) -> ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=n_jobs)
    return _POOL


def evaluate_config(config: MatcherConfig, trip_ids, method: str = "hmm",
                    n_jobs: int = 8, probs_file: str | None = None) -> list[dict]:
    args = [(t, config.to_dict(), method, probs_file) for t in trip_ids]
    if n_jobs <= 1:
        return [_worker(a) for a in args]
    # chunks keep each trip on a stable-ish worker; cache is per process
    return list(get_pool(n_jobs).map(_worker, args))


def summarize(rows: list[dict], keys=("arc_f1", "arc_precision", "arc_recall",
                                      "length_precision", "length_recall",
                                      "route_continuity", "obs_top1_accuracy",
                                      "obs_macro_f1", "candidate_miss_rate",
                                      "pl2_top1_accuracy", "runtime_ms")) -> dict:
    out = {"n_trips": len(rows)}
    for k in keys:
        v = np.array([r[k] for r in rows if r.get(k) is not None], dtype=float)
        out[f"{k}_mean"] = round(float(v.mean()), 4)
        out[f"{k}_median"] = round(float(np.median(v)), 4)
    return out
