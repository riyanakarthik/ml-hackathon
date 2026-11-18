"""Supervised candidate classifier: dataset construction, model zoo, metrics.

TRAINING ONLY (reads .route through src.labels / src.evaluation to build
targets). Features come from src.features, which is inference-safe.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import MatcherConfig
from .data_loader import load_trip
from .evaluation import load_reference_route
from .features import FEATURE_NAMES, candidate_features
from .labels import align_track_to_route, candidate_labels
from .matcher import generate_candidates
from .pipeline import build_network

SEED = 42


def trip_rows(args) -> dict:
    """Features + pseudo-labels for one trip (picklable for process pools)."""
    trip_id, cfg_dict = args
    cfg = MatcherConfig.from_dict(cfg_dict)
    trip = load_trip(trip_id)
    net, xy = build_network(trip)
    cands = generate_candidates(net, xy, k=cfg.k, radius_m=cfg.radius_m,
                                fallback_radius_m=cfg.fallback_radius_m)
    X, pidx, cidx = candidate_features(net, xy, trip.timestamps, cands, radius_m=cfg.radius_m)
    ref = load_reference_route(trip_id)              # labels only
    al = align_track_to_route(xy, trip.timestamps, ref, net)
    y, row_valid = candidate_labels(cands, al)
    label_in_cands = np.array([al.labels[t] in set(c.arc_ids.tolist()) for t, c in enumerate(cands)])
    return {"trip_id": trip_id, "X": X, "y": y, "row_valid": row_valid, "point_idx": pidx,
            "n_points": len(cands), "point_valid": al.valid, "label_in_cands": label_in_cands}


def build_dataset(trip_ids, cfg: MatcherConfig, n_jobs: int = 8) -> dict:
    with ProcessPoolExecutor(n_jobs) as ex:
        parts = list(ex.map(trip_rows, [(t, cfg.to_dict()) for t in trip_ids]))
    return {"trips": {p["trip_id"]: p for p in parts}, "feature_names": FEATURE_NAMES,
            "candidate_config": {k: cfg.to_dict()[k] for k in ("k", "radius_m", "fallback_radius_m")}}


def make_models(seed: int = SEED) -> dict:
    """Five distinct classifier families (initial, untuned configurations)."""
    return {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
        "decision_tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=20,
                                                max_features="sqrt", n_jobs=-1, random_state=seed),
        "extra_trees": ExtraTreesClassifier(n_estimators=150, max_depth=18, min_samples_leaf=20,
                                            max_features="sqrt", n_jobs=-1, random_state=seed),
        "hist_gb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                                  l2_regularization=1.0, early_stopping=False,
                                                  random_state=seed),
    }


def stack(ds: dict, trip_ids, valid_only: bool = True):
    Xs, ys, gs = [], [], []
    for t in trip_ids:
        p = ds["trips"][t]
        m = p["row_valid"] if valid_only else np.ones(len(p["y"]), bool)
        Xs.append(p["X"][m]); ys.append(p["y"][m]); gs.append(np.full(m.sum(), t))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)


def candidate_metrics(ds: dict, trip_ids, probs: dict[str, np.ndarray]) -> dict:
    """Classification metrics on valid-label rows + per-observation top-1
    against PSEUDO-labels (not route-level accuracy)."""
    y_all, p_all = [], []
    top1_hits, top1_n, top1_cov_hits, top1_cov_n = 0, 0, 0, 0
    for t in trip_ids:
        d = ds["trips"][t]
        p = probs[t]
        v = d["row_valid"]
        y_all.append(d["y"][v]); p_all.append(p[v])
        # per-observation top-1: highest-probability candidate at each point
        pid = d["point_idx"]
        if len(pid) == 0:
            continue
        order = np.lexsort((-p, pid))                    # by point, prob desc
        first = np.r_[True, pid[order][1:] != pid[order][:-1]]
        best_row = order[first]
        best_pt = pid[best_row]
        hit = d["y"][best_row] == 1
        valid_pt = d["point_valid"][best_pt]
        top1_hits += int(hit[valid_pt].sum()); top1_n += int(d["point_valid"].sum())
        cov = valid_pt & d["label_in_cands"][best_pt]
        top1_cov_hits += int(hit[cov].sum()); top1_cov_n += int((d["point_valid"] & d["label_in_cands"]).sum())
    y = np.concatenate(y_all); p = np.concatenate(p_all)
    yhat = (p >= 0.5).astype(int)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "precision": round(float(precision_score(y, yhat, zero_division=0)), 4),
        "recall": round(float(recall_score(y, yhat, zero_division=0)), 4),
        "f1": round(float(f1_score(y, yhat, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y, yhat, average="macro", zero_division=0)), 4),
        "log_loss": round(float(log_loss(y, pc, labels=[0, 1])), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pseudo_top1_accuracy": round(top1_hits / max(top1_n, 1), 4),
        "pseudo_top1_given_label_in_candidates": round(top1_cov_hits / max(top1_cov_n, 1), 4),
        "n_rows": int(len(y)), "positive_rate": round(float(y.mean()), 4),
    }


def fit_timed(model, X, y):
    t0 = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - t0


def predict_timed(model, X):
    t0 = time.perf_counter()
    p = model.predict_proba(X)[:, 1]
    return p, time.perf_counter() - t0
