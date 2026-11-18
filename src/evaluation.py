"""Reference-route evaluation. THE ONLY MODULE THAT READS ``.route`` FILES.

What these metrics measure
--------------------------
``.route`` is a chronological list of arc ids the vehicle drove. It is NOT
aligned to GPS points, so per-observation accuracy ("was point i matched to
the right arc?") cannot be computed from it. We therefore report route-level
metrics only:

* arc_precision  = |pred ∩ ref| / |pred|   (set of distinct arc ids)
* arc_recall     = |pred ∩ ref| / |ref|
* arc_f1
* length_precision / length_recall: same, weighted by arc length in metres
  (robust to many tiny arcs dominating the counts)
* route_continuity = fraction of consecutive predicted arcs (a, b) with
  to_node(a) == from_node(b) (1.0 = a fully connected path). This needs no
  reference; reference continuity is reported for comparison.

Directed arcs count as different arcs, so driving the right street in the
wrong direction is counted as wrong.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .data_loader import find_trip_dir


def load_reference_route(trip_id, data_dir: str | os.PathLike | None = None) -> np.ndarray | None:
    """Reference arc ids for a trip, or None if the trip has no .route file."""
    trip_dir, _ = find_trip_dir(trip_id, data_dir)
    path = Path(trip_dir) / f"{trip_dir.name}.route"
    if not path.exists():
        return None
    return np.loadtxt(path, dtype=np.int64, ndmin=1)


def route_continuity(route, arcs: np.ndarray) -> float | None:
    route = np.asarray(route, dtype=np.int64)
    if len(route) < 2:
        return None
    return float(np.mean(arcs[route[:-1], 1] == arcs[route[1:], 0]))


def route_metrics(pred_route, ref_route, arcs: np.ndarray,
                  arc_len: np.ndarray | None = None) -> dict:
    pred = set(int(a) for a in pred_route)
    ref = set(int(a) for a in ref_route)
    inter = pred & ref
    p = len(inter) / len(pred) if pred else 0.0
    r = len(inter) / len(ref) if ref else 0.0
    out = {
        "metric_type": "reference_route_arc_membership (not per-GPS-point accuracy)",
        "arc_precision": round(p, 4),
        "arc_recall": round(r, 4),
        "arc_f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
        "n_pred_arcs": len(pred),
        "n_ref_arcs": len(ref),
        "n_common_arcs": len(inter),
        "route_continuity": route_continuity(pred_route, arcs),
        "reference_continuity": route_continuity(ref_route, arcs),
    }
    if arc_len is not None:
        L = lambda s: float(sum(arc_len[a] for a in s))
        lp, lr = L(pred), L(ref)
        out["length_precision"] = round(L(inter) / lp, 4) if lp else 0.0
        out["length_recall"] = round(L(inter) / lr, 4) if lr else 0.0
    return out
