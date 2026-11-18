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


# ----------------------------------------------------------------------------
# Derived per-point reference labels (evaluation only)
# ----------------------------------------------------------------------------

def align_points_to_route(track_xy: np.ndarray, ref_route, nodes_xy: np.ndarray,
                          arcs: np.ndarray, sigma: float = 10.0):
    """Derive a reference arc for every GPS point by a monotone alignment of
    the track to the reference route (the vehicle moves forward along .route).

    Dynamic programme: state = position i in ref_route; score = sum of
    Gaussian log-likelihoods of point-to-arc distance; i may stay or advance.
    These are DERIVED labels (the dataset has no per-point labels) and inherit
    GPS noise; use them only for evaluation / training targets.

    Returns (labels (T,) arc ids, distance_m (T,) point-to-label-arc distance).
    """
    from .matcher import point_segment_projection
    ref = np.asarray(ref_route, dtype=np.int64)
    a = nodes_xy[arcs[ref, 0]][None, :, :]
    b = nodes_xy[arcs[ref, 1]][None, :, :]
    _, dist, _ = point_segment_projection(track_xy[:, None, :], a, b)  # (T, R)
    em = -0.5 * (dist / sigma) ** 2
    T, R = em.shape
    delta = em[0].copy()
    bp = np.zeros((T, R), dtype=np.int32)
    idx = np.arange(R)
    for t in range(1, T):
        # best predecessor for position i is argmax(delta[:i+1]) (monotone).
        cm = np.maximum.accumulate(delta)
        bp[t] = np.maximum.accumulate(np.where(delta >= cm, idx, 0))
        delta = em[t] + cm
    pos = np.empty(T, dtype=np.int64)
    pos[-1] = int(np.argmax(delta))
    for t in range(T - 1, 0, -1):
        pos[t - 1] = bp[t, pos[t]]
    return ref[pos], dist[np.arange(T), pos]


def observation_metrics(point_arc_ids, ref_labels, candidates=None) -> dict:
    """Per-GPS-point metrics against DERIVED labels (see align_points_to_route)."""
    pred = np.asarray(point_arc_ids)
    lab = np.asarray(ref_labels)
    out = {"obs_top1_accuracy": round(float(np.mean(pred == lab)), 4)}
    classes = np.union1d(np.unique(lab), np.unique(pred[pred >= 0]))
    f1s = []
    for c in classes:
        tp = np.sum((pred == c) & (lab == c))
        fp = np.sum((pred == c) & (lab != c))
        fn = np.sum((pred != c) & (lab == c))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    out["obs_macro_f1"] = round(float(np.mean(f1s)), 4)
    if candidates is not None:
        miss = [lab[t] not in set(candidates[t].arc_ids.tolist()) for t in range(len(lab))]
        out["candidate_miss_rate"] = round(float(np.mean(miss)), 4)
    return out


# ----------------------------------------------------------------------------
# Error diagnostics
# ----------------------------------------------------------------------------

def route_diagnostics(pred_route, is_fill, point_arc_ids, ref_route, arcs: np.ndarray,
                      arc_len: np.ndarray, candidates=None) -> dict:
    """Break down route-level errors. Each false-positive arc (distinct predicted
    arc not in .route) gets ONE category, checked in this order:

    * fp_reverse_direction : its reverse twin is in .route (right street, wrong way)
    * fp_outside_ref_span  : lies before the first / after the last correctly
                             predicted arc (start/end overshoot)
    * fp_gap_fill_only     : only inserted by shortest-path gap filling, never
                             directly matched to a GPS point
    * fp_matched           : directly matched to >= 1 GPS point (candidate /
                             decoding error)

    Repeated traversals (an arc occurring more than once in the predicted
    sequence) are counted separately; the set-based metric does NOT penalise them.
    False negatives are split into arcs that never appeared among any point's
    candidates (candidate generation miss) vs. available but not chosen.
    """
    pred_route = [int(a) for a in pred_route]
    ref = [int(a) for a in ref_route]
    ref_set = set(ref)
    pred_set = set(pred_route)
    matched = set(int(a) for a in point_arc_ids if a >= 0)
    rev_of = {(int(u), int(v)) for u, v in arcs[ref]}
    hit_pos = [i for i, a in enumerate(pred_route) if a in ref_set]
    lo, hi = (hit_pos[0], hit_pos[-1]) if hit_pos else (len(pred_route), -1)
    first_pos = {}
    for i, a in enumerate(pred_route):
        first_pos.setdefault(a, i)
    filled_only = {a for a in pred_set if a not in matched}

    cats = {"fp_reverse_direction": [], "fp_outside_ref_span": [],
            "fp_gap_fill_only": [], "fp_matched": []}
    for a in pred_set - ref_set:
        u, v = int(arcs[a, 0]), int(arcs[a, 1])
        if (v, u) in rev_of:
            cats["fp_reverse_direction"].append(a)
        elif first_pos[a] < lo or first_pos[a] > hi:
            cats["fp_outside_ref_span"].append(a)
        elif a in filled_only:
            cats["fp_gap_fill_only"].append(a)
        else:
            cats["fp_matched"].append(a)
    out = {k: len(v) for k, v in cats.items()}
    out.update({k + "_len_m": round(float(arc_len[v].sum()), 1) if v else 0.0
                for k, v in cats.items()})
    out["n_fp"] = len(pred_set - ref_set)
    out["repeated_traversals"] = len(pred_route) - len(pred_set)
    out["n_fill_arcs"] = int(sum(is_fill)) if is_fill else 0
    fn = ref_set - pred_set
    out["n_fn"] = len(fn)
    if candidates is not None:
        cand_arcs = set()
        for c in candidates:
            cand_arcs.update(c.arc_ids.tolist())
        out["fn_never_candidate"] = len(fn - cand_arcs)
        out["fn_candidate_not_chosen"] = len(fn & cand_arcs)
    return out


# ----------------------------------------------------------------------------
# Order-aware supplementary metric
# ----------------------------------------------------------------------------

def lcs_length(a, b) -> int:
    """Longest common subsequence length (O(len(a) * len(b)), small routes)."""
    a = [int(x) for x in a]
    b = [int(x) for x in b]
    if not a or not b:
        return 0
    prev = np.zeros(len(b) + 1, dtype=np.int32)
    bb = np.asarray(b)
    for x in a:
        match = np.r_[0, (bb == x)].astype(np.int32)
        cur = np.zeros_like(prev)
        # cur[j] = max(prev[j], cur[j-1], prev[j-1] + match[j])
        diag = prev[:-1] + match[1:]
        for j in range(1, len(prev)):
            cur[j] = max(prev[j], cur[j - 1], diag[j - 1])
        prev = cur
    return int(prev[-1])


def sequence_metrics(pred_route, ref_route) -> dict:
    """ORDER- and DUPLICATE-aware route agreement (supplementary to the
    set-based arc metrics): LCS between the predicted arc sequence (as
    returned, consecutive duplicates already collapsed by route assembly) and
    the reference sequence.

      seq_precision = LCS / len(pred)   seq_recall = LCS / len(ref)

    Unlike arc_f1, an arc predicted twice when driven once costs precision, and
    arcs predicted in the wrong order are not matched. Direction is respected
    because arc ids are directed."""
    pred = [int(a) for a in pred_route]
    ref = [int(a) for a in ref_route]
    lcs = lcs_length(pred, ref)
    p = lcs / len(pred) if pred else 0.0
    r = lcs / len(ref) if ref else 0.0
    return {"seq_precision": round(p, 4), "seq_recall": round(r, 4),
            "seq_f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}
