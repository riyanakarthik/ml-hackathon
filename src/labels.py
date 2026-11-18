"""Pseudo-labels: per-GPS-point reference arcs inferred from .route.

TRAINING / EVALUATION ONLY. ``.route`` is an ordered list of traversed arcs,
not one label per GPS point. We infer a label for each point with a monotone
alignment (dynamic programme) of the track to the route positions:

  state     i = position in the reference route (so repeated arcs and
              A -> reverse(A) direction changes are distinct states)
  emission  -0.5 (d / sigma)^2                       distance to arc i
            + kappa (cos(heading - bearing_i) - 1)    only when moving
  transition stay on i, or advance i -> j > i paying lambda * (length of the
            arcs strictly between i and j)  (discourages skipping long arcs
            that should have received GPS points)

Labels further than ``max_label_dist_m`` from their arc are marked invalid
(e.g. GPS recorded before the vehicle entered the routed network); invalid
points are excluded from training and from pseudo-label accuracy.

These are DERIVED labels and inherit GPS noise; they are not ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import MIN_HEADING_DISP_M, gps_kinematics
from .matcher import Candidates, RoadNetwork, point_segment_projection


@dataclass
class Alignment:
    labels: np.ndarray       # (T,) arc id per point
    position: np.ndarray     # (T,) index into the reference route (monotone)
    distance_m: np.ndarray   # (T,) point-to-label-arc distance
    cos_heading: np.ndarray  # (T,) cos(heading - label bearing); nan when not moving
    valid: np.ndarray        # (T,) bool: label usable


def align_track_to_route(track_xy: np.ndarray, times: np.ndarray, ref_route, net: RoadNetwork,
                         sigma: float = 10.0, heading_kappa: float = 2.0,
                         min_speed_mps: float = 2.0, skip_penalty_per_m: float = 0.01,
                         max_label_dist_m: float = 30.0) -> Alignment:
    ref = np.asarray(ref_route, dtype=np.int64)
    R, T = len(ref), len(track_xy)
    a = net.nodes_xy[net.arcs[ref, 0]]
    b = net.nodes_xy[net.arcs[ref, 1]]
    _, dist, _ = point_segment_projection(track_xy[:, None, :], a[None], b[None])  # (T, R)

    heading, speed, hvalid, _, _ = gps_kinematics(track_xy, times)
    moving = hvalid & (speed >= min_speed_mps)
    bearing = np.arctan2((b - a)[:, 1], (b - a)[:, 0])
    cosd = np.cos(heading[:, None] - bearing[None, :])  # (T, R)
    em = -0.5 * (dist / sigma) ** 2 + np.where(moving[:, None], heading_kappa * (cosd - 1.0), 0.0)

    # cumulative length: L[i] = total length of ref[:i]
    L = np.r_[0.0, np.cumsum(net.arc_len[ref])]
    lam = skip_penalty_per_m
    idx = np.arange(R)
    delta = em[0].copy()
    bp = np.zeros((T, R), dtype=np.int32)
    for t in range(1, T):
        # advance from i < j: delta[i] + lam * L[i+1] - lam * L[j]
        g = delta + lam * L[1:]                     # g[i]
        gmax = np.maximum.accumulate(g)
        garg = np.maximum.accumulate(np.where(g >= gmax, idx, 0))
        adv = np.full(R, -np.inf)
        adv_arg = np.zeros(R, dtype=np.int32)
        adv[1:] = gmax[:-1] - lam * L[1:-1]          # j >= 1: best over i <= j-1
        adv_arg[1:] = garg[:-1]
        stay_better = delta >= adv
        bp[t] = np.where(stay_better, idx, adv_arg)
        delta = em[t] + np.where(stay_better, delta, adv)
    pos = np.empty(T, dtype=np.int64)
    pos[-1] = int(np.argmax(delta))
    for t in range(T - 1, 0, -1):
        pos[t - 1] = bp[t, pos[t]]

    d = dist[np.arange(T), pos]
    ch = np.where(moving, cosd[np.arange(T), pos], np.nan)
    return Alignment(ref[pos], pos, d, ch, d <= max_label_dist_m)


def alignment_quality(al: Alignment, ref_route, net: RoadNetwork,
                      long_arc_m: float = 50.0) -> dict:
    """Computational checks of an alignment."""
    ref = np.asarray(ref_route)
    R = len(ref)
    visited = np.zeros(R, bool)
    visited[np.unique(al.position)] = True
    lens = net.arc_len[ref]
    ch = al.cos_heading[~np.isnan(al.cos_heading) & al.valid]
    rep_arcs = [a for a in set(ref.tolist()) if (ref == a).sum() > 1]
    rep_visits_with_points = 0
    rep_visits = 0
    for a in rep_arcs:
        for p in np.flatnonzero(ref == a):
            rep_visits += 1
            rep_visits_with_points += int(visited[p])
    return {
        "n_points": len(al.labels),
        "label_valid_frac": round(float(al.valid.mean()), 4),
        "dist_median_m": round(float(np.median(al.distance_m)), 2),
        "dist_p95_m": round(float(np.percentile(al.distance_m, 95)), 2),
        "heading_agree_frac": round(float((ch > 0).mean()), 4) if ch.size else None,
        "heading_strong_disagree_frac": round(float((ch < -0.5).mean()), 4) if ch.size else None,
        "monotone": bool(np.all(np.diff(al.position) >= 0)),
        "ref_positions_with_points": round(float(visited.mean()), 4),
        "ref_length_with_points": round(float(lens[visited].sum() / max(lens.sum(), 1e-9)), 4),
        "long_ref_arcs_without_points": int(((lens >= long_arc_m) & ~visited).sum()),
        "repeated_ref_arcs": len(rep_arcs),
        "repeated_visits_with_points": f"{rep_visits_with_points}/{rep_visits}",
        "first_pos": int(al.position[0]), "last_pos": int(al.position[-1]), "n_ref": R,
    }


def candidate_labels(cands: list[Candidates], al: Alignment) -> tuple[np.ndarray, np.ndarray]:
    """y (N,) = 1 if candidate arc equals the point's pseudo-label; and
    row_valid (N,) = the point's label is valid."""
    y = np.concatenate([(c.arc_ids == al.labels[t]).astype(np.int8) for t, c in enumerate(cands)])
    v = np.concatenate([np.full(len(c), al.valid[t]) for t, c in enumerate(cands)])
    return y, v
