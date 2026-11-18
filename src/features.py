"""Candidate features for the supervised road classifier.

INFERENCE-SAFE: inputs are only the GPS track (positions + timestamps), the
road network geometry and the generated candidates. This module must never
import src.evaluation / src.labels or read .route (enforced by a test).

One row per (GPS point t, candidate j); rows are ordered by t, then j, which
is exactly the order of ``Candidates`` lists, so per-point probability arrays
can be recovered with ``np.split(p, np.cumsum(counts)[:-1])``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .matcher import Candidates, RoadNetwork

SCHEMA_VERSION = 1

FEATURE_NAMES = [
    "dist_m",               # GPS point -> candidate arc (exact point-to-segment)
    "dist_rank",            # rank of this candidate by distance at the point (0 = nearest)
    "dist_minus_best_m",    # dist - nearest candidate's dist
    "dist_z",               # dist / per-trip GPS noise estimate (MAD of nearest-road distance)
    "n_candidates",         # number of candidates at the point
    "arc_len_m",            # candidate arc length
    "offset_frac",          # projection position along the arc in [0, 1]
    "dist_to_arc_end_m",    # remaining length to the arc's end node
    "dist_to_arc_start_m",  # length from the arc's start node
    "heading_valid",        # 1 if the vehicle moved enough to define a heading
    "cos_heading_diff",     # cos(GPS heading - directed arc bearing); 0 if heading invalid
    "abs_heading_diff_deg", # |heading - bearing| in [0, 180]; 90 if heading invalid
    "speed_mps",            # centred-difference speed
    "dt_prev_s",            # time gap to previous GPS point
    "dt_next_s",            # time gap to next GPS point
    "has_reverse_twin",     # the reverse-direction arc is also a candidate here
    "end_out_degree",       # arcs leaving the candidate's end node
    "start_in_degree",      # arcs entering the candidate's start node
    "window_support",       # fraction of points within +-W that also have this arc as candidate
    "window_mean_dist_m",   # mean distance to this arc over the window (radius where absent)
]

MIN_HEADING_DISP_M = 3.0


def gps_kinematics(track_xy: np.ndarray, times: np.ndarray):
    """Centred-difference heading (radians, math convention), speed, validity
    and time gaps for each GPS point."""
    T = len(track_xy)
    prev = np.r_[0, np.arange(T - 1)]
    nxt = np.r_[np.arange(1, T), T - 1]
    disp = track_xy[nxt] - track_xy[prev]
    dist = np.linalg.norm(disp, axis=1)
    dt = np.maximum(times[nxt] - times[prev], 1e-6)
    heading = np.arctan2(disp[:, 1], disp[:, 0])
    valid = dist >= MIN_HEADING_DISP_M
    speed = np.where(nxt != prev, dist / dt, 0.0)
    dt_prev = np.r_[0.0, np.diff(times)]
    dt_next = np.r_[np.diff(times), 0.0]
    return heading, speed, valid, dt_prev, dt_next


def candidate_features(net: RoadNetwork, track_xy: np.ndarray, times: np.ndarray,
                       cands: Sequence[Candidates], window: int = 5,
                       radius_m: float = 50.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X float32 (N, F), point_idx (N,), cand_idx (N,))."""
    T = len(cands)
    counts = np.array([len(c) for c in cands])
    N = int(counts.sum())
    X = np.zeros((N, len(FEATURE_NAMES)), dtype=np.float32)
    point_idx = np.repeat(np.arange(T), counts)
    cand_idx = np.concatenate([np.arange(n) for n in counts]) if N else np.zeros(0, int)
    if N == 0:
        return X, point_idx, cand_idx

    heading, speed, valid, dt_prev, dt_next = gps_kinematics(track_xy, times)
    nearest = np.array([c.distance_m[0] for c in cands if len(c)])
    sigma = float(np.clip(1.4826 * np.median(nearest), 3.0, 30.0)) if nearest.size else 10.0

    arc_ids = np.concatenate([c.arc_ids for c in cands])
    dist = np.concatenate([c.distance_m for c in cands])
    offset = np.concatenate([c.offset_m for c in cands])
    rank = cand_idx.astype(float)  # candidates are sorted by distance
    best = np.repeat([c.distance_m[0] if len(c) else 0.0 for c in cands], counts)

    a_from, a_to = net.arcs[arc_ids, 0], net.arcs[arc_ids, 1]
    vec = net.nodes_xy[a_to] - net.nodes_xy[a_from]
    bearing = np.arctan2(vec[:, 1], vec[:, 0])
    alen = net.arc_len[arc_ids]
    hv = valid[point_idx]
    diff = np.abs((heading[point_idx] - bearing + np.pi) % (2 * np.pi) - np.pi)

    out_deg = np.bincount(net.arcs[:, 0], minlength=len(net.nodes_xy))
    in_deg = np.bincount(net.arcs[:, 1], minlength=len(net.nodes_xy))

    # reverse twin present among the same point's candidates
    twin = np.zeros(N, dtype=np.float32)
    support = np.zeros(N, dtype=np.float32)
    wmean = np.zeros(N, dtype=np.float32)
    per_point = [dict(zip(c.arc_ids.tolist(), c.distance_m.tolist())) for c in cands]
    pairs = [set(zip(net.arcs[c.arc_ids, 0].tolist(), net.arcs[c.arc_ids, 1].tolist()))
             for c in cands]
    row = 0
    for t, c in enumerate(cands):
        lo, hi = max(0, t - window), min(T, t + window + 1)
        span = hi - lo
        for j in range(len(c)):
            a = int(c.arc_ids[j])
            u, v = int(net.arcs[a, 0]), int(net.arcs[a, 1])
            twin[row] = (v, u) in pairs[t]
            n_hit, s = 0, 0.0
            for w in range(lo, hi):
                d = per_point[w].get(a)
                if d is None:
                    s += radius_m
                else:
                    n_hit += 1
                    s += d
            support[row] = n_hit / span
            wmean[row] = s / span
            row += 1

    cols = {
        "dist_m": dist,
        "dist_rank": rank,
        "dist_minus_best_m": dist - best,
        "dist_z": dist / sigma,
        "n_candidates": counts[point_idx],
        "arc_len_m": alen,
        "offset_frac": np.where(alen > 0, offset / np.maximum(alen, 1e-9), 0.0),
        "dist_to_arc_end_m": alen - offset,
        "dist_to_arc_start_m": offset,
        "heading_valid": hv.astype(float),
        "cos_heading_diff": np.where(hv, np.cos(diff), 0.0),
        "abs_heading_diff_deg": np.where(hv, np.degrees(diff), 90.0),
        "speed_mps": speed[point_idx],
        "dt_prev_s": dt_prev[point_idx],
        "dt_next_s": dt_next[point_idx],
        "has_reverse_twin": twin,
        "end_out_degree": out_deg[a_to],
        "start_in_degree": in_deg[a_from],
        "window_support": support,
        "window_mean_dist_m": wmean,
    }
    for i, name in enumerate(FEATURE_NAMES):
        X[:, i] = cols[name]
    return X, point_idx, cand_idx
