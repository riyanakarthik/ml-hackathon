"""End-to-end prediction for one trip. Uses only track, nodes and arcs."""
from __future__ import annotations

import time

import numpy as np

from .data_loader import Trip
from .decoder import MatchResult, greedy_continuity_decode, hmm_viterbi_decode
from .matcher import LocalProjector, RoadNetwork, generate_candidates

METHODS = ("hmm", "baseline")


def predict_trip(trip: Trip, method: str = "hmm", k: int = 8,
                 radius_m: float = 50.0) -> tuple[MatchResult, RoadNetwork, dict]:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    timings = {}
    t0 = time.perf_counter()
    proj = LocalProjector.from_points(trip.track_lonlat)
    net = RoadNetwork.build(trip.nodes, trip.arcs, proj)
    track_xy = proj.to_xy(trip.track_lonlat)
    timings["network_ms"] = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    cands = generate_candidates(net, track_xy, k=k, radius_m=radius_m)
    timings["candidates_ms"] = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    if method == "hmm":
        result = hmm_viterbi_decode(cands, net, track_xy)
    else:
        result = greedy_continuity_decode(cands, net)
    timings["decode_ms"] = (time.perf_counter() - t2) * 1000

    result.info["points_without_candidates"] = int(sum(1 for c in cands if not len(c)))
    result.info["mean_candidates_per_point"] = round(float(np.mean([len(c) for c in cands])), 2)
    return result, net, {k_: round(v, 1) for k_, v in timings.items()}


def to_latlon(net: RoadNetwork, xy: np.ndarray) -> list:
    """(T, 2) metres -> list of [lat, lon] (None where unmatched)."""
    out = []
    ll = net.projector.to_lonlat(np.nan_to_num(xy))
    for (lon, lat), row in zip(ll, xy):
        out.append(None if np.isnan(row).any() else [round(float(lat), 7), round(float(lon), 7)])
    return out
