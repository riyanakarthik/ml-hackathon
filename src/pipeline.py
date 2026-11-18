"""End-to-end prediction for one trip. Uses only track, nodes and arcs."""
from __future__ import annotations

import logging
import time

import numpy as np

from .config import BASELINE_V1, MatcherConfig
from .data_loader import Trip
from .decoder import (MatchResult, emission_log_probs, estimate_sigma,
                      greedy_continuity_decode, hmm_viterbi_decode)
from .matcher import LocalProjector, RoadNetwork, generate_candidates

METHODS = ("hmm", "baseline")
log = logging.getLogger(__name__)


def build_network(trip: Trip) -> tuple[RoadNetwork, np.ndarray]:
    """Road network + projected track. Independent of MatcherConfig, so it can
    be cached across tuning trials."""
    proj = LocalProjector.from_points(trip.track_lonlat)
    net = RoadNetwork.build(trip.nodes, trip.arcs, proj)
    return net, proj.to_xy(trip.track_lonlat)


def predict_trip(trip: Trip, method: str = "hmm", k: int | None = None,
                 radius_m: float | None = None, config: MatcherConfig = BASELINE_V1,
                 network: tuple[RoadNetwork, np.ndarray] | None = None,
                 ml_probs: list[np.ndarray] | None = None,
                 ) -> tuple[MatchResult, RoadNetwork, dict]:
    """``ml_probs`` (experiments only): externally supplied per-point candidate
    probabilities, e.g. out-of-fold predictions; otherwise the model named in
    ``config.ml_model`` is used when ``config.emission`` is "ml"/"hybrid".
    If the ML emission cannot be computed, falls back to geometric emissions."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    cfg = config
    if k is not None or radius_m is not None:
        cfg = cfg.with_(k=k if k is not None else cfg.k,
                        radius_m=radius_m if radius_m is not None else cfg.radius_m)
    timings = {}
    t0 = time.perf_counter()
    net, track_xy = network if network is not None else build_network(trip)
    timings["network_ms"] = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    cands = generate_candidates(net, track_xy, k=cfg.k, radius_m=cfg.radius_m,
                                fallback_radius_m=cfg.fallback_radius_m)
    timings["candidates_ms"] = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    if method == "hmm":
        emissions, name, fallback = None, "hmm_viterbi", None
        if cfg.emission != "geometric":
            try:
                from .ml_emission import combine_emissions, load_model_bundle, predict_candidate_probs
                probs = ml_probs
                if probs is None:
                    if not cfg.ml_model:
                        raise ValueError("config.ml_model not set")
                    bundle = load_model_bundle(cfg.ml_model)
                    probs = predict_candidate_probs(bundle, net, track_xy, trip.timestamps,
                                                    cands, cfg.radius_m)
                sigma = (cfg.sigma_m if cfg.sigma_m is not None else estimate_sigma(cands)) \
                    * cfg.sigma_scale
                emissions = combine_emissions(probs, emission_log_probs(cands, sigma),
                                              cfg.emission, cfg.ml_weight, cfg.prob_floor)
                name = f"hmm_viterbi_{cfg.emission}_emission"
            except Exception as e:  # noqa: BLE001 - any model failure -> geometric fallback
                log.warning("ML emission unavailable (%s); falling back to geometric", e)
                fallback = str(e)
        result = hmm_viterbi_decode(cands, net, track_xy, config=cfg, emissions=emissions,
                                    method_name=name)
        result.info["emission"] = "geometric" if emissions is None else cfg.emission
        if fallback:
            result.info["emission_fallback"] = fallback
    else:
        result = greedy_continuity_decode(cands, net)
    timings["decode_ms"] = (time.perf_counter() - t2) * 1000

    result.info["points_without_candidates"] = int(sum(1 for c in cands if not len(c)))
    result.info["mean_candidates_per_point"] = round(float(np.mean([len(c) for c in cands])), 2)
    result.info["_candidates"] = cands  # for diagnostics; stripped by the API
    return result, net, {k_: round(v, 1) for k_, v in timings.items()}


def to_latlon(net: RoadNetwork, xy: np.ndarray) -> list:
    """(T, 2) metres -> list of [lat, lon] (None where unmatched)."""
    out = []
    ll = net.projector.to_lonlat(np.nan_to_num(xy))
    for (lon, lat), row in zip(ll, xy):
        out.append(None if np.isnan(row).any() else [round(float(lat), 7), round(float(lon), 7)])
    return out
