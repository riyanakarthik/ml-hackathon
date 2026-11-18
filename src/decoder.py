"""Decoders that turn per-point road candidates into a matched path.

* ``greedy_continuity_decode`` - fast baseline: per-point choice of the
  closest candidate with a bonus for staying on / continuing from the previous
  arc. Not an HMM.
* ``viterbi_log`` - generic log-space Viterbi over variable-size state sets,
  taking emission log-probabilities and a transition-score callback.
* ``hmm_viterbi_decode`` - a Newson & Krumm (2009)-style HMM: Gaussian
  emissions on GPS-to-road distance and exponential transitions on
  |network distance - straight-line distance|, decoded with ``viterbi_log``.

Both decoders then expand matched arcs into a connected arc route by filling
gaps with shortest paths on the road graph. No reference route is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.sparse.csgraph import dijkstra

from .matcher import Candidates, RoadNetwork

NEG_INF = -np.inf


@dataclass
class MatchResult:
    method: str
    point_arc_ids: np.ndarray        # (T,) matched arc per GPS point, -1 if none
    point_xy: np.ndarray             # (T, 2) matched position in metres (nan if none)
    route_arc_ids: list[int]         # connected, de-duplicated arc sequence
    info: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Emissions
# ----------------------------------------------------------------------------

def estimate_sigma(cands: Sequence[Candidates], lo: float = 3.0, hi: float = 30.0) -> float:
    """Robust GPS noise estimate (Newson & Krumm: 1.4826 * MAD of the distance
    to the nearest road). Uses only GPS + map geometry."""
    d = np.array([c.distance_m[0] for c in cands if len(c)])
    if d.size == 0:
        return 10.0
    return float(np.clip(1.4826 * np.median(d), lo, hi))


def emission_log_probs(cands: Sequence[Candidates], sigma: float) -> list[np.ndarray]:
    """Gaussian log-likelihood of the GPS-to-road distance (up to a constant).

    Hook for a learned model: replace with e.g. ``model.predict_log_proba`` on
    per-candidate features; any list of (k_t,) log-score arrays works."""
    return [-0.5 * (c.distance_m / sigma) ** 2 for c in cands]


# ----------------------------------------------------------------------------
# Generic Viterbi
# ----------------------------------------------------------------------------

TransitionFn = Callable[[int, int], np.ndarray]


def viterbi_log(emissions: Sequence[np.ndarray], transition_fn: TransitionFn):
    """Log-space Viterbi over steps with variable (possibly zero) state counts.

    emissions[t]       : (k_t,) log-probabilities; empty array = no candidates,
                         the step is skipped and stays unassigned.
    transition_fn(s, t): (k_s, k_t) log transition scores from the previous
                         *non-empty* step s to step t (-inf = impossible).

    If every transition into a step is impossible the chain is broken and
    decoding restarts from that step's emissions (the usual HMM "break").

    Returns (path, breaks): path[t] is the chosen state index or -1;
    breaks lists the steps where a new chain started.
    """
    T = len(emissions)
    path = np.full(T, -1, dtype=int)
    breaks: list[int] = []
    backptr: dict[int, np.ndarray] = {}
    chain: list[int] = []

    def backtrack(chain_steps: list[int], delta: np.ndarray) -> None:
        state = int(np.argmax(delta))
        for t in reversed(chain_steps):
            path[t] = state
            if t in backptr:
                state = int(backptr[t][state])

    delta = None
    prev = -1
    for t in range(T):
        em = np.asarray(emissions[t], dtype=float)
        if em.size == 0:
            continue
        if delta is None:
            delta, chain = em.copy(), [t]
            breaks.append(t)
        else:
            scores = delta[:, None] + np.asarray(transition_fn(prev, t), dtype=float)
            best = scores.max(axis=0)
            if not np.any(np.isfinite(best + em)):
                backtrack(chain, delta)
                delta, chain = em.copy(), [t]
                breaks.append(t)
            else:
                backptr[t] = scores.argmax(axis=0)
                delta = best + em
                chain.append(t)
        prev = t
    if delta is not None:
        backtrack(chain, delta)
    return path, breaks


# ----------------------------------------------------------------------------
# Network distances
# ----------------------------------------------------------------------------

class NetworkDistance:
    """Cached bounded single-source shortest paths on the node graph."""

    def __init__(self, network: RoadNetwork, limit_m: float = 2000.0):
        self.net = network
        self.limit = limit_m
        self._cache: dict[int, dict[int, float]] = {}

    def from_node(self, u: int, limit: float | None = None) -> dict[int, float]:
        if limit is None or limit <= self.limit:
            if u not in self._cache:
                self._cache[u] = self._run(u, self.limit)
            return self._cache[u]
        return self._run(u, limit)

    def _run(self, u: int, limit: float) -> dict[int, float]:
        d = dijkstra(self.net.graph, directed=True, indices=u, limit=limit)
        idx = np.flatnonzero(np.isfinite(d))
        return dict(zip(idx.tolist(), d[idx].tolist()))

    def path_arcs(self, u: int, v: int, limit: float) -> list[int] | None:
        """Arc ids of a shortest path u -> v, or None if unreachable."""
        if u == v:
            return []
        d, pred = dijkstra(self.net.graph, directed=True, indices=u,
                           limit=limit, return_predecessors=True)
        if not np.isfinite(d[v]):
            return None
        nodes = [v]
        while nodes[-1] != u:
            nodes.append(int(pred[nodes[-1]]))
        nodes.reverse()
        return [self.net.edge_arc[(a, b)] for a, b in zip(nodes[:-1], nodes[1:])]


def route_distance(net: RoadNetwork, nd: NetworkDistance, arc_a: int, off_a: float,
                   arc_b: int, off_b: float, limit: float | None = None,
                   backtrack_tol_m: float = 5.0) -> float:
    """Driving distance from a position on arc_a to a position on arc_b."""
    if arc_a == arc_b and off_b >= off_a - backtrack_tol_m:
        return max(off_b - off_a, 0.0)
    to_a = int(net.arcs[arc_a, 1])
    from_b = int(net.arcs[arc_b, 0])
    rest = net.arc_len[arc_a] - off_a
    mid = 0.0 if to_a == from_b else nd.from_node(to_a, limit).get(from_b, np.inf)
    return rest + mid + off_b


def make_hmm_transition_fn(net: RoadNetwork, cands: Sequence[Candidates],
                           track_xy: np.ndarray, nd: NetworkDistance,
                           beta: float = 15.0) -> TransitionFn:
    """Newson & Krumm transition: log p = -|d_route - d_straight| / beta.

    Hook for a learned model: any callable returning (k_s, k_t) log-scores
    (e.g. built from features like d_route, d_straight, dt, turn angle)."""

    def fn(s: int, t: int) -> np.ndarray:
        cs, ct = cands[s], cands[t]
        d_gc = float(np.linalg.norm(track_xy[t] - track_xy[s]))
        limit = max(nd.limit, 3.0 * d_gc + 200.0)
        out = np.full((len(cs), len(ct)), NEG_INF)
        for i in range(len(cs)):
            for j in range(len(ct)):
                d = route_distance(net, nd, int(cs.arc_ids[i]), float(cs.offset_m[i]),
                                   int(ct.arc_ids[j]), float(ct.offset_m[j]), limit)
                if np.isfinite(d):
                    out[i, j] = -abs(d - d_gc) / beta
        return out

    return fn


# ----------------------------------------------------------------------------
# Route assembly
# ----------------------------------------------------------------------------

def assemble_route(point_arc_ids: np.ndarray, net: RoadNetwork, nd: NetworkDistance,
                   max_gap_m: float = 3000.0) -> tuple[list[int], int]:
    """Collapse per-point arcs to a sequence and fill gaps between consecutive
    distinct arcs with shortest paths. Returns (route, n_unfilled_gaps)."""
    seq: list[int] = []
    for a in point_arc_ids:
        a = int(a)
        if a >= 0 and (not seq or seq[-1] != a):
            seq.append(a)
    if not seq:
        return [], 0
    route = [seq[0]]
    unfilled = 0
    for b in seq[1:]:
        a = route[-1]
        to_a, from_b = int(net.arcs[a, 1]), int(net.arcs[b, 0])
        if to_a != from_b:
            fill = nd.path_arcs(to_a, from_b, max_gap_m)
            if fill is None:
                unfilled += 1
            else:
                route.extend(x for x in fill if x != route[-1])
        if route[-1] != b:
            route.append(b)
    return route, unfilled


def _positions(cands: Sequence[Candidates], choice: np.ndarray):
    T = len(cands)
    arcs = np.full(T, -1, dtype=int)
    xy = np.full((T, 2), np.nan)
    for t, j in enumerate(choice):
        if j >= 0:
            arcs[t] = cands[t].arc_ids[j]
            xy[t] = cands[t].proj_xy[j]
    return arcs, xy


# ----------------------------------------------------------------------------
# Decoders
# ----------------------------------------------------------------------------

def greedy_continuity_decode(cands: Sequence[Candidates], net: RoadNetwork,
                             sigma: float | None = None, stay_bonus: float = 2.0,
                             successor_bonus: float = 1.5) -> MatchResult:
    """Baseline: pick the best candidate per point by Gaussian distance score
    plus a bonus for the previous arc or an arc that starts where it ends."""
    sigma = sigma or estimate_sigma(cands)
    em = emission_log_probs(cands, sigma)
    choice = np.full(len(cands), -1, dtype=int)
    prev_arc = -1
    for t, c in enumerate(cands):
        if not len(c):
            continue
        score = em[t].copy()
        if prev_arc >= 0:
            score[c.arc_ids == prev_arc] += stay_bonus
            score[net.arcs[c.arc_ids, 0] == net.arcs[prev_arc, 1]] += successor_bonus
        choice[t] = int(np.argmax(score))
        prev_arc = int(c.arc_ids[choice[t]])
    arcs, xy = _positions(cands, choice)
    nd = NetworkDistance(net)
    route, unfilled = assemble_route(arcs, net, nd)
    return MatchResult("geometric_greedy_continuity", arcs, xy, route,
                       {"sigma_m": sigma, "unfilled_gaps": unfilled})


def hmm_viterbi_decode(cands: Sequence[Candidates], net: RoadNetwork, track_xy: np.ndarray,
                       sigma: float | None = None, beta: float = 15.0) -> MatchResult:
    """Newson & Krumm-style HMM decoded with ``viterbi_log``."""
    sigma = sigma or estimate_sigma(cands)
    nd = NetworkDistance(net)
    em = emission_log_probs(cands, sigma)
    trans = make_hmm_transition_fn(net, cands, track_xy, nd, beta)
    choice, breaks = viterbi_log(em, trans)
    arcs, xy = _positions(cands, choice)
    route, unfilled = assemble_route(arcs, net, nd)
    return MatchResult("hmm_viterbi", arcs, xy, route,
                       {"sigma_m": sigma, "beta": beta, "hmm_breaks": len(breaks) - 1,
                        "unfilled_gaps": unfilled})
