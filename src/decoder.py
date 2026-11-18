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

from .config import BASELINE_V1, MatcherConfig
from .matcher import Candidates, RoadNetwork

NEG_INF = -np.inf


@dataclass
class MatchResult:
    method: str
    point_arc_ids: np.ndarray        # (T,) matched arc per GPS point, -1 if none
    point_xy: np.ndarray             # (T, 2) matched position in metres (nan if none)
    route_arc_ids: list[int]         # connected, de-duplicated arc sequence
    route_is_fill: list[bool] = field(default_factory=list)  # True = inserted by gap filling
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


def is_reverse_twin(net: RoadNetwork, a: int, b: int) -> bool:
    return a != b and net.arcs[a, 0] == net.arcs[b, 1] and net.arcs[a, 1] == net.arcs[b, 0]


def make_hmm_transition_fn(net: RoadNetwork, cands: Sequence[Candidates],
                           track_xy: np.ndarray, nd: NetworkDistance,
                           beta: float = 15.0, backtrack_tol_m: float = 5.0,
                           backtrack_penalty: float = 0.0,
                           uturn_penalty: float = 0.0) -> TransitionFn:
    """Newson & Krumm transition: log p = -|d_route - d_straight| / beta, plus
    optional penalties (all 0 in baseline_v1):

    * backtrack_penalty > 0: moving backwards on the same arc by more than
      backtrack_tol_m is scored as GPS noise (d_route = 0) with a penalty of
      backtrack_penalty per metre, instead of requiring a loop through the graph.
    * uturn_penalty: subtracted when moving onto the reverse twin of an arc.

    Hook for a learned model: any callable returning (k_s, k_t) log-scores
    (e.g. built from features like d_route, d_straight, dt, turn angle)."""

    def fn(s: int, t: int) -> np.ndarray:
        cs, ct = cands[s], cands[t]
        d_gc = float(np.linalg.norm(track_xy[t] - track_xy[s]))
        limit = max(nd.limit, 3.0 * d_gc + 200.0)
        out = np.full((len(cs), len(ct)), NEG_INF)
        for i in range(len(cs)):
            a, oa = int(cs.arc_ids[i]), float(cs.offset_m[i])
            for j in range(len(ct)):
                b, ob = int(ct.arc_ids[j]), float(ct.offset_m[j])
                if backtrack_penalty > 0 and a == b and ob < oa - backtrack_tol_m:
                    out[i, j] = (-d_gc / beta
                                 - backtrack_penalty * (oa - ob - backtrack_tol_m))
                    continue
                d = route_distance(net, nd, a, oa, b, ob, limit, backtrack_tol_m)
                if np.isfinite(d):
                    score = -abs(d - d_gc) / beta
                    if uturn_penalty and is_reverse_twin(net, a, b):
                        score -= uturn_penalty
                    out[i, j] = score
        return out

    return fn


# ----------------------------------------------------------------------------
# Route assembly
# ----------------------------------------------------------------------------

def assemble_route(point_arc_ids: np.ndarray, net: RoadNetwork, nd: NetworkDistance,
                   max_gap_m: float = 3000.0, return_fill_mask: bool = False):
    """Collapse per-point arcs to a sequence and fill gaps between consecutive
    distinct arcs with shortest paths. Returns (route, n_unfilled_gaps) or,
    with return_fill_mask, (route, n_unfilled_gaps, is_fill)."""
    seq: list[int] = []
    for a in point_arc_ids:
        a = int(a)
        if a >= 0 and (not seq or seq[-1] != a):
            seq.append(a)
    route: list[int] = []
    is_fill: list[bool] = []
    unfilled = 0
    if seq:
        route, is_fill = [seq[0]], [False]
    for b in seq[1:]:
        a = route[-1]
        to_a, from_b = int(net.arcs[a, 1]), int(net.arcs[b, 0])
        if to_a != from_b:
            fill = nd.path_arcs(to_a, from_b, max_gap_m)
            if fill is None:
                unfilled += 1
            else:
                for x in fill:
                    if x != route[-1]:
                        route.append(x)
                        is_fill.append(True)
        if route[-1] != b:
            route.append(b)
            is_fill.append(False)
        else:
            is_fill[-1] = False  # matched directly, not only filled
    if return_fill_mask:
        return route, unfilled, is_fill
    return route, unfilled


def remove_short_loops(route: list[int], is_fill: list[bool], arc_len: np.ndarray,
                       max_loop_m: float) -> tuple[list[int], list[bool]]:
    """Cut cycles A ... A from the route when the arcs travelled between the
    two visits total less than max_loop_m (typically decoder jitter, e.g.
    A -> reverse(A) -> A). Continuity is preserved because the route resumes
    from the first visit of A. Longer cycles are kept as real traversals."""
    if max_loop_m <= 0:
        return route, is_fill
    out: list[int] = []
    flags: list[bool] = []
    pos: dict[int, int] = {}
    for a, f in zip(route, is_fill):
        p = pos.get(a)
        if p is not None and float(arc_len[out[p + 1:]].sum()) < max_loop_m:
            for x in out[p + 1:]:
                if pos.get(x, -1) > p:
                    del pos[x]
            del out[p + 1:]
            del flags[p + 1:]
            flags[p] = flags[p] and f
            continue
        pos[a] = len(out)
        out.append(a)
        flags.append(f)
    return out, flags


def _positions(cands: Sequence[Candidates], choice: np.ndarray):
    T = len(cands)
    arcs = np.full(T, -1, dtype=int)
    xy = np.full((T, 2), np.nan)
    for t, j in enumerate(choice):
        if j >= 0:
            arcs[t] = cands[t].arc_ids[j]
            xy[t] = cands[t].proj_xy[j]
    return arcs, xy


def thin_observations(track_xy: np.ndarray, cands: Sequence[Candidates],
                      min_spacing_m: float) -> np.ndarray:
    """Boolean mask of points kept for decoding: a point is skipped when it is
    closer than min_spacing_m to the last kept point (first/last always kept)."""
    keep = np.zeros(len(track_xy), bool)
    last = None
    for t in range(len(track_xy)):
        if not len(cands[t]):
            continue
        if last is None or min_spacing_m <= 0 or \
                np.linalg.norm(track_xy[t] - track_xy[last]) >= min_spacing_m:
            keep[t] = True
            last = t
    if last is not None:
        keep[last] = True
    nz = [t for t in range(len(cands)) if len(cands[t])]
    if nz:
        keep[nz[-1]] = True
    return keep


def _fill_skipped(choice: np.ndarray, keep: np.ndarray, cands: Sequence[Candidates],
                  em: Sequence[np.ndarray]) -> np.ndarray:
    """Give thinned-out points the candidate on the previous decoded arc if
    available, otherwise their best-emission candidate."""
    choice = choice.copy()
    prev_arc = -1
    for t in range(len(choice)):
        if keep[t]:
            if choice[t] >= 0:
                prev_arc = int(cands[t].arc_ids[choice[t]])
            continue
        if not len(cands[t]):
            continue
        same = np.flatnonzero(cands[t].arc_ids == prev_arc)
        choice[t] = int(same[0]) if len(same) else int(np.argmax(em[t]))
    return choice


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
    route, unfilled, fill = assemble_route(arcs, net, nd, return_fill_mask=True)
    return MatchResult("geometric_greedy_continuity", arcs, xy, route, fill,
                       {"sigma_m": sigma, "unfilled_gaps": unfilled})


def hmm_viterbi_decode(cands: Sequence[Candidates], net: RoadNetwork, track_xy: np.ndarray,
                       sigma: float | None = None, beta: float | None = None,
                       config: MatcherConfig = BASELINE_V1,
                       emissions: Sequence[np.ndarray] | None = None,
                       method_name: str = "hmm_viterbi") -> MatchResult:
    """Newson & Krumm-style HMM decoded with ``viterbi_log``.

    ``sigma``/``beta`` arguments override the config (kept for compatibility).
    ``emissions`` (per-point log-scores aligned with ``cands``) replaces the
    geometric Gaussian emissions, e.g. classifier-based ones."""
    cfg = config
    if beta is None:
        beta = cfg.beta
    if sigma is None:
        sigma = cfg.sigma_m if cfg.sigma_m is not None else estimate_sigma(cands)
    sigma = sigma * cfg.sigma_scale
    nd = NetworkDistance(net, cfg.dijkstra_limit_m)
    em = emission_log_probs(cands, sigma) if emissions is None else list(emissions)
    keep = thin_observations(track_xy, cands, cfg.min_spacing_m)
    em_dec = [e if keep[t] else np.empty(0) for t, e in enumerate(em)]
    trans = make_hmm_transition_fn(net, cands, track_xy, nd, beta, cfg.backtrack_tol_m,
                                   cfg.backtrack_penalty, cfg.uturn_penalty)
    choice, breaks = viterbi_log(em_dec, trans)
    choice = _fill_skipped(choice, keep, cands, em)
    arcs, xy = _positions(cands, choice)
    route, unfilled, fill = assemble_route(arcs, net, nd, cfg.fill_max_gap_m, return_fill_mask=True)
    route, fill = remove_short_loops(route, fill, net.arc_len, cfg.max_loop_m)
    return MatchResult(method_name, arcs, xy, route, fill,
                       {"config": cfg.name, "sigma_m": round(sigma, 3), "beta": beta,
                        "hmm_breaks": len(breaks) - 1, "unfilled_gaps": unfilled,
                        "decoded_points": int(keep.sum())})
