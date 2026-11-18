"""Geometry: local metric projection, road segments and candidate generation.

Nothing here reads the reference route; inputs are only GPS track, nodes, arcs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

EARTH_RADIUS_M = 6_371_008.8


class LocalProjector:
    """Equirectangular projection around a reference point (accurate to well
    under 1 m over the few-km extent of a trip)."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self._kx = EARTH_RADIUS_M * np.cos(np.radians(self.lat0)) * np.pi / 180.0
        self._ky = EARTH_RADIUS_M * np.pi / 180.0

    @classmethod
    def from_points(cls, lonlat: np.ndarray) -> "LocalProjector":
        return cls(float(np.mean(lonlat[:, 0])), float(np.mean(lonlat[:, 1])))

    def to_xy(self, lonlat: np.ndarray) -> np.ndarray:
        lonlat = np.asarray(lonlat, dtype=float)
        return np.column_stack(((lonlat[..., 0] - self.lon0) * self._kx,
                                (lonlat[..., 1] - self.lat0) * self._ky))

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        return np.column_stack((xy[:, 0] / self._kx + self.lon0,
                                xy[:, 1] / self._ky + self.lat0))


def point_segment_projection(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Exact projection of point(s) p onto segment(s) a->b (all (..., 2)).

    Returns (projected_xy, distance, t) where t in [0, 1] is the fraction
    along the segment.
    """
    ab = b - a
    denom = np.einsum("...i,...i->...", ab, ab)
    t = np.where(denom > 0, np.einsum("...i,...i->...", p - a, ab) / np.where(denom > 0, denom, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[..., None] * ab
    dist = np.linalg.norm(p - proj, axis=-1)
    return proj, dist, t


@dataclass
class RoadNetwork:
    """Directed road graph in local metres plus a sampled-segment KD-tree."""

    projector: LocalProjector
    nodes_xy: np.ndarray       # (N, 2)
    arcs: np.ndarray           # (A, 2) from, to
    arc_len: np.ndarray        # (A,)
    sample_spacing: float
    _tree: cKDTree = field(repr=False)
    _sample_arc: np.ndarray = field(repr=False)
    out_arcs: list = field(repr=False)        # node -> arc ids leaving it
    graph: object = field(repr=False)         # CSR (N x N), min arc length per (u, v)
    edge_arc: dict = field(repr=False)        # (u, v) -> shortest arc id

    @classmethod
    def build(cls, nodes_lonlat: np.ndarray, arcs: np.ndarray,
              projector: LocalProjector, sample_spacing: float = 10.0) -> "RoadNetwork":
        nodes_xy = projector.to_xy(nodes_lonlat)
        a = nodes_xy[arcs[:, 0]]
        b = nodes_xy[arcs[:, 1]]
        arc_len = np.linalg.norm(b - a, axis=1)

        # Densify every arc so a KD-tree over sample points finds every
        # segment within radius r when queried with r + spacing / 2.
        n_samples = np.maximum(np.ceil(arc_len / sample_spacing).astype(int), 1) + 1
        sample_arc = np.repeat(np.arange(len(arcs)), n_samples)
        starts = np.cumsum(n_samples) - n_samples
        idx_in = np.arange(len(sample_arc)) - np.repeat(starts, n_samples)
        frac = idx_in / np.repeat(n_samples - 1, n_samples)
        samples = a[sample_arc] + frac[:, None] * (b - a)[sample_arc]
        tree = cKDTree(samples)

        out_arcs: list[list[int]] = [[] for _ in range(len(nodes_xy))]
        for aid, (u, _v) in enumerate(arcs):
            out_arcs[u].append(aid)

        # Node graph for shortest paths: keep the shortest arc between each
        # ordered node pair (CSR would otherwise *sum* duplicates).
        w = np.maximum(arc_len, 1e-3)  # explicit zeros would vanish from CSR
        order = np.lexsort((w, arcs[:, 1], arcs[:, 0]))
        keys = arcs[order]
        first = np.ones(len(order), bool)
        first[1:] = np.any(keys[1:] != keys[:-1], axis=1)
        keep = order[first]
        graph = coo_matrix((w[keep], (arcs[keep, 0], arcs[keep, 1])),
                           shape=(len(nodes_xy),) * 2).tocsr()
        edge_arc = {(int(arcs[k, 0]), int(arcs[k, 1])): int(k) for k in keep}

        return cls(projector, nodes_xy, arcs, arc_len, sample_spacing,
                   tree, sample_arc, out_arcs, graph, edge_arc)

    def arcs_near(self, xy: np.ndarray, radius: float) -> list[np.ndarray]:
        hits = self._tree.query_ball_point(xy, radius + self.sample_spacing / 2.0)
        return [np.unique(self._sample_arc[h]) if len(h) else np.empty(0, int) for h in hits]


@dataclass
class Candidates:
    """Road candidates for one GPS point, sorted by distance."""
    arc_ids: np.ndarray     # (k,)
    proj_xy: np.ndarray     # (k, 2) projected point in metres
    distance_m: np.ndarray  # (k,)
    offset_m: np.ndarray    # (k,) distance from arc start along the arc

    def __len__(self) -> int:
        return len(self.arc_ids)


def generate_candidates(network: RoadNetwork, track_xy: np.ndarray,
                        k: int = 8, radius_m: float = 50.0,
                        fallback_radius_m: float = 200.0) -> list[Candidates]:
    """For every GPS point, the k nearest arcs (exact point-to-segment distance)
    within ``radius_m``; retry with ``fallback_radius_m`` if none were found.
    Points with no arc even in the fallback radius get an empty Candidates."""
    near = network.arcs_near(track_xy, radius_m)
    out: list[Candidates] = []
    for i, arc_ids in enumerate(near):
        r = radius_m
        if len(arc_ids) == 0 and fallback_radius_m > radius_m:
            arc_ids = network.arcs_near(track_xy[i:i + 1], fallback_radius_m)[0]
            r = fallback_radius_m
        if len(arc_ids) == 0:
            out.append(Candidates(np.empty(0, int), np.empty((0, 2)), np.empty(0), np.empty(0)))
            continue
        a = network.nodes_xy[network.arcs[arc_ids, 0]]
        b = network.nodes_xy[network.arcs[arc_ids, 1]]
        proj, dist, t = point_segment_projection(track_xy[i][None, :], a, b)
        keep = dist <= r
        order = np.argsort(dist[keep], kind="stable")[:k]
        sel = np.flatnonzero(keep)[order]
        out.append(Candidates(arc_ids[sel], proj[sel], dist[sel],
                              t[sel] * network.arc_len[arc_ids[sel]]))
    return out
