"""Validate pseudo-label alignment on the train trips (computational + visual).

  python -m scripts.validate_alignment
Writes experiments/ml/alignment_quality.csv, PNG plots and Folium HTML maps
for selected trips under experiments/ml/alignment_plots/.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import list_trips, load_trip
from src.evaluation import align_points_to_route, load_reference_route
from src.experiment import EXPERIMENT_DIR
from src.labels import align_track_to_route, alignment_quality
from src.pipeline import build_network

OUT = EXPERIMENT_DIR / "ml"
PLOTS = OUT / "alignment_plots"


def plot_trip(trip, net, track_xy, ref, al, path, zoom_positions=None):
    fig, axes = plt.subplots(1, 2 if zoom_positions is not None else 1,
                             figsize=(16 if zoom_positions is not None else 9, 8), squeeze=False)
    for k, ax in enumerate(axes[0]):
        # Network context (light) and reference route (by position colour)
        nx = net.nodes_xy
        for a in ref:
            u, v = net.arcs[a]
            ax.plot(*nx[[u, v]].T, color="0.6", lw=3, zorder=1)
        cmap = plt.get_cmap("viridis")
        col = cmap(al.position / max(len(ref) - 1, 1))
        ax.scatter(*track_xy[al.valid].T, c=col[al.valid], s=9, zorder=3)
        ax.scatter(*track_xy[~al.valid].T, c="red", marker="x", s=20, zorder=4,
                   label=f"invalid label ({(~al.valid).sum()})")
        # connector from each point to its label arc's projection
        a = nx[net.arcs[al.labels, 0]]
        b = nx[net.arcs[al.labels, 1]]
        ab = b - a
        tt = np.clip(np.einsum("ij,ij->i", track_xy - a, ab) /
                     np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0, 1)
        proj = a + tt[:, None] * ab
        for p, q in zip(track_xy[::2], proj[::2]):
            ax.plot([p[0], q[0]], [p[1], q[1]], color="tab:orange", lw=0.5, zorder=2)
        ax.set_aspect("equal")
        if k == 1:
            sel = np.isin(al.position, zoom_positions)
            pts = np.vstack([track_xy[sel]] + [nx[net.arcs[ref[p]]] for p in zoom_positions])
            lo, hi = pts.min(0) - 40, pts.max(0) + 40
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
            ax.set_title("zoom: repeated / reversed reference arcs")
        else:
            ax.set_title(f"trip {trip.trip_id}: GPS coloured by aligned route position "
                         f"(grey = .route, orange = point->label)")
            ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def folium_map(trip, ref, al, path):
    import folium
    lat0, lon0 = trip.track[:, 1].mean(), trip.track[:, 0].mean()
    m = folium.Map(location=[lat0, lon0], zoom_start=15, tiles="OpenStreetMap")
    g_ref = folium.FeatureGroup("reference route (.route)")
    for a in ref:
        u, v = trip.arcs[a]
        folium.PolyLine([[trip.nodes[u, 1], trip.nodes[u, 0]], [trip.nodes[v, 1], trip.nodes[v, 0]]],
                        color="green", weight=5, opacity=0.6).add_to(g_ref)
    g_gps = folium.FeatureGroup("GPS (blue = valid label, red = invalid)")
    for t, (lon, lat, ts) in enumerate(trip.track):
        folium.CircleMarker([lat, lon], radius=2, color="blue" if al.valid[t] else "red",
                            tooltip=f"t={ts:.0f} label arc {al.labels[t]} pos {al.position[t]} "
                                    f"d={al.distance_m[t]:.1f}m").add_to(g_gps)
    g_ref.add_to(m); g_gps.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", nargs="*", default=None, help="trip ids to plot")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    rows, cache = [], {}
    for item in list_trips():
        if item["split"] != "train":
            continue
        tid = item["trip_id"]
        trip = load_trip(tid)
        net, xy = build_network(trip)
        ref = load_reference_route(tid)
        al = align_track_to_route(xy, trip.timestamps, ref, net)
        q = alignment_quality(al, ref, net)
        # sensitivity: agreement with perturbed settings and with v1 (distance-only)
        for name, kw in [("sigma5", {"sigma": 5.0}), ("sigma20", {"sigma": 20.0}),
                         ("no_heading", {"heading_kappa": 0.0}),
                         ("no_skip_penalty", {"skip_penalty_per_m": 0.0})]:
            alt = align_track_to_route(xy, trip.timestamps, ref, net, **kw)
            q[f"agree_{name}"] = round(float(np.mean(alt.labels == al.labels)), 4)
        v1, _ = align_points_to_route(xy, ref, net.nodes_xy, trip.arcs)
        q["agree_v1_labels"] = round(float(np.mean(v1 == al.labels)), 4)
        rows.append({"trip_id": tid, **q})
        cache[tid] = (trip, net, xy, ref, al)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "alignment_quality.csv", index=False)
    num = df.select_dtypes("number").drop(columns=["first_pos", "last_pos", "n_ref", "n_points"])
    print("alignment quality over", len(df), "train trips (mean / min):")
    for c in num.columns:
        print(f"  {c:34s} mean {num[c].mean():8.4f}   min {num[c].min():8.4f}   max {num[c].max():8.4f}")
    print("  monotone in all trips:", bool(df["monotone"].all()))
    rep = df[df["repeated_ref_arcs"] > 0][["trip_id", "repeated_ref_arcs", "repeated_visits_with_points",
                                           "heading_agree_frac", "label_valid_frac"]]
    print("\ntrips with repeated reference arcs:\n", rep.to_string(index=False))
    worst = df.sort_values("label_valid_frac").head(5)[["trip_id", "label_valid_frac", "dist_p95_m",
                                                        "heading_agree_frac", "first_pos", "last_pos", "n_ref"]]
    print("\nlowest label validity:\n", worst.to_string(index=False))

    to_plot = args.plot if args.plot else (
        ["00000038"] + rep["trip_id"].head(2).tolist() + worst["trip_id"].head(1).tolist())
    for tid in to_plot:
        trip, net, xy, ref, al = cache[tid]
        zoom = None
        dup = [a for a in set(ref.tolist()) if (ref == a).sum() > 1]
        twins = [p for p in range(len(ref)) if any(
            net.arcs[ref[q], 0] == net.arcs[ref[p], 1] and net.arcs[ref[q], 1] == net.arcs[ref[p], 0]
            for q in range(len(ref)))]
        if dup:
            zoom = [p for p in range(len(ref)) if ref[p] in dup]
        elif twins:
            zoom = twins
        plot_trip(trip, net, xy, ref, al, PLOTS / f"alignment_{tid}.png", zoom)
        folium_map(trip, ref, al, PLOTS / f"alignment_{tid}.html")
        print("plotted", tid)


if __name__ == "__main__":
    main()
