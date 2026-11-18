"""Build report-only assets from recorded data (no training, no tuning, no backend changes).

  .venv/bin/python reports/tooling/build_assets.py      (run from the repo root)
Writes reports/figures/*.png, reports/figures/maps/*.html, reports/data/*.json|csv.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = ROOT / "reports" / "figures"
DATA = ROOT / "reports" / "data"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "axes.axisbelow": True, "legend.frameon": False})


# ---------------------------------------------------------------- A. dataset overview
def dataset_overview():
    from src.data_loader import list_trips, load_trip
    from src.evaluation import load_reference_route
    rows = []
    for t in list_trips(ROOT / "Student folder"):
        trip = load_trip(t["trip_id"], ROOT / "Student folder")
        ref = load_reference_route(t["trip_id"], ROOT / "Student folder")
        dt = np.diff(trip.timestamps)
        rows.append({"split": t["split"], "trip_id": t["trip_id"], "gps_points": len(trip.track),
                     "nodes": len(trip.nodes), "arcs": len(trip.arcs), "route_arcs": len(ref),
                     "route_repeats": len(ref) - len(set(ref.tolist())),
                     "frac_1s": float(np.mean(dt == 1)), "max_gap_s": float(dt.max()),
                     "gaps_over_10s": int((dt > 10).sum()), "duration_s": float(trip.timestamps[-1] - trip.timestamps[0])})
    # evaluator_test: inventory recorded in the frozen manifest; route lengths from per-trip results
    man = json.load(open(ROOT / "experiments/final_evaluation/evaluation_manifest.json"))
    ev = pd.read_csv(ROOT / "experiments/final_evaluation/evaluation_per_trip.csv")
    ref_len = ev[ev.variant == "rf80_hmm_deployed"].set_index("trip_id")["n_ref_arcs"]
    for r in man["dataset_inventory"]:
        rows.append({"split": "evaluator_test", "trip_id": r["trip_id"], "gps_points": r["n_gps_points"],
                     "nodes": r["n_nodes"], "arcs": r["n_arcs"], "route_arcs": int(ref_len[int(r["trip_id"])]),
                     "route_repeats": None, "frac_1s": r["frac_1s_steps"], "max_gap_s": r["max_time_gap_s"],
                     "gaps_over_10s": None, "duration_s": r["duration_s"]})
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "dataset_overview_per_trip.csv", index=False)
    g = df.groupby("split", sort=False)
    summ = pd.DataFrame({
        "trips": g.size(), "gps_points_total": g.gps_points.sum(), "gps_points_median": g.gps_points.median(),
        "gps_points_min": g.gps_points.min(), "gps_points_max": g.gps_points.max(),
        "nodes_median": g.nodes.median(), "arcs_median": g.arcs.median(), "arcs_max": g.arcs.max(),
        "route_arcs_median": g.route_arcs.median(), "route_arcs_total": g.route_arcs.sum(),
        "frac_1s_mean": g.frac_1s.mean().round(4), "max_gap_s": g.max_gap_s.max(),
        "trips_with_repeated_route_arcs": g.route_repeats.apply(lambda s: int((s > 0).sum()) if s.notna().all() else None),
    }).reset_index()
    summ.to_csv(DATA / "dataset_overview_summary.csv", index=False)
    print(summ.to_string(index=False))

    order = ["train", "student_test", "evaluator_test"]
    lab = {"train": "train (80)", "student_test": "student_test (10)", "evaluator_test": "evaluator_test (10)"}
    col = {"train": S1, "student_test": S2, "evaluator_test": S3}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (c, title, log) in zip(axes, [("gps_points", "GPS observations per trip", True),
                                          ("arcs", "Directed arcs (road segments) per trip", True),
                                          ("route_arcs", "Reference .route length (arcs) per trip", False)]):
        for i, s in enumerate(order):
            v = df[df.split == s][c].values
            jitter = (np.random.default_rng(0).random(len(v)) - 0.5) * 0.35
            ax.scatter(np.full(len(v), i) + jitter, v, s=14, color=col[s], alpha=0.8, edgecolors="none")
            ax.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, lw=2)
        ax.set_xticks(range(3), [lab[s] for s in order], fontsize=8.5)
        ax.set_title(title, fontsize=10.5); ax.grid(axis="x", visible=False)
        if log:
            ax.set_yscale("log")
    fig.text(0.01, 0.01, "Each dot is one trip; black bar = median. Source: reports/data/dataset_overview_per_trip.csv "
             "(train/student_test from data files; evaluator_test from experiments/final_evaluation/evaluation_manifest.json).",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(FIG / "fig_dataset_overview.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- B. architecture diagram
def architecture():
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="white", ec=color, lw=1.8))
        ax.text(x + w / 2, y + h - 0.18, title, ha="center", va="top", fontsize=9.5, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.32 - 0.27 * (title.count("\n") + 1), body, ha="center", va="top",
                fontsize=7.6, color=INK2, linespacing=1.35)

    def arrow(x1, y1, x2, y2, label=None):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, color=INK2, lw=1.4))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", fontsize=7, color=INK2)

    ax.text(0.1, 4.45, "Laptop 1 — FastAPI backend (main.py)", fontsize=10, weight="bold", color=S1)
    ax.text(0.1, 1.15, "Laptop 2 — evaluator UI (frontend/app.py)", fontsize=10, weight="bold", color=S2)
    y = 2.45; h = 1.85; w = 1.9
    steps = [
        ("1. Load trip", ".track  lon lat t\n.nodes  lon lat\n.arcs   from to\n(validation)", S1),
        ("2. Project", "lon/lat -> local\nmetres\n(equirectangular)", S1),
        ("3. Candidates", "KD-tree over\ndensified arcs;\nexact point-to-\nsegment; k=12,\nr=52.7 m", S1),
        ("4. Features +\nclassifier", "20 features per\ncandidate; Random\nForest P(correct)", S1),
        ("5. HMM /\nViterbi", "emission = 4 log p\ntransition =\n-|d_route-d_gc|/beta\nlog-space Viterbi", S1),
        ("6. Route\nassembly", "collapse arcs;\nDijkstra gap\nstitching ->\nconnected route", S1),
    ]
    xs = [0.1 + i * 2.15 for i in range(len(steps))]
    for x, (t, b, c) in zip(xs, steps):
        box(x, y, w, h, t, b, c)
    for x in xs[:-1]:
        arrow(x + w, y + h / 2, x + 2.15, y + h / 2)
    # evaluation (server-side only)
    box(10.85, 0.15, 2.05, 1.85, "Evaluation\n(server only)", "reads .route AFTER\nprediction; arc\nP / R / F1 returned\nin response", "#6250d6")
    arrow(xs[-1] + w / 2, y, 11.9, 2.0)
    # UI
    box(0.1, 0.15, 3.4, 0.9, "Streamlit + Folium UI", "GET /trips   POST /predict {trip_id}", S2)
    box(4.0, 0.15, 3.6, 0.9, "JSON response", "[lat, lon] GPS + matched coords,\narc ids, metrics, method, runtime_ms", S2)
    arrow(3.5, 0.6, 4.0, 0.6)
    arrow(10.85, 0.6, 7.6, 0.6, "HTTP")
    ax.text(6.0, 2.15, "No .route data enters steps 1-6 (enforced by tests and a no-.route re-run)", ha="center",
            fontsize=8, color="#6250d6", style="italic")
    fig.tight_layout(); fig.savefig(FIG / "fig_architecture.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- C. decoder tuned vs untuned
def decoder_tuning():
    rows = []
    t = pd.read_csv(ROOT / "experiments/optuna_trials.csv")
    rows.append(("Tune 64 (selection set)", float(t.loc[t.number == 0, "value"].iloc[0]), float(t.value.max()),
                 "experiments/optuna_trials.csv (trial 0 vs best)"))
    for split, f in (("Holdout 16", "compare_baseline_v1_vs_tuned_candidate_holdout.csv"),
                     ("Student test 10", "compare_baseline_v1_vs_tuned_v2_student_test.csv")):
        d = pd.read_csv(ROOT / "experiments" / f)
        b = [c for c in d.columns if c.startswith("arc_f1__") and "baseline" not in c and "delta" not in c][0]
        rows.append((split, float(d["arc_f1__baseline_v1"].mean()), float(d[b].mean()), f"experiments/{f}"))
    out = pd.DataFrame(rows, columns=["split", "baseline_v1", "tuned_v2", "source"])
    out.to_csv(DATA / "decoder_tuned_vs_untuned.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(out)); h = 0.38
    ax.barh(y - h / 2, out.baseline_v1, height=h - 0.03, color=S2, label="untuned (baseline_v1)")
    ax.barh(y + h / 2, out.tuned_v2, height=h - 0.03, color=S1, label="Optuna-tuned (tuned_v2)")
    for yi, v in list(zip(y - h / 2, out.baseline_v1)) + list(zip(y + h / 2, out.tuned_v2)):
        ax.text(v + 0.003, yi, f"{v:.4f}", va="center", fontsize=8)
    ax.set_yticks(y, out.split); ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlim(0.8, 1.0)
    ax.set_xlabel("mean route arc F1 (geometric-emission HMM)")
    ax.set_title("HMM decoder: untuned vs Bayesian-tuned")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.text(0.01, 0.01, "Source: reports/data/decoder_tuned_vs_untuned.csv (from experiments/optuna_trials.csv and "
             "experiments/compare_baseline_v1_vs_*.csv).", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(FIG / "fig_decoder_tuned_vs_untuned.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- D. student-test route maps
def route_maps():
    import folium
    from matplotlib.lines import Line2D

    from src.config import get_config
    from src.data_loader import load_trip
    from src.evaluation import load_reference_route, route_metrics
    from src.pipeline import predict_trip
    rec = pd.read_csv(ROOT / "experiments/compare_tuned_v2_vs_ml_v3_student_test.csv",
                      dtype={"trip_id": str}).set_index("trip_id")
    gain = (rec["arc_f1__ml_v3"] - rec["arc_f1__tuned_v2"])
    picks = {gain.idxmax(): "largest gain of RF + HMM over geometric HMM",
             rec["arc_f1__ml_v3"].idxmin(): "lowest RF + HMM F1 on student test"}
    cfg = get_config("ml_v3")
    checks = []
    for tid, why in picks.items():
        tid = str(tid).zfill(8)
        trip = load_trip(tid, ROOT / "Student folder")
        res, net, _ = predict_trip(trip, "hmm", config=cfg)
        ref = load_reference_route(tid, ROOT / "Student folder")
        m = route_metrics(res.route_arc_ids, ref, trip.arcs, net.arc_len)
        recorded = float(rec.loc[tid, "arc_f1__ml_v3"])
        checks.append({"trip_id": tid, "why": why, "arc_f1_regenerated": m["arc_f1"], "arc_f1_recorded": recorded,
                       "match": abs(m["arc_f1"] - recorded) < 1e-9})
        pred, refset = res.route_arc_ids, set(ref.tolist())
        predset = set(pred)
        lon, lat = trip.track[:, 0], trip.track[:, 1]
        pad = 0.002
        inbox = ((trip.nodes[trip.arcs[:, 0], 0] > lon.min() - pad) & (trip.nodes[trip.arcs[:, 0], 0] < lon.max() + pad) &
                 (trip.nodes[trip.arcs[:, 0], 1] > lat.min() - pad) & (trip.nodes[trip.arcs[:, 0], 1] < lat.max() + pad))
        net_arcs = np.flatnonzero(inbox)
        ll = lambda n: [trip.nodes[n, 1], trip.nodes[n, 0]]
        # Folium (interactive)
        fm = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=15, tiles="OpenStreetMap")
        layers = {k: folium.FeatureGroup(k) for k in ("Road network (.arcs)", "Reference route (.route)",
                                                     "Predicted: in .route", "Predicted: not in .route",
                                                     "Reference arcs missed", "GPS observations (.track)")}
        for a in net_arcs:
            u, v = trip.arcs[a]
            folium.PolyLine([ll(u), ll(v)], color="#9a9890", weight=1, opacity=0.6).add_to(layers["Road network (.arcs)"])
        for a in ref:
            u, v = trip.arcs[a]
            folium.PolyLine([ll(u), ll(v)], color=S3, weight=9, opacity=0.45).add_to(layers["Reference route (.route)"])
        for a in pred:
            u, v = trip.arcs[a]
            ok = a in refset
            folium.PolyLine([ll(u), ll(v)], color=S1 if ok else S2, weight=4 if ok else 5, tooltip=f"arc {a}").add_to(
                layers["Predicted: in .route" if ok else "Predicted: not in .route"])
        for a in refset - predset:
            u, v = trip.arcs[a]
            folium.PolyLine([ll(u), ll(v)], color="#6250d6", weight=5, dash_array="6").add_to(layers["Reference arcs missed"])
        step = max(1, len(trip.track) // 800)
        for lo, la, ts in trip.track[::step]:
            folium.CircleMarker([la, lo], radius=2, color=INK2, fill=True, weight=0, fill_opacity=0.7,
                                tooltip=f"t={ts:.0f}").add_to(layers["GPS observations (.track)"])
        for g in layers.values():
            g.add_to(fm)
        folium.LayerControl(collapsed=False).add_to(fm)
        legend = (f"<b>Trip {tid}</b> (student_test, {why})<br>RF + HMM arc F1 {m['arc_f1']:.4f}<br>"
                  "grey = road network, green = .route, blue = predicted in .route,<br>"
                  "orange = predicted not in .route, purple dashed = missed, dots = GPS")
        fm.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;bottom:16px;left:16px;z-index:9999;background:#fcfcfb;padding:8px 10px;'
            f'font:12px sans-serif;border:1px solid #bbb">{legend}</div>'))
        fm.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
        fm.save(str(FIG / "maps" / f"student_test_{tid}_route_map.html"))
        # static rendering of the same layers
        span_x = (lon.max() - lon.min() + pad) * np.cos(np.radians(lat.mean()))
        span_y = lat.max() - lat.min() + pad
        fw = float(np.clip(7.0 * span_x / span_y, 4.8, 9.0))
        fig, ax = plt.subplots(figsize=(fw + 1.2, 7.6))
        for a in net_arcs:
            u, v = trip.arcs[a]; ax.plot(*trip.nodes[[u, v]].T, color="#bdbbb3", lw=0.6, zorder=0)
        for a in ref:
            u, v = trip.arcs[a]; ax.plot(*trip.nodes[[u, v]].T, color=S3, lw=6, alpha=0.45, solid_capstyle="round", zorder=1)
        for a in pred:
            u, v = trip.arcs[a]
            ax.plot(*trip.nodes[[u, v]].T, color=S1 if a in refset else S2, lw=1.8 if a in refset else 2.6, zorder=3)
        for a in refset - predset:
            u, v = trip.arcs[a]; ax.plot(*trip.nodes[[u, v]].T, color="#6250d6", lw=2.2, ls="--", zorder=3)
        ax.scatter(lon[::step], lat[::step], s=2, color=INK2, zorder=2)
        ax.set_xlim(lon.min() - pad / 2, lon.max() + pad / 2); ax.set_ylim(lat.min() - pad / 2, lat.max() + pad / 2)
        ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
        ax.set_title(f"Trip {tid} (student_test)\nRF + HMM arc F1 {m['arc_f1']:.4f}", fontsize=10.5)
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.ticklabel_format(useOffset=False)
        ax.legend(handles=[Line2D([], [], color="#bdbbb3", lw=1, label="road network (.arcs)"),
                           Line2D([], [], color=S3, lw=6, alpha=0.45, label="reference (.route)"),
                           Line2D([], [], color=S1, lw=2, label="predicted, in .route"),
                           Line2D([], [], color=S2, lw=2.6, label="predicted, not in .route"),
                           Line2D([], [], color="#6250d6", lw=2, ls="--", label="reference missed"),
                           Line2D([], [], color=INK2, marker=".", lw=0, label="GPS")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / f"fig_route_map_student_test_{tid}.png", dpi=170); plt.close(fig)
    pd.DataFrame(checks).to_csv(DATA / "route_map_f1_check.csv", index=False)
    print(pd.DataFrame(checks).to_string(index=False))


def copy_existing():
    src = {
        "experiments/final/figures": ["fig_cv_model_comparison.png", "fig_student_test_top1_macro_f1.png",
                                      "fig_decoder_optuna_history.png", "fig_rf_optuna_history.png",
                                      "fig_ablation_route_f1.png", "fig_classifier_only_vs_hmm.png",
                                      "fig_per_trip_delta.png", "fig_fp_breakdown_student_test.png",
                                      "fig_confusion_matrices.png", "fig_train_vs_cv.png"],
        "experiments/final_evaluation": ["route_f1_comparison.png", "runtime_distribution.png"],
        "experiments/ml/alignment_plots": ["alignment_00000018.png"],
    }
    for d, files in src.items():
        for f in files:
            shutil.copy(ROOT / d / f, FIG / f)
    trim_whitespace(FIG / "alignment_00000018.png")


def trim_whitespace(path, pad=12):
    """Crop uniform background margins (layout only; pixels inside are untouched)."""
    img = plt.imread(path)
    rgb = img[..., :3]
    bg = rgb[0, 0]
    mask = np.abs(rgb - bg).sum(axis=2) > 0.03
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
    plt.imsave(path, img[y0:y1, x0:x1])


if __name__ == "__main__":
    (FIG / "maps").mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    dataset_overview()
    architecture()
    decoder_tuning()
    route_maps()
    copy_existing()
