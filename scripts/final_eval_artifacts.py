"""Comparison table, plots and route maps for the final evaluation.

  python -m scripts.final_eval_artifacts   (run after scripts.final_evaluation)
Every number is read from recorded CSV/JSON files; nothing is typed in.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment import EXPERIMENT_DIR

FE = EXPERIMENT_DIR / "final_evaluation"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "axes.axisbelow": True, "legend.frameon": False})


def stats_row(split, n, model, clf_train, decoder, s, source, sp=None, sr=None):
    s = pd.Series(s, dtype=float)
    return {"split": split, "n_trips": n, "model": model, "classifier_training_trips": clf_train,
            "decoder_params": decoder, "arc_f1_mean": round(s.mean(), 4), "arc_f1_median": round(s.median(), 4),
            "arc_f1_min": round(s.min(), 4), "arc_f1_std": round(s.std(ddof=1), 4),
            "arc_precision_mean": None if sp is None else round(pd.Series(sp).mean(), 4),
            "arc_recall_mean": None if sr is None else round(pd.Series(sr).mean(), 4),
            "source_file": source}


def comparison():
    rows = []
    dec = "tuned_v2 (Optuna on 64 tune trips)"
    # tune64 OOF (classifier OOF within the 64 tune trips; summary means only were recorded)
    g = pd.read_csv(EXPERIMENT_DIR / "ml" / "oof_hmm_comparison.csv")
    e = pd.read_csv(EXPERIMENT_DIR / "ml" / "emission_grid_random_forest.csv")
    geo = g[g.model.str.startswith("tuned_v2")].iloc[0]
    rf = e[(e.emission == "ml") & (e.ml_weight == 4.0)].iloc[0]
    for name, r, tr, src in (("geometric HMM", geo, "none", "experiments/ml/oof_hmm_comparison.csv"),
                             ("RF + HMM (w=4)", rf, "5-fold OOF inside 64 (~51 per fold)",
                              "experiments/ml/emission_grid_random_forest.csv")):
        rows.append({"split": "train: tune64 (OOF; also used to select w and decoder params -> optimistic)",
                     "n_trips": int(r.n_trips), "model": name, "classifier_training_trips": tr,
                     "decoder_params": dec, "arc_f1_mean": r.arc_f1_mean, "arc_f1_median": r.arc_f1_median,
                     "arc_f1_min": None, "arc_f1_std": None, "arc_precision_mean": r.arc_precision_mean,
                     "arc_recall_mean": r.arc_recall_mean, "source_file": src})
    h = pd.read_csv(EXPERIMENT_DIR / "compare_tuned_v2_vs_ml_candidate_holdout.csv")
    src = "experiments/compare_tuned_v2_vs_ml_candidate_holdout.csv"
    rows.append(stats_row("train: holdout16 (never used for selection)", len(h), "geometric HMM", "none", dec,
                          h["arc_f1__tuned_v2"], src, h["arc_precision__tuned_v2"], h["arc_recall__tuned_v2"]))
    rows.append(stats_row("train: holdout16 (never used for selection)", len(h), "RF + HMM", "64 tune", dec,
                          h["arc_f1__ml_candidate"], src, h["arc_precision__ml_candidate"], h["arc_recall__ml_candidate"]))
    s = pd.read_csv(EXPERIMENT_DIR / "compare_tuned_v2_vs_ml_v3_student_test.csv")
    src = "experiments/compare_tuned_v2_vs_ml_v3_student_test.csv"
    rows.append(stats_row("student_test10", len(s), "geometric HMM", "none", dec, s["arc_f1__tuned_v2"], src,
                          s["arc_precision__tuned_v2"], s["arc_recall__tuned_v2"]))
    st = pd.read_csv(EXPERIMENT_DIR / "ml" / "student_test_model_report.csv").set_index("model")
    r = st.loc["random_forest__tune64"]
    rows.append({"split": "student_test10", "n_trips": int(r.n_trips), "model": "RF + HMM",
                 "classifier_training_trips": "64 tune", "decoder_params": dec,
                 "arc_f1_mean": r.arc_f1_mean, "arc_f1_median": r.arc_f1_median, "arc_f1_min": None,
                 "arc_f1_std": None, "arc_precision_mean": r.arc_precision_mean,
                 "arc_recall_mean": r.arc_recall_mean,
                 "source_file": "experiments/ml/student_test_model_report.csv"})
    rows.append(stats_row("student_test10", len(s), "RF + HMM (DEPLOYED ml_v3)", "80 train", dec,
                          s["arc_f1__ml_v3"], src, s["arc_precision__ml_v3"], s["arc_recall__ml_v3"]))
    ev = pd.read_csv(FE / "evaluation_per_trip.csv")
    names = {"nearest_road": ("nearest road (no classifier, no HMM)", "none"),
             "classifier_only_rf80": ("RF only (no HMM)", "80 train"),
             "geometric_hmm_tuned_v2": ("geometric HMM", "none"),
             "rf64_hmm_ml_candidate": ("RF + HMM", "64 tune"),
             "rf80_hmm_deployed": ("RF + HMM (DEPLOYED ml_v3)", "80 train")}
    for v, (nm, tr) in names.items():
        d = ev[ev.variant == v]
        rows.append(stats_row("evaluator_test10 (final, frozen model)", len(d), nm, tr, dec, d.arc_f1,
                              "experiments/final_evaluation/evaluation_per_trip.csv", d.arc_precision, d.arc_recall))
    df = pd.DataFrame(rows)
    df.to_csv(FE / "comparison_with_previous_splits.csv", index=False)
    print(df[["split", "model", "classifier_training_trips", "n_trips", "arc_f1_mean", "arc_f1_median"]].to_string(index=False))


def plots():
    ev = pd.read_csv(FE / "evaluation_per_trip.csv")
    # runtime distribution
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for v, lab, c, mk in (("geometric_hmm_tuned_v2", "geometric HMM", S2, "o"),
                          ("rf80_hmm_deployed", "RF + HMM (deployed)", S1, "s")):
        d = ev[ev.variant == v]
        ax.scatter(d.n_gps_points, d.runtime_ms_server, s=40, color=c, marker=mk, label=lab,
                   edgecolors=SURFACE, linewidths=1.5, zorder=3)
    d = ev[ev.variant == "rf80_hmm_deployed"]
    ax.set_xlabel("GPS points in trip"); ax.set_ylabel("server runtime per request (ms)")
    ax.set_title(f"Runtime per evaluator trip (deployed: median {d.runtime_ms_server.median():.0f} ms, "
                 f"max {d.runtime_ms_server.max():.0f} ms)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.text(0.01, 0.01, "Source: experiments/final_evaluation/evaluation_per_trip.csv; Apple M5, in-process "
             "FastAPI TestClient, model pre-loaded.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(FE / "runtime_distribution.png", dpi=200); plt.close(fig)

    # route F1 per trip, paired
    a = ev[ev.variant == "geometric_hmm_tuned_v2"].set_index("trip_id").arc_f1
    b = ev[ev.variant == "rf80_hmm_deployed"].set_index("trip_id").arc_f1
    order = b.sort_values().index
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(order))
    for yi, t in zip(y, order):
        ax.plot([a[t], b[t]], [yi, yi], color=GRID, lw=2, zorder=1)
    ax.scatter(a[order], y, s=46, color=S2, label=f"geometric HMM (mean {a.mean():.3f})", zorder=3,
               edgecolors=SURFACE, linewidths=1.5)
    ax.scatter(b[order], y, s=46, color=S1, marker="s", label=f"RF + HMM deployed (mean {b.mean():.3f})",
               zorder=3, edgecolors=SURFACE, linewidths=1.5)
    ax.set_yticks(y, [str(int(t)).zfill(8) for t in order]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("route-level arc F1 vs .route (directed arc sets)")
    s = json.load(open(FE / "evaluation_summary.json"))["deployed_vs_geometric_per_trip"]
    ax.set_title(f"Evaluator test: {s['better']} better, {s['worse']} worse, {s['same']} same "
                 f"(Wilcoxon p = {s['wilcoxon_p']})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.text(0.01, 0.01, "Source: experiments/final_evaluation/evaluation_per_trip.csv. Frozen models; "
             "no parameter changed after evaluation.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(FE / "route_f1_comparison.png", dpi=200); plt.close(fig)


def maps():
    import folium
    from src.data_loader import load_trip
    from src.evaluation import load_reference_route
    preds = json.load(open(FE / "predicted_routes.json"))["rf80_hmm_deployed"]
    ev = pd.read_csv(FE / "evaluation_per_trip.csv")
    d = ev[ev.variant == "rf80_hmm_deployed"].set_index("trip_id")
    a = ev[ev.variant == "geometric_hmm_tuned_v2"].set_index("trip_id").arc_f1
    worst = str(int(d.arc_f1.idxmin())).zfill(8)
    best_gain = str(int((d.arc_f1 - a).idxmax())).zfill(8)
    best = str(int(d.arc_f1.idxmax())).zfill(8)
    picks = {worst: "lowest F1", best_gain: "largest gain over geometric", best: "highest F1"}
    import os
    os.environ["PATHSNAP_DATA_DIR"] = str(EXPERIMENT_DIR.parent)
    for tid, why in picks.items():
        trip = load_trip(tid)
        ref = load_reference_route(tid)
        pred = preds[tid]
        refset, predset = set(ref.tolist()), set(pred)
        ll = lambda n: [trip.nodes[n, 1], trip.nodes[n, 0]]
        m = folium.Map(location=[trip.track[:, 1].mean(), trip.track[:, 0].mean()], zoom_start=15,
                       tiles="OpenStreetMap")
        layers = {k: folium.FeatureGroup(k) for k in
                  ("reference route (.route)", "predicted route: correct arcs",
                   "predicted route: extra arcs (FP)", "reference arcs missed (FN)", "GPS points")}
        for aid in ref:
            u, v = trip.arcs[aid]
            folium.PolyLine([ll(u), ll(v)], color="#1baf7a", weight=9, opacity=0.45).add_to(layers["reference route (.route)"])
        for aid in pred:
            u, v = trip.arcs[aid]
            ok = aid in refset
            folium.PolyLine([ll(u), ll(v)], color="#2a78d6" if ok else "#eb6834", weight=4 if ok else 5,
                            tooltip=f"arc {aid}" + ("" if ok else " (not in .route)")).add_to(
                layers["predicted route: correct arcs" if ok else "predicted route: extra arcs (FP)"])
        for aid in refset - predset:
            u, v = trip.arcs[aid]
            folium.PolyLine([ll(u), ll(v)], color="#6250d6", weight=5, dash_array="6",
                            tooltip=f"missed arc {aid}").add_to(layers["reference arcs missed (FN)"])
        step = max(1, len(trip.track) // 800)
        for lon, lat, ts in trip.track[::step]:
            folium.CircleMarker([lat, lon], radius=2, color="#52514e", fill=True, weight=0,
                                fill_opacity=0.7, tooltip=f"t={ts:.0f}").add_to(layers["GPS points"])
        for g in layers.values():
            g.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        title = (f"Trip {tid} ({why}): F1 {d.loc[int(tid), 'arc_f1']:.4f}, P {d.loc[int(tid), 'arc_precision']:.4f}, "
                 f"R {d.loc[int(tid), 'arc_recall']:.4f} — green=.route, blue=predicted correct, "
                 f"orange=extra, purple dashed=missed")
        m.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:8px;left:50px;z-index:9999;background:#fcfcfb;padding:6px 10px;'
            f'font:13px sans-serif;border:1px solid #ccc">{title}</div>'))
        m.save(str(FE / "maps" / f"trip_{tid}_overlay.html"))

        # static PNG for the report
        fig, ax = plt.subplots(figsize=(7, 7))
        for aid in ref:
            u, v = trip.arcs[aid]; ax.plot(*trip.nodes[[u, v]].T, color=S3, lw=6, alpha=0.45, solid_capstyle="round")
        for aid in pred:
            u, v = trip.arcs[aid]
            ax.plot(*trip.nodes[[u, v]].T, color=S1 if aid in refset else S2, lw=1.8 if aid in refset else 2.6)
        for aid in refset - predset:
            u, v = trip.arcs[aid]; ax.plot(*trip.nodes[[u, v]].T, color="#6250d6", lw=2.2, ls="--")
        ax.scatter(trip.track[::step, 0], trip.track[::step, 1], s=2, color=INK2, zorder=0)
        ax.set_aspect(1 / np.cos(np.radians(trip.track[:, 1].mean())))
        ax.set_title(f"Trip {tid} — {why}\nF1 {d.loc[int(tid), 'arc_f1']:.4f}", fontsize=11)
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color=S3, lw=6, alpha=0.45, label=".route (reference)"),
                           Line2D([], [], color=S1, lw=2, label="predicted, in .route"),
                           Line2D([], [], color=S2, lw=2.6, label="predicted, not in .route"),
                           Line2D([], [], color="#6250d6", lw=2, ls="--", label="missed"),
                           Line2D([], [], color=INK2, marker=".", lw=0, label="GPS")],
                  loc="best", fontsize=8, frameon=True, facecolor=SURFACE)
        fig.tight_layout(); fig.savefig(FE / "maps" / f"trip_{tid}_overlay.png", dpi=160); plt.close(fig)
        print("map", tid, why)


if __name__ == "__main__":
    comparison()
    plots()
    maps()
