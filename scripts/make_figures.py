"""Report figures generated ONLY from recorded experiment CSVs.

  python -m scripts.make_figures      -> experiments/final/figures/*.png
Every plotted number is read from a file under experiments/; nothing is typed in.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment import EXPERIMENT_DIR

OUT = EXPERIMENT_DIR / "final" / "figures"
ML = EXPERIMENT_DIR / "ml"
FINAL = EXPERIMENT_DIR / "final"

# validated reference palette (light): first three categorical slots pass all-pairs CVD checks
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
NICE = {"logreg": "Logistic Regression", "decision_tree": "Decision Tree",
        "random_forest": "Random Forest", "extra_trees": "Extra Trees", "hist_gb": "HistGradientBoosting"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False,
})


def save(fig, name, note):
    fig.text(0.01, 0.01, note, fontsize=7.5, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", name)


def hbar(ax, labels, values, color, fmt="{:.4f}", err=None, xlim=None):
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.6, color=color, xerr=err,
            error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    if xlim:
        ax.set_xlim(*xlim)
    for yi, v, e in zip(y, values, err if err is not None else [0] * len(values)):
        ax.text(v + (e or 0) + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01, yi, fmt.format(v),
                va="center", fontsize=8.5, color=INK)


def fig_optuna_history(csv, value_col, title, ylabel, name, minimize=False, note=""):
    d = pd.read_csv(csv)
    if "state" in d:
        d = d[d["state"] == "COMPLETE"]
    d = d.sort_values("number")
    best = d[value_col].cummin() if minimize else d[value_col].cummax()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.scatter(d["number"], d[value_col], s=16, color=S1, alpha=0.55, label="trial objective",
               edgecolors="none")
    ax.step(d["number"], best, where="post", color=S2, lw=2, label="best so far")
    first = d.iloc[0]
    ax.annotate(f"trial 0 (untuned): {first[value_col]:.4f}", (first["number"], first[value_col]),
                xytext=(10, -18 if not minimize else 12), textcoords="offset points", fontsize=8, color=INK2)
    b = d.loc[d[value_col].idxmin() if minimize else d[value_col].idxmax()]
    ax.annotate(f"best #{int(b['number'])}: {b[value_col]:.4f}", (b["number"], b[value_col]),
                xytext=(-10, 10 if not minimize else -16), textcoords="offset points", ha="right",
                fontsize=8, color=INK)
    ax.set_xlabel("trial"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(loc="lower right" if not minimize else "upper right")
    save(fig, name, note)


def fig_cv_models():
    f = pd.read_csv(ML / "cv_fold_metrics.csv")
    g = f.groupby("model")
    order = ["logreg", "decision_tree", "extra_trees", "random_forest", "hist_gb"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    hbar(axes[0], [NICE[m] for m in order], g["f1"].mean()[order].values, S1,
         err=g["f1"].std()[order].values, xlim=(0.6, 0.95))
    axes[0].set_title("CV F1 (positive class)"); axes[0].set_xlabel("mean ± sd over 5 trip-grouped folds")
    hbar(axes[1], [NICE[m] for m in order], g["log_loss"].mean()[order].values, S1,
         err=g["log_loss"].std()[order].values, xlim=(0, 0.25))
    axes[1].set_title("CV log loss (lower is better)"); axes[1].set_xlabel("mean ± sd over 5 folds")
    axes[1].set_yticklabels([])
    save(fig, "fig_cv_model_comparison.png",
         "Source: experiments/ml/cv_fold_metrics.csv. 64 tune trips, GroupKFold(5) by trip; "
         "labels are derived pseudo-labels (valid points only).")


def fig_student_test_models():
    d = pd.read_csv(ML / "student_test_model_report.csv")
    d = d[d["model"].str.endswith("__tune64")].copy()
    d["name"] = d["model"].str.replace("__tune64", "").map(NICE)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    hbar(axes[0], d["name"], d["clf_pseudo_top1_accuracy"].values, S1, xlim=(0.7, 0.95))
    axes[0].set_title("Student-test Top-1 accuracy (classifier only)")
    axes[0].set_xlabel("fraction of GPS points whose top candidate = pseudo-label")
    hbar(axes[1], d["name"], d["clf_macro_f1"].values, S1, xlim=(0.7, 0.95))
    axes[1].set_title("Student-test macro F1 (candidate level)"); axes[1].set_yticklabels([])
    save(fig, "fig_student_test_top1_macro_f1.png",
         "Source: experiments/ml/student_test_model_report.csv. All models trained on the same 64 tune trips; "
         "10 student-test trips; pseudo-labels.")


def fig_ablation():
    s = pd.read_csv(FINAL / "ablation_summary.csv")
    rows = [("nearest_road", "-", "Nearest road (no classifier, no HMM)"),
            ("classifier_only", "random_forest", "Random Forest only (no HMM)"),
            ("geometric_hmm", "-", "Geometric HMM"),
            ("ml_hmm", "random_forest", "Random Forest + HMM")]
    sets = [("tune64_OOF", "Tune 64 (OOF)", S3), ("holdout16", "Holdout 16", S1),
            ("student_test10", "Student test 10", S2)]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = np.arange(len(rows)); h = 0.26
    for k, (sname, lab, col) in enumerate(sets):
        vals = [s[(s.eval_set == sname) & (s.variant == v) & (s.model == m)]["arc_f1"].iloc[0]
                for v, m, _ in rows]
        ax.barh(y + (k - 1) * h, vals, height=h - 0.03, color=col, label=lab)
        for yi, v in zip(y + (k - 1) * h, vals):
            ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, [r[2] for r in rows]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlim(0.3, 1.05); ax.set_xlabel("mean route-level arc F1 vs .route")
    ax.set_title("Ablation: what each component contributes (equal data)")
    ax.legend(loc="lower right")
    save(fig, "fig_ablation_route_f1.png",
         "Source: experiments/final/ablation_summary.csv. Fixed candidates (tuned_v2), decoder, route assembly; "
         "classifier trained on the same 64 tune trips (OOF within them).")

    # classifier-only vs +HMM for every model, holdout and student test
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    order = ["logreg", "decision_tree", "random_forest", "extra_trees", "hist_gb"]
    for ax, (sname, lab) in zip(axes, [("holdout16", "Holdout 16"), ("student_test10", "Student test 10")]):
        y = np.arange(len(order)); h = 0.38
        co = [s[(s.eval_set == sname) & (s.variant == "classifier_only") & (s.model == m)]["arc_f1"].iloc[0] for m in order]
        hm = [s[(s.eval_set == sname) & (s.variant == "ml_hmm") & (s.model == m)]["arc_f1"].iloc[0] for m in order]
        ax.barh(y - h / 2, co, height=h - 0.03, color=S2, label="classifier only")
        ax.barh(y + h / 2, hm, height=h - 0.03, color=S1, label="classifier + HMM")
        for yi, v in list(zip(y - h / 2, co)) + list(zip(y + h / 2, hm)):
            ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=7.5)
        geo = s[(s.eval_set == sname) & (s.variant == "geometric_hmm")]["arc_f1"].iloc[0]
        ax.axvline(geo, color=INK2, lw=1.2, ls="--")
        ax.text(geo, -0.75, f"geometric HMM {geo:.3f}", fontsize=7.5, color=INK2, ha="right")
        ax.set_yticks(y, [NICE[m] for m in order]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        ax.set_xlim(0.4, 1.05); ax.set_title(lab); ax.set_xlabel("mean route arc F1")
    axes[0].legend(loc="lower left")
    save(fig, "fig_classifier_only_vs_hmm.png",
         "Source: experiments/final/ablation_summary.csv. All classifiers trained on the same 64 tune trips.")


def fig_per_trip_delta():
    d = pd.read_csv(FINAL / "ablation_per_trip.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (sname, lab) in zip(axes, [("holdout16", "Holdout 16"), ("student_test10", "Student test 10")]):
        a = d[(d.eval_set == sname) & (d.variant == "geometric_hmm")].set_index("trip_id")["arc_f1"]
        b = d[(d.eval_set == sname) & (d.variant == "ml_hmm") & (d.model == "random_forest")].set_index("trip_id")["arc_f1"]
        delta = (b - a).sort_values()
        cols = [S1 if v > 1e-9 else (S2 if v < -1e-9 else GRID) for v in delta]
        ax.bar(range(len(delta)), delta.values, color=cols, width=0.7)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(range(len(delta)), [t[-2:] for t in delta.index], fontsize=8)
        ax.set_xlabel("trip (last two digits of id)"); ax.set_ylabel("Δ arc F1 (RF+HMM − geometric)")
        w, l = int((delta > 1e-9).sum()), int((delta < -1e-9).sum())
        ax.set_title(f"{lab}: {w} better, {l} worse, {len(delta) - w - l} same")
    save(fig, "fig_per_trip_delta.png",
         "Source: experiments/final/ablation_per_trip.csv. RF trained on the 64 tune trips (equal-data). "
         "Blue = improved, orange = worse.")


def fig_fp_breakdown():
    d = pd.read_csv(FINAL / "ablation_per_trip.csv")
    d = d[d.eval_set == "student_test10"]
    cats = [("fp_reverse_direction", "wrong direction"), ("fp_matched", "wrong road matched"),
            ("fp_gap_fill_only", "gap-fill only")]
    groups = [("geometric_hmm", "-", "Geometric HMM", S2), ("ml_hmm", "random_forest", "RF + HMM", S1)]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    y = np.arange(len(cats)); h = 0.38
    for k, (v, m, lab, col) in enumerate(groups):
        vals = [int(d[(d.variant == v) & (d.model == m)][c].sum()) for c, _ in cats]
        ax.barh(y + (k - 0.5) * h, vals, height=h - 0.03, color=col, label=lab)
        for yi, val in zip(y + (k - 0.5) * h, vals):
            ax.text(val + 1, yi, str(val), va="center", fontsize=8)
    ax.set_yticks(y, [c[1] for c in cats]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("false-positive arcs, summed over 10 student-test trips")
    ax.set_title("Where the extra arcs come from"); ax.legend(loc="lower right")
    save(fig, "fig_fp_breakdown_student_test.png",
         "Source: experiments/final/ablation_per_trip.csv (RF trained on 64 tune trips).")


def fig_rf_tuning():
    f = ML / "rf_tuning_comparison.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    sets = [("tune64_5fold_OOF", "Tune 64 (OOF)"), ("holdout16", "Holdout 16"), ("student_test10", "Student test 10")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (col, title, lim) in zip(axes, [("clf_log_loss", "Classifier log loss (lower better)", None),
                                            ("arc_f1_mean", "Route arc F1 with HMM", (0.9, 1.0))]):
        y = np.arange(len(sets)); h = 0.38
        for k, (m, lab, c) in enumerate([("rf_untuned", "untuned", S2), ("rf_tuned", "Optuna-tuned", S1)]):
            vals = [d[(d.model == m) & (d.eval_set == s)][col].iloc[0] for s, _ in sets]
            ax.barh(y + (k - 0.5) * h, vals, height=h - 0.03, color=c, label=lab)
            for yi, v in zip(y + (k - 0.5) * h, vals):
                ax.text(v, yi, f" {v:.4f}", va="center", fontsize=7.5)
        ax.set_yticks(y, [s[1] for s in sets]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        if lim:
            ax.set_xlim(*lim)
        ax.set_title(title)
    axes[0].legend(loc="lower right")
    save(fig, "fig_rf_untuned_vs_tuned.png",
         "Source: experiments/ml/rf_tuning_comparison.csv. Both trained on identical trips "
         "(OOF folds / 64 tune trips); same decoder.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_optuna_history(EXPERIMENT_DIR / "optuna_trials.csv", "value",
                       "Decoder Bayesian optimisation (Optuna TPE, 150 trials)",
                       "mean route arc F1 (64 tune trips)", "fig_decoder_optuna_history.png",
                       note="Source: experiments/optuna_trials.csv. Trial 0 = baseline_v1.")
    if (ML / "rf_optuna_trials.csv").exists():
        fig_optuna_history(ML / "rf_optuna_trials.csv", "value",
                           "Random Forest Bayesian optimisation (Optuna TPE)",
                           "mean 5-fold log loss (64 tune trips)", "fig_rf_optuna_history.png",
                           minimize=True,
                           note="Source: experiments/ml/rf_optuna_trials.csv (completed trials). "
                                "Trial 0 = untuned configuration.")
    fig_cv_models()
    fig_student_test_models()
    fig_ablation()
    fig_per_trip_delta()
    fig_fp_breakdown()
    fig_rf_tuning()


if __name__ == "__main__":
    main()
