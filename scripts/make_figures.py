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

RECORD: list[dict] = []   # every plotted value -> figures/plotted_values.csv


def rec(fig, series, labels, values, source):
    for l, v in zip(labels, values):
        RECORD.append({"figure": fig, "series": series, "label": str(l), "value": float(v), "source": source})


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
    rec(name, "trial objective", d["number"].astype(int), d[value_col], str(csv.relative_to(EXPERIMENT_DIR.parent)))
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
                xytext=(-10, 10), textcoords="offset points", ha="right",
                fontsize=8, color=INK)
    ax.set_xlabel("trial"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(loc="lower right" if not minimize else "upper right")
    save(fig, name, note)


def fig_cv_models():
    f = pd.read_csv(ML / "cv_fold_metrics.csv")
    g = f.groupby("model")
    order = ["logreg", "decision_tree", "extra_trees", "random_forest", "hist_gb"]
    name = "fig_cv_model_comparison.png"
    src = "experiments/ml/cv_fold_metrics.csv"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    mf, ms = g["macro_f1"].mean()[order].values, g["macro_f1"].std()[order].values
    hbar(axes[0], [NICE[m] for m in order], mf, S1, err=ms, xlim=(0.8, 0.95))
    rec(name, "cv macro F1 mean", order, mf, src); rec(name, "cv macro F1 sd", order, ms, src)
    axes[0].set_title("CV macro F1 (candidate level)"); axes[0].set_xlabel("mean ± sd over 5 trip-grouped folds")
    lm, ls_ = g["log_loss"].mean()[order].values, g["log_loss"].std()[order].values
    hbar(axes[1], [NICE[m] for m in order], lm, S1, err=ls_, xlim=(0, 0.25))
    rec(name, "cv log loss mean", order, lm, src); rec(name, "cv log loss sd", order, ls_, src)
    axes[1].set_title("CV log loss (lower is better)"); axes[1].set_xlabel("mean ± sd over 5 folds")
    axes[1].set_yticklabels([])
    save(fig, name,
         "Source: experiments/ml/cv_fold_metrics.csv. 64 tune trips, GroupKFold(5) by trip; "
         "labels are derived pseudo-labels (valid points only).")


def fig_student_test_models():
    d = pd.read_csv(ML / "student_test_model_report.csv")
    d = d[d["model"].str.endswith("__tune64")].copy()
    d["name"] = d["model"].str.replace("__tune64", "").map(NICE)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    src = "experiments/ml/student_test_model_report.csv"
    rec("fig_student_test_top1_macro_f1.png", "top1", d["model"], d["clf_pseudo_top1_accuracy"], src)
    rec("fig_student_test_top1_macro_f1.png", "macro_f1", d["model"], d["clf_macro_f1"], src)
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
        rec("fig_ablation_route_f1.png", sname, [f"{v}|{m}" for v, m, _ in rows], vals,
            "experiments/final/ablation_summary.csv")
        ax.barh(y + (k - 1) * h, vals, height=h - 0.03, color=col, label=lab)
        for yi, v in zip(y + (k - 1) * h, vals):
            ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, [r[2] for r in rows]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlim(0.3, 1.05); ax.set_xlabel("mean route-level arc F1 vs .route")
    ax.set_title("Ablation: what each component contributes (equal data)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
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
        rec("fig_classifier_only_vs_hmm.png", sname, [f"classifier_only|{m}" for m in order], co,
            "experiments/final/ablation_summary.csv")
        rec("fig_classifier_only_vs_hmm.png", sname, [f"ml_hmm|{m}" for m in order], hm,
            "experiments/final/ablation_summary.csv")
        ax.barh(y - h / 2, co, height=h - 0.03, color=S2, label="classifier only")
        ax.barh(y + h / 2, hm, height=h - 0.03, color=S1, label="classifier + HMM")
        for yi, v in list(zip(y - h / 2, co)) + list(zip(y + h / 2, hm)):
            ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=7.5)
        geo = s[(s.eval_set == sname) & (s.variant == "geometric_hmm")]["arc_f1"].iloc[0]
        ax.axvline(geo, color=INK2, lw=1.2, ls="--")
        ax.set_yticks(y, [NICE[m] for m in order]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        ax.set_xlim(0.4, 1.05); ax.set_xlabel("mean route arc F1")
        ax.set_title(f"{lab}\ndashed line = geometric HMM ({geo:.3f})", fontsize=10.5)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.14), ncol=2)
    save(fig, "fig_classifier_only_vs_hmm.png",
         "Source: experiments/final/ablation_summary.csv. All classifiers trained on the same 64 tune trips.")


def fig_per_trip_delta():
    d = pd.read_csv(FINAL / "ablation_per_trip.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (sname, lab) in zip(axes, [("holdout16", "Holdout 16"), ("student_test10", "Student test 10")]):
        a = d[(d.eval_set == sname) & (d.variant == "geometric_hmm")].set_index("trip_id")["arc_f1"]
        b = d[(d.eval_set == sname) & (d.variant == "ml_hmm") & (d.model == "random_forest")].set_index("trip_id")["arc_f1"]
        delta = (b - a).sort_values()
        rec("fig_per_trip_delta.png", sname, delta.index, delta.values, "experiments/final/ablation_per_trip.csv")
        cols = [S1 if v > 1e-9 else (S2 if v < -1e-9 else GRID) for v in delta]
        ax.bar(range(len(delta)), delta.values, color=cols, width=0.7)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(range(len(delta)), [str(int(t)).zfill(8)[-2:] for t in delta.index], fontsize=8)
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
        rec("fig_fp_breakdown_student_test.png", f"{v}|{m}", [c for c, _ in cats], vals,
            "experiments/final/ablation_per_trip.csv")
        ax.barh(y + (k - 0.5) * h, vals, height=h - 0.03, color=col, label=lab)
        for yi, val in zip(y + (k - 0.5) * h, vals):
            ax.text(val + 1, yi, str(val), va="center", fontsize=8)
    ax.set_yticks(y, [c[1] for c in cats]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("false-positive arcs, summed over 10 student-test trips")
    ax.set_title("Where the extra arcs come from")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
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


def confusion_and_fit_tables():
    """From STORED out-of-fold predictions and SAVED 64-trip models (no training):
    confusion matrices (threshold 0.5) and training-vs-CV metrics. Writes the
    tables used by the figures to experiments/final/ (committed evidence)."""
    import json as _json

    import joblib
    from sklearn.metrics import confusion_matrix, f1_score, log_loss

    from src.ml_emission import load_model_bundle
    from src.ml_train import stack
    ds = joblib.load(ML / "dataset.joblib")
    tune = sorted(_json.load(open(EXPERIMENT_DIR / "train_partition.json"))["tune"])
    cm_rows, fit_rows = [], []
    for m in ["logreg", "decision_tree", "random_forest", "extra_trees", "hist_gb"]:
        oof = joblib.load(ML / f"oof_{m}.joblib")
        y = np.concatenate([ds["trips"][t]["y"][ds["trips"][t]["row_valid"]] for t in tune])
        p = np.concatenate([oof[t][ds["trips"][t]["row_valid"]] for t in tune])
        tn, fp, fn, tp = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1]).ravel()
        cm_rows.append({"model": m, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                        "source": f"experiments/ml/oof_{m}.joblib (5-fold OOF, 64 tune trips, valid rows)"})
        X, ytr, _ = stack(ds, tune)
        ptr = load_model_bundle(f"{m}__tune64")["model"].predict_proba(X)[:, 1]
        fit_rows.append({"model": m,
                         "train_macro_f1": round(f1_score(ytr, ptr >= 0.5, average="macro"), 4),
                         "train_log_loss": round(log_loss(ytr, np.clip(ptr, 1e-7, 1 - 1e-7), labels=[0, 1]), 4),
                         "cv_macro_f1": round(f1_score(y, p >= 0.5, average="macro"), 4),
                         "cv_log_loss": round(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]), 4),
                         "train_source": f"models/{m}__tune64.joblib scored on its own 64 training trips",
                         "cv_source": f"experiments/ml/oof_{m}.joblib"})
    pd.DataFrame(cm_rows).to_csv(FINAL / "confusion_matrices_oof.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(FINAL / "train_vs_cv_metrics.csv", index=False)


def fig_confusion():
    d = pd.read_csv(FINAL / "confusion_matrices_oof.csv")
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, (_, r) in zip(axes, d.iterrows()):
        mat = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        rec("fig_confusion_matrices.png", r.model, ["tn", "fp", "fn", "tp"], mat.ravel(),
            "experiments/final/confusion_matrices_oof.csv")
        norm = mat / mat.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:,}\n{norm[i, j]:.1%}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else INK)
        ax.set_xticks([0, 1], ["pred wrong", "pred road"]); ax.set_yticks([0, 1], ["true wrong", "true road"])
        ax.set_title(NICE[r.model], fontsize=10); ax.grid(False)
    save(fig, "fig_confusion_matrices.png",
         "Source: experiments/final/confusion_matrices_oof.csv (out-of-fold, threshold 0.5, 64 tune trips, "
         "pseudo-labels). Row-normalised colour; counts shown.")


def fig_train_vs_cv():
    d = pd.read_csv(FINAL / "train_vs_cv_metrics.csv")
    order = ["logreg", "decision_tree", "extra_trees", "random_forest", "hist_gb"]
    d = d.set_index("model").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (tr, cv, title, lim) in zip(axes, [("train_macro_f1", "cv_macro_f1", "Macro F1: training vs CV", (0.8, 1.0)),
                                               ("train_log_loss", "cv_log_loss", "Log loss: training vs CV", (0, 0.2))]):
        y = np.arange(len(order)); h = 0.38
        rec("fig_train_vs_cv.png", tr, order, d[tr].values, "experiments/final/train_vs_cv_metrics.csv")
        rec("fig_train_vs_cv.png", cv, order, d[cv].values, "experiments/final/train_vs_cv_metrics.csv")
        ax.barh(y - h / 2, d[tr], height=h - 0.03, color=S2, label="training (in-sample)")
        ax.barh(y + h / 2, d[cv], height=h - 0.03, color=S1, label="5-fold CV (out-of-fold)")
        for yi, v in list(zip(y - h / 2, d[tr])) + list(zip(y + h / 2, d[cv])):
            ax.text(v, yi, f" {v:.3f}", va="center", fontsize=7.5)
        ax.set_yticks(y, [NICE[m] for m in order]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        ax.set_xlim(*lim); ax.set_title(title)
    axes[1].set_yticklabels([])
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.12), ncol=2, fontsize=9)
    save(fig, "fig_train_vs_cv.png",
         "Source: experiments/final/train_vs_cv_metrics.csv. Training = saved 64-trip model on its own rows; "
         "CV = out-of-fold. A large gap indicates overfitting.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_optuna_history(EXPERIMENT_DIR / "optuna_trials.csv", "value",
                       "Decoder Bayesian optimisation (Optuna TPE, 150 trials)",
                       "mean route arc F1 (64 tune trips)", "fig_decoder_optuna_history.png",
                       note="Source: experiments/optuna_trials.csv. Trial 0 = baseline_v1.")
    rf_csv = ML / "rf_optuna_trials.csv"
    if not rf_csv.exists():
        rf_csv = ML / "rf_optuna_trials_partial.csv"
    if rf_csv.exists():
        n_done = int((pd.read_csv(rf_csv)["state"] == "COMPLETE").sum())
        fig_optuna_history(rf_csv, "value",
                           f"Random Forest Bayesian optimisation (TPE)\nsearch stopped after {n_done} completed trials",
                           "mean 5-fold log loss (64 tune trips)", "fig_rf_optuna_history.png",
                           minimize=True,
                           note=f"Source: {rf_csv.relative_to(EXPERIMENT_DIR.parent)} (completed trials only). "
                                "Trial 0 = untuned; tuned model NOT deployed.")
    fig_cv_models()
    fig_student_test_models()
    fig_ablation()
    fig_per_trip_delta()
    fig_fp_breakdown()
    fig_rf_tuning()
    if (ML / "dataset.joblib").exists():
        confusion_and_fit_tables()
    fig_confusion()
    fig_train_vs_cv()
    pd.DataFrame(RECORD).to_csv(OUT / "plotted_values.csv", index=False)
    print(f"recorded {len(RECORD)} plotted values")


if __name__ == "__main__":
    main()
