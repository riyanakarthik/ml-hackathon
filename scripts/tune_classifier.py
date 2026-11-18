"""Bayesian hyperparameter optimisation of the selected classifier (Random Forest).

  python -m scripts.tune_classifier --trials 30 --timeout 3600
  python -m scripts.tune_classifier --stage evaluate      # untuned vs tuned, same partitions

Protocol
  * Only the 64 TUNE trips (experiments/train_partition.json); the same 5
    trip-grouped folds as the model comparison (experiments/ml/cv_folds.json),
    so no trip contributes rows to both the training and validation side of a fold.
  * Objective: mean validation log loss over the 5 folds (minimise). Log loss is a
    proper scoring rule; the probabilities are used directly as HMM emissions.
  * Optuna TPESampler(seed=42); MedianPruner prunes trials after any fold.
  * Holdout (16) and student_test (10) trips are never used here except in
    --stage evaluate, which only REPORTS untuned vs tuned on fixed partitions.
Outputs (experiments/ml/): rf_optuna_trials.csv, rf_optuna_study.db, rf_tuned_params.json,
  rf_tuning_comparison.csv
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

from src.config import get_config
from src.experiment import evaluate_config, load_partition, summarize
from src.ml_train import SEED, candidate_metrics, make_models, stack

OUT_NAME = "experiments/ml"
DATASET = f"{OUT_NAME}/dataset.joblib"


def rf_from_params(p: dict) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                  min_samples_leaf=p["min_samples_leaf"],
                                  max_features=p["max_features"], max_samples=p["max_samples"],
                                  bootstrap=True, n_jobs=-1, random_state=SEED)


FINAL_DIR = f"{OUT_NAME}/rf_bayesian_final"


def search(n_trials: int, timeout: int, resume: bool = False, target_complete: int | None = None):
    """resume=True continues the existing 'rf_classifier' study in rf_optuna_study.db with the SAME
    objective, search space, sampler settings and pruner; all earlier trials are kept. A trial left
    RUNNING by an interrupted process is marked FAIL (it never finished). Stops once
    `target_complete` trials are COMPLETE (or after n_trials / timeout)."""
    ds = joblib.load(DATASET)
    folds = json.load(open(f"{OUT_NAME}/cv_folds.json"))
    tune = sorted(load_partition()["tune"])
    assert sorted(t for f in folds for t in f) == tune
    data = []
    for val in folds:
        tr = [t for t in tune if t not in val]
        Xtr, ytr, _ = stack(ds, tr)
        Xva, yva, _ = stack(ds, val)
        data.append((Xtr, ytr, Xva, yva))

    def objective(trial: optuna.Trial) -> float:
        p = {"n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
             "max_depth": trial.suggest_int("max_depth", 6, 30),
             "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 200, log=True),
             "max_features": trial.suggest_categorical("max_features", ["sqrt", "0.33", "0.5"]),
             "max_samples": trial.suggest_float("max_samples", 0.2, 1.0)}
        p["max_features"] = p["max_features"] if p["max_features"] == "sqrt" else float(p["max_features"])
        losses, t0 = [], time.perf_counter()
        for k, (Xtr, ytr, Xva, yva) in enumerate(data):
            m = rf_from_params(p).fit(Xtr, ytr)
            losses.append(log_loss(yva, np.clip(m.predict_proba(Xva)[:, 1], 1e-7, 1 - 1e-7), labels=[0, 1]))
            trial.report(float(np.mean(losses)), k)
            if trial.should_prune():
                trial.set_user_attr("fold_logloss", [round(x, 5) for x in losses])
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_logloss", [round(x, 5) for x in losses])
        trial.set_user_attr("fold_logloss_std", round(float(np.std(losses)), 5))
        trial.set_user_attr("seconds", round(time.perf_counter() - t0, 1))
        print(f"trial {trial.number:2d} logloss {np.mean(losses):.5f}  {p}  "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        return float(np.mean(losses))

    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(study_name="rf_classifier", direction="minimize", sampler=sampler,
                                pruner=pruner, storage=f"sqlite:///{OUT_NAME}/rf_optuna_study.db",
                                load_if_exists=resume)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if resume:
        for t in study.trials:
            if t.state == optuna.trial.TrialState.RUNNING:
                study.tell(t.number, state=optuna.trial.TrialState.FAIL)
                print(f"trial {t.number} was left RUNNING by an interrupted process -> marked FAIL", flush=True)
    else:
        # trial 0 = the untuned configuration used so far (max_samples=1.0 ~ default bootstrap)
        study.enqueue_trial({"n_estimators": 150, "max_depth": 18, "min_samples_leaf": 20,
                             "max_features": "sqrt", "max_samples": 1.0})

    def n_complete(st):
        return sum(t.state == optuna.trial.TrialState.COMPLETE for t in st.trials)

    def stop_cb(st, _trial):
        print(f"  [{n_complete(st)} complete / {len(st.trials)} total]", flush=True)
        if target_complete and n_complete(st) >= target_complete:
            st.stop()

    if not (target_complete and n_complete(study) >= target_complete):
        study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[stop_cb])
    study.trials_dataframe().to_csv(f"{OUT_NAME}/rf_optuna_trials.csv", index=False)
    best = dict(study.best_params)
    best["max_features"] = best["max_features"] if best["max_features"] == "sqrt" else float(best["max_features"])
    json.dump({"best_trial": study.best_trial.number, "best_mean_fold_logloss": study.best_value,
               "params": best, "n_trials": len(study.trials),
               "n_complete": sum(t.state.name == "COMPLETE" for t in study.trials),
               "n_pruned": sum(t.state.name == "PRUNED" for t in study.trials),
               "untuned_trial0_logloss": study.trials[0].value},
              open(f"{OUT_NAME}/rf_tuned_params.json", "w"), indent=1)
    print("best", study.best_trial.number, study.best_value, best)


def evaluate():
    """Untuned vs tuned RF on identical partitions:
    (a) 5-fold OOF classification + OOF HMM route F1 on the 64 tune trips,
    (b) trained on 64 -> holdout 16, (c) trained on 64 -> student_test 10 (report only)."""
    from src.data_loader import list_trips
    from src.ml_train import build_dataset
    ds = joblib.load(DATASET)
    folds = json.load(open(f"{OUT_NAME}/cv_folds.json"))
    part = load_partition()
    tune, hold = sorted(part["tune"]), sorted(part["holdout"])
    st = sorted(t["trip_id"] for t in list_trips() if t["split"] == "student_test")
    ds_st = build_dataset(st, get_config("tuned_v2"))
    best = json.load(open(f"{OUT_NAME}/rf_tuned_params.json"))["params"]
    models = {"rf_untuned": make_models()["random_forest"], "rf_tuned": rf_from_params(best)}
    base = get_config("tuned_v2").with_(emission="ml", ml_weight=4.0)
    rows = []
    for name, proto in models.items():
        oof = {}
        for val in folds:
            m = clone(proto).fit(*stack(ds, [t for t in tune if t not in val])[:2])
            for t in val:
                oof[t] = m.predict_proba(ds["trips"][t]["X"])[:, 1]
        f = f"{OUT_NAME}/oof_{name}.joblib"; joblib.dump(oof, f)
        cm = candidate_metrics(ds, tune, oof)
        hm = summarize(evaluate_config(base.with_(name=name), tune, probs_file=f))
        rows.append({"model": name, "eval_set": "tune64_5fold_OOF", "train_trips": "64 (4/5 per fold)",
                     **{f"clf_{k}": v for k, v in cm.items()}, **hm})
        t0 = time.perf_counter()
        m64 = clone(proto).fit(*stack(ds, tune)[:2])
        fit_s = time.perf_counter() - t0
        for set_name, ids, dsx in (("holdout16", hold, ds), ("student_test10", st, ds_st)):
            probs = {t: m64.predict_proba(dsx["trips"][t]["X"])[:, 1] for t in ids}
            pf = f"{OUT_NAME}/probs_{name}_{set_name}.joblib"; joblib.dump(probs, pf)
            cm = candidate_metrics(dsx, ids, probs)
            hm = summarize(evaluate_config(base.with_(name=name), ids, probs_file=pf))
            rows.append({"model": name, "eval_set": set_name, "train_trips": "64 tune",
                         "fit_s": round(fit_s, 1), **{f"clf_{k}": v for k, v in cm.items()}, **hm})
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_NAME}/rf_tuning_comparison.csv", index=False)
    print(df[["model", "eval_set", "clf_log_loss", "clf_f1", "clf_pseudo_top1_accuracy",
              "arc_f1_mean", "arc_f1_median", "arc_precision_mean", "arc_recall_mean"]].to_string(index=False))


# ============================================================================
# Final tuned-vs-untuned comparison (Deliverables §4)
#   python -m scripts.tune_classifier --stage final
# Same 64 tuning trips, same 5 GroupKFold folds (experiments/ml/cv_folds.json),
# same features / pseudo-labels (experiments/ml/dataset.joblib), same decoder.
# Student test is evaluated once per model; nothing is selected on it.
# evaluator_test is never touched.
# ============================================================================

def final_comparison():
    import datetime as dt
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score

    from scripts.train_classifiers import _bundle
    from src.data_loader import list_trips
    from src.ml_emission import MODEL_DIR, load_model_bundle
    from src.ml_train import build_dataset

    os.makedirs(FINAL_DIR, exist_ok=True)
    ds = joblib.load(DATASET)
    folds = json.load(open(f"{OUT_NAME}/cv_folds.json"))
    tune = sorted(load_partition()["tune"])
    assert sorted(t for f in folds for t in f) == tune
    st_ids = sorted(t["trip_id"] for t in list_trips() if t["split"] == "student_test")
    assert not set(st_ids) & set(tune)

    # ---- best trial: chosen ONLY by the training objective, before any evaluation
    study = optuna.load_study(study_name="rf_classifier", storage=f"sqlite:///{OUT_NAME}/rf_optuna_study.db")
    df = study.trials_dataframe()
    df.to_csv(f"{FINAL_DIR}/optuna_trials.csv", index=False)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = min(complete, key=lambda t: t.value)
    t0_trial = study.trials[0]
    bp = dict(best.params)
    bp["max_features"] = bp["max_features"] if bp["max_features"] == "sqrt" else float(bp["max_features"])
    counts = {s: sum(t.state.name == s for t in study.trials) for s in ("COMPLETE", "PRUNED", "FAIL", "RUNNING")}
    best_json = {
        "study": "rf_classifier (experiments/ml/rf_optuna_study.db)",
        "selection_rule": "lowest mean 5-fold validation log loss among COMPLETE trials (training objective only)",
        "best_trial": best.number, "best_mean_fold_logloss": best.value, "params": bp,
        "untuned_trial0_mean_fold_logloss": t0_trial.value,
        "trial_counts": counts, "n_trials_total": len(study.trials),
        "search_space": {"n_estimators": "int 50-300 step 25", "max_depth": "int 6-30",
                         "min_samples_leaf": "int 1-200 log", "max_features": "{sqrt, 0.33, 0.5}",
                         "max_samples": "float 0.2-1.0", "bootstrap": True, "random_state": SEED},
        "sampler": "TPESampler(seed=42, n_startup_trials=8)",
        "pruner": "MedianPruner(n_startup_trials=5, n_warmup_steps=1)",
        "objective": "mean validation log loss over the 5 trip-grouped folds of the 64 tuning trips",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
    }
    json.dump(best_json, open(f"{FINAL_DIR}/best_hyperparameters.json", "w"), indent=1)
    print("best trial", best.number, best.value, bp, counts, flush=True)

    models = {"untuned": make_models()["random_forest"], "tuned": rf_from_params(bp)}

    # ---- 5-fold OOF on the 64 tuning trips
    cv_rows, oof = [], {k: {} for k in models}
    for name, proto in models.items():
        fold_ll, fold_mf1, fit_s = [], [], 0.0
        for val in folds:
            tr = [t for t in tune if t not in val]
            Xtr, ytr, _ = stack(ds, tr)
            t0 = time.perf_counter()
            m = clone(proto).fit(Xtr, ytr)
            fit_s += time.perf_counter() - t0
            Xva, yva, _ = stack(ds, val)
            pva = m.predict_proba(Xva)[:, 1]
            fold_ll.append(log_loss(yva, np.clip(pva, 1e-7, 1 - 1e-7), labels=[0, 1]))
            fold_mf1.append(f1_score(yva, (pva >= 0.5).astype(int), average="macro"))
            for t in val:
                oof[name][t] = m.predict_proba(ds["trips"][t]["X"])[:, 1]
        pooled = candidate_metrics(ds, tune, oof[name])
        cv_rows.append({"model": name, "eval": "5-fold grouped CV, 64 tuning trips (out-of-fold)",
                        "cv_macro_f1_mean_over_folds": round(float(np.mean(fold_mf1)), 4),
                        "cv_macro_f1_sd": round(float(np.std(fold_mf1, ddof=1)), 4),
                        "cv_log_loss_mean_over_folds": round(float(np.mean(fold_ll)), 5),
                        "cv_log_loss_sd": round(float(np.std(fold_ll, ddof=1)), 5),
                        **{f"cv_pooled_{k}": v for k, v in pooled.items() if k in
                           ("precision", "recall", "f1", "macro_f1", "log_loss", "roc_auc", "pseudo_top1_accuracy")},
                        "fit_s_5_folds": round(fit_s, 1)})
        print(name, "CV done", cv_rows[-1]["cv_log_loss_mean_over_folds"], flush=True)
        joblib.dump(oof[name], f"{FINAL_DIR}/oof_{name}.joblib")
    cv = pd.DataFrame(cv_rows)
    # reproducibility: mean fold log loss must equal the stored Optuna values
    cv["optuna_recorded_value"] = [t0_trial.value, best.value]
    cv["reproduces_optuna_value"] = (cv.cv_log_loss_mean_over_folds - cv.optuna_recorded_value).abs() < 5e-6
    cv.to_csv(f"{FINAL_DIR}/tuned_vs_untuned_cv.csv", index=False)

    # ---- fit on all 64 tuning trips -> student test (once)
    ds_st = build_dataset(st_ids, get_config("tuned_v2"))
    base = get_config("tuned_v2").with_(emission="ml", ml_weight=4.0)
    X64, y64, _ = stack(ds, tune)
    st_rows, rt_rows = [], []
    Xst_all = np.vstack([ds_st["trips"][t]["X"] for t in st_ids])
    for name, proto in models.items():
        t0 = time.perf_counter()
        m64 = clone(proto).fit(X64, y64)
        fit_s = time.perf_counter() - t0
        if name == "untuned":  # integrity: refit equals the saved 64-trip model
            saved = load_model_bundle("random_forest__tune64")["model"]
            assert np.allclose(saved.predict_proba(Xst_all[:5000])[:, 1], m64.predict_proba(Xst_all[:5000])[:, 1])
        else:
            path = MODEL_DIR / "random_forest_optuna_tuned__tune64.joblib"
            assert not path.exists() or "optuna_tuned" in path.name
            b = _bundle(m64, "random_forest_optuna_tuned", tune, None)
            b["optuna_trial"] = best.number
            b["hyperparameters"] = bp
            joblib.dump(b, path, compress=3)
        probs = {t: m64.predict_proba(ds_st["trips"][t]["X"])[:, 1] for t in st_ids}
        pf = f"{FINAL_DIR}/student_test_probs_{name}.joblib"
        joblib.dump(probs, pf)
        cm = candidate_metrics(ds_st, st_ids, probs)
        route = summarize(evaluate_config(base.with_(name=f"rf_{name}"), st_ids, probs_file=pf))
        st_rows.append({"model": name, "trained_on": "64 tuning trips", "eval": "student_test (10 trips)",
                        "top1_accuracy": cm["pseudo_top1_accuracy"], "macro_f1": cm["macro_f1"],
                        "log_loss": cm["log_loss"], "precision": cm["precision"], "recall": cm["recall"],
                        "f1": cm["f1"], "roc_auc": cm["roc_auc"], "n_candidate_rows": cm["n_rows"],
                        "route_arc_f1_mean": route["arc_f1_mean"], "route_arc_f1_median": route["arc_f1_median"],
                        "route_precision_mean": route["arc_precision_mean"],
                        "route_recall_mean": route["arc_recall_mean"]})
        # inference time: predict_proba on all student-test candidate rows, 5 repeats, median
        reps = []
        for _ in range(5):
            t1 = time.perf_counter()
            m64.predict_proba(Xst_all)
            reps.append(time.perf_counter() - t1)
        n_nodes = sum(e.tree_.node_count for e in m64.estimators_)
        rt_rows.append({"model": name, "rows": len(Xst_all), "repeats": 5,
                        "inference_ms_per_1k_rows_median": round(1e6 * float(np.median(reps)) / len(Xst_all), 4),
                        "inference_ms_per_1k_rows_min": round(1e6 * float(np.min(reps)) / len(Xst_all), 4),
                        "fit_s_64_trips": round(fit_s, 1), "n_estimators": len(m64.estimators_),
                        "total_tree_nodes": int(n_nodes),
                        "hardware": "Apple M5, 10 cores, n_jobs=-1"})
        print(name, "student test done", st_rows[-1]["macro_f1"], st_rows[-1]["route_arc_f1_mean"], flush=True)
    pd.DataFrame(st_rows).to_csv(f"{FINAL_DIR}/tuned_vs_untuned_student_test.csv", index=False)
    pd.DataFrame(rt_rows).to_csv(f"{FINAL_DIR}/tuned_vs_untuned_runtime.csv", index=False)
    _figures(df, best, t0_trial, cv, pd.DataFrame(st_rows), pd.DataFrame(rt_rows), plt)


def _figures(df, best, t0_trial, cv, stt, rt, plt):
    SURF, INK2, GRID = "#fcfcfb", "#52514e", "#e4e3df"
    S1, S2 = "#2a78d6", "#eb6834"
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "axes.axisbelow": True, "legend.frameon": False,
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 10})
    # history
    c = df[df.state == "COMPLETE"].sort_values("number")
    pr = df[df.state == "PRUNED"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.scatter(c.number, c.value, s=22, color=S1, label=f"completed trial ({len(c)})", zorder=3)
    ax.step(c.number, c.value.cummin(), where="post", color=S2, lw=2, label="best so far")
    for _, r in pr.iterrows():
        ax.axvline(r.number, color=GRID, lw=1, zorder=1)
    ax.plot([], [], color=GRID, lw=1, label=f"pruned trial ({len(pr)}; partial-fold values not comparable)")
    ax.annotate(f"trial 0 (untuned): {t0_trial.value:.5f}", (0, t0_trial.value), xytext=(8, 10),
                textcoords="offset points", fontsize=8, color=INK2)
    ax.annotate(f"best #{best.number}: {best.value:.5f}", (best.number, best.value), xytext=(0, -18),
                textcoords="offset points", fontsize=8, ha="center")
    ax.set_xlabel("trial number"); ax.set_ylabel("mean 5-fold validation log loss")
    ax.set_title("Random Forest Bayesian optimisation (Optuna TPE)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)
    fig.text(0.01, 0.01, "Objective on the 64 tuning trips, 5 trip-grouped folds. Source: "
             "experiments/ml/rf_bayesian_final/optuna_trials.csv", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{FINAL_DIR}/optuna_optimisation_history.png", dpi=200)
    plt.close(fig)
    # comparison (small multiples: one axis per metric)
    cvi, sti, rti = cv.set_index("model"), stt.set_index("model"), rt.set_index("model")
    panels = [("CV macro F1\n(64 trips, OOF)", "higher is better", [cvi.loc[m, "cv_macro_f1_mean_over_folds"] for m in ("untuned", "tuned")], "{:.4f}"),
              ("CV log loss\n(mean of 5 folds)", "lower is better", [cvi.loc[m, "cv_log_loss_mean_over_folds"] for m in ("untuned", "tuned")], "{:.5f}"),
              ("Student-test\nTop-1 accuracy", "higher is better", [sti.loc[m, "top1_accuracy"] for m in ("untuned", "tuned")], "{:.4f}"),
              ("Student-test\nmacro F1", "higher is better", [sti.loc[m, "macro_f1"] for m in ("untuned", "tuned")], "{:.4f}"),
              ("Student-test\nlog loss", "lower is better", [sti.loc[m, "log_loss"] for m in ("untuned", "tuned")], "{:.4f}"),
              ("Inference ms per\n1k candidate rows", "lower is better", [rti.loc[m, "inference_ms_per_1k_rows_median"] for m in ("untuned", "tuned")], "{:.3f}"),
              ("Student-test route F1\n(+ frozen HMM)", "higher is better", [sti.loc[m, "route_arc_f1_mean"] for m in ("untuned", "tuned")], "{:.4f}")]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 3.6))
    for ax, (title, sub, vals, fmt) in zip(axes, panels):
        ax.bar([0, 1], vals, color=[S2, S1], width=0.65)
        lo, hi = min(vals), max(vals); span = (hi - lo) or abs(hi) * 0.05 or 1
        ax.set_ylim(lo - span * 3, hi + span * 2.2)
        for x, v in zip([0, 1], vals):
            ax.text(x, v + span * 0.25, fmt.format(v), ha="center", fontsize=8)
        ax.set_xticks([0, 1], ["untuned", "tuned"]); ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=9.5); ax.set_xlabel(sub, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
    fig.text(0.01, 0.01, "Both models fitted on the same 64 tuning trips and folds; student test = 10 trips, evaluated once. "
             "Note the truncated y-axes: differences are small. Sources: experiments/ml/rf_bayesian_final/tuned_vs_untuned_*.csv",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f"{FINAL_DIR}/tuned_vs_untuned_comparison.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="search", choices=["search", "evaluate", "final"])
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--resume", action="store_true", help="continue the existing study")
    ap.add_argument("--target-complete", type=int, default=None)
    a = ap.parse_args()
    if a.stage == "search":
        search(a.trials, a.timeout, a.resume, a.target_complete)
    elif a.stage == "final":
        final_comparison()
    else:
        evaluate()
