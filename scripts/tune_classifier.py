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


def search(n_trials: int, timeout: int):
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
                                load_if_exists=False)
    # trial 0 = the untuned configuration used so far (max_samples=1.0 ~ default bootstrap)
    study.enqueue_trial({"n_estimators": 150, "max_depth": 18, "min_samples_leaf": 20,
                         "max_features": "sqrt", "max_samples": 1.0})
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="search", choices=["search", "evaluate"])
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    search(a.trials, a.timeout) if a.stage == "search" else evaluate()
