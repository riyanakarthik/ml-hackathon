"""Bayesian hyperparameter optimisation of the HMM decoder (Optuna TPE).

  python -m scripts.tune --trials 120 --seed 42

Data protocol
  * Only the 80 TRAIN trips are used. They are split once (seed 42, saved in
    experiments/train_partition.json) into 64 `tune` trips and 16 `holdout` trips.
  * The objective is mean route-level arc F1 over the 64 tune trips. The
    decoder has no fitted weights, so "cross-validation" here means reporting
    the objective per trip-level fold (4 folds x 16 trips) for stability;
    the optimiser itself sees the mean over all tune trips.
  * Holdout trips and student_test are never evaluated inside this script.

Outputs
  experiments/optuna_trials.csv     every trial: params, all summary metrics, fold F1s
  experiments/optuna_study.db       Optuna SQLite storage (full search record)
  configs/tuned_candidate.json      best params (NOT yet validated)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import optuna
import pandas as pd

from src.config import BASELINE_V1, CONFIG_DIR
from src.experiment import EXPERIMENT_DIR, evaluate_config, load_partition, summarize

SEARCH_SPACE_DOC = {
    "k": "int [3, 12]",
    "radius_m": "float [20, 100]",
    "sigma_scale": "float log [0.5, 3]",
    "beta": "float log [2, 100]",
    "backtrack_tol_m": "float [0, 30]",
    "backtrack_penalty": "0 (v1) or float log [0.01, 2]",
    "uturn_penalty": "float [0, 20]",
    "min_spacing_m": "float [0, 40]",
    "max_loop_m": "0 (off) or float log [20, 2000]",
}


def suggest(trial: optuna.Trial):
    p = {
        "k": trial.suggest_int("k", 3, 12),
        "radius_m": trial.suggest_float("radius_m", 20.0, 100.0),
        "sigma_scale": trial.suggest_float("sigma_scale", 0.5, 3.0, log=True),
        "beta": trial.suggest_float("beta", 2.0, 100.0, log=True),
        "backtrack_tol_m": trial.suggest_float("backtrack_tol_m", 0.0, 30.0),
        "uturn_penalty": trial.suggest_float("uturn_penalty", 0.0, 20.0),
        "min_spacing_m": trial.suggest_float("min_spacing_m", 0.0, 40.0),
    }
    p["backtrack_penalty"] = (trial.suggest_float("backtrack_penalty_value", 0.01, 2.0, log=True)
                              if trial.suggest_categorical("use_backtrack_penalty", [False, True])
                              else 0.0)
    p["max_loop_m"] = (trial.suggest_float("max_loop_value", 20.0, 2000.0, log=True)
                       if trial.suggest_categorical("use_loop_removal", [False, True]) else 0.0)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--study", default="hmm_decoder_v2")
    args = ap.parse_args()

    part = load_partition()
    tune_ids = part["tune"]
    assert not set(tune_ids) & set(part["holdout"])
    fold_of = {t: i for i, f in enumerate(part["folds"]) for t in f}
    EXPERIMENT_DIR.mkdir(exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        cfg = BASELINE_V1.with_(name=f"trial_{trial.number}", **params)
        t0 = time.perf_counter()
        rows = evaluate_config(cfg, tune_ids, n_jobs=args.jobs)
        wall = time.perf_counter() - t0
        s = summarize(rows)
        for k, v in s.items():
            trial.set_user_attr(k, v)
        for i in range(len(part["folds"])):
            f1 = [r["arc_f1"] for r in rows if fold_of[r["trip_id"]] == i]
            trial.set_user_attr(f"fold{i}_arc_f1", round(float(np.mean(f1)), 4))
        trial.set_user_attr("wall_s", round(wall, 1))
        for k, v in params.items():
            trial.set_user_attr(f"cfg_{k}", v)
        return s["arc_f1_mean"]

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=20, multivariate=True)
    storage = f"sqlite:///{EXPERIMENT_DIR / 'optuna_study.db'}"
    study = optuna.create_study(study_name=args.study, direction="maximize", sampler=sampler,
                                storage=storage, load_if_exists=False)
    # Trial 0 = original baseline_v1 configuration, for reference.
    study.enqueue_trial({"k": BASELINE_V1.k, "radius_m": BASELINE_V1.radius_m,
                         "sigma_scale": 1.0, "beta": BASELINE_V1.beta,
                         "backtrack_tol_m": BASELINE_V1.backtrack_tol_m, "uturn_penalty": 0.0,
                         "min_spacing_m": 0.0, "use_backtrack_penalty": False,
                         "use_loop_removal": False})
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def log_cb(study, trial):
        print(f"trial {trial.number:3d}  F1={trial.value:.4f}  best={study.best_value:.4f} "
              f"(#{study.best_trial.number})  P={trial.user_attrs['arc_precision_mean']:.4f} "
              f"R={trial.user_attrs['arc_recall_mean']:.4f}  {trial.user_attrs['wall_s']}s",
              flush=True)

    study.optimize(objective, n_trials=args.trials, callbacks=[log_cb])

    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "duration"))
    df.to_csv(EXPERIMENT_DIR / "optuna_trials.csv", index=False)
    best = study.best_trial
    cfg_params = {k[4:]: v for k, v in best.user_attrs.items() if k.startswith("cfg_")}
    cand = BASELINE_V1.with_(name="tuned_candidate", **cfg_params)
    CONFIG_DIR.mkdir(exist_ok=True)
    cand.save(CONFIG_DIR / "tuned_candidate.json")
    print(f"\nbest trial #{best.number}: tune-set mean F1 = {best.value:.4f}")
    print(cand.to_dict())
    print("search space:", SEARCH_SPACE_DOC)


if __name__ == "__main__":
    main()
