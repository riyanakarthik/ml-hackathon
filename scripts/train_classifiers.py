"""Train / compare candidate classifiers and evaluate them inside the HMM.

Stages (run in order; each persists its outputs under experiments/ml/):

  python -m scripts.train_classifiers --stage dataset      # features + pseudo-labels, 80 train trips
  python -m scripts.train_classifiers --stage cv           # GroupKFold(5) by trip on the 64 tune trips
  python -m scripts.train_classifiers --stage oof_hmm      # OOF probs -> HMM -> route-level F1
  python -m scripts.train_classifiers --stage emission_grid --model <best>
  python -m scripts.train_classifiers --stage fit_tune     # all 5 models on 64 tune trips -> models/
  python -m scripts.train_classifiers --stage final --model <best>   # refit on all 80 train trips
  python -m scripts.train_classifiers --stage student_test           # report only, after freezing

Data protocol: only TRAIN trips. The 16 holdout trips (experiments/train_partition.json)
are excluded from every stage before `final`; student_test is never used here.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from src.config import CONFIG_DIR, get_config
from src.experiment import EXPERIMENT_DIR, evaluate_config, load_partition, summarize
from src.features import FEATURE_NAMES, SCHEMA_VERSION
from src.ml_emission import MODEL_DIR
from src.ml_train import (SEED, build_dataset, candidate_metrics, fit_timed, make_models,
                          predict_timed, stack)

OUT = EXPERIMENT_DIR / "ml"
DATASET = OUT / "dataset.joblib"
BASE_CONFIG = "tuned_v2"        # candidate generation + HMM transitions come from here
N_FOLDS = 5
LABEL_CONFIG = {"sigma": 10.0, "heading_kappa": 2.0, "min_speed_mps": 2.0,
                "skip_penalty_per_m": 0.01, "max_label_dist_m": 30.0}


def load_ds():
    return joblib.load(DATASET)


def stage_dataset():
    part = load_partition()
    ids = sorted(part["tune"] + part["holdout"])
    ds = build_dataset(ids, get_config(BASE_CONFIG))
    joblib.dump(ds, DATASET, compress=3)
    n = sum(len(p["y"]) for p in ds["trips"].values())
    nv = sum(int(p["row_valid"].sum()) for p in ds["trips"].values())
    pos = sum(int(p["y"][p["row_valid"]].sum()) for p in ds["trips"].values())
    pts = sum(p["n_points"] for p in ds["trips"].values())
    ptv = sum(int(p["point_valid"].sum()) for p in ds["trips"].values())
    cov = sum(int((p["point_valid"] & p["label_in_cands"]).sum()) for p in ds["trips"].values())
    print(f"dataset: {len(ids)} trips, {pts} GPS points ({ptv} with valid pseudo-label), "
          f"{n} candidate rows ({nv} valid), positives {pos} ({pos / nv:.3%}); "
          f"label among candidates for {cov / ptv:.2%} of valid points")


def stage_cv():
    ds = load_ds()
    part = load_partition()
    tune = sorted(part["tune"])
    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = [sorted(np.array(tune)[va].tolist()) for _, va in gkf.split(tune, groups=tune)]
    for i, f in enumerate(folds):           # leakage guard
        assert not set(f) & set(t for j, g in enumerate(folds) if j != i for t in g)
    (OUT / "cv_folds.json").write_text(json.dumps(folds, indent=1))

    fold_rows, summary = [], []
    for name, proto in make_models().items():
        oof: dict[str, np.ndarray] = {}
        fit_s, pred_s, n_pred = 0.0, 0.0, 0
        for k, val in enumerate(folds):
            train = [t for t in tune if t not in val]
            X, y, _ = stack(ds, train)
            model, ft = fit_timed(clone(proto), X, y)
            fit_s += ft
            for t in val:
                p, pt = predict_timed(model, ds["trips"][t]["X"])   # all rows, incl. invalid labels
                oof[t] = p
                pred_s += pt; n_pred += len(p)
            m = candidate_metrics(ds, val, oof)
            fold_rows.append({"model": name, "fold": k, "n_train_trips": len(train),
                              "n_val_trips": len(val), "fit_s": round(ft, 2), **m})
            print(f"  {name:14s} fold {k}: F1={m['f1']:.4f} logloss={m['log_loss']:.4f} "
                  f"top1(pseudo)={m['pseudo_top1_accuracy']:.4f}  fit {ft:.1f}s", flush=True)
        joblib.dump(oof, OUT / f"oof_{name}.joblib", compress=3)
        pooled = candidate_metrics(ds, tune, oof)
        fm = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        summary.append({"model": name, **{f"oof_{k}": v for k, v in pooled.items()},
                        "fold_f1_mean": round(fm["f1"].mean(), 4), "fold_f1_std": round(fm["f1"].std(), 4),
                        "fold_logloss_mean": round(fm["log_loss"].mean(), 4),
                        "fit_s_total": round(fit_s, 1),
                        "predict_ms_per_1k_rows": round(1000 * pred_s / n_pred * 1000, 3)})
    pd.DataFrame(fold_rows).to_csv(OUT / "cv_fold_metrics.csv", index=False)
    df = pd.DataFrame(summary)
    df.to_csv(OUT / "cv_model_comparison.csv", index=False)
    print(df.to_string(index=False))


def stage_oof_hmm(weight: float = 1.0, mode: str = "ml"):
    part = load_partition()
    tune = sorted(part["tune"])
    base = get_config(BASE_CONFIG)
    rows = []
    geo = evaluate_config(base, tune)
    rows.append({"model": f"{BASE_CONFIG} (geometric emission)", **summarize(geo)})
    per_trip = pd.DataFrame(geo)[["trip_id", "arc_f1"]].rename(columns={"arc_f1": "geometric"})
    for name in make_models():
        cfg = base.with_(name=f"oof_{name}", emission=mode, ml_weight=weight)
        r = evaluate_config(cfg, tune, probs_file=str(OUT / f"oof_{name}.joblib"))
        rows.append({"model": name, "emission": mode, "ml_weight": weight, **summarize(r)})
        per_trip = per_trip.merge(pd.DataFrame(r)[["trip_id", "arc_f1"]].rename(columns={"arc_f1": name}))
        print(f"  {name:14s} OOF+HMM route F1 {rows[-1]['arc_f1_mean']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "oof_hmm_comparison.csv", index=False)
    per_trip.to_csv(OUT / "oof_hmm_per_trip_f1.csv", index=False)
    cols = ["model", "arc_f1_mean", "arc_f1_median", "arc_precision_mean", "arc_recall_mean",
            "route_continuity_mean", "pl2_top1_accuracy_mean", "runtime_ms_mean"]
    print(df[cols].to_string(index=False))


def stage_emission_grid(model: str):
    part = load_partition()
    tune = sorted(part["tune"])
    base = get_config(BASE_CONFIG)
    rows = []
    for mode in ("ml", "hybrid"):
        for w in (0.25, 0.5, 1.0, 2.0, 4.0):
            cfg = base.with_(name=f"grid_{mode}_{w}", emission=mode, ml_weight=w)
            r = evaluate_config(cfg, tune, probs_file=str(OUT / f"oof_{model}.joblib"))
            s = summarize(r)
            rows.append({"model": model, "emission": mode, "ml_weight": w, **s})
            print(f"  {mode:6s} w={w:<5} F1 {s['arc_f1_mean']:.4f} (median {s['arc_f1_median']:.4f}) "
                  f"P {s['arc_precision_mean']:.4f} R {s['arc_recall_mean']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"emission_grid_{model}.csv", index=False)


def _bundle(model, name, trips, cv_row=None):
    ds_meta = joblib.load(DATASET)["candidate_config"]
    return {"model": model, "model_name": name, "feature_names": FEATURE_NAMES,
            "schema_version": SCHEMA_VERSION, "candidate_config": ds_meta,
            "label_config": LABEL_CONFIG, "trained_on_trips": sorted(trips),
            "cv_metrics": cv_row, "sklearn_version": sklearn.__version__,
            "python": platform.python_version(), "seed": SEED,
            "created": dt.datetime.now().isoformat(timespec="seconds")}


def stage_fit_tune():
    ds = load_ds()
    tune = sorted(load_partition()["tune"])
    cv = pd.read_csv(OUT / "cv_model_comparison.csv").set_index("model")
    MODEL_DIR.mkdir(exist_ok=True)
    X, y, _ = stack(ds, tune)
    for name, proto in make_models().items():
        model, ft = fit_timed(clone(proto), X, y)
        path = MODEL_DIR / f"{name}__tune64.joblib"
        joblib.dump(_bundle(model, name, tune, cv.loc[name].to_dict()), path, compress=3)
        print(f"  saved {path.name} ({path.stat().st_size / 1e6:.1f} MB, fit {ft:.1f}s)")


def stage_final(model_name: str, emission: str, weight: float):
    ds = load_ds()
    part = load_partition()
    all_train = sorted(part["tune"] + part["holdout"])
    X, y, _ = stack(ds, all_train)
    model, ft = fit_timed(clone(make_models()[model_name]), X, y)
    cv = pd.read_csv(OUT / "cv_model_comparison.csv").set_index("model")
    path = MODEL_DIR / f"{model_name}__train80.joblib"
    joblib.dump(_bundle(model, model_name, all_train, cv.loc[model_name].to_dict()), path, compress=3)
    cfg = get_config(BASE_CONFIG).with_(name="ml_v3", emission=emission, ml_weight=weight,
                                       ml_model=f"{model_name}__train80")
    cfg.save(CONFIG_DIR / "ml_v3.json")
    print(f"saved {path.name} (fit {ft:.1f}s on {len(all_train)} trips) and configs/ml_v3.json")


def stage_student_test():
    """AFTER the approach is frozen: report every saved model on student_test.
    Classifier-only metrics (pseudo-labels) and classifier+HMM route metrics.
    Nothing here is used for training or selection."""
    from src.data_loader import list_trips
    from src.ml_emission import load_model_bundle
    st = sorted(t["trip_id"] for t in list_trips() if t["split"] == "student_test")
    ds = build_dataset(st, get_config(BASE_CONFIG))
    base = get_config(BASE_CONFIG)
    rows = [{"model": "geometric HMM (tuned_v2)", **summarize(evaluate_config(base, st))}]
    names = [f"{m}__tune64" for m in make_models()] + ["random_forest__train80"]
    for name in names:
        b = load_model_bundle(name)
        probs, pred_s, n = {}, 0.0, 0
        for t in st:
            p, ps = predict_timed(b["model"], ds["trips"][t]["X"])
            probs[t] = p; pred_s += ps; n += len(p)
        cm = candidate_metrics(ds, st, probs)
        pf = OUT / f"student_test_probs_{name}.joblib"
        joblib.dump(probs, pf)
        cfg = base.with_(name=f"st_{name}", emission="ml", ml_weight=4.0)
        hm = summarize(evaluate_config(cfg, st, probs_file=str(pf)))
        rows.append({"model": name, **{f"clf_{k}": v for k, v in cm.items()},
                     "clf_predict_ms_per_1k_rows": round(1e6 * pred_s / n, 3), **hm})
        print(f"  {name:24s} clf F1 {cm['f1']:.4f} logloss {cm['log_loss']:.4f} "
              f"top1(pseudo) {cm['pseudo_top1_accuracy']:.4f} | +HMM route F1 {hm['arc_f1_mean']:.4f}",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "student_test_model_report.csv", index=False)
    print(df[["model", "arc_f1_mean", "arc_f1_median", "arc_precision_mean", "arc_recall_mean",
              "pl2_top1_accuracy_mean"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["dataset", "cv", "oof_hmm", "emission_grid", "fit_tune", "final",
                             "student_test"])
    ap.add_argument("--model")
    ap.add_argument("--emission", default="ml")
    ap.add_argument("--weight", type=float, default=1.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "dataset":
        stage_dataset()
    elif a.stage == "cv":
        stage_cv()
    elif a.stage == "oof_hmm":
        stage_oof_hmm(a.weight, a.emission)
    elif a.stage == "emission_grid":
        stage_emission_grid(a.model)
    elif a.stage == "fit_tune":
        stage_fit_tune()
    elif a.stage == "final":
        stage_final(a.model, a.emission, a.weight)
    elif a.stage == "student_test":
        stage_student_test()


if __name__ == "__main__":
    main()
