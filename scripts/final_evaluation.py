"""Final evaluation of the FROZEN deployed model on the evaluator_test partition.

  python -m scripts.final_evaluation [--data-root .] [--split evaluator_test]

Order of operations (nothing here trains, tunes or edits a model/config):
  1. inventory + SHA-256 of deployed model, configs and source  (before predicting)
  2. predictions for every trip through the real FastAPI app (POST /predict),
     deployed config, twice per trip (determinism) -> stored
  3. ONLY THEN: reference routes are read (src.evaluation) and metrics computed
  4. frozen comparison baselines: geometric HMM (tuned_v2, via API), RF trained on
     64 trips + HMM (ml_candidate, via API; comparison only), classifier-only and
     nearest-road (same candidates + route assembly), no parameters changed
  5. leakage check: predictions repeated on copies WITHOUT .route must be identical
Outputs: experiments/final_evaluation/
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "final_evaluation"
SRC_FILES = ["main.py", "src/config.py", "src/data_loader.py", "src/matcher.py", "src/features.py",
             "src/ml_emission.py", "src/decoder.py", "src/pipeline.py", "src/evaluation.py",
             "src/labels.py"]
log = logging.getLogger("final_eval")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def inventory(split_dir: Path) -> list[dict]:
    from src.data_loader import TripDataError, load_trip
    rows = []
    for d in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        tid = d.name
        files = {ext: (d / f"{tid}.{ext}") for ext in ("track", "nodes", "arcs", "route")}
        r = {"trip_id": tid, **{f"has_{e}": f.exists() for e, f in files.items()},
             **{f"bytes_{e}": (f.stat().st_size if f.exists() else None) for e, f in files.items()}}
        try:
            trip = load_trip(tid, split_dir.parent)
            ts = trip.timestamps
            r.update({"parse_ok": True, "n_gps_points": len(trip.track), "n_nodes": len(trip.nodes),
                      "n_arcs": len(trip.arcs), "duration_s": float(ts[-1] - ts[0]),
                      "max_time_gap_s": float(np.diff(ts).max()) if len(ts) > 1 else 0.0,
                      "frac_1s_steps": round(float(np.mean(np.diff(ts) == 1)), 4) if len(ts) > 1 else None,
                      "extra_files": sorted(p.name for p in d.iterdir()
                                            if p.suffix.lstrip(".") not in files)})
        except TripDataError as e:
            r.update({"parse_ok": False, "error": str(e)})
        rows.append(r)
    return rows


def connected_fraction(route, arcs):
    if len(route) < 2:
        return None
    r = np.asarray(route)
    return float(np.mean(arcs[r[:-1], 1] == arcs[r[1:], 0]))


def predict_all(client, trip_ids, config, label):
    out = {}
    for tid in trip_ids:
        body = {"trip_id": tid} if config is None else {"trip_id": tid, "config": config}
        t0 = time.perf_counter()
        r1 = client.post("/predict", json=body)
        wall = (time.perf_counter() - t0) * 1000
        r2 = client.post("/predict", json=body)
        b1 = r1.json()
        ok = r1.status_code == 200
        det = ok and r2.status_code == 200 and \
            b1["predicted_arc_ids"] == r2.json()["predicted_arc_ids"] and \
            b1["predicted_coordinates"] == r2.json()["predicted_coordinates"]
        out[tid] = {"status": r1.status_code, "body": b1, "wall_ms": wall, "deterministic": det}
        log.info("%s trip=%s status=%s wall_ms=%.1f server_ms=%s deterministic=%s", label, tid,
                 r1.status_code, wall, b1.get("runtime_ms"), det)
    return out


def classifier_only_and_nearest(trip_ids, data_root):
    """Frozen baselines without HMM (same candidates/route assembly as tuned_v2/ml_v3)."""
    from src.config import get_config
    from src.data_loader import load_trip
    from src.decoder import NetworkDistance, _positions, assemble_route
    from src.matcher import generate_candidates
    from src.ml_emission import load_model_bundle, predict_candidate_probs
    from src.pipeline import build_network
    cfg = get_config("ml_v3")
    bundle = load_model_bundle(cfg.ml_model)
    res = {}
    for tid in trip_ids:
        trip = load_trip(tid, data_root)
        net, xy = build_network(trip)
        cands = generate_candidates(net, xy, k=cfg.k, radius_m=cfg.radius_m,
                                    fallback_radius_m=cfg.fallback_radius_m)
        for name in ("classifier_only_rf80", "nearest_road"):
            t0 = time.perf_counter()
            if name == "nearest_road":
                choice = np.array([0 if len(c) else -1 for c in cands])
            else:
                probs = predict_candidate_probs(bundle, net, xy, trip.timestamps, cands, cfg.radius_m)
                choice = np.array([int(np.argmax(p)) if len(p) else -1 for p in probs])
            arcs, _ = _positions(cands, choice)
            route, _, fill = assemble_route(arcs, net, NetworkDistance(net, cfg.dijkstra_limit_m),
                                            cfg.fill_max_gap_m, return_fill_mask=True)
            res[(name, tid)] = {"route": route, "fill": fill, "point_arcs": arcs,
                                "ms": (time.perf_counter() - t0) * 1000}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT))
    ap.add_argument("--split", default="evaluator_test")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "maps").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(OUT / "evaluation_run.log", mode="w"),
                                  logging.StreamHandler(sys.stdout)])
    data_root = Path(args.data_root).resolve()
    split_dir = data_root / args.split
    os.environ["PATHSNAP_DATA_DIR"] = str(data_root)

    # ---------------------------------------------------------------- 1. manifest (before predicting)
    from src.config import get_config
    import main as api
    deployed = get_config(api.DEFAULT_CONFIG)
    model_path = ROOT / "models" / f"{deployed.ml_model}.joblib"
    inv = inventory(split_dir)
    trip_ids = [r["trip_id"] for r in inv if r.get("parse_ok")]
    manifest = {
        "evaluation_started": dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "git_dirty_files": subprocess.check_output(["git", "status", "--short"], cwd=ROOT).decode().split("\n")[:40],
        "python": sys.version.split()[0], "platform": platform.platform(),
        "data_root": str(data_root), "split": args.split, "n_trip_dirs": len(inv),
        "n_parse_ok": len(trip_ids), "all_have_route": all(r["has_route"] for r in inv),
        "deployed_config": api.DEFAULT_CONFIG, "deployed_config_values": deployed.to_dict(),
        "hashes": {"model": {str(model_path.relative_to(ROOT)): sha(model_path)},
                   "configs": {f"configs/{n}.json": sha(ROOT / "configs" / f"{n}.json")
                               for n in ("ml_v3", "tuned_v2", "ml_candidate", "baseline_v1")},
                   "source": {f: sha(ROOT / f) for f in SRC_FILES}},
        "frozen_manifest_match": {},
        "dataset_inventory": inv,
    }
    frozen = json.load(open(ROOT / "experiments" / "final" / "frozen_artifacts_sha256.json"))
    for f, h in frozen["files"].items():
        if f in ("models/random_forest__train80.joblib", "configs/ml_v3.json", "configs/tuned_v2.json"):
            manifest["frozen_manifest_match"][f] = (sha(ROOT / f) == h)
    json.dump(manifest, open(OUT / "evaluation_manifest.json", "w"), indent=1, default=str)
    log.info("inventory: %d trips, all parse=%s, all have .route=%s; frozen hashes match=%s",
             len(inv), len(trip_ids) == len(inv), manifest["all_have_route"],
             manifest["frozen_manifest_match"])

    # ---------------------------------------------------------------- 2. predictions via API
    from fastapi.testclient import TestClient
    from src.data_loader import load_trip
    from src.ml_emission import load_model_bundle
    b = load_model_bundle(deployed.ml_model)
    from src.features import FEATURE_NAMES, SCHEMA_VERSION
    model_check = {"model_name": b["model_name"], "n_train_trips": len(b["trained_on_trips"]),
                   "feature_order_matches_training": list(b["feature_names"]) == FEATURE_NAMES,
                   "schema_version": b["schema_version"] == SCHEMA_VERSION,
                   "classes": [int(c) for c in b["model"].classes_],
                   "eval_trip_in_training_set": sorted(set(trip_ids) & set(b["trained_on_trips"])),
                   "calibration": "none (raw RandomForest predict_proba, clipped at prob_floor, "
                                  "per-point normalised; identical to training-time CV usage)"}
    log.info("model check: %s", model_check)
    with TestClient(api.app) as client:
        listed = client.get("/trips").json()
        log.info("/trips lists %d trips", listed["count"])
        runs = {"rf80_hmm_deployed": predict_all(client, trip_ids, None, "deployed"),
                "geometric_hmm_tuned_v2": predict_all(client, trip_ids, "tuned_v2", "geometric"),
                "rf64_hmm_ml_candidate": predict_all(client, trip_ids, "ml_candidate", "rf64")}
    no_hmm = classifier_only_and_nearest(trip_ids, data_root)

    # leakage check: same predictions without .route present
    tmp = Path(tempfile.mkdtemp())
    try:
        for tid in trip_ids:
            dst = tmp / args.split / tid
            dst.mkdir(parents=True)
            for ext in ("track", "nodes", "arcs"):
                shutil.copy(split_dir / tid / f"{tid}.{ext}", dst)
        os.environ["PATHSNAP_DATA_DIR"] = str(tmp)
        with TestClient(api.app) as client:
            noroute = predict_all(client, trip_ids, None, "no_route_copy")
    finally:
        os.environ["PATHSNAP_DATA_DIR"] = str(data_root)
        shutil.rmtree(tmp, ignore_errors=True)
    leak_ok = {t: noroute[t]["body"].get("predicted_arc_ids") ==
               runs["rf80_hmm_deployed"][t]["body"].get("predicted_arc_ids")
               and noroute[t]["body"].get("metrics") is None for t in trip_ids}

    # ---------------------------------------------------------------- 3. scoring (after prediction)
    from src.evaluation import load_reference_route, route_diagnostics, route_metrics, sequence_metrics
    from src.labels import align_track_to_route
    from src.pipeline import build_network
    per, err = [], []
    variants = list(runs) + ["classifier_only_rf80", "nearest_road"]
    for tid in trip_ids:
        trip = load_trip(tid, data_root)
        net, xy = build_network(trip)
        ref = load_reference_route(tid, data_root)
        al = align_track_to_route(xy, trip.timestamps, ref, net) if ref is not None else None
        for v in variants:
            if v in runs:
                r = runs[v][tid]
                bd = r["body"]
                if r["status"] != 200:
                    per.append({"trip_id": tid, "variant": v, "failure": f"HTTP {r['status']}: {bd}"})
                    continue
                route = bd["predicted_arc_ids"]
                point_arcs = np.array(bd["point_arc_ids"])
                runtime, wall = bd["runtime_ms"], round(r["wall_ms"], 1)
                fill = None
                det = r["deterministic"]
                missing = bd["details"].get("points_without_candidates")
                unmatched = sum(c is None for c in bd["predicted_coordinates"])
                emission = bd["details"].get("emission")
                api_metrics = bd["metrics"]
            else:
                x = no_hmm[(v, tid)]
                route, point_arcs, fill = x["route"], x["point_arcs"], x["fill"]
                runtime = wall = round(x["ms"], 1)
                det, missing, unmatched, emission, api_metrics = None, None, int((point_arcs < 0).sum()), None, None
            valid_ids = all(0 <= a < len(trip.arcs) for a in route)
            row = {"trip_id": tid, "variant": v, "n_gps_points": len(trip.track),
                   "n_nodes": len(trip.nodes), "n_arcs": len(trip.arcs),
                   "pred_route_len": len(route), "pred_distinct_arcs": len(set(route)),
                   "runtime_ms_server": runtime, "latency_ms_wall": wall, "deterministic": det,
                   "points_without_candidates": missing, "unmatched_points": unmatched,
                   "all_arc_ids_valid": valid_ids,
                   "path_continuity": connected_fraction(route, trip.arcs) if valid_ids else None,
                   "emission": emission, "failure": ""}
            if ref is not None and valid_ids:
                m = route_metrics(route, ref, trip.arcs, net.arc_len)
                row.update({k: m[k] for k in ("arc_precision", "arc_recall", "arc_f1", "length_precision",
                                              "length_recall", "n_ref_arcs", "n_common_arcs")})
                row.update(sequence_metrics(route, ref))
                if api_metrics is not None:
                    row["api_metric_matches_independent"] = api_metrics["arc_f1"] == m["arc_f1"]
                v_ = al.valid
                row["pseudo_top1_accuracy"] = round(float(np.mean(point_arcs[v_] == al.labels[v_])), 4) \
                    if v_.any() else None
                row["label_valid_frac"] = round(float(v_.mean()), 4)
                if v == "rf80_hmm_deployed" or v == "geometric_hmm_tuned_v2":
                    is_fill = fill
                    if is_fill is None:  # API does not return fill flags: recompute provenance
                        is_fill = [a not in set(point_arcs.tolist()) for a in route]
                    d = route_diagnostics(route, is_fill, point_arcs, ref, trip.arcs, net.arc_len)
                    err.append({"trip_id": tid, "variant": v, **d,
                                "ref_route_len": len(ref), "ref_has_repeats": len(ref) != len(set(ref.tolist())),
                                "label_valid_frac": row["label_valid_frac"],
                                # GPS points aligned to the route's last arc but > 30 m from it:
                                # the vehicle kept recording after the reference route ends
                                "gps_beyond_route_end_frac": round(float(np.mean(
                                    (al.position == len(ref) - 1) & ~al.valid)), 4),
                                "points_without_candidates": missing})
            per.append(row)
    df = pd.DataFrame(per)
    df.to_csv(OUT / "evaluation_per_trip.csv", index=False)
    preds = {v: {t: runs[v][t]["body"].get("predicted_arc_ids") for t in trip_ids} for v in runs}
    preds.update({v: {t: [int(a) for a in no_hmm[(v, t)]["route"]] for t in trip_ids}
                  for v in ("classifier_only_rf80", "nearest_road")})
    json.dump(preds, open(OUT / "predicted_routes.json", "w"))
    pd.DataFrame(err).to_csv(OUT / "route_error_analysis.csv", index=False)

    # ---------------------------------------------------------------- 4. summary
    def stats(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return {"mean": round(s.mean(), 4), "median": round(s.median(), 4), "min": round(s.min(), 4),
                "max": round(s.max(), 4), "std": round(s.std(ddof=1), 4), "n": int(len(s))} if len(s) else None
    summary = {"split": args.split, "n_trips": len(trip_ids), "deployed_config": api.DEFAULT_CONFIG,
               "reference_routes_available": manifest["all_have_route"],
               "metric_definition": "arc_* = set-based over DISTINCT DIRECTED arc ids (duplicates "
                                    "collapsed, order ignored, gap-fill arcs included); seq_* = "
                                    "order/duplicate-aware LCS; pseudo_top1 vs derived per-point labels",
               "model_check": model_check, "trips_listed_by_api": listed["count"],
               "leakage_check_predictions_identical_without_route": f"{sum(leak_ok.values())}/{len(leak_ok)}",
               "variants": {}}
    for v in variants:
        d = df[df.variant == v]
        summary["variants"][v] = {
            "failures": int((d.failure.fillna("") != "").sum()),
            "all_deterministic": None if d.deterministic.isna().all() else bool(d.deterministic.fillna(False).all()),
            "all_arc_ids_valid": bool(d.all_arc_ids_valid.fillna(False).all()),
            **{k: stats(d[k]) for k in ("arc_f1", "arc_precision", "arc_recall", "seq_f1",
                                         "length_precision", "length_recall", "path_continuity",
                                         "pseudo_top1_accuracy", "runtime_ms_server", "latency_ms_wall")
               if k in d}}
        if "api_metric_matches_independent" in d and d.api_metric_matches_independent.notna().any():
            summary["variants"][v]["api_metrics_match_independent"] = bool(d.api_metric_matches_independent.dropna().all())
    a = df[df.variant == "geometric_hmm_tuned_v2"].set_index("trip_id")["arc_f1"]
    bb = df[df.variant == "rf80_hmm_deployed"].set_index("trip_id")["arc_f1"]
    if a.notna().all():
        delta = (bb - a)
        from scipy.stats import wilcoxon
        nz = delta[delta.abs() > 1e-9]
        summary["deployed_vs_geometric_per_trip"] = {
            "better": int((delta > 1e-9).sum()), "worse": int((delta < -1e-9).sum()),
            "same": int((delta.abs() <= 1e-9).sum()), "mean_delta": round(float(delta.mean()), 4),
            "wilcoxon_p": round(float(wilcoxon(nz).pvalue), 4) if len(nz) >= 5 else None}
    summary["evaluation_finished"] = dt.datetime.now().isoformat(timespec="seconds")
    json.dump(summary, open(OUT / "evaluation_summary.json", "w"), indent=1, default=str)
    log.info("summary written; deployed mean arc F1 = %s",
             (summary["variants"]["rf80_hmm_deployed"].get("arc_f1") or {}).get("mean"))


if __name__ == "__main__":
    main()
