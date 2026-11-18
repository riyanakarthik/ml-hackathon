"""Collect read-only evidence for the report into reports/data/ (no training, no backend changes).

  .venv/bin/python reports/tooling/collect_evidence.py   (repo root)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
DATA = ROOT / "reports" / "data"

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def tests():
    out = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "-p", "no:warnings"],
                         capture_output=True, text=True, cwd=ROOT)
    tail = out.stdout.strip().splitlines()[-1]
    co = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "--co", "-q"],
                        capture_output=True, text=True, cwd=ROOT).stdout
    per_file = {}
    for line in co.splitlines():
        if "::" in line:
            f = line.split("::")[0]
            per_file[f] = per_file.get(f, 0) + 1
    passed = int(re.search(r"(\d+) passed", tail).group(1)) if "passed" in tail else 0
    failed = int(re.search(r"(\d+) failed", tail).group(1)) if "failed" in tail else 0
    res = {"summary_line": tail, "passed": passed, "failed": failed, "per_file": per_file,
           "git_head": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()}
    json.dump(res, open(DATA / "test_run.json", "w"), indent=1)
    print("tests:", tail, per_file)


def api_example():
    os.environ["PATHSNAP_DATA_DIR"] = str(ROOT / "Student folder")
    from fastapi.testclient import TestClient

    import main as api
    with TestClient(api.app) as c:
        health = c.get("/health").json()
        trips = c.get("/trips").json()
        b = c.post("/predict", json={"trip_id": "00000091"}).json()
    short = {k: (v[:3] + ["..."] if isinstance(v, list) and len(v) > 3 else v) for k, v in b.items()}
    short["details"] = {k: v for k, v in b["details"].items() if k in
                        ("config", "emission", "sigma_m", "beta", "hmm_breaks", "unfilled_gaps",
                         "points_without_candidates", "mean_candidates_per_point", "decoded_points", "timings_ms")}
    short["metrics"] = {k: b["metrics"][k] for k in ("arc_precision", "arc_recall", "arc_f1",
                                                     "route_continuity", "n_pred_arcs", "n_ref_arcs")}
    json.dump({"health": health, "trips_count": trips["count"], "predict_00000091": short,
               "list_lengths": {k: len(v) for k, v in b.items() if isinstance(v, list)}},
              open(DATA / "api_example.json", "w"), indent=1)
    print("api example: method", b["method"], "config", b["details"]["config"])


def training_dataset():
    ds = joblib.load(ROOT / "experiments/ml/dataset.joblib")
    t = ds["trips"].values()
    rows = sum(len(p["y"]) for p in t)
    valid_rows = sum(int(p["row_valid"].sum()) for p in t)
    pos = sum(int(p["y"][p["row_valid"]].sum()) for p in t)
    pts = sum(p["n_points"] for p in t)
    ptv = sum(int(p["point_valid"].sum()) for p in t)
    cov = sum(int((p["point_valid"] & p["label_in_cands"]).sum()) for p in t)
    res = {"trips": len(ds["trips"]), "gps_points": pts, "gps_points_valid_label": ptv,
           "candidate_rows": rows, "candidate_rows_valid": valid_rows, "positives_valid": pos,
           "positive_rate_valid": round(pos / valid_rows, 4), "label_in_candidates_frac": round(cov / ptv, 4),
           "candidate_config": ds["candidate_config"], "n_features": len(ds["feature_names"])}
    json.dump(res, open(DATA / "training_dataset_stats.json", "w"), indent=1)
    print("dataset:", res)


def paired_tests():
    """Paired per-trip comparisons recomputed from recorded per-trip CSVs."""
    E = ROOT / "experiments"
    out = []

    def add(name, a, b, src):
        d = (np.asarray(b) - np.asarray(a))
        nz = d[np.abs(d) > 1e-9]
        out.append({"comparison": name, "n_trips": len(d), "mean_delta": round(float(d.mean()), 4),
                    "better": int((d > 1e-9).sum()), "worse": int((d < -1e-9).sum()),
                    "same": int((np.abs(d) <= 1e-9).sum()),
                    "sign_test_p": round(float(binomtest(int((nz > 0).sum()), len(nz)).pvalue), 4) if len(nz) else None,
                    "wilcoxon_p": round(float(wilcoxon(nz).pvalue), 4) if len(nz) >= 5 else None, "source": src})
    h = pd.read_csv(E / "compare_tuned_v2_vs_ml_candidate_holdout.csv")
    add("holdout16: geometric HMM -> RF(64)+HMM", h["arc_f1__tuned_v2"], h["arc_f1__ml_candidate"],
        "experiments/compare_tuned_v2_vs_ml_candidate_holdout.csv")
    s = pd.read_csv(E / "compare_tuned_v2_vs_ml_v3_student_test.csv")
    add("student_test10: geometric HMM -> RF(80)+HMM", s["arc_f1__tuned_v2"], s["arc_f1__ml_v3"],
        "experiments/compare_tuned_v2_vs_ml_v3_student_test.csv")
    o = pd.read_csv(E / "ml/oof_hmm_per_trip_f1.csv")
    add("tune64 OOF (w=1): geometric HMM -> RF+HMM", o["geometric"], o["random_forest"],
        "experiments/ml/oof_hmm_per_trip_f1.csv")
    ev = pd.read_csv(E / "final_evaluation/evaluation_per_trip.csv")
    g = ev[ev.variant == "geometric_hmm_tuned_v2"].set_index("trip_id")["arc_f1"]
    r = ev[ev.variant == "rf80_hmm_deployed"].set_index("trip_id")["arc_f1"]
    add("evaluator_test10: geometric HMM -> RF(80)+HMM", g, r[g.index],
        "experiments/final_evaluation/evaluation_per_trip.csv")
    for set_, f in (("holdout16", "compare_baseline_v1_vs_tuned_candidate_holdout.csv"),
                    ("student_test10", "compare_baseline_v1_vs_tuned_v2_student_test.csv")):
        d = pd.read_csv(E / f)
        b = [c for c in d.columns if c.startswith("arc_f1__") and "baseline" not in c and "delta" not in c][0]
        add(f"{set_}: baseline_v1 -> tuned_v2 decoder", d["arc_f1__baseline_v1"], d[b], f"experiments/{f}")
    pd.DataFrame(out).to_csv(DATA / "paired_tests.csv", index=False)
    print(pd.DataFrame(out).drop(columns="source").to_string(index=False))


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    tests()
    training_dataset()
    paired_tests()
    api_example()


def connectivity():
    """Road-network connectivity per train/student_test trip (read-only)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from src.data_loader import list_trips, load_trip
    rows = []
    for t in list_trips(ROOT / "Student folder"):
        tr = load_trip(t["trip_id"], ROOT / "Student folder")
        n = len(tr.nodes)
        g = coo_matrix((np.ones(len(tr.arcs)), (tr.arcs[:, 0], tr.arcs[:, 1])), shape=(n, n)).tocsr()
        nw, lw = connected_components(g, directed=True, connection="weak")
        ns, ls = connected_components(g, directed=True, connection="strong")
        deg = np.bincount(tr.arcs.ravel(), minlength=n)
        rev = set(map(tuple, tr.arcs[:, ::-1].tolist()))
        rows.append({"split": t["split"], "trip_id": t["trip_id"], "nodes": n, "arcs": len(tr.arcs),
                     "weak_components": int(nw), "largest_weak_frac": round(np.bincount(lw).max() / n, 4),
                     "largest_strong_frac": round(np.bincount(ls).max() / n, 4),
                     "isolated_nodes": int((deg == 0).sum()),
                     "two_way_arc_frac": round(float(np.mean([tuple(a) in rev for a in tr.arcs.tolist()])), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "network_connectivity_per_trip.csv", index=False)
    s = df.groupby("split").agg(trips=("trip_id", "size"), weak_components_median=("weak_components", "median"),
                                weak_components_max=("weak_components", "max"),
                                largest_weak_frac_min=("largest_weak_frac", "min"),
                                largest_strong_frac_median=("largest_strong_frac", "median"),
                                isolated_nodes_total=("isolated_nodes", "sum"),
                                two_way_arc_frac_median=("two_way_arc_frac", "median")).reset_index()
    s.to_csv(DATA / "network_connectivity_summary.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__" and "--connectivity" in sys.argv:
    connectivity()
