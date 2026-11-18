"""Post-freeze regression check: evaluator_test predictions must be unchanged.

  python -m scripts.verify_post_freeze
Re-runs POST /predict (deployed config) for every evaluator trip with
PATHSNAP_DATA_DIR set (a) to the repo root and (b) directly to evaluator_test/,
and compares predicted_arc_ids and arc F1 with the FROZEN records in
experiments/final_evaluation/ (which this script never modifies).
Writes experiments/final_evaluation/post_freeze_checks/.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FE = ROOT / "experiments" / "final_evaluation"
OUT = FE / "post_freeze_checks"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    frozen_routes = json.load(open(FE / "predicted_routes.json"))["rf80_hmm_deployed"]
    frozen = pd.read_csv(FE / "evaluation_per_trip.csv")
    frozen = frozen[frozen.variant == "rf80_hmm_deployed"].set_index("trip_id")
    from fastapi.testclient import TestClient

    import main as api
    rows = []
    for mode, data_dir in (("repo_root", ROOT), ("direct_split_dir", ROOT / "evaluator_test")):
        os.environ["PATHSNAP_DATA_DIR"] = str(data_dir)
        with TestClient(api.app) as c:
            listed = [t["trip_id"] for t in c.get("/trips").json()["trips"]]
            for tid in sorted(frozen_routes):
                b = c.post("/predict", json={"trip_id": tid}).json()
                rows.append({"mode": mode, "trip_id": tid, "listed_by_trips": tid in listed,
                             "routes_identical": b["predicted_arc_ids"] == frozen_routes[tid],
                             "arc_f1_now": b["metrics"]["arc_f1"],
                             "arc_f1_frozen": float(frozen.loc[int(tid), "arc_f1"]),
                             "config": b["details"]["config"], "emission": b["details"]["emission"],
                             "response_keys": ",".join(sorted(b))})
    os.environ.pop("PATHSNAP_DATA_DIR", None)
    df = pd.DataFrame(rows)
    df["f1_identical"] = (df.arc_f1_now - df.arc_f1_frozen).abs() < 1e-12
    df.to_csv(OUT / "post_freeze_prediction_check.csv", index=False)
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    rep = {
        "checked_at": dt.datetime.now().isoformat(timespec="seconds"),
        "git_head_at_check": rev,
        "note": "post-freeze SOFTWARE SETUP change only (trip discovery in src/data_loader.py::list_trips); "
                "prediction model, configs and inference code unchanged",
        "src/data_loader.py_sha256_now": hashlib.sha256((ROOT / "src/data_loader.py").read_bytes()).hexdigest(),
        "src/data_loader.py_sha256_frozen": json.load(open(FE / "evaluation_manifest.json"))["hashes"]["source"]["src/data_loader.py"],
        "all_routes_identical": bool(df.routes_identical.all()),
        "all_f1_identical": bool(df.f1_identical.all()),
        "all_listed_by_trips_in_both_modes": bool(df.listed_by_trips.all()),
        "n_checks": len(df),
        "response_keys": sorted(set(df.response_keys))[0].split(","),
    }
    json.dump(rep, open(OUT / "post_freeze_check.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
