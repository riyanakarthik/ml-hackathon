"""Final end-to-end smoke test of the FastAPI backend (in-process TestClient).

  python -m scripts.smoke_test
Checks every permitted trip (80 train + 10 student_test), simulated evaluator
trips (student_test copies WITHOUT .route), schema, coordinate order, arc-id
ranges, determinism, latency, model artifact integrity and error handling.
Writes experiments/final/smoke_test_trips.csv and smoke_test_report.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "final"
REQUIRED = ["trip_id", "gps_coordinates", "predicted_coordinates", "predicted_arc_ids",
            "metrics", "method", "runtime_ms"]


def check_response(body: dict, trip, expect_metrics: bool) -> list[str]:
    errs = []
    for k in REQUIRED:
        if k not in body:
            errs.append(f"missing key {k}")
    if errs:
        return errs
    T = len(trip.track)
    if len(body["gps_coordinates"]) != T:
        errs.append("gps length")
    if len(body["predicted_coordinates"]) != T:
        errs.append("predicted length")
    lat0, lon0 = body["gps_coordinates"][0]
    if not (abs(lat0 - trip.track[0, 1]) < 1e-9 and abs(lon0 - trip.track[0, 0]) < 1e-9):
        errs.append("gps not [lat, lon] / not in file order")
    pc = [c for c in body["predicted_coordinates"] if c is not None]
    if len(pc) != T:
        errs.append(f"{T - len(pc)} unmatched points")
    if pc:
        a = np.array(pc)
        if (np.abs(a[:, 0]) > 90).any() or (np.abs(a[:, 1]) > 180).any():
            errs.append("predicted coords out of range")
        g = np.array(body["gps_coordinates"])[[c is not None for c in body["predicted_coordinates"]]]
        # snapped point must be near its GPS point (<= 200 m fallback radius, in degrees approx)
        dlat = (a[:, 0] - g[:, 0]) * 111_195
        dlon = (a[:, 1] - g[:, 1]) * 111_195 * np.cos(np.radians(g[:, 0]))
        if np.hypot(dlat, dlon).max() > 201:
            errs.append("predicted coordinate > 200 m from GPS point")
    ids = body["predicted_arc_ids"]
    if not ids or not all(isinstance(i, int) and 0 <= i < len(trip.arcs) for i in ids):
        errs.append("arc ids invalid/out of range")
    if (body["metrics"] is not None) != expect_metrics:
        errs.append(f"metrics presence {body['metrics'] is not None} != expected {expect_metrics}")
    if not body["method"].startswith("hmm_viterbi"):
        errs.append(f"unexpected method {body['method']}")
    return errs


def run_all(client, trips, label, expect_metrics, results, store):
    from src.data_loader import load_trip
    for item in trips:
        tid = item["trip_id"]
        trip = load_trip(tid)
        t0 = time.perf_counter()
        r1 = client.post("/predict", json={"trip_id": tid})
        lat_ms = (time.perf_counter() - t0) * 1000
        r2 = client.post("/predict", json={"trip_id": tid})
        b1, b2 = r1.json(), r2.json()
        errs = [] if r1.status_code == 200 else [f"HTTP {r1.status_code}: {b1}"]
        if not errs:
            errs += check_response(b1, trip, expect_metrics)
            if b1["predicted_arc_ids"] != b2["predicted_arc_ids"] or \
                    b1["predicted_coordinates"] != b2["predicted_coordinates"]:
                errs.append("non-deterministic")
            d = b1.get("details", {})
            if d.get("emission") != "ml" or "emission_fallback" in d:
                errs.append(f"ML emission not used: {d.get('emission')} {d.get('emission_fallback')}")
        store[(label, tid)] = b1.get("predicted_arc_ids")
        results.append({"set": label, "trip_id": tid, "split": item["split"], "status": r1.status_code,
                        "ok": not errs, "errors": "; ".join(errs), "latency_ms": round(lat_ms, 1),
                        "server_runtime_ms": b1.get("runtime_ms"), "n_points": len(trip.track),
                        "arc_f1": (b1.get("metrics") or {}).get("arc_f1"),
                        "config": (b1.get("details") or {}).get("config")})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    real_dir = ROOT / "Student folder"
    os.environ["PATHSNAP_DATA_DIR"] = str(real_dir)
    from fastapi.testclient import TestClient

    import main as api
    from src.config import get_config
    from src.data_loader import list_trips
    from src.ml_emission import load_model_bundle

    report = {"python": sys.version.split()[0], "platform": platform.platform(),
              "default_config": api.DEFAULT_CONFIG}
    # --- artifacts
    man = json.load(open(OUT / "frozen_artifacts_sha256.json"))
    art = {}
    for f, h in man["files"].items():
        ok_hash = hashlib.sha256(open(ROOT / f, "rb").read()).hexdigest() == h
        entry = {"sha256_matches_frozen": ok_hash}
        if f.endswith(".joblib"):
            try:
                t0 = time.perf_counter()
                load_model_bundle.cache_clear()
                b = load_model_bundle(str(ROOT / f))
                entry.update({"loads": True, "load_s": round(time.perf_counter() - t0, 2),
                              "model": b["model_name"], "n_train_trips": len(b["trained_on_trips"])})
            except Exception as e:  # noqa: BLE001
                entry.update({"loads": False, "error": str(e)})
        art[f] = entry
    report["artifacts"] = art
    cfg = get_config(api.DEFAULT_CONFIG)
    report["default_model"] = cfg.ml_model

    results, store = [], {}
    with TestClient(api.app) as client:          # runs startup (model preload)
        h = client.get("/health").json()
        report["health"] = h
        trips = client.get("/trips").json()["trips"]
        report["n_trips_listed"] = len(trips)
        client.post("/predict", json={"trip_id": trips[0]["trip_id"]})  # warm-up
        run_all(client, trips, "permitted", True, results, store)

        # --- errors
        errs = {}
        errs["unknown_trip"] = client.post("/predict", json={"trip_id": "99999999"}).status_code
        errs["non_numeric_trip"] = client.post("/predict", json={"trip_id": "abc"}).status_code
        errs["missing_trip_id"] = client.post("/predict", json={}).status_code
        errs["bad_config"] = client.post("/predict", json={"trip_id": "38", "config": "../x"}).status_code
        errs["bad_method"] = client.post("/predict", json={"trip_id": "38", "method": "x"}).status_code
        errs["rollback_tuned_v2"] = client.post("/predict", json={"trip_id": "38", "config": "tuned_v2"}).json()["details"]["emission"]

    # --- simulated evaluator partition: student_test copies without .route
    tmp = Path(tempfile.mkdtemp())
    try:
        st = [t for t in list_trips(real_dir) if t["split"] == "student_test"]
        for t in st:
            src = real_dir / "student_test" / t["trip_id"]
            dst = tmp / "evaluator_test" / t["trip_id"]
            dst.mkdir(parents=True)
            for ext in ("track", "nodes", "arcs"):
                shutil.copy(src / f"{t['trip_id']}.{ext}", dst)
        # corrupted trip: arcs reference a node that does not exist
        bad = tmp / "evaluator_test" / "00000999"
        shutil.copytree(tmp / "evaluator_test" / st[0]["trip_id"], bad)
        for ext in ("track", "nodes", "arcs"):
            (bad / f"{st[0]['trip_id']}.{ext}").rename(bad / f"00000999.{ext}")
        (bad / "00000999.arcs").write_text("0 999999\n")
        os.environ["PATHSNAP_DATA_DIR"] = str(tmp)
        with TestClient(api.app) as client:
            ev = [t for t in client.get("/trips").json()["trips"] if t["trip_id"] != "00000999"]
            assert all(not t["has_reference_route"] for t in ev)
            run_all(client, ev, "evaluator_sim_no_route", False, results, store)
            r = client.post("/predict", json={"trip_id": "00000999"})
            errs["corrupted_trip_file"] = f"{r.status_code}: {r.json().get('detail')}"
    finally:
        os.environ["PATHSNAP_DATA_DIR"] = str(real_dir)
        shutil.rmtree(tmp, ignore_errors=True)
    report["error_handling"] = errs
    same = [store[("permitted", t["trip_id"])] == store[("evaluator_sim_no_route", t["trip_id"])]
            for t in st]
    report["route_file_absent_predictions_identical"] = f"{sum(same)}/{len(same)}"

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(OUT / "smoke_test_trips.csv", index=False)
    lat = df["latency_ms"].tolist()
    report["summary"] = {
        "requests_checked": len(df), "all_ok": bool(df["ok"].all()),
        "failures": df[~df["ok"]][["set", "trip_id", "errors"]].to_dict("records"),
        "latency_ms_p50": round(statistics.median(lat), 1),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
        "latency_ms_max": round(max(lat), 1),
        "max_trip_points": int(df["n_points"].max()),
        "latency_ms_for_max_trip": float(df.loc[df["n_points"].idxmax(), "latency_ms"]),
    }
    json.dump(report, open(OUT / "smoke_test_report.json", "w"), indent=1, default=str)
    print(json.dumps({k: report[k] for k in ("summary", "error_handling",
                                              "route_file_absent_predictions_identical")}, indent=1))
    print("artifacts:", {k: (v.get("sha256_matches_frozen"), v.get("loads")) for k, v in art.items()})


if __name__ == "__main__":
    main()
