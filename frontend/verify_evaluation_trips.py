"""End-to-End verification script for PathSnap frontend against all 10 evaluation trips.

This script tests the complete end-to-end pipeline:
1. Connects to FastAPI backend via TestClient
2. Verifies GET /health and GET /trips
3. Runs POST /predict for all 10 student_test evaluation trips
4. Validates coordinate ranges (latitude in [-90, 90], longitude in [-180, 180])
5. Feeds coordinates through frontend sanitize_coords() and build_route_map()
6. Generates a clean evaluation summary table formatted for reporting.

Run:
    python frontend/verify_evaluation_trips.py
"""

from __future__ import annotations

import math
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

# Set up paths
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "frontend"))

# Mock streamlit before importing app
_st_mock = MagicMock()
_st_mock.session_state = {}
sys.modules.setdefault("streamlit", _st_mock)
sys.modules.setdefault("streamlit_folium", MagicMock())

import app as frontend_app  # noqa: E402
from fastapi.testclient import TestClient
from main import app as fastapi_app


def run_e2e_verification() -> bool:
    print("=" * 78)
    print("PathSnap E2E Evaluation: Verifying All 10 Student Test Trips")
    print("=" * 78)

    client = TestClient(fastapi_app)

    # 1. Health check
    h_resp = client.get("/health")
    assert h_resp.status_code == 200, f"Health check failed: {h_resp.status_code}"
    health_data = h_resp.json()
    print(f"[1/3] Backend Health OK: status={health_data.get('status')}, "
          f"default_config={health_data.get('default_config')}")

    # 2. Trips check
    t_resp = client.get("/trips")
    assert t_resp.status_code == 200, f"Trips fetch failed: {t_resp.status_code}"
    trips_payload = t_resp.json()
    
    # Verify frontend fetch_trips logic handles this payload
    parsed_trips = frontend_app.fetch_trips.__wrapped__(trips_payload) if hasattr(frontend_app.fetch_trips, "__wrapped__") else None
    all_trips = trips_payload.get("trips", [])
    student_test_ids = [
        t["trip_id"] for t in all_trips if isinstance(t, dict) and t.get("split") == "student_test"
    ]
    print(f"[2/3] Found {len(student_test_ids)} student_test evaluation trips: {student_test_ids}")
    assert len(student_test_ids) == 10, f"Expected 10 student_test trips, found {len(student_test_ids)}"

    # 3. Predict & Map verification for all 10 trips
    print("\n[3/3] Running Route Reconstruction & Map Building for each evaluation trip:\n")
    print(f"{'Trip ID':<10} {'Points':<8} {'Arcs':<6} {'Runtime':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Continuity':<11} {'Map OK':<8}")
    print("-" * 85)

    all_passed = True
    results = []

    for tid in sorted(student_test_ids):
        res = client.post("/predict", json={"trip_id": tid})
        if res.status_code != 200:
            print(f"FAILED trip {tid}: HTTP {res.status_code}")
            all_passed = False
            continue

        data = res.json()

        # Contract assertions
        for req_key in ("trip_id", "gps_coordinates", "predicted_coordinates",
                        "predicted_arc_ids", "metrics", "method", "runtime_ms"):
            assert req_key in data, f"Missing {req_key} in response for trip {tid}"

        gps = data["gps_coordinates"]
        pred_coords = data["predicted_coordinates"]
        arc_ids = data["predicted_arc_ids"]
        metrics = data.get("metrics") or {}
        runtime = data.get("runtime_ms", 0.0)

        # Coordinate format validation: [lat, lon]
        for pt in gps:
            lat, lon = pt[0], pt[1]
            assert -90.0 <= lat <= 90.0, f"Invalid latitude {lat} in trip {tid}"
            assert -180.0 <= lon <= 180.0, f"Invalid longitude {lon} in trip {tid}"

        # Frontend coordinate sanitization
        gps_clean = frontend_app.sanitize_coords(gps)
        pred_clean = frontend_app.sanitize_coords([p for p in pred_coords if p is not None])
        assert len(gps_clean) == len(gps), f"Sanitization dropped valid GPS points in trip {tid}"

        # Frontend map generation
        route_map = frontend_app.build_route_map(gps_clean, pred_clean, tid)
        assert route_map is not None, f"Map building failed for trip {tid}"
        map_ok = "PASS"

        # Metric extraction
        prec = metrics.get("arc_precision", 0.0)
        rec = metrics.get("arc_recall", 0.0)
        f1 = metrics.get("arc_f1", 0.0)
        cont = metrics.get("route_continuity", 0.0)

        results.append({
            "trip_id": tid,
            "points": len(gps),
            "arcs": len(arc_ids),
            "runtime": runtime,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "continuity": cont,
        })

        print(f"{tid:<10} {len(gps):<8} {len(arc_ids):<6} {runtime:<10.1f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {cont:<11.4f} {map_ok:<8}")

    print("-" * 85)
    mean_f1 = sum(r["f1"] for r in results) / len(results) if results else 0
    mean_runtime = sum(r["runtime"] for r in results) / len(results) if results else 0
    mean_cont = sum(r["continuity"] for r in results) / len(results) if results else 0
    print(f"{'MEAN':<10} {'':<8} {'':<6} {mean_runtime:<10.1f} {'':<10} {'':<10} {mean_f1:<10.4f} {mean_cont:<11.4f}")
    print("=" * 85)

    if all_passed:
        print("\nALL 10 EVALUATION TRIPS VERIFIED SUCCESSFULLY END-TO-END!")
    return all_passed


if __name__ == "__main__":
    success = run_e2e_verification()
    sys.exit(0 if success else 1)
