"""Robustness of the deployed API (default config, real saved model) on synthetic fixtures."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.test_pathsnap import write_trip

LON0, LAT0 = 10.0, 50.0


def _track(n=16, t0=1.0, dt=1.0, north_off=0.00002):
    east = [[LON0 + f * 0.002, LAT0 + north_off, t0 + i * dt] for i, f in enumerate(np.linspace(0.05, 0.95, n // 2))]
    north = [[LON0 + 0.00203, LAT0 + f * 0.002, t0 + (n // 2 + i) * dt]
             for i, f in enumerate(np.linspace(0.05, 0.95, n // 2))]
    return np.array(east + north)


@pytest.fixture
def client_for(tmp_path, monkeypatch):
    def make(track, tid="00000001", empty=False):
        d = write_trip(tmp_path, tid, track=track if not empty else _track())
        if empty:
            (d / f"{tid}.track").write_text("")
        monkeypatch.setenv("PATHSNAP_DATA_DIR", str(tmp_path))
        from main import app
        return TestClient(app)
    return make


def _ok(r, n_points):
    assert r.status_code == 200, r.text
    b = r.json()
    assert len(b["gps_coordinates"]) == len(b["predicted_coordinates"]) == n_points
    return b


def test_missing_gps_samples(client_for):
    tr = np.delete(_track(), [3, 4, 5, 10, 11], axis=0)   # dropped observations
    b = _ok(client_for(tr).post("/predict", json={"trip_id": "1"}), len(tr))
    assert b["predicted_arc_ids"] == [0, 1]


def test_large_timestamp_gap(client_for):
    tr = _track()
    tr[8:, 2] += 900.0                                    # 15-minute gap mid-trip
    b = _ok(client_for(tr).post("/predict", json={"trip_id": "1"}), len(tr))
    assert b["predicted_arc_ids"] == [0, 1]


def test_single_point_track(client_for):
    tr = _track()[:1]
    b = _ok(client_for(tr).post("/predict", json={"trip_id": "1"}), 1)
    assert b["predicted_arc_ids"] and b["predicted_coordinates"][0] is not None


def test_no_candidates_anywhere(client_for):
    tr = _track(north_off=0.02)                            # ~2.2 km from every road (> 200 m fallback)
    tr[8:, 1] += 0.02
    b = _ok(client_for(tr).post("/predict", json={"trip_id": "1"}), len(tr))
    assert b["predicted_arc_ids"] == []
    assert all(c is None for c in b["predicted_coordinates"])
    assert b["details"]["points_without_candidates"] == len(tr)


def test_sparse_candidates_use_fallback_radius(client_for):
    tr = _track(north_off=0.0009)                          # ~100 m off the east-west road
    b = _ok(client_for(tr).post("/predict", json={"trip_id": "1"}), len(tr))
    assert b["details"]["points_without_candidates"] == 0
    assert b["predicted_arc_ids"]


def test_empty_track_file_is_a_clean_error(client_for):
    r = client_for(None, empty=True).post("/predict", json={"trip_id": "1"})
    assert r.status_code == 404 and "Empty file" in r.json()["detail"]


def test_unknown_and_malformed_trip_ids(client_for):
    c = client_for(_track())
    assert c.post("/predict", json={"trip_id": "424242"}).status_code == 404
    assert c.post("/predict", json={"trip_id": "abc"}).status_code == 404
    assert c.post("/predict", json={}).status_code == 422


def test_repeated_calls_are_deterministic(client_for):
    c = client_for(_track())
    bodies = [c.post("/predict", json={"trip_id": "1"}).json() for _ in range(3)]
    assert bodies[0]["details"]["emission"] == "ml" and "emission_fallback" not in bodies[0]["details"]
    for k in ("predicted_arc_ids", "predicted_coordinates", "point_arc_ids", "metrics"):
        assert bodies[0][k] == bodies[1][k] == bodies[2][k]
