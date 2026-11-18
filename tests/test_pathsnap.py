import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.data_loader import TripDataError, list_trips, load_trip, normalize_trip_id
from src.decoder import viterbi_log
from src.evaluation import route_metrics
from src.matcher import LocalProjector, RoadNetwork, generate_candidates, point_segment_projection
from src.pipeline import predict_trip, to_latlon

REAL_DATA = Path(__file__).resolve().parent.parent / "Student folder"
has_real = pytest.mark.skipif(not (REAL_DATA / "train" / "00000038").is_dir(),
                              reason="real dataset not present")


def write_trip(root: Path, tid="00000001", split="train", route=True, track=None):
    """Synthetic L-shaped road: n0 -> n1 -> n2 (east then north) + reverse arcs."""
    d = root / split / tid
    d.mkdir(parents=True)
    lon0, lat0 = 10.0, 50.0
    nodes = np.array([[lon0, lat0], [lon0 + 0.002, lat0], [lon0 + 0.002, lat0 + 0.002]])
    arcs = np.array([[0, 1], [1, 2], [1, 0], [2, 1]])
    if track is None:
        # Drive east along arc 0 then north along arc 1, ~2 m north/east noise.
        east = [[lon0 + f * 0.002, lat0 + 0.00002, i + 1] for i, f in enumerate(np.linspace(0.05, 0.95, 8))]
        north = [[lon0 + 0.00203, lat0 + f * 0.002, 9 + i] for i, f in enumerate(np.linspace(0.05, 0.95, 8))]
        track = np.array(east + north)
    np.savetxt(d / f"{tid}.nodes", nodes, fmt="%.6f", delimiter="\t")
    np.savetxt(d / f"{tid}.arcs", arcs, fmt="%d", delimiter="\t")
    np.savetxt(d / f"{tid}.track", track, fmt="%.6f", delimiter="\t")
    if route:
        np.savetxt(d / f"{tid}.route", np.array([0, 1]), fmt="%d")
    return d


# ---------------------------------------------------------------- file loading

def test_load_synthetic_trip(tmp_path):
    write_trip(tmp_path)
    trip = load_trip(1, tmp_path)
    assert trip.trip_id == "00000001" and trip.split == "train"
    assert trip.track.shape == (16, 3) and trip.nodes.shape == (3, 2) and trip.arcs.shape == (4, 2)
    assert trip.arcs.dtype == np.int64
    # Files are lon, lat: longitude column is ~10, latitude ~50.
    assert np.allclose(trip.nodes[0], [10.0, 50.0])


def test_normalize_and_missing(tmp_path):
    assert normalize_trip_id(38) == "00000038"
    assert normalize_trip_id("00000091") == "00000091"
    with pytest.raises(TripDataError):
        normalize_trip_id("abc")
    with pytest.raises(TripDataError):
        load_trip("999", tmp_path)


def test_invalid_arc_node_reference(tmp_path):
    d = write_trip(tmp_path)
    np.savetxt(d / "00000001.arcs", np.array([[0, 7]]), fmt="%d")
    with pytest.raises(TripDataError, match="outside"):
        load_trip(1, tmp_path)


def test_list_trips(tmp_path):
    write_trip(tmp_path, "00000001", "train")
    write_trip(tmp_path, "00000002", "student_test", route=False)
    items = {t["trip_id"]: t for t in list_trips(tmp_path)}
    assert items["00000002"]["split"] == "student_test"
    assert items["00000002"]["has_reference_route"] is False


@has_real
def test_real_trip_shapes():
    trip = load_trip("38", REAL_DATA)
    assert trip.split == "train"
    assert trip.arcs.max() < len(trip.nodes)
    t91 = load_trip(91, REAL_DATA)
    assert t91.split == "student_test"


# ---------------------------------------------------------------- geometry / arc ids

def test_point_segment_projection():
    proj, dist, t = point_segment_projection(np.array([[5.0, 3.0]]), np.array([[0.0, 0.0]]),
                                             np.array([[10.0, 0.0]]))
    assert np.allclose(proj, [[5.0, 0.0]]) and np.allclose(dist, 3.0) and np.allclose(t, 0.5)
    _, dist, t = point_segment_projection(np.array([[-4.0, 3.0]]), np.array([[0.0, 0.0]]),
                                          np.array([[10.0, 0.0]]))
    assert np.allclose(dist, 5.0) and np.allclose(t, 0.0)


def test_projector_roundtrip():
    p = LocalProjector(10.0, 50.0)
    ll = np.array([[10.001, 50.002], [9.99, 49.995]])
    assert np.allclose(p.to_lonlat(p.to_xy(ll)), ll, atol=1e-9)
    # 0.001 deg latitude ~ 111 m
    assert abs(p.to_xy(np.array([[10.0, 50.001]]))[0, 1] - 111.2) < 0.5


def test_candidate_arc_ids_are_row_indices(tmp_path):
    write_trip(tmp_path)
    trip = load_trip(1, tmp_path)
    proj = LocalProjector.from_points(trip.track_lonlat)
    net = RoadNetwork.build(trip.nodes, trip.arcs, proj)
    cands = generate_candidates(net, proj.to_xy(trip.track_lonlat), k=8, radius_m=30)
    # First point lies on the east-west road: arcs 0 and 2 (its reverse).
    assert set(cands[0].arc_ids.tolist()) == {0, 2}
    assert np.all(np.diff(cands[0].distance_m) >= 0)
    assert all(len(c) <= 8 for c in cands)


# ---------------------------------------------------------------- decoding

def test_viterbi_with_missing_candidates():
    em = [np.array([0.0, -5.0]), np.array([]), np.array([-5.0, 0.0])]
    calls = []

    def trans(s, t):
        calls.append((s, t))
        return np.array([[0.0, -1.0], [-1.0, 0.0]])

    # Weak switch penalty (-1) < emission gain (5): switch to state 1 at t=2.
    path, breaks = viterbi_log(em, trans)
    assert calls == [(0, 2)]  # empty step 1 skipped, transition bridges 0 -> 2
    assert path.tolist() == [0, -1, 1] and breaks == [0]
    # Strong transitions force staying in state 0 despite emission at t=2.
    path, _ = viterbi_log(em, lambda s, t: np.array([[0.0, -100.0], [-100.0, 0.0]]))
    assert path.tolist() == [0, -1, 0]


def test_viterbi_break_on_impossible_transition():
    em = [np.array([0.0]), np.array([0.0])]
    path, breaks = viterbi_log(em, lambda s, t: np.array([[-np.inf]]))
    assert path.tolist() == [0, 0] and breaks == [0, 1]


@pytest.mark.parametrize("method", ["hmm", "baseline"])
def test_synthetic_trajectory_output(tmp_path, method):
    write_trip(tmp_path)
    trip = load_trip(1, tmp_path)
    res, net, _ = predict_trip(trip, method)
    assert res.route_arc_ids == [0, 1]
    assert len(res.point_arc_ids) == len(trip.track)
    coords = to_latlon(net, res.point_xy)
    assert len(coords) == len(trip.track)
    lat, lon = coords[0]
    assert 49 < lat < 51 and 9 < lon < 11  # [lat, lon] order


def test_prediction_never_reads_route(tmp_path):
    write_trip(tmp_path, "00000001", "train", route=True)
    write_trip(tmp_path, "00000002", "train", route=False)
    r1, _, _ = predict_trip(load_trip(1, tmp_path))
    r2, _, _ = predict_trip(load_trip(2, tmp_path))
    assert r1.route_arc_ids == r2.route_arc_ids


def test_route_metrics():
    arcs = np.array([[0, 1], [1, 2], [2, 3], [1, 0]])
    m = route_metrics([0, 1, 3], [0, 1, 2], arcs)
    assert m["arc_precision"] == round(2 / 3, 4) and m["arc_recall"] == round(2 / 3, 4)
    assert m["route_continuity"] == 0.5 and m["reference_continuity"] == 1.0


# ---------------------------------------------------------------- API

def test_api_synthetic(tmp_path, monkeypatch):
    write_trip(tmp_path)
    monkeypatch.setenv("PATHSNAP_DATA_DIR", str(tmp_path))
    from main import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/trips").json()["count"] == 1
    r = c.post("/predict", json={"trip_id": "1"})
    assert r.status_code == 200
    body = r.json()
    for key in ("trip_id", "gps_coordinates", "predicted_coordinates", "predicted_arc_ids",
                "metrics", "method", "runtime_ms"):
        assert key in body
    assert body["method"] == "hmm_viterbi"
    assert body["gps_coordinates"][0] == pytest.approx([50.00002, 10.0001], abs=1e-5)  # [lat, lon]
    assert body["predicted_arc_ids"] == [0, 1]
    assert body["metrics"]["arc_recall"] == 1.0
    assert c.post("/predict", json={"trip_id": "424242"}).status_code == 404


@has_real
def test_api_real_trip_38(monkeypatch):
    monkeypatch.setenv("PATHSNAP_DATA_DIR", str(REAL_DATA))
    from main import app
    body = TestClient(app).post("/predict", json={"trip_id": "00000038"}).json()
    assert len(body["gps_coordinates"]) == len(body["predicted_coordinates"])
    lat, lon = body["gps_coordinates"][0]
    assert lat == pytest.approx(53.844471) and lon == pytest.approx(27.494492)
