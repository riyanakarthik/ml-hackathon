"""Tests for the supervised candidate-classifier extension."""
import ast
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import BASELINE_V1
from src.data_loader import load_trip
from src.features import FEATURE_NAMES, SCHEMA_VERSION, candidate_features
from src.labels import align_track_to_route, candidate_labels
from src.matcher import generate_candidates
from src.ml_emission import ModelSchemaError, combine_emissions, load_model_bundle, predict_candidate_probs
from src.pipeline import build_network, predict_trip
from tests.test_pathsnap import write_trip

ROOT = Path(__file__).resolve().parent.parent


def _write_out_and_back(root: Path, tid="00000003"):
    """Drive east on arc 0 (n0->n1), U-turn, drive back west on arc 2 (n1->n0)."""
    lon0, lat0 = 10.0, 50.0
    east = [[lon0 + f * 0.002, lat0 + 0.00002, i + 1] for i, f in enumerate(np.linspace(0.05, 0.95, 10))]
    west = [[lon0 + f * 0.002, lat0 - 0.00002, 11 + i] for i, f in enumerate(np.linspace(0.95, 0.05, 10))]
    d = write_trip(root, tid, track=np.array(east + west))
    np.savetxt(d / f"{tid}.route", np.array([0, 2]), fmt="%d")
    return d


def _assets(root, tid):
    trip = load_trip(tid, root)
    net, xy = build_network(trip)
    return trip, net, xy


# ---------------------------------------------------------------- alignment

def test_alignment_direction_change(tmp_path):
    _write_out_and_back(tmp_path)
    trip, net, xy = _assets(tmp_path, 3)
    al = align_track_to_route(xy, trip.timestamps, [0, 2], net)
    assert al.labels.tolist() == [0] * 10 + [2] * 10   # same geometry, told apart by heading
    assert np.all(np.diff(al.position) >= 0)
    assert al.valid.all()


def test_alignment_repeated_arc_and_invalid(tmp_path):
    _write_out_and_back(tmp_path)
    trip, net, xy = _assets(tmp_path, 3)
    # route 0, 2, 0: arc 0 repeated -> second visit is a distinct state
    al = align_track_to_route(xy, trip.timestamps, [0, 2, 0], net)
    assert al.position.max() <= 2 and np.all(np.diff(al.position) >= 0)
    # far-away points are flagged invalid
    far = xy + np.array([0.0, 500.0])
    al2 = align_track_to_route(far, trip.timestamps, [0, 2], net)
    assert not al2.valid.any()


def test_candidate_labels_shape(tmp_path):
    write_trip(tmp_path)
    trip, net, xy = _assets(tmp_path, 1)
    cands = generate_candidates(net, xy, k=8, radius_m=30)
    al = align_track_to_route(xy, trip.timestamps, [0, 1], net)
    y, v = candidate_labels(cands, al)
    assert len(y) == len(v) == sum(len(c) for c in cands)
    # exactly one positive per point (label always among candidates here)
    starts = np.cumsum([0] + [len(c) for c in cands])
    assert all(y[s:e].sum() == 1 for s, e in zip(starts[:-1], starts[1:]))


# ---------------------------------------------------------------- leakage

def test_inference_modules_do_not_import_reference_code():
    """Feature extraction and ML emission must not touch .route / labels."""
    forbidden = {"src.evaluation", "src.labels", "evaluation", "labels", "src.ml_train", "ml_train"}
    for mod in ("src/features.py", "src/ml_emission.py", "src/matcher.py", "src/decoder.py"):
        tree = ast.parse((ROOT / mod).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                name = (node.module or "")
                assert name.lstrip(".") not in {f.split(".")[-1] for f in forbidden}, (mod, name)
            if isinstance(node, ast.Import):
                assert not any(a.name in forbidden for a in node.names), mod
        # no string literal in code (docstrings excluded) refers to .route files
        docstrings = {id(n.value) for n in ast.walk(tree)
                      if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in docstrings:
                assert ".route" not in node.value, (mod, node.value)


def test_features_identical_without_route_file(tmp_path):
    write_trip(tmp_path / "a", "00000001", route=True)
    write_trip(tmp_path / "b", "00000001", route=False)
    out = []
    for sub in ("a", "b"):
        trip, net, xy = _assets(tmp_path / sub, 1)
        cands = generate_candidates(net, xy, k=8, radius_m=30)
        X, _, _ = candidate_features(net, xy, trip.timestamps, cands)
        out.append(X)
    assert out[0].shape[1] == len(FEATURE_NAMES)
    np.testing.assert_array_equal(out[0], out[1])


def test_group_folds_have_no_trip_overlap():
    import json
    f = ROOT / "experiments" / "ml" / "cv_folds.json"
    if not f.exists():
        pytest.skip("CV not run")
    folds = json.loads(f.read_text())
    flat = [t for g in folds for t in g]
    assert len(flat) == len(set(flat))
    part = json.loads((ROOT / "experiments" / "train_partition.json").read_text())
    assert not set(flat) & set(part["holdout"])   # holdout never in CV


# ---------------------------------------------------------------- probabilities / models

def _tiny_bundle(tmp_path, schema=SCHEMA_VERSION, names=None):
    write_trip(tmp_path)
    trip, net, xy = _assets(tmp_path, 1)
    cands = generate_candidates(net, xy, k=8, radius_m=30)
    X, _, _ = candidate_features(net, xy, trip.timestamps, cands)
    al = align_track_to_route(xy, trip.timestamps, [0, 1], net)
    y, _ = candidate_labels(cands, al)
    model = LogisticRegression(max_iter=500).fit(X, y)
    path = tmp_path / "tiny.joblib"
    joblib.dump({"model": model, "feature_names": names or FEATURE_NAMES,
                 "schema_version": schema}, path)
    return path, trip, net, xy, cands


def test_probability_shapes_and_clipping(tmp_path):
    path, trip, net, xy, cands = _tiny_bundle(tmp_path)
    bundle = load_model_bundle(str(path))
    probs = predict_candidate_probs(bundle, net, xy, trip.timestamps, cands, 30.0)
    assert [len(p) for p in probs] == [len(c) for c in cands]
    assert all(((p >= 0) & (p <= 1)).all() for p in probs)
    geo = [-0.5 * c.distance_m ** 2 for c in cands]
    zero = [np.zeros(len(c)) for c in cands]          # p = 0 must not give -inf
    for mode in ("ml", "hybrid"):
        em = combine_emissions(zero, geo, mode, 1.0, 1e-4)
        assert all(np.isfinite(e).all() for e in em)
        assert [len(e) for e in em] == [len(c) for c in cands]
    em = combine_emissions(probs, geo, "ml", 1.0)
    assert all(np.isclose(np.exp(e).sum(), 1.0) for e in em)   # per-point normalised


def test_schema_mismatch_rejected(tmp_path):
    path, *_ = _tiny_bundle(tmp_path, schema=SCHEMA_VERSION + 99)
    with pytest.raises(ModelSchemaError):
        load_model_bundle(str(path))


def test_ml_config_falls_back_to_geometric_when_model_missing(tmp_path):
    write_trip(tmp_path)
    trip = load_trip(1, tmp_path)
    cfg = BASELINE_V1.with_(emission="ml", ml_model="does_not_exist")
    res, _, _ = predict_trip(trip, "hmm", config=cfg)
    assert res.info["emission"] == "geometric" and "emission_fallback" in res.info
    assert res.route_arc_ids == [0, 1]


def test_ml_emission_decodes_synthetic(tmp_path):
    path, trip, *_ = _tiny_bundle(tmp_path)
    cfg = BASELINE_V1.with_(emission="hybrid", ml_model=str(path))
    res, _, _ = predict_trip(trip, "hmm", config=cfg)
    assert res.info["emission"] == "hybrid" and res.method == "hmm_viterbi_hybrid_emission"
    assert res.route_arc_ids == [0, 1]


def test_saved_models_load():
    models = sorted((ROOT / "models").glob("*.joblib")) if (ROOT / "models").is_dir() else []
    if not models:
        pytest.skip("no trained models saved")
    for p in models:
        b = load_model_bundle(str(p))
        assert hasattr(b["model"], "predict_proba")
        assert b["trained_on_trips"] and all(t.isdigit() for t in b["trained_on_trips"])
