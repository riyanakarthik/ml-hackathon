"""Contract tests for the PathSnap Streamlit frontend.

These tests verify that the frontend's API-calling helpers and map-building
logic correctly handle the FastAPI backend contract — without requiring a
running backend.  All mocked payloads live exclusively in this file.

Run:
    python -m pytest frontend/test_contract.py -v
    # or simply
    python -m unittest frontend/test_contract.py -v
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import MagicMock, patch

# Import the helpers under test from the sibling app module.
# We use importlib so this test file works regardless of how the
# working directory or sys.path is set up.
import importlib
import sys
import os

# Ensure the frontend package is importable
_frontend_dir = os.path.dirname(os.path.abspath(__file__))
if _frontend_dir not in sys.path:
    sys.path.insert(0, _frontend_dir)

# We need to mock streamlit before importing app, because app.py calls
# st.set_page_config at module level.
_st_mock = MagicMock()
_st_mock.session_state = {}
sys.modules.setdefault("streamlit", _st_mock)
sys.modules.setdefault("streamlit_folium", MagicMock())

import app as frontend_app  # noqa: E402


# ---------------------------------------------------------------------------
# Realistic mock payloads (live exclusively in this file)
# ---------------------------------------------------------------------------

MOCK_HEALTH_OK = {"status": "ok", "data_dir": "/data"}

MOCK_TRIPS_WRAPPED = {"count": 3, "trips": ["00000038", "00000040", "00000052"]}

MOCK_TRIPS_LIST = ["00000038", "00000040", "00000052"]

MOCK_TRIPS_ALT_KEY = {"trip_ids": ["00000038", "00000040"]}

MOCK_PREDICT_RESPONSE = {
    "trip_id": "00000038",
    "gps_coordinates": [
        [51.5074, -0.1278],
        [51.5080, -0.1270],
        [51.5085, -0.1265],
    ],
    "predicted_coordinates": [
        [51.5075, -0.1277],
        [51.5081, -0.1269],
        [51.5086, -0.1264],
    ],
    "predicted_arc_ids": [12, 15, 18, 22],
    "metrics": {
        "arc_precision": 0.8571,
        "arc_recall": 0.9876,
        "arc_f1": 0.9180,
        "length_precision": 0.9060,
        "length_recall": 0.9960,
        "route_continuity": 0.9990,
    },
    "method": "hmm_viterbi",
    "runtime_ms": 221.3,
}

BASE_URL = "http://localhost:8000"


# ===================================================================
# Test: GET /health
# ===================================================================
class TestHealthCheck(unittest.TestCase):
    """Verify check_health handles success, non-200, and connection errors."""

    @patch("app.requests.get")
    def test_health_ok(self, mock_get: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = MOCK_HEALTH_OK
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        result = frontend_app.check_health(BASE_URL)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "ok")
        mock_get.assert_called_once_with(f"{BASE_URL}/health", timeout=5)

    @patch("app.requests.get")
    def test_health_non_200(self, mock_get: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 503
        resp.json.return_value = {"detail": "service unavailable"}
        exc = __import__("requests").exceptions.HTTPError(response=resp)
        resp.raise_for_status.side_effect = exc
        mock_get.return_value = resp

        result = frontend_app.check_health(BASE_URL)
        self.assertIsInstance(result, str)
        self.assertIn("503", result)

    @patch("app.requests.get")
    def test_health_connection_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = __import__("requests").exceptions.ConnectionError()

        result = frontend_app.check_health(BASE_URL)
        self.assertIsInstance(result, str)
        self.assertIn("Connection refused", result)


# ===================================================================
# Test: GET /trips
# ===================================================================
class TestFetchTrips(unittest.TestCase):
    """Verify fetch_trips parses various response shapes and handles errors."""

    @patch("app.requests.get")
    def test_trips_wrapped_dict(self, mock_get: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = MOCK_TRIPS_WRAPPED
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        result = frontend_app.fetch_trips(BASE_URL)
        self.assertIsInstance(result, list)
        self.assertEqual(result, ["00000038", "00000040", "00000052"])

    @patch("app.requests.get")
    def test_trips_raw_list(self, mock_get: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = MOCK_TRIPS_LIST
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        result = frontend_app.fetch_trips(BASE_URL)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

    @patch("app.requests.get")
    def test_trips_alt_key(self, mock_get: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = MOCK_TRIPS_ALT_KEY
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        result = frontend_app.fetch_trips(BASE_URL)
        self.assertIsInstance(result, list)
        self.assertEqual(result, ["00000038", "00000040"])

    @patch("app.requests.get")
    def test_trips_offline(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = __import__("requests").exceptions.ConnectionError()

        result = frontend_app.fetch_trips(BASE_URL)
        self.assertIsInstance(result, str)
        self.assertIn("connection refused", result.lower())


# ===================================================================
# Test: POST /predict
# ===================================================================
class TestRunPrediction(unittest.TestCase):
    """Verify run_prediction validates contract and handles errors."""

    @patch("app.requests.post")
    def test_predict_success(self, mock_post: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = MOCK_PREDICT_RESPONSE.copy()
        resp.raise_for_status = MagicMock()
        mock_post.return_value = resp

        result = frontend_app.run_prediction(BASE_URL, "00000038")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["trip_id"], "00000038")
        self.assertEqual(result["method"], "hmm_viterbi")
        self.assertIsInstance(result["gps_coordinates"], list)
        self.assertIsInstance(result["predicted_arc_ids"], list)

        # Verify exact request URL and body
        mock_post.assert_called_once_with(
            f"{BASE_URL}/predict",
            json={"trip_id": "00000038"},
            timeout=120,
        )

    @patch("app.requests.post")
    def test_predict_missing_key(self, mock_post: MagicMock) -> None:
        """If a required key is missing the function returns an error string."""
        required_keys = [
            "trip_id", "gps_coordinates", "predicted_coordinates",
            "predicted_arc_ids", "metrics", "method", "runtime_ms",
        ]
        for drop_key in required_keys:
            payload = {k: v for k, v in MOCK_PREDICT_RESPONSE.items() if k != drop_key}
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = payload
            resp.raise_for_status = MagicMock()
            mock_post.return_value = resp

            result = frontend_app.run_prediction(BASE_URL, "00000038")
            self.assertIsInstance(result, str, f"Expected error string when '{drop_key}' is missing")
            self.assertIn(drop_key, result)

    @patch("app.requests.post")
    def test_predict_connection_error(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = __import__("requests").exceptions.ConnectionError()

        result = frontend_app.run_prediction(BASE_URL, "00000038")
        self.assertIsInstance(result, str)
        self.assertIn("connection refused", result.lower())


# ===================================================================
# Test: Coordinate sanitization
# ===================================================================
class TestSanitizeCoords(unittest.TestCase):
    """Verify coordinate validation rejects invalid / out-of-range values."""

    def test_valid_coords(self) -> None:
        raw = [[51.5, -0.1], [48.8, 2.3]]
        clean = frontend_app.sanitize_coords(raw)
        self.assertEqual(len(clean), 2)

    def test_filters_nan_inf(self) -> None:
        raw = [[51.5, -0.1], [float("nan"), 2.0], [float("inf"), 1.0]]
        clean = frontend_app.sanitize_coords(raw)
        self.assertEqual(len(clean), 1)

    def test_filters_out_of_range(self) -> None:
        raw = [[91.0, 0.0], [-91.0, 0.0], [0.0, 181.0], [0.0, -181.0], [45.0, 90.0]]
        clean = frontend_app.sanitize_coords(raw)
        self.assertEqual(len(clean), 1)

    def test_filters_non_numeric(self) -> None:
        raw = [["foo", "bar"], None, 42, [51.0]]
        clean = frontend_app.sanitize_coords(raw)
        self.assertEqual(len(clean), 0)


# ===================================================================
# Test: build_route_map
# ===================================================================
class TestBuildRouteMap(unittest.TestCase):
    """Verify build_route_map runs without errors on contract data."""

    def test_map_from_mock_payload(self) -> None:
        gps = frontend_app.sanitize_coords(MOCK_PREDICT_RESPONSE["gps_coordinates"])
        pred = frontend_app.sanitize_coords(MOCK_PREDICT_RESPONSE["predicted_coordinates"])
        m = frontend_app.build_route_map(gps, pred, "00000038")
        self.assertIsInstance(m, __import__("folium").Map)

    def test_map_empty_coords(self) -> None:
        m = frontend_app.build_route_map([], [], "empty")
        self.assertIsInstance(m, __import__("folium").Map)

    def test_map_single_point(self) -> None:
        m = frontend_app.build_route_map([(51.5, -0.1)], [], "single")
        self.assertIsInstance(m, __import__("folium").Map)


if __name__ == "__main__":
    unittest.main()

