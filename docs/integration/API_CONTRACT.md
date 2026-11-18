# PathSnap API Contract Specification

**Document Version:** 1.0.0  
**Team ID:** PT-056 (UE24CS352A — PES University, RR Campus)  
**Members:** RIYANA KARTHIK VISHALAKSHI (PES1UG24AM226), ROHANTH GANESHA C (PES1UG24AM227), ROHIT CHAKI (PES1UG24AM228), ROHITH KARNING (PES1UG24AM229)  
**Backend:** FastAPI (`main.py`)  
**Frontend:** Streamlit (`frontend/app.py`)  
**Port:** `8000` (Backend), `8501` (Frontend)

---

## 1. Overview & Coordinate Standard

This document formally specifies the communication contract between the PathSnap FastAPI inference backend (running on Laptop 1) and the Streamlit evaluator dashboard (running on Laptop 2).

> [!IMPORTANT]
> **Coordinate Standard:** All geographical coordinates returned across all endpoints are formatted strictly as:
> `[latitude, longitude]` (decimal degrees, WGS84).
> This matches Folium / Leaflet expectations. Note that raw dataset `.track` and `.nodes` files store data as `[longitude, latitude]`; the backend projection pipeline (`src.pipeline.to_latlon`) converts all coordinates to `[latitude, longitude]` prior to returning the API payload.

---

## 2. Endpoints

### 2.1 Health Check

* **Path:** `GET /health`
* **Purpose:** Heartbeat check to verify Laptop 1 backend reachability, data root resolution, and available HMM configurations.
* **Request:** None
* **Response (HTTP 200 OK):**
```json
{
  "status": "ok",
  "data_dir": "C:\\Users\\...\\Student folder",
  "default_config": "tuned_v2",
  "configs": [
    "baseline_v1",
    "tuned_candidate",
    "tuned_v2"
  ]
}
```

---

### 2.2 Available Trips

* **Path:** `GET /trips`
* **Purpose:** Dynamically retrieve all trips available across partitions. Allows the evaluator dashboard to populate the trip selection dropdown without hardcoding.
* **Request:** None
* **Response (HTTP 200 OK):**
```json
{
  "count": 90,
  "trips": [
    {
      "trip_id": "00000000",
      "split": "student_test",
      "has_reference_route": true
    },
    {
      "trip_id": "00000001",
      "split": "train",
      "has_reference_route": true
    }
  ]
}
```
* **Frontend Handling:** The frontend parser extracts clean, 8-character zero-padded strings (`"00000000"`) from each trip object, ensuring leading zeros are preserved.

---

### 2.3 Route Prediction & Server-Side Evaluation

* **Path:** `POST /predict`
* **Headers:** `Content-Type: application/json`
* **Request Payload:**
```json
{
  "trip_id": "00000038",
  "method": "hmm",
  "config": "tuned_v2"
}
```
* **Field Specifications:**
  * `trip_id` (string | integer, required): The target trip identifier. Normalized to 8-digit zero-padded string server-side.
  * `method` (string, optional, default `"hmm"`): `"hmm"` (Newson & Krumm HMM + Viterbi) or `"baseline"` (greedy geometric nearest-road continuity).
  * `config` (string, optional, default `"tuned_v2"`): HMM configuration profile name. Permitted values: `"tuned_v2"`, `"baseline_v1"`, `"tuned_candidate"`.

* **Response Payload (HTTP 200 OK):**
```json
{
  "trip_id": "00000038",
  "gps_coordinates": [
    [53.844471, 27.494492],
    [53.844512, 27.494580]
  ],
  "predicted_coordinates": [
    [53.844465, 27.494488],
    [53.844508, 27.494575]
  ],
  "predicted_arc_ids": [104, 105, 218, 219],
  "point_arc_ids": [104, 104, 105, 105],
  "metrics": {
    "metric_type": "reference_route_arc_membership (not per-GPS-point accuracy)",
    "arc_precision": 0.8571,
    "arc_recall": 0.9876,
    "arc_f1": 0.9180,
    "length_precision": 0.9060,
    "length_recall": 0.9960,
    "route_continuity": 1.0,
    "reference_continuity": 1.0,
    "n_pred_arcs": 42,
    "n_ref_arcs": 39,
    "n_common_arcs": 36
  },
  "method": "hmm_viterbi",
  "runtime_ms": 145.2,
  "details": {
    "split": "train",
    "timings_ms": {
      "network_ms": 12.1,
      "candidates_ms": 18.4,
      "decode_ms": 114.7
    },
    "points_without_candidates": 0,
    "mean_candidates_per_point": 5.4,
    "sigma_m": 7.42,
    "beta": 15.0,
    "hmm_breaks": 0,
    "unfilled_gaps": 0
  }
}
```

---

## 3. Data Protection & Academic Integrity Guarantees

1. **Prediction Independence:** The prediction pipeline consumes only `.track`, `.nodes`, and `.arcs`. It never accesses `.route`.
2. **Server-Side Metrics Only:** The reference `.route` file is loaded exclusively by `src.evaluation` on Laptop 1 *after* predictions have been computed, solely to calculate verification metrics.
3. **No Ground Truth Leakage to UI:** The UI receives summary metrics and predicted coordinates; it never receives raw evaluator `.route` files or ground-truth arrays.
4. **Missing Ground Truth Handling:** If a trip has no reference route (such as unlabelled evaluation trips), `metrics` is returned as `null`, and the UI displays an informational notice without erroring.
