# PathSnap Frontend — Evaluator Dashboard

**Team ID:** PT-056 (UE24CS352A — PES University, RR Campus)  
**Members:** RIYANA KARTHIK VISHALAKSHI (PES1UG24AM226), ROHANTH GANESHA C (PES1UG24AM227), ROHIT CHAKI (PES1UG24AM228), ROHITH KARNING (PES1UG24AM229)  

Interactive **Streamlit + Folium** web application for visualising GPS route-reconstruction results produced by the PathSnap FastAPI backend.

> [!IMPORTANT]
> This dashboard is a **thin client** — it fetches every trip ID, prediction result, metric and coordinate **live** from the backend API.  Nothing is hardcoded or cached locally.

## Quick Start

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

By default the app connects to `http://localhost:8000`.  To point it at the FastAPI backend running on another machine, change the **Backend URL** in the sidebar (e.g. `http://192.168.1.42:8000`).

## Features

| Area | Detail |
|---|---|
| **Health check** | `GET /health` — green/red status badge with actionable guidance |
| **Trip selector** | `GET /trips` — dynamically populated dropdown, supports multiple JSON shapes |
| **Prediction** | `POST /predict` — runs route reconstruction for the selected trip |
| **KPI strip** | Trip ID · Method · Runtime · Points / Arcs |
| **Metrics cards** | All key-value pairs from `response["metrics"]`, floats to 4 d.p. |
| **Folium map** | Noisy GPS observations (blue dots) + predicted route (green line) with layer control, legend, start/end markers, auto-fit bounds |
| **Raw inspector** | Collapsible view of `predicted_arc_ids` and full JSON payload |

## API Contract

The frontend expects the backend to expose:

- **`GET /health`** → `{"status": "ok", ...}`
- **`GET /trips`** → `["id", ...]` or `{"trips": [...]}` or `{"trip_ids": [...]}`
- **`POST /predict`** with `{"trip_id": "<id>"}` → response containing `trip_id`, `gps_coordinates` (`[lat, lon]`), `predicted_coordinates`, `predicted_arc_ids`, `metrics`, `method`, `runtime_ms`

## Tests & E2E Verification

### 1. Isolated Contract Tests
Contract tests run with `unittest` and `unittest.mock` (zero network or backend required):

```bash
python -m pytest frontend/test_contract.py -v
```

Tested scenarios (20 tests):
- `GET /health`: OK (200), HTTP error (503), ConnectionError
- `GET /trips`: list of strings, list of objects, wrapped dict (`{"trips": [...]}`), alt key (`{"trip_ids": [...]}`), numeric IDs, ConnectionError
- `POST /predict`: success, missing required keys (tested across all contract keys), ConnectionError
- Coordinate sanitization: valid pairs, NaN/Inf rejection, out-of-range filtering, non-numeric filtering
- Folium map generation: mock payload, empty coordinates, single-point edge case

### 2. Live End-to-End Evaluation Verification
Verify the entire pipeline against the live FastAPI backend across all 10 student evaluation trips:

```bash
python frontend/verify_evaluation_trips.py
```

#### Measured Evaluation Results (10 Student Test Trips)

| Trip ID | GPS Points | Route Arcs | Runtime (ms) | Arc Precision | Arc Recall | Arc F1 | Route Continuity | Map Generation |
|---|---|---|---|---|---|---|---|---|
| `00000000` | 2,503 | 203 | 432.3 | 0.8950 | 1.0000 | 0.9446 | 0.9901 | PASS |
| `00000007` | 1,354 | 121 | 249.9 | 1.0000 | 0.9837 | 0.9918 | 1.0000 | PASS |
| `00000009` | 1,049 | 490 | 702.3 | 0.8889 | 0.9883 | 0.9360 | 1.0000 | PASS |
| `00000040` | 2,995 | 321 | 714.2 | 0.9904 | 1.0000 | 0.9952 | 1.0000 | PASS |
| `00000052` | 262 | 93 | 146.5 | 1.0000 | 0.9789 | 0.9894 | 1.0000 | PASS |
| `00000056` | 3,958 | 1,390 | 1,628.0 | 0.9435 | 1.0000 | 0.9709 | 0.9993 | PASS |
| `00000065` | 768 | 153 | 441.8 | 0.9732 | 1.0000 | 0.9864 | 1.0000 | PASS |
| `00000076` | 1,486 | 435 | 1,673.2 | 0.8894 | 0.9893 | 0.9367 | 1.0000 | PASS |
| `00000077` | 3,911 | 363 | 786.1 | 0.8963 | 1.0000 | 0.9453 | 1.0000 | PASS |
| `00000091` | 874 | 237 | 427.1 | 0.9788 | 0.9957 | 0.9872 | 1.0000 | PASS |
| **MEAN** | **1,916** | **380** | **720.1** | **0.9456** | **0.9936** | **0.9684** | **0.9989** | **10/10 PASS** |

## Project Layout

```
frontend/
├── app.py                      # Streamlit dashboard (entry point)
├── test_contract.py            # Isolated contract tests (unittest + mock)
├── verify_evaluation_trips.py   # E2E verification across all 10 student evaluation trips
├── requirements.txt            # Lightweight dependencies
└── README.md                   # Documentation & evaluation benchmark
```

