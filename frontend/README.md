# PathSnap Frontend — Evaluator Dashboard

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

## Tests

Contract tests run with the standard library — no extra test dependencies:

```bash
python -m pytest frontend/test_contract.py -v
```

Tested scenarios: health OK / error / offline, trips (list / dict / alt-key / offline), prediction (success / missing keys / connection error), coordinate sanitization, and map building.

## Project Layout

```
frontend/
├── app.py               # Streamlit dashboard (entry point)
├── test_contract.py      # Isolated contract tests (unittest + mock)
├── requirements.txt      # Lightweight dependencies
└── README.md             # This file
```

