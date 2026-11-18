# PathSnap: GPS-to-road map matching

FastAPI backend that snaps a GPS track onto a road graph and returns the matched arc route.

## Setup and run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                                      # tests
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000              # server
```

Data is read from `Student folder/` by default. Override it with `PATHSNAP_DATA_DIR=/path/to/data`; the layout is `<dir>/<split>/<trip_id>/<trip_id>.{track,nodes,arcs[,route]}`.

```bash
curl localhost:8000/health
curl localhost:8000/trips
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d '{"trip_id": "00000038"}'                  # optional: "method": "baseline"
.venv/bin/python -m scripts.evaluate --split train --method hmm    # aggregate metrics
```

## `/predict` response

| field | meaning |
|---|---|
| `gps_coordinates` | raw track, `[lat, lon]` |
| `predicted_coordinates` | each GPS point projected onto its matched arc, `[lat, lon]` (`null` if unmatched) |
| `predicted_arc_ids` | connected matched route (arc ids = row index in `.arcs`), gaps filled by shortest path |
| `point_arc_ids` | matched arc per GPS point (`-1` if none) |
| `metrics` | route-level metrics against `.route` (`null` when no `.route` exists) |
| `method` | `hmm_viterbi` (default) or `geometric_greedy_continuity` |
| `runtime_ms`, `details` | timings, estimated GPS sigma, HMM breaks, etc. |

## Pipeline

1. **`src/data_loader.py`** loads and validates `.track`/`.nodes`/`.arcs`. It never reads `.route`.
2. **`src/matcher.py`** projects lon/lat to local metres (equirectangular) and densifies arcs into a `cKDTree`. For each point it computes exact point-to-segment distances and keeps up to 8 candidates within 50 m (200 m fallback).
3. **`src/decoder.py`**
   - `greedy_continuity_decode`: the baseline. It scores each point independently, with bonuses for staying on the same arc or moving to a successor arc. It is **not** an HMM.
   - `viterbi_log`: a generic log-space Viterbi. It handles variable candidate counts, skips empty steps and restarts the chain on impossible transitions.
   - `hmm_viterbi_decode`: a Newson & Krumm (2009)-style HMM. Emissions are Gaussian on distance, with σ estimated per trip from the GPS-to-nearest-road MAD. Transitions use `exp(-|network distance − straight-line distance| / β)` with β = 15 m, and network distances come from bounded Dijkstra.
4. **`src/evaluation.py`** is the **only** code that reads `.route`, and it runs after prediction.
5. **`src/pipeline.py`** and **`main.py`** handle orchestration and the API.

### Plugging in learned models
- Emissions: replace `emission_log_probs` with any function that returns one `(k_t,)` array of log-scores per point, for example `clf.predict_log_proba(features)[:, 1]`.
- Transitions: replace `make_hmm_transition_fn` with any `(s, t) -> (k_s, k_t)` log-score callable. `viterbi_log` stays as it is.

## Metrics (read carefully)

`.route` lists the arcs driven in order but is **not aligned to GPS points**, so we cannot compute per-observation accuracy. Reported metrics are route-level:
- `arc_precision` / `arc_recall` / `arc_f1` on the sets of distinct arc ids (direction matters).
- `length_precision` / `length_recall`: the same, weighted by arc length.
- `route_continuity`: the fraction of consecutive predicted arcs that connect (`to(a) == from(b)`).

Results measured with `scripts/evaluate.py` (means over trips):

| split | method | P | R | F1 | len-P | len-R | continuity | runtime |
|---|---|---|---|---|---|---|---|---|
| train (80) | baseline | 0.586 | 0.973 | 0.708 | 0.655 | 0.982 | 0.988 | 145 ms |
| train (80) | hmm | 0.785 | 0.986 | 0.861 | 0.849 | 0.989 | 0.999 | 335 ms |
| student_test (10) | hmm | 0.857 | 0.988 | 0.916 | 0.906 | 0.996 | 0.999 | 221 ms |

HMM parameters (k = 8, radius 50 m, β = 15 m) are hand-set defaults. They were not tuned on either split.

## Frontend — Evaluator Dashboard

An interactive **Streamlit + Folium** web UI that connects to this FastAPI backend over the network.  See [`frontend/README.md`](frontend/README.md) for details.

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py          # connects to http://localhost:8000 by default
```
