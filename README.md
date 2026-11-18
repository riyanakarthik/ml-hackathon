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
     -d '{"trip_id": "00000038"}'     # optional: "method": "baseline", "config": "baseline_v1"
.venv/bin/python -m scripts.evaluate --split train --method hmm    # aggregate metrics
```

### Decoder configurations
The HMM is driven by `MatcherConfig` ([src/config.py](src/config.py)), saved as JSON in `configs/`.

| config | status |
|---|---|
| `baseline_v1` | original hand-set decoder (k=8, radius 50 m, β=15 m); kept for **rollback** |
| `tuned_v2` | **API default**: Optuna-selected, validated on held-out train trips |
| `tuned_candidate` | raw output of the last tuning run (not validated by itself) |

To roll back, set `DEFAULT_CONFIG = "baseline_v1"` in `main.py`, or send `"config": "baseline_v1"` with a request.

### Tuning and experiment commands
```bash
.venv/bin/python -m scripts.diagnose --config baseline_v1 --trips train_all   # error breakdown
.venv/bin/python -m scripts.tune --trials 150 --seed 42                       # Optuna TPE
.venv/bin/python -m scripts.compare --a baseline_v1 --b tuned_candidate --trips holdout
.venv/bin/python -m scripts.compare --a baseline_v1 --b tuned_v2 --trips student_test   # final
```
Outputs are written to `experiments/`. Files there include `optuna_trials.csv` (every trial), `optuna_study.db`, `train_partition.json`, `compare_*.csv` (trip by trip), `diagnostics_*.csv`, `comparisons_summary.csv` and `tune_log.txt`.

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

Observation-level metrics (`obs_top1_accuracy`, `obs_macro_f1`, `candidate_miss_rate`) are computed against **derived** per-point labels. These come from a monotone alignment of the GPS points to `.route` (`align_points_to_route`, evaluation-only code). They are not dataset ground truth.

Error diagnostics (`route_diagnostics`) assign each false-positive arc to exactly one cause: reverse direction, outside the reference span, gap-fill-only, or directly matched. Repeated traversals are counted separately and are **not** penalised by the set metric. Note that 9 of the 80 train reference routes legitimately repeat arcs.

## Hyperparameter optimisation (train trips only)

- **Split.** The 80 train trips get a seeded (42) split into 64 *tune* trips (4 trip-level folds) and 16 *held-out* trips; see `experiments/train_partition.json`. Student-test trips were not used for tuning or selection.
- **Optimiser.** Optuna `TPESampler(seed=42, multivariate=True, n_startup_trials=20)`, 150 trials. Trial 0 is baseline_v1.
- **Objective.** Mean arc F1 over the 64 tune trips. The decoder has no fitted weights, so per-fold F1 is recorded for stability only.
- **Search space:** k ∈ [3,12]; radius ∈ [20,100] m; σ-scale ∈ [0.5,3] (log); β ∈ [2,100] m (log); same-arc backtrack tolerance ∈ [0,30] m; optional backtrack penalty; U-turn penalty ∈ [0,20]; observation thinning ∈ [0,40] m; optional short-loop removal.
- **Best trial (#133):** k=12, radius=52.7 m, σ-scale=0.866, β=2.34 m, backtrack_tol=24.7 m, uturn_penalty=13.7, min_spacing=11.6 m. Backtrack penalty and loop removal were off. β, k and backtrack_tol ended near the edges of their ranges.

| set | config | F1 mean | F1 median | P mean | R mean | obs Top-1 (derived) | runtime mean |
|---|---|---|---|---|---|---|---|
| tune (64) | baseline_v1 | 0.8466 | | 0.7651 | 0.9845 | | |
| tune (64) | tuned (#133) | 0.9398 | | 0.9043 | 0.9874 | | |
| held-out train (16) | baseline_v1 | 0.9181 | 0.9542 | 0.8652 | 0.9928 | 0.8540 | 270 ms |
| held-out train (16) | tuned_v2 | 0.9477 | 0.9740 | 0.9184 | 0.9902 | 0.8736 | 291 ms |
| student_test (10, run once) | baseline_v1 | 0.9155 | 0.9027 | 0.8566 | 0.9883 | 0.8840 | 240 ms |
| student_test (10, run once) | tuned_v2 | 0.9683 | 0.9787 | 0.9455 | 0.9936 | 0.9034 | 274 ms |

On the held-out trips, 13 of 16 improved, 2 were unchanged and 1 was slightly worse (−0.0013). On student-test, 9 of 10 improved and 1 was unchanged. Observation-level Macro F1 fell slightly (held-out 0.833 → 0.827). Runtimes use 8 worker processes and exclude the network build.

HMM parameters (k = 8, radius 50 m, β = 15 m) are hand-set defaults. They were not tuned on either split.

## Frontend — Evaluator Dashboard

An interactive **Streamlit + Folium** web UI that connects to this FastAPI backend over the network.  See [`frontend/README.md`](frontend/README.md) for details.

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py          # connects to http://localhost:8000 by default
```
