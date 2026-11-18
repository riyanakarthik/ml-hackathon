# PathSnap: Distributed Map-Matching & Route Reconstruction

An end-to-end distributed Machine Learning system that matches noisy GPS observation trajectories to a directed road network using **supervised Random Forest candidate classification**, **Bayesian-optimized Hidden Markov Models (HMM)**, and **Viterbi sequential decoding**.

Built for **UE24CS352A — Machine Learning**, PES University.

---

## 🚀 Quick Start: Two-Laptop Demonstration Setup

The system is deployed across two machines connected over Wi-Fi/LAN:
* **Laptop 1 (Backend Server):** Hosts FastAPI inference server on port `8000`, road network graph, spatial indexer, Random Forest model artifacts, and evaluation logic.
* **Laptop 2 (Evaluator Dashboard):** Runs Streamlit + Folium map dashboard on port `8501`, connecting live to Laptop 1 over the network.

```text
+------------------------------------+           +------------------------------------+
|             LAPTOP 2               |           |             LAPTOP 1               |
|      (Evaluator UI / Presenter)    |           |       (Backend Inference Server)   |
|                                    |           |                                    |
|  Streamlit Dashboard               |  HTTP GET |  FastAPI Server (Port 8000)        |
|  (frontend/app.py on Port 8501)    | --------> |  Local Tangent Plane Projector     |
|                                    |  /health  |  k-NN Candidate Generator          |
|  Folium Map Renderer               |  /trips   |  Random Forest Model Artifacts     |
|  (Blue GPS points, Green Route)    |           |  HMM + Viterbi Decoder (Optuna)    |
|                                    |  HTTP POST|  Server-Side Metric Calculator     |
|  Evaluation Metrics & KPIs         | --------> |                                    |
|  (arc_f1, continuity, runtime_ms)  |  /predict |                                    |
+------------------------------------+           +------------------------------------+
                   \                               /
                    \                             /
                     \                           /
                      Shared Local Network (Wi-Fi)
```

---

### 🖥️ LAPTOP 1: Backend Setup & Run (Person 1)

1. **Clone & Setup Environment:**
   * **Windows (PowerShell):**
     ```powershell
     git clone https://github.com/riyanakarthik/ml-hackathon.git
     cd ml-hackathon
     python -m venv .venv
     .venv\Scripts\activate
     pip install -r requirements.txt
     ```
   * **macOS / Linux:**
     ```bash
     git clone https://github.com/riyanakarthik/ml-hackathon.git
     cd ml-hackathon
     python3 -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     ```

2. **Find Laptop 1's Wi-Fi IP Address:**
   * **Windows:** Run `ipconfig` $\to$ Look for the IPv4 Address under the active Wi-Fi adapter (e.g. `192.168.1.45`).
   * **macOS / Linux:** Run `ifconfig | grep "inet "` or `ip addr`.

3. **(Windows Only) Allow Port 8000 Through Firewall:**
   Open PowerShell as Administrator:
   ```powershell
   New-NetFirewallRule -DisplayName "PathSnap FastAPI Port 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow -Profile Private
   ```

4. **Launch the FastAPI Server:**
   * **Windows (PowerShell):**
     ```powershell
     .venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 8000
     ```
   * **macOS / Linux:**
     ```bash
     .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
     ```
   *Keep this terminal visible during evaluation so evaluators can inspect incoming inference request logs!*

---

### 💻 LAPTOP 2: Evaluator Dashboard Setup & Run (Person 3 & Person 4)

1. **Clone & Setup Environment:**
   * **Windows (PowerShell):**
     ```powershell
     git clone https://github.com/riyanakarthik/ml-hackathon.git
     cd ml-hackathon
     python -m venv .venv
     .venv\Scripts\activate
     pip install -r requirements.txt
     ```
   * **macOS / Linux:**
     ```bash
     git clone https://github.com/riyanakarthik/ml-hackathon.git
     cd ml-hackathon
     python3 -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     ```

2. **Verify Connectivity to Laptop 1:**
   ```bash
   curl http://<LAPTOP_1_IP>:8000/health
   # Expected response: {"status":"ok","default_config":"ml_v3",...}
   ```

3. **Launch the Streamlit Evaluator Dashboard:**
   * **Windows (PowerShell):**
     ```powershell
     .venv\Scripts\streamlit run frontend/app.py
     ```
   * **macOS / Linux:**
     ```bash
     .venv/bin/streamlit run frontend/app.py
     ```
   Open your browser to `http://localhost:8501`.

4. **Connect UI to Laptop 1:**
   * In the left sidebar under **Backend Connection**, enter:
     `http://<LAPTOP_1_IP>:8000` (e.g. `http://192.168.1.45:8000`).
   * Click **Check Backend Status** $\to$ status turns green: `✅ Backend online`.
   * Click **Refresh Trips** $\to$ select any evaluation trip (e.g. `00000000` or `00000040`).
   * Click **🚀 Run Route Reconstruction** to view the live reconstructed route on the Folium map.

---

### 💻 Single-Laptop Local Fallback (Running Everything on One Machine)

If demonstrating on a single machine:
1. **Terminal 1 (Backend):** `uvicorn main:app --host 0.0.0.0 --port 8000`
2. **Terminal 2 (Frontend):** `streamlit run frontend/app.py`
3. In your browser at `http://localhost:8501`, keep backend URL as `http://localhost:8000`.

---

### 🧪 Automated Testing & Verification

Run all unit, contract, and machine learning tests:
```bash
pytest -v
```
Run end-to-end verification across all 10 evaluation trips:
```bash
python frontend/verify_evaluation_trips.py
```

---

## Technical Overview & Data Layout

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
| `ml_v3` | **API default**: HMM with Random-Forest emissions (trained on all 80 train trips) |
| `baseline_v1` | original hand-set decoder (k=8, radius 50 m, β=15 m); kept for **rollback** |
| `tuned_v2` | Optuna-tuned geometric-emission HMM; the ML **fallback** and rollback option |
| `tuned_candidate` | raw output of the last tuning run (not validated by itself) |

To roll back, set `DEFAULT_CONFIG` in `main.py` to `"tuned_v2"` or `"baseline_v1"`, or send `"config": ...` with a request. If the ML model can't be loaded, prediction falls back to geometric emissions and reports `details.emission_fallback`.

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

## Supervised candidate classifier (ML + HMM)

### Pipeline
1. **Pseudo-labels** ([src/labels.py](src/labels.py), training/evaluation only). `.route` is not one label per GPS point, so each point is aligned *monotonically* to a position in the ordered reference route. The alignment scores distance, adds a heading-agreement term when the vehicle is moving, and penalises skipping over long arcs. Repeated arcs and A→reverse(A) U-turns are separate route positions. A label is *invalid* if it lies more than 30 m from its arc; invalid labels are excluded from training and from pseudo-label accuracy.
   - **Checks** (`python -m scripts.validate_alignment`, 80 train trips): always monotone; 93.4% of moving points agree with the arc direction; 96.7% of labels valid on average; ≥96% agreement under σ=5/20 m.
   - **Plots and Folium maps** are in `experiments/ml/alignment_plots/`.
   - **Finding:** in some trips the `.route` ends early while the GPS keeps going (trip 54: 91% of points after the route end; trip 33: 71%). Those points get invalid labels, and route-level precision on these trips penalises arcs that may be matched correctly.
2. **Features** ([src/features.py](src/features.py), inference-safe; never reads `.route`, enforced by tests). There are 20 per-candidate features: distance, rank and gap to the best candidate, noise-normalised distance, heading vs arc bearing (cos and degrees, plus a validity flag), speed, time gaps, arc length, projection position and distances to the arc ends, reverse-twin presence, node degrees, and ±5-point window support and mean distance.
3. **Models** ([src/ml_train.py](src/ml_train.py)): Logistic Regression, Decision Tree, Random Forest, Extra Trees and HistGradientBoosting. They are trained on valid-label rows with `GroupKFold(5)` by trip over the 64 tune trips, and fold disjointness is asserted.
4. **Emissions** ([src/ml_emission.py](src/ml_emission.py)): `clip(p, 1e-4, 1)`, normalised per point, then `log`, multiplied by `ml_weight`. Per-point normalisation doesn't change the Viterbi path; it keeps scores interpretable. The transition model is unchanged from `tuned_v2`.

```bash
python -m scripts.train_classifiers --stage dataset
python -m scripts.train_classifiers --stage cv
python -m scripts.train_classifiers --stage oof_hmm
python -m scripts.train_classifiers --stage emission_grid --model random_forest
python -m scripts.train_classifiers --stage fit_tune
python -m scripts.compare --a tuned_v2 --b ml_candidate --trips holdout
python -m scripts.train_classifiers --stage final --model random_forest --emission ml --weight 4
python -m scripts.compare --a tuned_v2 --b ml_v3 --trips student_test
python -m scripts.train_classifiers --stage student_test
```

Models are saved in `models/*.joblib`. Each bundle holds the model (with its scaler, for logistic regression), the feature names, the schema version, the candidate configuration, the label configuration, the training trip ids, the CV metrics and the library versions. Results are in `experiments/ml/*.csv`.

### Results (real runs; seed 42)
**Two kinds of accuracy.** *Pseudo-label* metrics compare against derived per-point labels. *Route-level* metrics (arc F1) compare against `.route` itself.

Candidate classification, grouped 5-fold CV on the 64 tune trips (OOF, valid-label rows):

| model | precision | recall | F1 | macro F1 | log loss | pseudo top-1 | fit (5 folds) | predict / 1k rows |
|---|---|---|---|---|---|---|---|---|
| logreg | 0.807 | 0.771 | 0.789 | 0.878 | 0.147 | 0.824 | 2.6 s | 0.03 ms |
| decision_tree | 0.847 | 0.792 | 0.818 | 0.895 | 0.130 | 0.830 | 32 s | 0.03 ms |
| random_forest | 0.862 | 0.794 | 0.827 | 0.900 | 0.111 | 0.833 | 147 s | 1.4 ms |
| extra_trees | 0.864 | 0.775 | 0.817 | 0.894 | 0.121 | 0.827 | 47 s | 1.5 ms |
| hist_gb | 0.852 | 0.806 | 0.828 | 0.901 | 0.111 | 0.835 | 30 s | 1.5 ms |

For reference, the nearest-candidate rule scores a pseudo top-1 of 0.538.

OOF classifier + HMM, route-level, 64 tune trips (`ml` emission, w=1):

| emission | F1 mean | median | P | R |
|---|---|---|---|---|
| geometric | 0.9398 | 0.9745 | 0.904 | 0.987 |
| logreg | 0.9428 | 0.9768 | 0.910 | 0.988 |
| decision_tree | 0.9451 | 0.9782 | 0.913 | 0.988 |
| random_forest | 0.9458 | 0.9776 | 0.915 | 0.989 |
| extra_trees | 0.9439 | 0.9755 | 0.912 | 0.988 |
| hist_gb | 0.9449 | 0.9776 | 0.913 | 0.988 |

Random Forest was selected on this OOF route F1. The emission grid on OOF gave the best result with `ml`, w=4 (F1 0.9492); w=8 gave 0.9421, and the best hybrid gave 0.9467.

| set | geometric HMM (`tuned_v2`) | RF + HMM | per trip |
|---|---|---|---|
| held-out train (16), model trained on the 64 tune trips | F1 0.9477 (median 0.9740) | **0.9511** (0.9793) | 10 better, 3 worse, 3 same; Wilcoxon p = 0.041 |
| student_test (10, run once after freezing), model trained on 80 | F1 0.9683 (median 0.9787) | **0.9755** (0.9787) | 6 better, 0 worse, 4 same |

Runtime is about +90–110 ms per trip, for feature extraction and RF inference. Gains come mostly from fewer wrong-direction arcs: 49 → 22 false positives on student test. Gap-fill false positives are unchanged.

Every saved model on student test (reporting only, after freezing; `experiments/ml/student_test_model_report.csv`), shown as route F1 with the HMM: geometric 0.9683, logreg 0.9671, decision_tree 0.9737, random_forest 0.9730, extra_trees 0.9738, hist_gb 0.9742, and random_forest trained on 80 trips 0.9755.
