# PathSnap: ML-Assisted GPS Map Matching and Route Reconstruction

**Final Technical Report**

UE24CS352A: Machine Learning — PathSnap Hackathon (PES University, Department of CSE, RR Campus)  
Repository: github.com/riyanakarthik/ml-hackathon  
**Team ID: PT-056**  
RIYANA KARTHIK VISHALAKSHI (PES1UG24AM226)  
ROHANTH GANESHA C (PES1UG24AM227)  
ROHIT CHAKI (PES1UG24AM228)  
ROHITH KARNING (PES1UG24AM229)  

*Table of contents: see the PDF/DOCX versions.*

## 1  Executive Summary

PathSnap reconstructs the road path driven by a vehicle from a noisy GPS trace and a directed road network. The deployed system (configuration **ml_v3**) generates nearby road candidates for every GPS observation, scores each candidate with a **Random Forest** classifier trained on 80 training trips, and decodes the most plausible connected path with a **Hidden Markov Model (HMM) and log-space Viterbi** decoder whose transition parameters were tuned with Bayesian optimisation (Optuna TPE). Gaps between matched arcs are stitched with shortest paths, so every returned route is a connected directed path. A FastAPI backend serves predictions to a Streamlit/Folium UI.

- **Five classifiers** (Logistic Regression, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting) were compared with 5-fold trip-grouped cross-validation on 64 training trips. Candidate-level CV macro F1 ranged 0.8776–0.9006; tree models outperformed Logistic Regression.
- **The HMM is the dominant component**: on held-out training trips, classifier-only routes reach arc F1 0.763, the same classifier inside the HMM reaches 0.951 (Table 13).
- **Classifier emissions give a small, consistent gain over geometric emissions** on development data (holdout 16: 0.9477 → 0.9511, Wilcoxon p = 0.041).
- **Final evaluation on the 10 reserved evaluator_test trips** (model frozen beforehand): mean directed-arc-set F1 **0.9620** (median 0.9802) versus 0.9502 for the geometric HMM; 6 trips better, 3 worse, 1 equal — the difference is **not statistically significant** on 10 trips (Wilcoxon p = 0.43).
- **Limitations**: per-observation labels are inferred by aligning GPS to the ordered reference route (pseudo-labels); the route metric ignores order and duplicates; Bayesian optimisation of the Random Forest was stopped after 10 completed trials and its tuned configuration was never evaluated or deployed; UI screenshots are not yet available.

## 2  Problem Statement and Objectives

GPS receivers report positions with metre-scale noise, occasional dropouts and irregular timestamps, so a raw trace drifts onto parallel streets, cuts corners and jumps between nearby roads. Map matching assigns each observation to a road segment and reconstructs a route that is consistent with the road network: consecutive segments must connect head-to-tail and respect one-way directions. In PathSnap the road network is a directed graph whose edges (arcs) are identified by their row index in the trip's .arcs file, and the reference answer is an ordered list of arc ids.

Objectives, as set by the PathSnap guidelines:

- Understand and explore the raw trip files and their relationships.
- Train and compare at least five supervised models that score road candidates for GPS observations.
- Select a model on experimental evidence and apply Bayesian hyperparameter optimisation.
- Reconstruct routes with an HMM and Viterbi decoding, handling missing candidates, gaps and disconnected segments.
- Deploy a FastAPI backend (Laptop 1) and an evaluator UI (Laptop 2) that works for all ten reserved trips, keeping ground truth server-side.
- Report verified results without leakage, fabricated numbers or hardcoded predictions.

| Deliverables requirement | Where answered |
|---|---|
| §2.2 Dataset understanding and EDA (missing values, trajectory and network plots, Folium map, partitions) | Sections 3, 5 and 14 |
| §3.1 Overall model comparison table | Section 6, Table 6 |
| §3.2 Compulsory questions for every model (A1–A2, B5–B6, C7–C9) | Sections 6.1–6.5 |
| §3.3 Model visualisations (CV macro F1, student-test Top-1 and macro F1, log loss, confusion matrices) | Section 10 |
| §4 Bayesian hyperparameter optimisation and tuned vs untuned comparison | Section 7 (classifier comparison missing — stated) |
| §5 HMM and Viterbi; classifier-only vs classifier + HMM | Sections 8 and 11 |
| §6 Geographical visualisation using Folium | Section 14 |
| §7 API, UI and deployment screenshots | Section 15 (screenshots missing — stated) |
| §8 Academic integrity and AI disclosure | Section 19 |

*Table 1. Mapping of the official report structure (Deliverables.pdf §2–§8) to this report.*

## 3  Dataset Exploration (Deliverables §2.2)

Each trip directory holds four whitespace-separated files without headers (verified by the loader in src/data_loader.py, which validates column counts, finite values, coordinate ranges and node-id ranges):

| File | Columns | Meaning |
|---|---|---|
| .track | longitude, latitude, timestamp (s) | GPS observations of the trip (one row per observation) |
| .nodes | longitude, latitude | Road-network nodes; the 0-based row index is the node id |
| .arcs | from_node_id, to_node_id | Directed road segments; the 0-based row index is the arc id |
| .route | arc_id | Chronological reference sequence of traversed arcs — NOT one label per GPS observation |

*Table 2. Raw file types (Dataset Exploration document §3, verified against the files).*

| Split | Trips | GPS obs. | Median obs./trip | Range | Median nodes | Median arcs | Median .route arcs | Mean share 1-s steps | Max gap (s) |
|---|---|---|---|---|---|---|---|---|---|
| train | 80 | 208,417 | 1,529 | 361–19,313 | 9,608 | 19,526 | 280 | 0.938 | 153,064 |
| student_test | 10 | 19,160 | 1,420 | 262–3,958 | 6,596 | 13,145 | 250 | 0.982 | 210 |
| evaluator_test | 10 | 19,674 | 1,245 | 458–6,854 | 12,391 | 24,121 | 373 | 0.988 | 220 |

*Table 3. Dataset inventory per partition. Source: reports/data/dataset_overview_summary.csv (train/student_test computed from the files; evaluator_test from the frozen evaluation manifest).*

![Figure 1. Per-trip size of the GPS trace, the road network and the reference route for each partition (log scale for the first two panels).](figures\fig_dataset_overview.png)

*Figure 1. Per-trip size of the GPS trace, the road network and the reference route for each partition (log scale for the first two panels).*

| Split | Trips | Missing files | Non-finite values | Invalid node refs | Trips with a gap > 10 s | Largest gap (s) |
|---|---|---|---|---|---|---|
| train | 80 | 0 | 0 | 0 | 40 | 153,064 |
| student_test | 10 | 0 | 0 | 0 | 2 | 210 |
| evaluator_test | 10 | 0 | 0 | 0 | not computed | 220 |

*Table 4. Missing-value and anomaly analysis. Zeros are guaranteed by the loader, which rejects any trip with a missing file, non-finite value, out-of-range coordinate or arc endpoint outside the node table (all 100 trips loaded). Source: reports/data/dataset_overview_per_trip.csv.*

**Missing and invalid values.** All 100 trips have all four files and parse with zero non-finite values, zero out-of-range coordinates and zero arc endpoints outside the node table (the loader rejects any violation). Sampling is nominally 1 Hz (median share of 1-second steps in train: 0.991), but 40 of 80 training trips contain at least one gap above 10 s and two trips contain very long interruptions (trip 00000090: 153,064 s; trip 00000078: 52,810 s).

**Road-network connectivity** (reports/data/network_connectivity_summary.csv). No train or student_test network has isolated nodes. Networks are extracts around each trip and split into several weakly connected components (train median 20, max 421); the largest strongly connected component covers a median 96.4% of nodes, and a median 95.1% of arcs have a reverse twin (two-way streets are two arcs with identical geometry). Direction therefore cannot be inferred from geometry alone.

**Reference routes.** Most .route files are connected directed paths (the to-node of one arc equals the from-node of the next), but not all: 10 of 80 training routes (13 breaks in total) and 1 of 10 student-test routes contain a discontinuity (lowest continuity 0.981; reports/data/reference_route_continuity.csv; evaluator_test routes were not examined); neither the label alignment nor the route metrics assume a connected reference. 9 of 80 training routes traverse some arc more than once (e.g. roundabout loops). In some training trips the GPS trace continues well beyond the end of the reference route (trip 00000054: 91% of observations; trip 00000033: 71%), which matters for labelling and scoring (Sections 5 and 13).

**Partitions used for each analysis.** Table 3 and Figure 1 cover all partitions; connectivity, gaps and route properties were computed on train and student_test only (evaluator_test was not explored before the frozen final evaluation; its inventory comes from that evaluation's manifest). Feature choices motivated by these observations — heading versus arc bearing for two-way streets, time-gap features for irregular sampling — are listed in Section 5.3.

## 4  System Architecture

![Figure 2. End-to-end pipeline. Steps 1–6 use only .track, .nodes and .arcs; the reference .route is read only by the server-side evaluation after prediction.](figures\fig_architecture.png)

*Figure 2. End-to-end pipeline. Steps 1–6 use only .track, .nodes and .arcs; the reference .route is read only by the server-side evaluation after prediction.*

- **Ingestion** (src/data_loader.py): trip discovery, parsing and validation; never reads .route.
- **Candidate generation** (src/matcher.py): local metric projection, KD-tree over densified arcs, exact point-to-segment distance and projection.
- **Features and classifier** (src/features.py, src/ml_emission.py): 20 inference-safe features per candidate; the saved Random Forest bundle is loaded once at API start-up.
- **HMM/Viterbi and route assembly** (src/decoder.py): log-space Viterbi with network-distance transitions, Dijkstra gap stitching, optional geometric fallback if the model cannot be loaded.
- **API** (main.py): GET /health, GET /trips, POST /predict; request logs printed on the server terminal.
- **UI** (frontend/app.py): Streamlit + Folium dashboard on Laptop 2; it receives coordinates and metrics only.
- **Evaluation** (src/evaluation.py): the only module that reads .route; called after prediction.

## 5  Data Preprocessing and Label Alignment (Deliverables §2.2)

### 5.1 Spatial projection and candidate search

Longitude/latitude are angular coordinates, so all geometry is computed after an equirectangular projection to local metres around the trip's mean position (accurate to well under a metre at trip scale). Every arc is densified with sample points every 10 m and indexed in a scipy cKDTree; a radius query (plus half the sample spacing) returns every arc within the search radius, after which the exact point-to-segment distance, projected point and offset along the arc are computed. The deployed configuration keeps up to k = 12 candidates within 52.7 m, retrying at 200 m if none are found. Directed arcs and their reverse twins are separate candidates.

### 5.2 Monotonic alignment of .route to GPS observations (pseudo-labels)

Because .route is an ordered arc sequence and not a label per observation, supervised targets were inferred by a monotone dynamic-programming alignment (src/labels.py). The hidden state is a position i in the reference route, so repeated arcs and A → reverse(A) U-turns are distinct states. The score combines a Gaussian term on point-to-arc distance (σ = 10 m), a heading-agreement term κ(cos Δθ − 1) applied only when the vehicle moves (κ = 2, speed ≥ 2 m/s), and a penalty of 0.01 per metre of reference arcs skipped between consecutive positions. Labels more than 30 m from their arc are marked invalid and excluded from training and from pseudo-label accuracy.

| Check (80 training trips) | Result |
|---|---|
| Aligned positions non-decreasing | 80/80 |
| Label equals the .route arc at its aligned position | 80/80 |
| Label sequence is an ordered subsequence of .route | 80/80 |
| Mean share of observations with a valid label | 0.9668 |
| Moving observations whose heading agrees with the label arc direction | 0.9344 |
| Agreement of labels when σ = 5 m / σ = 20 m | 0.9753 / 0.9632 |
| Agreement without the heading term | 0.9504 |
| Valid labels whose arc is among the generated candidates | 0.9980 |

*Table 5. Alignment quality. Sources: experiments/final/label_verification.csv, experiments/ml/alignment_quality.csv, reports/data/training_dataset_stats.json.*

![Figure 3. Alignment on training trip 00000018, whose route repeats arcs around a roundabout. GPS points are coloured by aligned route position; orange lines join points to their label arc.](figures\alignment_00000018.png)

*Figure 3. Alignment on training trip 00000018, whose route repeats arcs around a roundabout. GPS points are coloured by aligned route position; orange lines join points to their label arc.*

> **Disclosure.** These labels are derived, not ground truth. They inherit GPS noise, depend on the alignment parameters (about 2–5% of labels change under the perturbations above) and are ambiguous at junctions, on short arcs and where the GPS continues after the reference route ends. All per-candidate and per-observation metrics in this report are measured against these pseudo-labels; route-level metrics are measured against .route itself.

### 5.3 Feature construction

Twenty features are computed per candidate (src/features.py, schema version 1) from the GPS track, the network and the candidate set only: distance to the arc, its rank and gap to the nearest candidate, distance normalised by a per-trip noise estimate; arc length, projection position and distances to both arc ends; centred GPS heading versus directed arc bearing (cosine and degrees, with a validity flag for slow movement); speed and time gaps to the previous/next observation; presence of the reverse twin among the candidates; out-degree of the end node and in-degree of the start node; and two context features over ±5 observations (share of neighbouring observations that also have this arc as a candidate, and their mean distance to it). The heading features target the two-way-street ambiguity found in Section 3; the time-gap features target irregular sampling; the window features use temporal context that a single observation lacks. The training table has 1,589,846 candidate rows from 208,417 observations (1,327,875 rows with valid labels, positive rate 13.7%).

## 6  Five Machine-Learning Models (Deliverables §3.1–3.2)

Every model solves the same binary task — is this candidate arc the road the vehicle was on? — on the same features, the same valid-label rows and the same five trip-grouped folds of the 64 tune trips (GroupKFold by trip; fold disjointness asserted in code, experiments/ml/cv_folds.json). No preprocessing is fitted except the StandardScaler inside the Logistic Regression pipeline, which is fitted within each training fold. All models use random_state = 42. The optimisation objective of every learner is its standard binary log-loss / impurity criterion (below); model selection used out-of-fold route-level F1 after HMM decoding (Section 9).

| Model | Algorithm family | CV macro F1 | Top-1 accuracy (student test) | Route continuity (student test, +HMM) | Student-test macro F1 | Log loss CV / student test | Inference ms per 1k candidates |
|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear / probabilistic | 0.8776 | 0.8339 | 0.9990 | 0.8900 | 0.1468 / 0.1244 | 0.029 |
| Decision Tree | Tree-based | 0.8949 | 0.8469 | 0.9989 | 0.9056 | 0.1304 / 0.1011 | 0.028 |
| Random Forest | Ensemble (bagging) | 0.9000 | 0.8506 | 0.9989 | 0.9091 | 0.1108 / 0.0914 | 1.429 |
| Extra Trees | Ensemble (randomised trees) | 0.8944 | 0.8402 | 0.9989 | 0.9022 | 0.1214 / 0.1028 | 1.478 |
| HistGradientBoosting | Boosting | 0.9006 | 0.8535 | 0.9989 | 0.9103 | 0.1114 / 0.0905 | 1.496 |

*Table 6. Overall model comparison (Deliverables §3.1). CV = out-of-fold over 64 tune trips; student-test columns use models trained on the same 64 trips; Top-1, macro F1 and log loss are against pseudo-labels; route continuity is route-level. Sources: experiments/ml/cv_model_comparison.csv, experiments/ml/student_test_model_report.csv, experiments/final/ablation_summary.csv.*

Key configuration of each model is given in its subsection and in Appendix A.1. Inference time is the classifier's predict_proba time only, measured during cross-validation on the same machine.

### 6.1 Logistic Regression

- **A1 — Suitability.** A linear probabilistic baseline: if candidate correctness were mostly a monotone function of distance, heading agreement and context, a calibrated linear model would suffice and be fast.
- **A2 — Strengths and limitations.** Strengths: fastest training (2.6 s for 5 folds), inference as fast as a single tree, probabilistic outputs, interpretable coefficients. Limitations: cannot model interactions such as 'heading matters only when moving' or non-linear distance effects without manual feature crosses.
- **B5 — Hyperparameters (initial values, not tuned):** StandardScaler + LogisticRegression(C=1.0, max_iter=2000); random_state = 42.
- **B6 — Objective.** L2-regularised logistic (binary cross-entropy) loss.
- **C7 — Student-test performance** (trained on 64 tune trips): with the HMM, mean route arc F1 0.9671; classifier-only route arc F1 0.7136.
- **C8 — Observation/candidate level on student test:** Top-1 accuracy 0.8339, macro F1 0.8900, log loss 0.1244 (pseudo-labels).
- **C9 — Over/underfitting:** training macro F1 0.8793 vs CV 0.8776; training log loss 0.1449 vs CV 0.1468 — almost no gap, but the lowest ceiling of the five — mild underfitting. Fold-to-fold F1 standard deviation 0.0466.

### 6.2 Decision Tree

- **A1 — Suitability.** Captures threshold rules and interactions (e.g. distance only matters relative to the nearest candidate) and is directly interpretable.
- **A2 — Strengths and limitations.** Strengths: interactions, no scaling, fast inference. Limitations: high variance and coarse, piecewise-constant probabilities (leaf frequencies) as emission scores.
- **B5 — Hyperparameters (initial values, not tuned):** DecisionTreeClassifier(max_depth=12, min_samples_leaf=50); random_state = 42.
- **B6 — Objective.** Gini impurity reduction at each split.
- **C7 — Student-test performance** (trained on 64 tune trips): with the HMM, mean route arc F1 0.9737; classifier-only route arc F1 0.7025.
- **C8 — Observation/candidate level on student test:** Top-1 accuracy 0.8469, macro F1 0.9056, log loss 0.1011 (pseudo-labels).
- **C9 — Over/underfitting:** training macro F1 0.9089 vs CV 0.8949; training log loss 0.1007 vs CV 0.1304 — a modest gap of 0.014 macro F1 — mild overfitting, controlled by the leaf-size limits. Fold-to-fold F1 standard deviation 0.0495.

### 6.3 Random Forest

- **A1 — Suitability.** Averages many decorrelated trees, keeping the interaction modelling of a tree while reducing variance and producing smoother probabilities for HMM emissions.
- **A2 — Strengths and limitations.** Strengths: robust, little tuning, good probability ranking. Limitations: large model file (the deployed bundle is about 47 MB), slower inference than linear or single-tree models.
- **B5 — Hyperparameters (initial values, not tuned):** RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=20, max_features='sqrt'); random_state = 42.
- **B6 — Objective.** Gini impurity per tree; probabilities averaged over trees.
- **C7 — Student-test performance** (trained on 64 tune trips): with the HMM, mean route arc F1 0.9730; classifier-only route arc F1 0.7079.
- **C8 — Observation/candidate level on student test:** Top-1 accuracy 0.8506, macro F1 0.9091, log loss 0.0914 (pseudo-labels).
- **C9 — Over/underfitting:** training macro F1 0.9203 vs CV 0.9000; training log loss 0.0872 vs CV 0.1108 — a modest gap of 0.020 macro F1 — mild overfitting, controlled by the leaf-size limits. Fold-to-fold F1 standard deviation 0.0481.

### 6.4 Extra Trees

- **A1 — Suitability.** Randomised split thresholds give a different bias–variance trade-off from Random Forest at lower training cost.
- **A2 — Strengths and limitations.** Strengths: about 3x faster to train than Random Forest here (46.5 s vs 147.1 s for 5 folds). Limitations: more conservative on the positive class (lowest recall of the tree models here).
- **B5 — Hyperparameters (initial values, not tuned):** ExtraTreesClassifier(n_estimators=150, max_depth=18, min_samples_leaf=20, max_features='sqrt'); random_state = 42.
- **B6 — Objective.** Gini impurity with randomised thresholds; probabilities averaged.
- **C7 — Student-test performance** (trained on 64 tune trips): with the HMM, mean route arc F1 0.9738; classifier-only route arc F1 0.7044.
- **C8 — Observation/candidate level on student test:** Top-1 accuracy 0.8402, macro F1 0.9022, log loss 0.1028 (pseudo-labels).
- **C9 — Over/underfitting:** training macro F1 0.9005 vs CV 0.8944; training log loss 0.1132 vs CV 0.1214 — a modest gap of 0.006 macro F1 — mild overfitting, controlled by the leaf-size limits. Fold-to-fold F1 standard deviation 0.0497.

### 6.5 HistGradientBoosting

- **A1 — Suitability.** Gradient boosting on binned features corrects errors sequentially and is efficient on about a million rows.
- **A2 — Strengths and limitations.** Strengths: the best CV F1, macro F1 and Top-1 of the five (log loss within 0.001 of Random Forest), the fastest ensemble to train here, and a very small model file. Limitations: more hyperparameters, sequential training, risk of overfitting without early stopping (disabled here for reproducibility).
- **B5 — Hyperparameters (initial values, not tuned):** HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False); random_state = 42.
- **B6 — Objective.** Binary log-loss minimised by gradient boosting.
- **C7 — Student-test performance** (trained on 64 tune trips): with the HMM, mean route arc F1 0.9742; classifier-only route arc F1 0.7228.
- **C8 — Observation/candidate level on student test:** Top-1 accuracy 0.8535, macro F1 0.9103, log loss 0.0905 (pseudo-labels).
- **C9 — Over/underfitting:** training macro F1 0.9211 vs CV 0.9006; training log loss 0.0889 vs CV 0.1114 — a modest gap of 0.021 macro F1 — mild overfitting, controlled by the leaf-size limits. Fold-to-fold F1 standard deviation 0.0463.

## 7  Bayesian Hyperparameter Optimisation (Deliverables §4)

Two components were optimised separately, and only one search is complete. The HMM decoder was tuned with Optuna TPE; the Random Forest search was started but stopped early at the team's request. The ML emission weight was chosen by a small grid, not by Bayesian optimisation.

### 7.1 HMM decoder (completed)

| Hyperparameter | Search space | Untuned (baseline_v1) | Best (trial #133) |
|---|---|---|---|
| k (candidates per point) | int [3, 12] | 8 | 12 |
| search radius | float [20, 100] m | 50 m | 52.72 m |
| σ scale (emission noise multiplier) | float [0.5, 3], log | 1.0 | 0.866 |
| β (transition scale) | float [2, 100] m, log | 15 m | 2.335 m |
| same-arc backtrack tolerance | float [0, 30] m | 5 m | 24.69 m |
| backtrack penalty | 0 or float [0.01, 2], log | 0 (off) | 0.0 (off) |
| U-turn penalty | float [0, 20] | 0 | 13.74 |
| observation thinning distance | float [0, 40] m | 0 m | 11.64 m |
| short-loop removal | 0 (off) or float [20, 2000] m, log | 0 (off) | 0 (off) |

*Table 7. Decoder search: Optuna TPESampler(seed=42, multivariate, 20 start-up trials), 150 trials; objective = mean route arc F1 on the 64 tune trips; trial 0 = untuned configuration. Sources: experiments/optuna_trials.csv, experiments/optuna_study.db, configs/tuned_v2.json.*

![Figure 4. Decoder optimisation history: objective rose from 0.8466 (trial 0, untuned) to 0.9398 (trial 133).](figures\fig_decoder_optuna_history.png)

*Figure 4. Decoder optimisation history: objective rose from 0.8466 (trial 0, untuned) to 0.9398 (trial 133).*

![Figure 5. Tuned versus untuned decoder on the selection set and on two sets not used for tuning (holdout 16 training trips, student test).](figures\fig_decoder_tuned_vs_untuned.png)

*Figure 5. Tuned versus untuned decoder on the selection set and on two sets not used for tuning (holdout 16 training trips, student test).*

The tuned decoder outperformed the untuned one on data not used for tuning: holdout 16 0.9181 → 0.9477 (13 better, 1 worse, Wilcoxon p = 0.0002); student test 0.9155 → 0.9683 (9 better, 0 worse). Diagnostics showed why: the untuned decoder treated small backward GPS jitter on an arc as a loop through the network, producing oscillation between an arc and its reverse twin. The best values for β, k and backtrack tolerance lie near the edges of their ranges, so a wider search might improve further.

### 7.2 ML emission weight (grid search, not Bayesian)

| Emission mode | Weight | Route F1 mean | Median | Precision | Recall |
|---|---|---|---|---|---|
| hybrid | 0.25 | 0.9426 | 0.9758 | 0.9088 | 0.9880 |
| hybrid | 0.5 | 0.9438 | 0.9769 | 0.9108 | 0.9881 |
| hybrid | 1 | 0.9460 | 0.9783 | 0.9149 | 0.9881 |
| hybrid | 2 | 0.9467 | 0.9800 | 0.9161 | 0.9882 |
| hybrid | 4 | 0.9435 | 0.9800 | 0.9119 | 0.9861 |
| ml | 0.25 | 0.9330 | 0.9666 | 0.8923 | 0.9880 |
| ml | 0.5 | 0.9381 | 0.9695 | 0.9015 | 0.9881 |
| ml | 1 | 0.9458 | 0.9776 | 0.9145 | 0.9885 |
| ml | 2 | 0.9487 | 0.9814 | 0.9193 | 0.9887 |
| ml | 4 | 0.9492 | 0.9819 | 0.9199 | 0.9886 |
| ml | 8 | 0.9421 | 0.9770 | 0.9091 | 0.9862 |

*Table 8. Random Forest emission grid on out-of-fold probabilities of the 64 tune trips. 'ml' = weight × log p; 'hybrid' = geometric log-likelihood + weight × log p. Selected: ml, weight 4. Source: experiments/ml/emission_grid_random_forest.csv.*

### 7.3 Random Forest classifier (incomplete)

Search space: n_estimators 50–300 (step 25), max_depth 6–30, min_samples_leaf 1–200 (log), max_features ∈ {sqrt, 0.33, 0.5}, max_samples 0.2–1.0; objective = mean 5-fold log loss on the 64 tune trips (same folds as Section 6); Optuna TPESampler(seed=42) with a median pruner. The search was **stopped after 10 completed trials** (5 pruned, 1 interrupted). Best completed trial #10: log loss 0.11033 versus 0.11071 for the untuned configuration (trial 0), a difference of 0.00038, smaller than the fold-to-fold variation (standard deviation of the untuned model's fold log loss: 0.0292). Best parameters: n_estimators 125, max_depth 16, min_samples_leaf 5, max_features 0.33, max_samples 0.363.

> **Not done:** the tuned Random Forest was never refitted, evaluated on student test, or deployed, so the untuned-versus-tuned comparison of the classifier required by Deliverables §4 (CV macro F1, student-test Top-1 and macro F1, log loss, inference time) does not exist. The deployed Random Forest uses the initial hyperparameters in Section 6.3. Sources: experiments/ml/rf_optuna_trials_partial.csv, experiments/ml/rf_optuna_study.db.

![Figure 6. Random Forest search history (completed trials only).](figures\fig_rf_optuna_history.png)

*Figure 6. Random Forest search history (completed trials only).*

## 8  HMM and Viterbi Route Reconstruction (Deliverables §5)

- **Hidden states:** at observation t, the candidate set C_t (directed arc, projected point, offset along the arc). The candidate-to-arc correspondence is preserved through decoding and returned per observation (point_arc_ids).
- **Emissions:** geometric mode uses −½(d/σ)², with σ = 0.866 × the per-trip noise estimate 1.4826 · median(distance to nearest arc), clipped to [3, 30] m (Newson and Krumm, 2009). ML mode (deployed) uses 4 · log q, where q is the Random Forest probability clipped at 0.0001 and normalised over the observation's candidates (normalisation does not change the Viterbi path; it keeps scores interpretable).
- **Transitions:** log p = −|d_route − d_straight| / β with β = 2.335 m, where d_route is the driving distance along the directed network (same-arc forward movement, otherwise remaining arc + bounded Dijkstra + offset) and d_straight the straight-line distance between the observations. Backward movement on the same arc up to 24.7 m is treated as noise; moving onto an arc's reverse twin costs 13.74.
- **Viterbi:** log-space recursion δ_t(j) = e_t(j) + max_i[δ_{t−1}(i) + a_{ij}] with back-pointers (src/decoder.py::viterbi_log). Observations closer than 11.6 m to the previous decoded one are skipped and later assigned to the decoded arc.
- **Missing candidates and gaps:** observations with no candidate within the fallback radius are skipped (no state, null coordinates in the response); the transition is computed between the surrounding decoded observations. Dijkstra limits grow with the straight-line gap (max(2000 m, 3 × gap + 200 m)), so long time gaps remain decodable.
- **Disconnected segments:** if every transition into an observation is impossible, the chain restarts from that observation's emissions (a recorded 'HMM break'). After decoding, consecutive distinct arcs that do not connect are joined by the shortest directed path (≤ 3000 m), so the returned route is connected whenever a path exists.

**Did HMM decoding improve the route?** Yes, by a large margin. In the equal-data ablation (Section 11), classifier-only routes for the five models score 0.749–0.779 route F1 on the holdout trips and 0.947–0.951 after HMM decoding. Independent per-observation choices oscillate between parallel arcs and reverse twins, and every switch adds gap-stitching detours; the HMM removes most of these. Cases where it did not help are discussed in Section 11.

## 9  Experimental Protocol

| Data | Trips | Used for | Never used for |
|---|---|---|---|
| train — tune | 64 | Classifier fitting in 5 trip-grouped CV folds; decoder Optuna objective; emission-weight grid; model selection (out-of-fold route F1) | — |
| train — holdout | 16 | Validation of each selected change (decoder, then ML emissions) after it was frozen; finally included in the 80-trip refit | Any tuning or selection |
| student_test | 10 | Development comparison after each freeze; reporting of all five models | Training, tuning, selection |
| evaluator_test | 10 | One final evaluation of the frozen deployed model | Training, tuning, selection, exploration |

*Table 9. Data use. The 64/16 split is seeded (seed 42, experiments/train_partition.json).*

- **Selection rule** (stated before evaluation): the classifier with the best out-of-fold route F1 inside the HMM on the 64 tune trips; Random Forest scored 0.9458 at emission weight 1 versus 0.9449 for HistGradientBoosting and 0.9451 for the Decision Tree.
- **Freeze, then refit:** the RF + HMM configuration was validated on the holdout trips with the model trained on 64 trips, the choice to deploy it was fixed, and the Random Forest was refitted on all 80 training trips (models/random_forest__train80.joblib) before student test and evaluator_test were scored.
- **Repeated inspection of student test:** student-test scores were viewed several times (decoder v2, RF + HMM, every saved model, the ablation). They were not used to choose models or parameters, but repeated viewing still weakens student test as an unbiased estimate; the guidelines permit its use for development comparison.
- **Evaluator test:** scored once with the frozen model (SHA-256 recorded before prediction). Later re-runs only checked determinism, leakage (identical predictions with .route removed) and a post-freeze setup fix; all reproduced the frozen routes and scores exactly.

### 9.1 Metric definitions

| Level | Unit | Metrics | Compared against |
|---|---|---|---|
| Per-candidate classification | one (observation, candidate arc) row | precision, recall, F1 (positive class), macro F1, log loss, threshold 0.5 | pseudo-labels |
| Per-observation | one GPS observation | Top-1 accuracy (top-scored or decoded candidate = label), candidate miss rate | pseudo-labels (valid labels only) |
| Route level (primary) | one trip | arc precision / recall / F1 on the SETS of distinct directed arc ids; route continuity | .route |
| Route level (supplementary) | one trip | sequence F1 = longest common subsequence of the ordered arc sequences | .route |

*Table 10. Three levels of evaluation. Arc F1 collapses duplicates, ignores order, treats reverse twins as different arcs and counts gap-stitched arcs as predictions. The PathSnap documents define no official route formula.*

## 10  Model Comparison and Visualisations (Deliverables §3.3)

| Model | Precision | Recall | F1 | Macro F1 | Log loss | ROC AUC | Top-1 | Fold F1 sd | Fit s (5 folds) |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8068 | 0.7710 | 0.7885 | 0.8776 | 0.1468 | 0.9708 | 0.8239 | 0.0466 | 2.6 |
| Decision Tree | 0.8466 | 0.7917 | 0.8182 | 0.8949 | 0.1304 | 0.9780 | 0.8298 | 0.0495 | 31.8 |
| Random Forest | 0.8621 | 0.7944 | 0.8269 | 0.9000 | 0.1108 | 0.9834 | 0.8334 | 0.0481 | 147.1 |
| Extra Trees | 0.8638 | 0.7746 | 0.8168 | 0.8944 | 0.1214 | 0.9812 | 0.8265 | 0.0497 | 46.5 |
| HistGradientBoosting | 0.8521 | 0.8056 | 0.8282 | 0.9006 | 0.1114 | 0.9835 | 0.8352 | 0.0463 | 30.4 |

*Table 11. Cross-validation on 64 tune trips (out-of-fold, valid-label rows, threshold 0.5; pseudo-labels). Source: experiments/ml/cv_model_comparison.csv.*

![Figure 7. Five-model cross-validation comparison: macro F1 and log loss (mean ± sd over five trip-grouped folds).](figures\fig_cv_model_comparison.png)

*Figure 7. Five-model cross-validation comparison: macro F1 and log loss (mean ± sd over five trip-grouped folds).*

| Model (training trips) | Precision | Recall | F1 | Macro F1 | Log loss | Top-1 | Route F1 with HMM |
|---|---|---|---|---|---|---|---|
| Logistic Regression (64) | 0.8206 | 0.7946 | 0.8074 | 0.8900 | 0.1244 | 0.8339 | 0.9671 |
| Decision Tree (64) | 0.8544 | 0.8156 | 0.8346 | 0.9056 | 0.1011 | 0.8469 | 0.9737 |
| Random Forest (64) | 0.8723 | 0.8107 | 0.8404 | 0.9091 | 0.0914 | 0.8506 | 0.9730 |
| Extra Trees (64) | 0.8752 | 0.7853 | 0.8278 | 0.9022 | 0.1028 | 0.8402 | 0.9738 |
| HistGradientBoosting (64) | 0.8689 | 0.8177 | 0.8425 | 0.9103 | 0.0905 | 0.8535 | 0.9742 |
| Random Forest (80, deployed) | 0.8718 | 0.8095 | 0.8395 | 0.9086 | 0.0916 | 0.8497 | 0.9755 |
| Geometric HMM (no classifier) | — | — | — | — | — | — | 0.9683 |

*Table 12. Student-test comparison (10 trips), reported after models were frozen. Classifier columns use pseudo-labels; the last column is route-level against .route. The 80-trip row is not an equal-data comparison with the 64-trip rows. Source: experiments/ml/student_test_model_report.csv.*

![Figure 8. Student-test Top-1 accuracy and macro F1 for the five models (all trained on the same 64 trips).](figures\fig_student_test_top1_macro_f1.png)

*Figure 8. Student-test Top-1 accuracy and macro F1 for the five models (all trained on the same 64 trips).*

![Figure 9. Out-of-fold confusion matrices at threshold 0.5 (64 tune trips, valid-label rows).](figures\fig_confusion_matrices.png)

*Figure 9. Out-of-fold confusion matrices at threshold 0.5 (64 tune trips, valid-label rows).*

![Figure 10. Training (in-sample) versus out-of-fold CV metrics — evidence for over/underfitting.](figures\fig_train_vs_cv.png)

*Figure 10. Training (in-sample) versus out-of-fold CV metrics — evidence for over/underfitting.*

### 10.1 Why Random Forest + HMM was deployed

HistGradientBoosting has marginally better standalone classifier scores (CV F1 0.8282 vs 0.8269; student-test F1 0.8425 vs 0.8404 with both trained on 64 trips). The deployment decision, however, followed the pre-stated rule of best out-of-fold **route** F1 inside the HMM, because the route is the deliverable: Random Forest 0.9458 vs HistGradientBoosting 0.9449 at emission weight 1. The emission weight (4) was then tuned for Random Forest only. In the later equal-data ablation at weight 4 the two remain within noise: out-of-fold 0.9492 vs 0.9472, holdout 0.9511 vs 0.9511, student test 0.9730 vs 0.9742. The deployed model's student-test score (0.9755) comes from 80 training trips and must not be compared with 64-trip results as if the data were equal. HistGradientBoosting would be a defensible alternative with a far smaller model file.

## 11  Ablation Study: Classifier-only vs Classifier + HMM (Deliverables §5)

All variants share the candidate generation, decoder parameters, emission weight and route assembly of the deployed configuration; every classifier is trained on the same 64 tune trips (out-of-fold within them for the tune row). No evaluator data, retraining or tuning is involved (scripts/ablation.py).

| Set | Nearest road | RF only (no HMM) | Geometric HMM | RF + HMM | RF + HMM sequence F1 |
|---|---|---|---|---|---|
| Tune 64 (OOF) | 0.5428 | 0.6989 | 0.9398 | 0.9492 | 0.9158 |
| Holdout 16 | 0.5762 | 0.7631 | 0.9477 | 0.9511 | 0.9379 |
| Student test 10 | 0.5615 | 0.7079 | 0.9684 | 0.9730 | 0.9629 |

*Table 13. Ablation, mean route arc F1 (last column: order-aware sequence F1). Source: experiments/final/ablation_summary.csv.*

![Figure 11. What each component contributes, with equal training data.](figures\fig_ablation_route_f1.png)

*Figure 11. What each component contributes, with equal training data.*

![Figure 12. Classifier-only versus classifier + HMM for every model; dashed line = geometric HMM.](figures\fig_classifier_only_vs_hmm.png)

*Figure 12. Classifier-only versus classifier + HMM for every model; dashed line = geometric HMM.*

The HMM is the decisive component (+0.17 to +0.27 route F1 over classifier-only on holdout and student test). Tree-model emissions add a small gain over geometric emissions (+0.003 to +0.006 on the same sets), while Logistic Regression emissions give none (-0.0003 holdout, -0.0013 student test). Figure 13 shows the per-trip effect of replacing geometric emissions by Random Forest emissions: some trips get worse, so the gain is not uniform.

![Figure 13. Per-trip change in route F1 from geometric HMM to RF + HMM (RF trained on 64 trips).](figures\fig_per_trip_delta.png)

*Figure 13. Per-trip change in route F1 from geometric HMM to RF + HMM (RF trained on 64 trips).*

## 12  Final Evaluation on evaluator_test

The 10 evaluator_test trips (ids disjoint from all other partitions) include .route reference files. The guidelines state that the API computes accuracy from internal reference data, so scoring is permitted; references were read only after all predictions were made. The deployed model, configuration and inference code were frozen and hashed before prediction (experiments/final_evaluation/evaluation_manifest.json); nothing was changed after seeing the results.

| Variant | F1 mean | Median | Min | Max | SD | Precision | Recall | Seq. F1 | ms/trip |
|---|---|---|---|---|---|---|---|---|---|
| RF + HMM (deployed, 80 trips) | 0.9620 | 0.9802 | 0.8617 | 0.9989 | 0.0448 | 0.9371 | 0.9908 | 0.9511 | 334 |
| RF + HMM (64 trips, comparison) | 0.9618 | 0.9802 | 0.8617 | 0.9989 | 0.0448 | 0.9368 | 0.9908 | 0.9474 | 359 |
| Geometric HMM | 0.9502 | 0.9772 | 0.8028 | 1.0000 | 0.0648 | 0.9151 | 0.9941 | 0.9134 | 280 |
| RF only, no HMM (80 trips) | 0.7504 | 0.7524 | 0.5620 | 0.9368 | 0.1135 | 0.6135 | 0.9966 | 0.5049 | 66 |
| Nearest road | 0.6227 | 0.6179 | 0.5152 | 0.7402 | 0.0813 | 0.4579 | 0.9933 | 0.3092 | 45 |

*Table 14. evaluator_test results (directed arc-set metrics vs .route; runtime = server time per request, model pre-loaded, Apple M5). Source: experiments/final_evaluation/evaluation_summary.json.*

![Figure 14. Per-trip route F1 on evaluator_test: geometric HMM versus deployed RF + HMM.](figures\route_f1_comparison.png)

*Figure 14. Per-trip route F1 on evaluator_test: geometric HMM versus deployed RF + HMM.*

| Trip | GPS obs. | Arcs | Route len. | Distinct | F1 | Geo. F1 | Prec. | Recall | Seq. F1 | Contin. | No cand. | ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 00000015 | 6,854 | 3,225 | 197 | 175 | 0.8617 | 0.8774 | 0.7657 | 0.9853 | 0.8048 | 1.00 | 9 | 398 |
| 00000017 | 1,346 | 76,425 | 851 | 769 | 0.9603 | 0.9359 | 0.9272 | 0.9958 | 0.9623 | 1.00 | 0 | 755 |
| 00000023 | 458 | 10,596 | 68 | 66 | 0.9120 | 0.8028 | 0.8636 | 0.9661 | 0.8976 | 1.00 | 0 | 80 |
| 00000030 | 654 | 24,218 | 374 | 362 | 0.9442 | 0.9415 | 0.9116 | 0.9792 | 0.9283 | 1.00 | 0 | 257 |
| 00000049 | 2,642 | 47,542 | 409 | 409 | 0.9988 | 0.9988 | 1.0000 | 0.9976 | 0.9988 | 1.00 | 0 | 299 |
| 00000059 | 1,031 | 24,024 | 486 | 476 | 0.9774 | 0.9764 | 0.9559 | 1.0000 | 0.9671 | 1.00 | 0 | 358 |
| 00000067 | 916 | 4,485 | 82 | 82 | 0.9878 | 1.0000 | 0.9878 | 0.9878 | 0.9878 | 1.00 | 0 | 95 |
| 00000073 | 1,144 | 25,631 | 240 | 239 | 0.9958 | 0.9916 | 0.9916 | 1.0000 | 0.9958 | 1.00 | 0 | 222 |
| 00000087 | 1,399 | 107,451 | 454 | 454 | 0.9989 | 1.0000 | 1.0000 | 0.9978 | 0.9989 | 1.00 | 0 | 456 |
| 00000094 | 3,230 | 14,931 | 640 | 623 | 0.9829 | 0.9781 | 0.9679 | 0.9983 | 0.9695 | 1.00 | 0 | 416 |

*Table 15. Per-trip evaluator_test results for the deployed model. Source: experiments/final_evaluation/evaluation_per_trip.csv.*

All 10 trips were predicted without failure, deterministically, with valid arc ids and fully connected paths (continuity 1.00); API-reported metrics equal an independent recomputation, and predictions were identical when .route files were removed (10/10). The deployed model is +0.0117 ahead of the geometric HMM on average (6 better, 3 worse, 1 equal), but with p = 0.43 (Wilcoxon) and p = 0.51 (sign test) this is not statistically significant on 10 trips. RF trained on 64 versus 80 trips gives practically identical results (0.9618 vs 0.9620).

| Split | Model | Classifier training trips | Trips | F1 mean | F1 median |
|---|---|---|---|---|---|
| train: tune64 | geometric HMM | none | 64 | 0.9398 | 0.9745 |
| train: tune64 | RF + HMM (w=4) | 5-fold OOF inside 64 (~51 per fold) | 64 | 0.9492 | 0.9819 |
| train: holdout16 | geometric HMM | none | 16 | 0.9477 | 0.9740 |
| train: holdout16 | RF + HMM | 64 tune | 16 | 0.9511 | 0.9794 |
| student_test10 | geometric HMM | none | 10 | 0.9683 | 0.9786 |
| student_test10 | RF + HMM | 64 tune | 10 | 0.9730 | 0.9768 |
| student_test10 | RF + HMM (DEPLOYED ml_v3) | 80 train | 10 | 0.9755 | 0.9786 |
| evaluator_test10 | nearest road (no classifier, no HMM) | none | 10 | 0.6227 | 0.6179 |
| evaluator_test10 | RF only (no HMM) | 80 train | 10 | 0.7504 | 0.7524 |
| evaluator_test10 | geometric HMM | none | 10 | 0.9502 | 0.9772 |
| evaluator_test10 | RF + HMM | 64 tune | 10 | 0.9618 | 0.9802 |
| evaluator_test10 | RF + HMM (DEPLOYED ml_v3) | 80 train | 10 | 0.9620 | 0.9802 |

*Table 16. Route F1 across splits with training sizes labelled (tune-64 rows are optimistic: the same trips selected the decoder parameters and emission weight). Source: experiments/final_evaluation/comparison_with_previous_splits.csv.*

## 13  Error Analysis

| Error category (evaluator_test, 10 trips) | Geometric HMM | RF + HMM (deployed) |
|---|---|---|
| Wrong direction (reverse twin of a .route arc) | 67 | 35 |
| Gap-stitching detour only | 112 | 92 |
| Matched to a wrong road | 66 | 50 |
| Before/after the reference span | 3 | 5 |
| Missed reference arcs | 11 | 18 |
| Repeated traversals (not penalised) | 323 | 146 |

*Table 17. Each false-positive arc gets exactly one category. Source: experiments/final_evaluation/route_error_analysis.csv.*

![Figure 15. False-positive sources on student test (RF trained on 64 trips).](figures\fig_fp_breakdown_student_test.png)

*Figure 15. False-positive sources on student test (RF trained on 64 trips).*

- **Gap stitching** is the largest source of extra arcs and is unchanged by the classifier: when consecutive decoded arcs do not touch, the shortest connecting path is inserted and may differ from the road actually driven.
- **Direction errors** fall from 67 to 35 arcs with RF emissions, consistent with the heading features.
- **Difficult trips:** evaluator trip 00000015 has the lowest F1 (0.8617): 9 observations have no road within 200 m, and its 41 extra arcs split into 16 gap-fill, 12 direction and 13 wrong-road arcs — short stubs at junctions rather than wrong streets. Trips 00000015, 00000067 and 00000087 are slightly worse with RF emissions than with geometric emissions.
- **Candidate failures** are rare: the pseudo-label arc is among the candidates for 99.8% of valid training observations.
- **Reference truncation** (training trips 00000033 and 00000054) makes correct matches after the route end count as false positives; route precision on such trips understates quality.
- **Computational cost:** deployed mean 334 ms per evaluator trip (max 755 ms) versus 280 ms for the geometric HMM; the extra time is feature extraction and Random Forest inference.

![Figure 16. Server runtime per evaluator trip against trip length.](figures\runtime_distribution.png)

*Figure 16. Server runtime per evaluator trip against trip length.*

## 14  Geographical Visualisation Using Folium (Deliverables §6)

Interactive maps are generated by code in the repository from the deployed model's actual predictions (reports/tooling/build_assets.py; earlier map code: scripts/validate_alignment.py, scripts/final_eval_artifacts.py). Each map shows the original GPS observations, the road-network geometry around the trip, the reference route and the predicted route, colour-coded into arcs that are in .route, extra arcs and missed reference arcs, with a legend, a layer control and an extent fitted to the trip. Student-test trips are used so the maps can be published; maps of evaluator trips exist locally but are not reproduced because they draw evaluator ground truth.

![Figure 17. Folium map, student-test trip 00000077 (largest RF + HMM gain over the geometric HMM; arc F1 0.9693, identical to the recorded value). Headless-browser screenshot of reports/figures/maps/student_test_00000077_route_map.html.](figures\folium_screenshot_student_test_00000077.png)

*Figure 17. Folium map, student-test trip 00000077 (largest RF + HMM gain over the geometric HMM; arc F1 0.9693, identical to the recorded value). Headless-browser screenshot of reports/figures/maps/student_test_00000077_route_map.html.*

![Figure 18. Folium map, student-test trip 00000076 (lowest RF + HMM F1 on student test, 0.9512). Screenshot of reports/figures/maps/student_test_00000076_route_map.html.](figures\folium_screenshot_student_test_00000076.png)

*Figure 18. Folium map, student-test trip 00000076 (lowest RF + HMM F1 on student test, 0.9512). Screenshot of reports/figures/maps/student_test_00000076_route_map.html.*

At city scale the predicted route (blue) covers the reference (green) almost everywhere; errors are short orange stubs at junctions, visible when zooming into the interactive maps or in the static renderings in Appendix A.4.

## 15  API, UI and Deployment (Deliverables §7)

| Endpoint | Request | Response |
|---|---|---|
| GET /health | — | status, data_dir, default_config, available configs |
| GET /trips | — | count and list of {trip_id, split, has_reference_route} |
| POST /predict | {"trip_id": "00000091"} (optional "config", "method") | trip_id, gps_coordinates [lat, lon], predicted_coordinates [lat, lon] (null if unmatched), predicted_arc_ids, point_arc_ids, metrics (null without .route), method, runtime_ms, details |

*Table 18. FastAPI endpoints (main.py). Coordinates are always [latitude, longitude].*

- **Model loading:** the deployed bundle (models/random_forest__train80.joblib: model, feature names, schema version, training trip ids, CV metrics, library versions) is loaded at start-up; a schema mismatch or load failure falls back to geometric emissions and is reported in details.emission_fallback.
- **Running:** PATHSNAP_DATA_DIR=<folder> uvicorn main:app --host 0.0.0.0 --port 8000. The folder may be the parent of split folders or a split folder itself (post-freeze setup fix, commit 0fcc9b8, predictions verified identical).
- **Frontend:** frontend/app.py (Streamlit) calls the three endpoints over HTTP and draws GPS observations and the predicted route with Folium; it never receives .route data. A live HTTP check with the frontend's own client functions returned HTTP 200 for all 10 evaluator trips (experiments/final_evaluation/server_http_check.log).

```
{
  "trip_id": "00000091",
  "method": "hmm_viterbi_ml_emission",
  "runtime_ms": 185.4,
  "metrics": {"arc_precision": 0.9872, "arc_recall": 0.9957, "arc_f1": 0.9914, 
      "route_continuity": 1.0, "n_pred_arcs": 234, "n_ref_arcs": 232},
  "predicted_arc_ids": [345, 347, 3620, "..."],
  "point_arc_ids": [345, 345, 345, "..."],
  "gps_coordinates": [[47.099623, 2.418679], [47.099593, 2.418516], [47.099536, 2.418324], 
      "..."],
  "predicted_coordinates": [[47.0995607, 2.418734], [47.0995037, 2.4185947], [47.0994309, 
      2.4184166], "..."]
}
```

Example: a real response for student-test trip 00000091 (lists truncated; full lengths: gps_coordinates 874, predicted_coordinates 874, predicted_arc_ids 236, point_arc_ids 874). Source: reports/data/api_example.json.

```
2026-11-18 16:10:57,482 INFO pathsnap: predict trip=00000015 split=evaluator_test
    method=hmm_viterbi_ml_emission config=ml_v3 points=6854 route_arcs=197 runtime_ms=399.9
    metrics={'arc_precision': 0.7657, 'arc_recall': 0.9853, 'arc_f1': 0.8617}
2026-11-18 16:10:58,263 INFO pathsnap: predict trip=00000017 split=evaluator_test
    method=hmm_viterbi_ml_emission config=ml_v3 points=1346 route_arcs=851 runtime_ms=746.6
    metrics={'arc_precision': 0.9272, 'arc_recall': 0.9958, 'arc_f1': 0.9603}
```

Server log lines printed by the API for two evaluator requests (wrapped, truncated at 300 characters; the metrics field shows route scores computed server-side).

> **Missing material:** screenshots of the running UI on Laptop 2 and of API responses on Laptop 1's terminal (Deliverables §7) have not been captured yet; the figures above are static renderings of real predictions, not UI screenshots. The Folium figures in Section 14 are real map screenshots of model output, not of the UI.

## 16  Testing and Reproducibility

| Test file | Tests | Covers |
|---|---|---|
| frontend/test_contract.py | 20 | frontend parsing of /health, /trips and /predict responses |
| tests/test_ml.py | 11 | alignment (direction change, repeated arc), leakage guards, probability shape, model loading and fallback |
| tests/test_pathsnap.py | 26 | loading, arc ids, coordinate order, decoding, metric definitions, API, data-dir layouts |
| tests/test_robustness.py | 8 | missing samples, 15-min gap, single point, no candidates, sparse candidates, empty file, bad ids, determinism |

*Table 19. Automated tests: 65 passed in 2.51s (python -m pytest -q, git 627c448). Source: reports/data/test_run.json.*

**Environment:** Python 3.14.7 on macOS-26.5.1-arm64-arm-64bit-Mach-O (Apple M5, 10 cores); numpy 2.5.5, scipy 1.18.1, scikit-learn 1.9.1, optuna 5.1.0, pandas 3.1.1, fastapi 0.163.2, folium 1.0.0. The full pinned list is experiments/final/pip_freeze.txt; requirements.txt lists minimum versions. Seeds: 42 for the partition, CV models, Optuna samplers and all estimators.

**Reproduction** (from the repository root; complete list in Appendix A.5):

```
python -m pytest -q
python -m scripts.train_classifiers --stage dataset   # then cv, oof_hmm, emission_grid, 
      fit_tune, final
python -m scripts.tune --trials 150 --seed 42          # decoder Optuna
python -m scripts.ablation && python -m scripts.make_figures && python -m scripts.verify_figures
python -m scripts.final_evaluation                      # needs evaluator_test/ beside the repo 
      code
PATHSNAP_DATA_DIR=/path/to/evaluator_test uvicorn main:app --host 0.0.0.0 --port 8000
```

All 294 values plotted in the experiment figures were re-derived from their source CSVs by an independent script (experiments/final/figures/plotted_values_verification.csv: 294/294 match).

## 17  Limitations and Future Work

- **Pseudo-labels.** Per-observation targets come from an alignment with its own parameters; ambiguity at junctions and after truncated routes limits the meaning of Top-1 and candidate metrics. Future work: confidence-weighted labels.
- **Scoring metric.** Arc F1 on directed sets ignores order and repeated traversals; the order-aware LCS score is reported but no official formula was provided. Precision penalises correct matches beyond truncated references.
- **Gap stitching.** Shortest-path insertion is the main source of extra arcs; using time-consistent or classifier-guided path choice could reduce it.
- **Statistical power.** Ten evaluator trips cannot separate RF + HMM from the geometric HMM (p = 0.43).
- **Optimisation.** Random Forest Bayesian optimisation is incomplete (10 trials) and never evaluated; decoder optima lie near search-range edges; the emission weight was grid-searched for Random Forest only.
- **Calibration.** Classifier probabilities are used uncalibrated (clipped and normalised); calibration was not assessed.
- **Reference quality.** Some reference routes are truncated or contain discontinuities (Section 3), so .route is a reference, not verified ground truth.
- **Generalisation and data selection.** Trips come from several cities with very different network sizes; decoder parameters were tuned on the same 64 trips used for classifier CV, so tune-set numbers are optimistic.

## 18  Conclusion

PathSnap delivers a working, leakage-checked map-matching system: candidate generation on a directed network, five compared classifiers, a Bayesian-tuned HMM/Viterbi decoder and a FastAPI/Streamlit deployment that served all ten reserved trips deterministically. The HMM is responsible for most of the route quality; Random Forest emissions add a small gain that was significant on held-out training trips but not on the ten evaluator trips, where the deployed model reached a mean directed-arc-set F1 of 0.9620. Remaining work is listed in Section 17 and in the missing-material notes above.

## 19  AI-Assistance Disclosure (Deliverables §8)

As required by Deliverables §8, the team discloses that an AI coding assistant (Anthropic Claude, used through Claude Code) was used extensively to write source code, tests and experiment scripts, to run experiments, and to draft this report from the recorded experiment files. All numbers in this report are generated from files in the repository by reports/tooling/build_report.py; the team remains responsible for verifying and explaining every claim.

## 20  References

1. Newson, P. and Krumm, J. (2009). Hidden Markov map matching through noise and sparseness. In Proc. 17th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, pp. 336–343.
2. Viterbi, A. J. (1967). Error bounds for convolutional codes and an asymptotically optimum decoding algorithm. IEEE Transactions on Information Theory, 13(2), 260–269.
3. Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. Numerische Mathematik, 1, 269–271.
4. Akiba, T., Sano, S., Yanase, T., Ohta, T. and Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In Proc. 25th ACM SIGKDD, pp. 2623–2631.
5. Bergstra, J., Bardenet, R., Bengio, Y. and Kégl, B. (2011). Algorithms for hyper-parameter optimization. Advances in Neural Information Processing Systems 24.
6. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
7. Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.
8. Geurts, P., Ernst, D. and Wehenkel, L. (2006). Extremely randomized trees. Machine Learning, 63(1), 3–42.
9. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189–1232.
10. Virtanen, P. et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nature Methods, 17, 261–272.
11. Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics Bulletin, 1(6), 80–83.
12. PathSnap hackathon documents (2026): Main Guidelines; Deliverables; Models and Training Guidelines; Dataset Exploration; Assessment and Plagiarism Policy. PES University, UE24CS352A.
13. Software used: FastAPI, Uvicorn, Folium, Streamlit, pandas, matplotlib, joblib (versions in experiments/final/pip_freeze.txt).

## Appendix A

### A.1 Model hyperparameters (initial, deployed for Random Forest)

| Model | Configuration (scikit-learn 1.9.1) |
|---|---|
| Logistic Regression | StandardScaler + LogisticRegression(C=1.0, max_iter=2000) |
| Decision Tree | DecisionTreeClassifier(max_depth=12, min_samples_leaf=50) |
| Random Forest | RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=20, max_features='sqrt') |
| Extra Trees | ExtraTreesClassifier(n_estimators=150, max_depth=18, min_samples_leaf=20, max_features='sqrt') |
| HistGradientBoosting | HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False) |

*Table 20. Exact estimator settings from src/ml_train.py::make_models (random_state = 42).*

### A.2 Deployed configuration configs/ml_v3.json

```
{
 "name": "ml_v3",
 "k": 12,
 "radius_m": 52.72439791668979,
 "fallback_radius_m": 200.0,
 "sigma_m": null,
 "sigma_scale": 0.8662467912517785,
 "beta": 2.335001654519724,
 "backtrack_tol_m": 24.693182489954108,
 "backtrack_penalty": 0.0,
 "uturn_penalty": 13.74033717036362,
 "dijkstra_limit_m": 2000.0,
 "min_spacing_m": 11.642490335126803,
 "fill_max_gap_m": 3000.0,
 "emission": "ml",
 "ml_model": "random_forest__train80",
 "ml_weight": 4.0,
 "prob_floor": 0.0001,
 "max_loop_m": 0.0
}
```

### A.3 Evidence and post-freeze record

- Frozen model SHA-256 (models/random_forest__train80.joblib): e0267fb4ed25839219264d02aa709e447e9db99a4f469822ac97779a93a14c61
- Frozen commit: 8459a045f9c3949852db2c75ecd0e87dcf7521a3; evaluation started 2026-11-18T16:08:17.
- Post-freeze setup change: commit 0fcc9b8 (src/data_loader.py::list_trips). SHA-256 95a49ac17086e130… → 735d39d91b16ed9f…; 20 re-checks, routes identical: True, F1 identical: True.
- Evidence index for every number and figure: reports/report_evidence_index.md.

### A.4 Static renderings of the route maps

![Figure 19. Route map for student-test trip 00000077 (largest gain of RF + HMM over geometric HMM): road network, GPS, reference and predicted route. Static rendering of the same layers as the interactive Folium map reports/figures/maps/student_test_00000077_route_map.html.](figures\fig_route_map_student_test_00000077.png)

*Figure 19. Route map for student-test trip 00000077 (largest gain of RF + HMM over geometric HMM): road network, GPS, reference and predicted route. Static rendering of the same layers as the interactive Folium map reports/figures/maps/student_test_00000077_route_map.html.*

![Figure 20. Route map for student-test trip 00000076 (lowest RF + HMM F1 on student test). Interactive version: reports/figures/maps/student_test_00000076_route_map.html.](figures\fig_route_map_student_test_00000076.png)

*Figure 20. Route map for student-test trip 00000076 (lowest RF + HMM F1 on student test). Interactive version: reports/figures/maps/student_test_00000076_route_map.html.*

### A.5 Command-line reproduction

```
# setup
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# classifier pipeline (train trips only)
python -m scripts.train_classifiers --stage dataset
python -m scripts.train_classifiers --stage cv
python -m scripts.train_classifiers --stage oof_hmm
python -m scripts.train_classifiers --stage emission_grid --model random_forest
python -m scripts.train_classifiers --stage fit_tune
python -m scripts.compare --a tuned_v2 --b ml_candidate --trips holdout
python -m scripts.train_classifiers --stage final --model random_forest --emission ml --weight 4
python -m scripts.train_classifiers --stage student_test
# Bayesian optimisation
python -m scripts.tune --trials 150 --seed 42
python -m scripts.tune_classifier   # stopped early
# verification and figures
python -m scripts.verify_labels
python -m scripts.ablation
python -m scripts.make_figures && python -m scripts.verify_figures
# final evaluation (frozen model)
python -m scripts.final_evaluation && python -m scripts.final_eval_artifacts
python -m scripts.verify_post_freeze
# this report
.venv/bin/python reports/tooling/build_assets.py
.venv/bin/python reports/tooling/collect_evidence.py
reports/tooling/.venv/bin/python reports/tooling/build_report.py
node reports/tooling/build_docx.js
```
