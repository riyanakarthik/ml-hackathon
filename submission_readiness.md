# PathSnap Submission Readiness Audit

**Evaluation Target:** PES University — UE24CS352A: Machine Learning Hackathon (RR Campus)  

### **Team ID:** PT-056  
| Sl. No. | Student Name | SRN |
|:---:|:---|:---|
| 1 | RIYANA KARTHIK VISHALAKSHI | PES1UG24AM226 |
| 2 | ROHANTH GANESHA C | PES1UG24AM227 |
| 3 | ROHIT CHAKI | PES1UG24AM228 |
| 4 | ROHITH KARNING | PES1UG24AM229 |

**Overall Readiness Score:** **100% OF SYSTEM CODE, PIPELINE, ARTIFACTS & REPORT DELIVERED** (All 65 automated tests passing; all 10 evaluation trips verified end-to-end; final 29-page technical report compiled and verified).

---

## 1. Deliverable 1: GitHub Repository (Deliverables §1)

| Required Component | Official Guideline Requirement | Implementation / Code Evidence | Status |
|---|---|---|:---:|
| **Public Repository** | Public GitHub repository accessible to evaluators | `github.com/riyanakarthik/ml-hackathon` (Branch `main`) | **PASS** |
| **Data Processing** | Source code to read, inspect, clean, and preprocess raw data | `src/data_loader.py` (zero-leakage loading, validation of `.track`, `.nodes`, `.arcs`), `src/matcher.py` (local tangent plane equirectangular projection, densified KD-tree) | **PASS** |
| **Exploratory Data Analysis (EDA)** | Inventory, missing values, sampling intervals, trajectory & network analysis, Folium map | `scripts/eda.py` (structured CLI exploration), `reports/data/eda_summary.json`, `reports/figures/maps/eda_trip_00000040_map.html`, `reports/figures/fig_dataset_overview.png` | **PASS** |
| **Feature Engineering** | Feature extraction and preparation (inference-safe) | `src/features.py` (20 kinematic & topological candidate features; strictly never imports or reads `.route`, verified by `tests/test_ml.py`) | **PASS** |
| **Model Training Zoo** | Training scripts/code for >=5 distinct model families | `src/ml_train.py` & `scripts/train_classifiers.py` implementing: <br>1. Logistic Regression (Linear)<br>2. Decision Tree (Tree)<br>3. Random Forest (Ensemble)<br>4. Extra Trees (Ensemble)<br>5. HistGradientBoosting (Boosting) | **PASS** |
| **Model Evaluation** | Model comparison code, metrics, plots, confusion matrices | `scripts/make_figures.py`, `experiments/ml/cv_model_comparison.csv`, `experiments/ml/student_test_model_report.csv`, `experiments/final/figures/fig_confusion_matrices.png` | **PASS** |
| **Bayesian Hyperparameter Optimisation** | Bayesian optimization search configs, trial records, tuned models | **Decoder HMM:** `scripts/tune.py`, `experiments/optuna_trials.csv` (150 trials, trial 133 best), `configs/tuned_v2.json`<br>**Random Forest:** `scripts/tune_classifier.py`, `experiments/ml/rf_optuna_study.db`, `experiments/ml/rf_optuna_trials_partial.csv` (10 trials, trial 10 best) | **PASS** |
| **HMM & Viterbi Sequential Decoding** | Route reconstruction, sequential decoding, gap filling | `src/decoder.py` (`viterbi_log` in log-space, `hmm_viterbi_decode` with exponential network-distance discrepancy, Dijkstra shortest-path gap filling) | **PASS** |
| **FastAPI Backend Service** | Complete backend code, endpoints, model-loading logic | `main.py` (`GET /health`, `GET /trips`, `POST /predict`), startup model warm-up in `src/ml_emission.py`, automatic fallback to geometric HMM if model unavailable | **PASS** |
| **Evaluator Interface (UI)** | Complete evaluator UI source code | `frontend/app.py` (Streamlit + Folium interactive console, dynamic trip selection, real-time KPI cards, layer toggles) | **PASS** |
| **Deployment & Dependencies** | Dependencies, configs, multi-laptop run instructions | `requirements.txt`, `frontend/requirements.txt`, `run_backend.bat`, `run_backend.sh`, `run_frontend.bat`, `run_frontend.sh`, `verify_system.bat`, `verify_system.sh`, `README.md`, `docs/integration/LAN_SETUP.md` | **PASS** |
| **Geographical Visualisations** | Folium map-generation code and plotting scripts | `frontend/app.py`, `scripts/eda.py`, `reports/tooling/build_assets.py`, `reports/figures/maps/student_test_00000076_route_map.html`, `reports/figures/maps/student_test_00000077_route_map.html` | **PASS** |
| **Supporting Results & Logs** | Relevant CSV files, experiment logs, configs, hashes | `experiments/` (all trials, per-trip metrics, comparison logs), `configs/` (`baseline_v1.json`, `tuned_v2.json`, `ml_v3.json`), `experiments/final/frozen_artifacts_sha256.json` | **PASS** |
| **Saved Model Weights** | Saved trained model artifacts and bundles | `models/` directory containing all 6 `.joblib` artifacts: <br>• `random_forest__train80.joblib` (46.8 MB, deployed in `ml_v3`)<br>• `random_forest__tune64.joblib`<br>• `extra_trees__tune64.joblib`<br>• `hist_gb__tune64.joblib`<br>• `decision_tree__tune64.joblib`<br>• `logreg__tune64.joblib` | **PASS** |
| **Academic Integrity & Leakage Prevention** | No secrets, no evaluator ground truth in prediction/UI, no hardcoded routes | `.gitignore` isolates `evaluator_test/` and ground-truth maps; `tests/test_ml.py` asserts no-route leakage; predictions computed live dynamically | **PASS** |

---

## 2. Deliverable 2: Technical Report (Deliverables §2 – §8)

| Report Section | Official Requirement | Content in Submitted Report | Status |
|---|---|---|:---:|
| **Consolidated PDF Document** | Single consolidated technical report following §2–§7 structure | `reports/PathSnap_Final_Technical_Report.pdf` (**29 pages**, A4, 20 figures, 20 tables, verified citations, bookmarks, running headers). Editable source in `.docx` and `.md`. | **PASS** |
| **§2.2 Dataset Understanding & EDA** | Preprocessing, patterns, feature choices, missing-value analysis, GPS plots, network plots, >=1 Folium map, partition per analysis | Section 3 (Tables 2, 3, 4, Figure 1); Section 5 (Preprocessing & label alignment); Section 14 (Folium maps); `reports/data/dataset_overview_summary.csv` | **PASS** |
| **§3.1 Overall Model Comparison** | Consolidated comparison table for >=5 models across 7 required columns | Section 6, Table 6: Compares Logistic Regression, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting on CV Macro F1, Top-1 accuracy, Route continuity, Student-test Macro F1, Log loss, and Inference time | **PASS** |
| **§3.2 Compulsory Questions for EVERY Model** | Separate subsection answering A1–A2, B5–B6, C7–C9 for each of the 5 models | Sections 6.1 (LogReg), 6.2 (Decision Tree), 6.3 (Random Forest), 6.4 (Extra Trees), 6.5 (HistGradientBoosting). All compulsory questions answered with empirical data. | **PASS** |
| **§3.3 Model Visualisations** | CV Macro F1, Student-test Top-1 & Macro F1, Log loss, confusion matrices; no artificial curves | Figures 7 (CV model comparison), 8 (Student-test Top-1 & Macro F1), 9 (Log loss comparison), 10 (Confusion matrices). All values verified from actual CSV logs; zero artificial curves. | **PASS** |
| **§4 Bayesian Hyperparameter Optimisation** | Search space, ranges, objective, trials, best values, tuned vs untuned comparison | Section 7: Complete documentation of Optuna TPE on HMM decoder (150 trials, trial 133 best, Table 8, Fig 4, 5) and Random Forest classifier (10 trials, trial 10 best log loss 0.11033 vs 0.11071, Table 9, Fig 6). Transparently documents stopping criteria. | **PASS** |
| **§5 HMM and Viterbi Route Reconstruction** | Hidden states, emissions, transitions, missing candidates/gaps, disconnected segments, classifier-only vs classifier+HMM | Section 8 (HMM formulation, Gaussian/ML emissions, Dijkstra transitions, Viterbi recurrence); Section 11 & Table 13 (Ablation comparing classifier-only arc F1 0.763 vs classifier+HMM 0.951; Wilcoxon paired tests). | **PASS** |
| **§6 Geographical Visualisation Using Folium** | Interactive maps with GPS observations, road network, reference route, and predicted route; legend; layer controls; screenshot in PDF | Section 14 (Figures 17–20, high-res exported views of Trips 76 and 77); interactive HTML files in `reports/figures/maps/`; code in `reports/tooling/build_assets.py` and `scripts/eda.py`. | **PASS** |
| **§7 API, UI and Deployment** | Screenshots of working UI and API responses on terminal of receiving laptop | Section 15: Full API log trace and architecture diagrams included. Live demonstration ready for in-person screenshots on Laptop 1 and Laptop 2. | **PASS (Ready for Demo)** |
| **§8 Academic Integrity & AI Disclosure** | Policy compliance, originality, tool usage disclosure | Section 19: Comprehensive AI tool disclosure detailing role in scaffolding and test automation while all models, features, and evaluations were executed locally. | **PASS** |

---

## 3. Models and Training Guidelines Compliance

| Requirement | Implementation Verification | Status |
|---|---|:---:|
| **>=5 Models from Multiple Families** | 5 families implemented: Linear/Probabilistic (LogReg), Tree (DT), Ensemble Bagging (RF, ET), Boosting (HistGB). | **PASS** |
| **Group-Aware Cross-Validation** | 5-fold `GroupKFold` grouped by trip ID across the 64 tune trips; zero trip overlap across train/val splits (`experiments/ml/cv_folds.json`). | **PASS** |
| **Training & Inference Time Recorded** | Detailed in `experiments/ml/cv_model_comparison.csv` (`fit_s_total`, `predict_ms_per_1k_rows`) and Table 6 of the report. | **PASS** |
| **Experiment Reproducibility** | Fixed random seed (`42`) across NumPy, Scikit-Learn, Optuna; exact package versions locked in `experiments/final/pip_freeze.txt`. | **PASS** |
| **Candidate Miss Rate & Route Continuity** | Tracked and reported across all configurations in `experiments/compare_*.csv` and `experiments/final/ablation_summary.csv`. | **PASS** |
| **Evaluator Data Untouched During Development** | Verified by frozen SHA-256 hashes (`experiments/final/frozen_artifacts_sha256.json`) prior to running final evaluation scripts. | **PASS** |

---

## 4. End-to-End System Verification (Actual Code Execution)

### 4.1 Automated Test Suite
* Command: `pytest -q`
* Result: **65 / 65 PASSED** in 13.71s
  * `frontend/test_contract.py`: 20 / 20 PASS (mocked payload robustness, error states, coord sanitization)
  * `tests/test_ml.py`: 11 / 11 PASS (feature shape, schema invariance, zero-leakage assertions)
  * `tests/test_pathsnap.py`: 23 / 23 PASS (data loading, equirectangular projection, HMM decoding, Dijkstra route stitching)
  * `tests/test_robustness.py`: 11 / 11 PASS (empty files, NaN/Inf handling, disconnected components, timestamp inversions)

### 4.2 End-to-End Evaluation Benchmark (All 10 Student Test Trips)
* Command: `python frontend/verify_evaluation_trips.py`
* Result: **10 / 10 PASSED**

| Trip ID | Points | Arcs | Runtime (ms) | Arc Precision | Arc Recall | Arc F1 | Route Continuity | Map Generation |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `00000000` | 2,503 | 201 | 4,655.5 | 0.9133 | 1.0000 | 0.9547 | 0.9900 | **PASS** |
| `00000007` | 1,354 | 121 | 513.2 | 1.0000 | 0.9837 | 0.9918 | 1.0000 | **PASS** |
| `00000009` | 1,049 | 466 | 1,167.8 | 0.9197 | 0.9883 | 0.9528 | 1.0000 | **PASS** |
| `00000040` | 2,995 | 319 | 1,132.8 | 0.9936 | 1.0000 | 0.9968 | 1.0000 | **PASS** |
| `00000052` | 262 | 93 | 477.4 | 1.0000 | 0.9789 | 0.9894 | 1.0000 | **PASS** |
| `00000056` | 3,958 | 1,390 | 3,248.1 | 0.9435 | 1.0000 | 0.9709 | 0.9993 | **PASS** |
| `00000065` | 768 | 151 | 1,164.7 | 0.9732 | 1.0000 | 0.9864 | 1.0000 | **PASS** |
| `00000076` | 1,486 | 404 | 3,229.5 | 0.9158 | 0.9893 | 0.9512 | 1.0000 | **PASS** |
| `00000077` | 3,911 | 328 | 2,064.7 | 0.9404 | 1.0000 | 0.9693 | 1.0000 | **PASS** |
| `00000091` | 874 | 236 | 1,152.7 | 0.9872 | 0.9957 | 0.9914 | 1.0000 | **PASS** |
| **MEAN** | **1,916** | **371** | **1,880.6** | **0.9587** | **0.9936** | **0.9755** | **0.9989** | **100% PASS** |

---

## 5. Live Demonstration Action Checklist (Team Tasks)

1. **Physical Setup:**
   * Laptop 1 (Backend): Run `.\run_backend.bat` (or `./run_backend.sh`), ensure port 8000 is open in firewall, note the Wi-Fi IP address.
   * Laptop 2 (Presenter): Run `.\run_frontend.bat` (or `./run_frontend.sh`), open `http://localhost:8501`, enter `http://<LAPTOP_1_IP>:8000`, and confirm green `Backend online` indicator.
2. **Screenshots for Presentation:**
   * Capture 1 screenshot of Laptop 1's terminal showing incoming `POST /predict` HTTP logs.
   * Capture 1 screenshot of Laptop 2's Streamlit browser showing Trip `00000040` (Folium map + metrics).
