# PathSnap submission readiness

Status for every requirement in the official PathSnap documents (`PathSnap_student_materials/*.pdf`), checked on 2026-11-18.
**PASS** means the evidence exists at the path shown. **FAIL** means it is missing. **TEAM** means a manual task only the team can do.

## Deliverable 1: GitHub repository (Deliverables.pdf §1)
| Requirement | Status | Evidence |
|---|---|---|
| Repository is public | PASS | `gh repo view` → PUBLIC (github.com/riyanakarthik/ml-hackathon) |
| Data processing: read, inspect, clean, preprocess | PASS | `src/data_loader.py`, `src/matcher.py`, `scripts/validate_alignment.py` |
| **Dataset exploration / EDA code** (inventory, missing values, sampling intervals, trajectory and network plots) | **FAIL** | No EDA script or notebook exists |
| Feature engineering code | PASS | `src/features.py` |
| Training scripts for every model | PASS | `src/ml_train.py`, `scripts/train_classifiers.py` |
| Model comparison, metrics, plots, confusion matrices | PASS | `scripts/make_figures.py`, `experiments/final/figures/fig_confusion_matrices.png`, `experiments/ml/cv_model_comparison.csv` |
| Bayesian optimisation code, search config, results | PASS (decoder) / partial (RF) | `scripts/tune.py`, `experiments/optuna_trials.csv`; `scripts/tune_classifier.py`, `experiments/ml/rf_optuna_trials_partial.csv` |
| HMM and Viterbi code | PASS | `src/decoder.py` |
| API with model loading | PASS | `main.py`, `src/ml_emission.py`, `src/pipeline.py` |
| UI source | PASS | `frontend/app.py` |
| Deployment config, dependencies, instructions | PASS | `requirements.txt`, `frontend/requirements.txt`, `README.md`, `frontend/README.md` |
| Folium map-generation code and plotting scripts | PASS | `scripts/validate_alignment.py`, `scripts/final_eval_artifacts.py`, `scripts/make_figures.py` |
| Supporting CSVs, logs, configs | PASS | `experiments/`, `configs/` |
| Saved weights and configs | PASS | `models/*.joblib`, `configs/ml_v3.json` |
| README with install, run and deploy steps | PASS | `README.md` |
| No secrets committed | PASS | Pattern scan of tracked files found nothing |
| Evaluator ground truth not published | PASS | `evaluator_test/` and `experiments/final_evaluation/maps/` git-ignored; unpushed commit rebuilt without the maps |

## Deliverable 2: technical report (PDF)
| Requirement | Status | Evidence / source for the report |
|---|---|---|
| PDF exists | FAIL (not started, by design) | Source material: `experiments/final_evaluation/REPORT_INPUT.md` |
| §2.2 EDA: missing values, GPS trajectory plots, road-network plots, ≥1 Folium map, partition per analysis | **FAIL** | Only alignment maps exist (`experiments/ml/alignment_plots/`); no EDA outputs |
| §3.1 five-model table (CV macro F1, Top-1, continuity, student-test macro F1, log loss, inference time) | PASS | `experiments/ml/cv_model_comparison.csv`, `experiments/ml/student_test_model_report.csv`, `experiments/final/ablation_summary.csv` |
| §3.2 per model: hyperparameters, objective, student-test Top-1 / macro F1 / log loss, over/underfitting evidence | PASS | `src/ml_train.py::make_models`, `experiments/final/train_vs_cv_metrics.csv` |
| §3.3 figures: CV macro F1, student-test Top-1 and macro F1, log loss, model-selection plots (training curves N/A: no epoch-based models) | PASS | `experiments/final/figures/` (294/294 values verified) |
| §4 BO: space, ranges, objective, trials, best values | PASS (decoder) / partial (RF: 10 completed trials) | `experiments/optuna_trials.csv`, `experiments/ml/rf_optuna_trials_partial.csv` |
| §4 **tuned vs untuned comparison of the chosen model** (CV macro F1, student-test Top-1 / macro F1, log loss, inference time) | **FAIL for the RF** (decoder comparison exists) | Decoder: `experiments/compare_baseline_v1_vs_tuned_*.csv`. RF: none, because the tuned RF was never refit or evaluated |
| §5 HMM: states, emissions, transitions, gaps; classifier-only vs classifier+HMM | PASS | `src/decoder.py`, `src/ml_emission.py`, `experiments/final/figures/fig_classifier_only_vs_hmm.png` |
| §6 Folium map in the repo with GPS, **road network**, reference route and predicted route, legend, layer control | **FAIL (partial)** | Evaluator maps have GPS, reference and predicted layers but no network layer and are local-only; no committed map has all four layers |
| §7 screenshots of the UI and API responses on Laptop 1's terminal | TEAM | Run the two-laptop demo and capture screenshots |
| §8 AI-assistance disclosure | TEAM | Write in the report as the event policy requires |

## Models and Training Guidelines
| Requirement | Status | Evidence |
|---|---|---|
| ≥5 models from different families, trained and compared | PASS | `experiments/ml/cv_model_comparison.csv` |
| Group-aware CV on the 80 train trips | PASS | `experiments/ml/cv_folds.json` (GroupKFold by trip, 64 tune trips) |
| Training and inference time recorded | PASS | `cv_model_comparison.csv` (`fit_s_total`, `predict_ms_per_1k_rows`) |
| Logs, seeds, configs retained | PASS | `experiments/`, `experiments/final/pip_freeze.txt`, seed 42 throughout |
| Candidate miss rate and route continuity | PASS | `experiments/compare_*.csv` (`candidate_miss_rate`, `route_continuity`) |
| Refit the selected model with the tuned configuration, then evaluate on student test | **FAIL** | Deployed RF uses initial hyperparameters |
| Evaluator partition untouched until the final evaluation | PASS | `experiments/final/frozen_artifacts_sha256.json`, `experiments/final_evaluation/evaluation_manifest.json` |

## Final evaluation and integrity (Main Guidelines §6, §7)
| Requirement | Status | Evidence |
|---|---|---|
| API on Laptop 1 runs the full pipeline for all 10 reserved trips | PASS | `experiments/final_evaluation/evaluation_per_trip.csv` (0 failures) |
| UI lists the ten trips and displays route and results | PASS | Live HTTP check with frontend functions: `experiments/final_evaluation/server_http_check.log`; `/trips` works in both data-dir layouts (`post_freeze_checks/`) |
| API computes accuracy server-side; ground truth never reaches the model or UI | PASS | `tests/test_ml.py` leakage tests; no-`.route` re-run identical (10/10) |
| API request logs visible | PASS | `main.py` logs every `/predict` (`experiments/final_evaluation/evaluation_run.log`) |
| No hardcoded predictions | PASS | Predictions recomputed per request; identical across runs |
| All automated tests pass | PASS | `python -m pytest -q`: 65 passed, 0 failed |

## Blockers before writing the PDF
1. **EDA:** an exploration script and its outputs (inventory, missing values and time gaps, trajectory plots, road-network and connectivity plots, at least one Folium map, partition stated). Uses train and student-test data only.
2. **Folium report map with all four layers:** GPS, road network, reference route and predicted route, with legend and layer control, for a train or student-test trip so it can be committed publicly.
3. **RF tuned-vs-untuned comparison (§4):** `python -m scripts.tune_classifier --stage evaluate` fits a comparison RF with the best params from the partial search on the 64 tune trips, then reports CV, holdout and student-test results. It doesn't touch the deployed model or evaluator data. Alternatively, finish the search first. If neither is done, the report must state that §4 is incomplete for the classifier.
4. **Team tasks:** demo screenshots (§7) and the AI-use disclosure (§8).
