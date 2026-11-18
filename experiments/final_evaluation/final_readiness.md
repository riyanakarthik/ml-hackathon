# PathSnap — final readiness checklist

Status as verified on 2026-11-18 by the commands named in each line. Numbers come from the files in this folder.

## Model and integrity
- [x] Deployed config `ml_v3` (RF + HMM, RF trained on 80 train trips). Its SHA-256 matches the frozen manifest: `evaluation_manifest.json` → `frozen_manifest_match`.
- [x] Inference code and deployed artifacts are byte-identical to commit `8459a04` (`git diff 8459a04 -- main.py src/{config,data_loader,matcher,features,ml_emission,decoder,pipeline}.py models configs/ml_v3.json configs/tuned_v2.json` is empty).
- [x] No retraining, tuning or config change after seeing evaluator_test results.
- [x] The model loads with training-consistent feature order and schema; there is no calibration step; no evaluator trip is in its training set (`evaluation_summary.json` → `model_check`).
- [x] `.route` is read only by `src/evaluation.py::load_reference_route`. It is called after prediction (in `main.py`), in experiments and in training-label code, and never in candidates, features, ML inference, HMM decoding or gap filling. This is checked statically (`tests/test_ml.py::test_inference_modules_do_not_import_reference_code`) and at runtime: predictions are identical with `.route` removed, 10/10.
- [x] Training labels follow the chronological `.route` order on 80/80 trips (`experiments/final/label_verification.csv`).

## Final evaluation (evaluator_test, 10 trips, reference routes present)
- [x] All 10 trips went through `POST /predict` (the real FastAPI app) with 0 failures and deterministic outputs (each request sent twice).
- [x] Every predicted arc ID exists, and every predicted route is a connected directed path (continuity 1.0 on all trips).
- [x] Metrics were computed after prediction, in evaluation code, and the API's own metrics equal the independent recomputation.
- [x] Deployed RF+HMM: mean route F1 0.9620 (median 0.9802, min 0.8617, max 0.9989, sd 0.0448). Geometric HMM: 0.9502. Classifier-only: 0.7504. Nearest road: 0.6227.
- [x] Honest significance check: 6 trips better, 3 worse, 1 same. Wilcoxon p = 0.4258, so the gain is **not significant** on 10 trips.
- [x] Comparison across splits, with the classifier's training size labelled: `comparison_with_previous_splits.csv`.

## Robustness and API contract
- [x] Automated tests: **65 passed, 0 failed** after the post-freeze fix (64 at freeze) (`python -m pytest -q`, which includes the 20 frontend contract tests). This covers missing GPS samples, a 15-minute timestamp gap, a single-point track, no candidates anywhere, sparse candidates (fallback radius), an empty track file (clean 404), unknown, non-numeric and missing trip IDs, and determinism (`tests/test_robustness.py`).
- [x] Laptop 2 contract is unchanged. A live uvicorn server, called over HTTP with the frontend's own `check_health`, `fetch_trips` and `run_prediction`, listed and predicted all 10 evaluator trips (HTTP 200). Trip 15 has 9 `null` predicted points, which have no road within 200 m; `sanitize_coords` drops them as designed.
- [x] Latency (Apple M5, model preloaded): deployed mean ≈ 330 ms per trip, max ≈ 755 ms (`runtime_distribution.png`).

## Demo-day actions
- [ ] Start the API: `PATHSNAP_DATA_DIR=/path/to/evaluator_test uvicorn main:app --host 0.0.0.0 --port 8000`. The parent folder also works. The `/trips` discovery issue was **fixed post-freeze** in commit `0fcc9b8` (a setup-only change). Predictions were verified identical: `post_freeze_checks/post_freeze_check.json`.
- [ ] Open the server terminal so evaluators can see the request logs, as the guidelines require.
- [ ] Map tiles need internet on Laptop 2.

## Known open items (not done)
- [ ] Random Forest Bayesian optimisation is **incomplete**: 10 trials completed, then stopped on request. The best log loss is 0.11033 vs 0.11071 untuned, and the tuned model was not evaluated or deployed.
- [x] The equal-data ablation and the five-model figures have been run, and all 294 plotted values were verified against their sources (`experiments/final/`).
- [ ] The PDF report is not written (REPORT_INPUT.md is its source).
- [ ] Changes since `8459a04` are uncommitted: evaluation code, tests, scripts and results. The model and inference code are unchanged.
- [x] `evaluator_test/` (contains ground-truth `.route`) is now in `.gitignore`, so it is not published in the public repo.
