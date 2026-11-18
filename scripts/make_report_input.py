"""Write experiments/final_evaluation/REPORT_INPUT.md and final_readiness.md.

  python -m scripts.make_report_input
All tables are generated from recorded CSV/JSON files (paths cited inline).
"""
from __future__ import annotations

import json

import pandas as pd

from src.config import get_config
from src.experiment import EXPERIMENT_DIR, load_partition

FE = EXPERIMENT_DIR / "final_evaluation"
ML = EXPERIMENT_DIR / "ml"
NICE = {"logreg": "Logistic Regression", "decision_tree": "Decision Tree", "random_forest": "Random Forest",
        "extra_trees": "Extra Trees", "hist_gb": "HistGradientBoosting"}


def _np(o):
    return o.item() if hasattr(o, "item") else str(o)


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.4f}" if isinstance(v, float) else str(v))
                                     for v in r.values) + " |")
    return "\n".join(out)


def main() -> None:
    import joblib
    part = load_partition()
    summ = json.load(open(FE / "evaluation_summary.json"))
    man = json.load(open(FE / "evaluation_manifest.json"))
    cfg = get_config("ml_v3").to_dict()
    bundle = joblib.load(f"models/{cfg['ml_model']}.joblib")
    rf = bundle["model"].get_params()
    rf_keep = {k: rf[k] for k in ("n_estimators", "max_depth", "min_samples_leaf", "max_features",
                                  "bootstrap", "max_samples", "random_state", "class_weight")}

    cv = pd.read_csv(ML / "cv_model_comparison.csv")
    cvt = pd.DataFrame({"Model": cv.model.map(NICE), "CV macro F1": cv.oof_macro_f1,
                        "CV F1 (pos.)": cv.oof_f1, "CV precision": cv.oof_precision,
                        "CV recall": cv.oof_recall, "CV log loss": cv.oof_log_loss,
                        "CV top-1 (pseudo)": cv.oof_pseudo_top1_accuracy,
                        "fold F1 sd": cv.fold_f1_std, "fit s (5 folds)": cv.fit_s_total})
    oh = pd.read_csv(ML / "oof_hmm_comparison.csv")
    oht = pd.DataFrame({"Emission": oh.model.map(lambda m: NICE.get(m, m)), "route F1 mean": oh.arc_f1_mean,
                        "median": oh.arc_f1_median, "precision": oh.arc_precision_mean,
                        "recall": oh.arc_recall_mean, "continuity": oh.route_continuity_mean})
    st = pd.read_csv(ML / "student_test_model_report.csv")
    stt = pd.DataFrame({
        "Model": st.model.map(lambda m: NICE.get(m.split("__")[0], m) + (" [80 trips]" if "train80" in m else
                                                                         (" [64 trips]" if "__" in m else ""))),
        "Top-1 (pseudo)": st.clf_pseudo_top1_accuracy, "Macro F1 (cand.)": st.clf_macro_f1,
        "Log loss": st.clf_log_loss, "+HMM route F1": st.arc_f1_mean,
        "+HMM continuity": st.route_continuity_mean})
    comp = pd.read_csv(FE / "comparison_with_previous_splits.csv")
    compt = comp[["split", "model", "classifier_training_trips", "n_trips", "arc_f1_mean", "arc_f1_median",
                  "arc_precision_mean", "arc_recall_mean"]]
    ev = pd.read_csv(FE / "evaluation_per_trip.csv")
    dep = ev[ev.variant == "rf80_hmm_deployed"].copy()
    geo = ev[ev.variant == "geometric_hmm_tuned_v2"].set_index("trip_id")
    dep["geometric F1"] = dep.trip_id.map(geo.arc_f1)
    pert = dep[["trip_id", "n_gps_points", "n_arcs", "pred_route_len", "pred_distinct_arcs", "arc_f1",
                "arc_precision", "arc_recall", "seq_f1", "geometric F1", "path_continuity",
                "points_without_candidates", "runtime_ms_server"]].copy()
    pert["trip_id"] = pert.trip_id.map(lambda t: str(t).zfill(8))
    vs = []
    for v, d in summ["variants"].items():
        f = d.get("arc_f1") or {}
        vs.append({"variant": v, "mean": f.get("mean"), "median": f.get("median"), "min": f.get("min"),
                   "max": f.get("max"), "sd": f.get("std"),
                   "precision": (d.get("arc_precision") or {}).get("mean"),
                   "recall": (d.get("arc_recall") or {}).get("mean"),
                   "seq F1": (d.get("seq_f1") or {}).get("mean"),
                   "runtime ms (mean)": (d.get("runtime_ms_server") or {}).get("mean")})
    vst = pd.DataFrame(vs)
    err = pd.read_csv(FE / "route_error_analysis.csv")
    errt = err[err.variant == "rf80_hmm_deployed"][["fp_reverse_direction", "fp_outside_ref_span",
                                                    "fp_gap_fill_only", "fp_matched", "n_fn",
                                                    "repeated_traversals"]].sum()
    errg = err[err.variant == "geometric_hmm_tuned_v2"][errt.index].sum()
    errtab = pd.DataFrame({"category": errt.index, "geometric HMM": errg.values.astype(int),
                           "RF + HMM (deployed)": errt.values.astype(int)})
    dec = pd.read_csv(EXPERIMENT_DIR / "optuna_trials.csv")
    dbest = dec.loc[dec.value.idxmax()]
    dparams = {k.replace("user_attrs_cfg_", ""): dbest[k] for k in dec.columns if k.startswith("user_attrs_cfg_")}
    rft = pd.read_csv(ML / "rf_optuna_trials_partial.csv")
    rfc = rft[rft.state == "COMPLETE"]
    rbest = rfc.loc[rfc.value.idxmin()]
    rparams = {k.replace("params_", ""): rbest[k] for k in rft.columns if k.startswith("params_")}
    dvg = summ["deployed_vs_geometric_per_trip"]
    lab = pd.read_csv(EXPERIMENT_DIR / "final" / "label_verification.csv")
    sm = summ["model_check"]
    FIN = EXPERIMENT_DIR / "final"
    pf = json.load(open(FE / "post_freeze_checks" / "post_freeze_check.json"))
    ab = pd.read_csv(FIN / "ablation_summary.csv")
    abt = ab[ab.eval_set.isin(["holdout16", "student_test10"])][
        ["eval_set", "variant", "model", "n_trips", "arc_f1", "arc_f1_median", "arc_precision", "arc_recall",
         "seq_f1", "route_continuity", "pseudo_top1_accuracy"]]
    tvc = pd.read_csv(FIN / "train_vs_cv_metrics.csv")[["model", "train_macro_f1", "cv_macro_f1",
                                                         "train_log_loss", "cv_log_loss"]]
    tvc["model"] = tvc.model.map(NICE)
    cmx = pd.read_csv(FIN / "confusion_matrices_oof.csv")[["model", "tn", "fp", "fn", "tp"]]
    cmx["model"] = cmx.model.map(NICE)
    ver = pd.read_csv(FIN / "figures" / "plotted_values_verification.csv")
    vtab = ver.groupby("figure")["match"].agg(["sum", "count"]).reset_index()
    vtab.columns = ["figure (experiments/final/figures/)", "values matching source", "values plotted"]

    text = f"""# PathSnap — report input (final evaluation handoff)

Generated by `python -m scripts.make_report_input` from recorded files only. Evaluation run:
{man['evaluation_started']} → {summ['evaluation_finished']}; git commit `{man['git_commit']}` (inference code and
deployed artifacts byte-identical to this commit; later changes are evaluation/test-only).

**Post-freeze software setup change (not a model change):** commit `0fcc9b8` changed only
`src/data_loader.py::list_trips` so `GET /trips` also discovers trip folders when `PATHSNAP_DATA_DIR` points
directly at a split folder. SHA-256 {pf['src/data_loader.py_sha256_frozen'][:16]}… (frozen) →
{pf['src/data_loader.py_sha256_now'][:16]}… (now). Re-running all 10 evaluator trips in both data-dir modes gave
identical routes ({pf['all_routes_identical']}) and identical F1 ({pf['all_f1_identical']}), {pf['n_checks']} checks;
response keys unchanged ([post_freeze_check.json](post_freeze_checks/post_freeze_check.json)). The frozen evaluator
results below are the original, unmodified records.

## 1. Final deployed model
- **Config `ml_v3`** ([configs/ml_v3.json](../../configs/ml_v3.json)): HMM/Viterbi with Random-Forest emissions.
```json
{json.dumps(cfg, indent=1, default=_np)}
```
- **Classifier** `models/{cfg['ml_model']}.joblib`: RandomForestClassifier {json.dumps(rf_keep, default=_np)},
  trained on valid-label candidate rows of all **80 train trips**; 20 features (schema v{bundle['schema_version']},
  order verified identical to training: {sm['feature_order_matches_training']}). No probability calibration
  (raw `predict_proba`, clipped at 1e-4, per-point normalised, log, × weight 4).
- **Fallback / rollback**: `tuned_v2` (same decoder, geometric Gaussian emissions), `baseline_v1` (original).
- SHA-256 of model/configs/source at evaluation time: [evaluation_manifest.json](evaluation_manifest.json).

## 2. Data splits
| partition | trips | use |
|---|---|---|
| train → tune | {len(part['tune'])} | classifier CV (GroupKFold 5 by trip), decoder Optuna, emission-weight grid |
| train → holdout | {len(part['holdout'])} | untouched until model/decoder selection was frozen (seed {part['seed']}, `experiments/train_partition.json`) |
| student_test | 10 | development comparison after freezing; never trained on |
| evaluator_test | 10 | final evaluation only (this run); IDs disjoint from all other splits |

Training dataset: 1,589,846 candidate rows from 208,417 GPS points (80 trips); positives = candidates equal to the
pseudo-label. Pseudo-labels: monotone alignment of GPS points to `.route` positions (distance + heading + skip
penalty), labels > 30 m from their arc marked invalid. Verified on all 80 train trips
([label_verification.csv](../final/label_verification.csv)): positions non-decreasing
{int(lab.positions_non_decreasing.sum())}/80, label = route arc at its position
{int(lab.labels_equal_route_at_position.sum())}/80, label sequence an ordered subsequence of `.route`
{int(lab.label_sequence_is_ordered_subsequence_of_route.sum())}/80.

## 3. Cross-validation setup and five-model comparison
GroupKFold(5) by trip over the 64 tune trips (`experiments/ml/cv_folds.json`); fold disjointness asserted; metrics on
valid-label rows; threshold 0.5 for P/R/F1. **Pseudo-label metrics**, not route accuracy.
Source: `experiments/ml/cv_model_comparison.csv`.

{md(cvt)}

Out-of-fold classifier + HMM (route-level vs `.route`, emission weight 1, 64 tune trips;
`experiments/ml/oof_hmm_comparison.csv`):

{md(oht)}

Student test (10 trips; models trained on 64 trips unless marked; reported after freezing;
`experiments/ml/student_test_model_report.csv`):

{md(stt)}

Overfitting evidence (saved 64-trip model scored on its own training rows vs out-of-fold CV;
`experiments/final/train_vs_cv_metrics.csv`, figure `fig_train_vs_cv.png`). Gaps are small (≤ 0.02 macro F1);
Logistic Regression shows almost no gap but the lowest ceiling (mild underfitting).

{md(tvc)}

Confusion matrices (out-of-fold, threshold 0.5, valid-label rows of the 64 tune trips;
`experiments/final/confusion_matrices_oof.csv`, figure `fig_confusion_matrices.png`):

{md(cmx)}

### Equal-data ablation (classifier-only vs geometric HMM vs classifier + HMM)
Fixed candidates (`tuned_v2`), decoder, emission weight 4 and route assembly; every classifier trained on the same 64
tune trips; no evaluator data; nothing retrained or tuned (`python -m scripts.ablation`;
`experiments/final/ablation_summary.csv`, per trip `ablation_per_trip.csv`). Key reading: the HMM is the dominant
component (classifier-only → +HMM: +0.17 to +0.27 route F1 on holdout/student test); ML emissions add a small gain over geometric emissions
for tree models (+0.003 to +0.006 on holdout/student test), none for Logistic Regression (−0.000 holdout, −0.001 student test). RF and HistGB tie on holdout;
RF was selected earlier by the pre-registered OOF route-F1 rule (0.9492 vs 0.9472).

{md(abt)}

### Figures (all generated from recorded results, every value independently re-verified)
`python -m scripts.make_figures` then `python -m scripts.verify_figures`
(`experiments/final/figures/plotted_values_verification.csv`):

{md(vtab)}

## 4. Bayesian hyperparameter optimisation
**Decoder (completed).** Optuna TPESampler(seed=42, multivariate, 20 startup), {len(dec)} trials, objective = mean
route arc F1 on the 64 tune trips; trial 0 = original config. Best trial #{int(dbest.number)}: {dbest.value:.4f}
(trial 0: {dec.loc[dec.number == 0, 'value'].iloc[0]:.4f}). Best params: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in dparams.items()}, default=_np)}.
Record: `experiments/optuna_trials.csv`, `experiments/optuna_study.db`.
Emission weight: grid {{0.25,0.5,1,2,4,8}} × {{ml, hybrid}} on OOF tune trips (`experiments/ml/emission_grid_random_forest.csv`); not Bayesian.

**Random Forest (INCOMPLETE — stopped on request).** TPESampler(seed=42), MedianPruner, objective = mean 5-fold
log loss on the 64 tune trips. {len(rft)} trials started: {int((rft.state == 'COMPLETE').sum())} complete,
{int((rft.state == 'PRUNED').sum())} pruned, {int((rft.state == 'RUNNING').sum())} interrupted. Best complete trial
#{int(rbest.number)}: {rbest.value:.5f} vs untuned trial 0: {rft.loc[rft.number == 0, 'value'].iloc[0]:.5f}
(difference {rft.loc[rft.number == 0, 'value'].iloc[0] - rbest.value:.5f}). Params {json.dumps(rparams, default=_np)}.
The tuned RF was **not** retrained, evaluated downstream, or deployed; the deployed RF uses the initial parameters.
Consequently **no untuned-vs-tuned comparison of the classifier exists** (guideline §4 asks for one on CV macro F1,
student-test Top-1/macro F1, log loss and inference time). The decoder does have a tuned-vs-untuned comparison:
baseline_v1 vs tuned_v2, holdout F1 0.9181 → 0.9477, student test 0.9155 → 0.9683
(`experiments/compare_baseline_v1_vs_tuned_*.csv`).
Record: `experiments/ml/rf_optuna_trials_partial.csv`, `experiments/ml/rf_optuna_study.db`. Pruned-trial values are
partial-fold means and are not comparable to completed trials.

## 5. RF + HMM vs geometric HMM across splits (different training sizes labelled)
Source: [comparison_with_previous_splits.csv](comparison_with_previous_splits.csv).

{md(compt)}

## 6. Final evaluation on evaluator_test (frozen model)
Reference `.route` files were present for all 10 trips and were read only after all predictions (scoring is
permitted: the guidelines state the API computes accuracy from internal reference data). No model, config or
threshold was changed after seeing these results.

{md(vst)}

Deployed vs geometric per trip: {dvg['better']} better, {dvg['worse']} worse, {dvg['same']} same; mean Δ
{dvg['mean_delta']:+.4f}; Wilcoxon p = {dvg['wilcoxon_p']} → **not statistically significant on 10 trips.**

Per trip (deployed RF + HMM; [evaluation_per_trip.csv](evaluation_per_trip.csv)):

{md(pert)}

Error analysis, summed over 10 trips ([route_error_analysis.csv](route_error_analysis.csv)):

{md(errtab)}

Integrity: all {summ['n_trips']} trips predicted without failure; deterministic across repeated calls; every predicted
arc id exists and consecutive arcs form a connected directed path (continuity 1.0); API metrics equal the
independent recomputation; predictions identical with `.route` removed
({summ['leakage_check_predictions_identical_without_route']}); no evaluator trip in the model's training set.
Plots: [route_f1_comparison.png](route_f1_comparison.png), [runtime_distribution.png](runtime_distribution.png);
maps: [maps/](maps/) (lowest F1, largest gain, highest F1; HTML + PNG).

## 7. Metric definitions and limitations
- **arc precision/recall/F1** (primary; `src/evaluation.py::route_metrics`): computed on the **sets of distinct
  directed arc ids** of the predicted route and `.route`. Duplicates are collapsed (repeated traversals are not
  penalised), order is ignored, direction matters (reverse twin = different arc), shortest-path gap-fill arcs count
  as predictions. No official formula is given in the PathSnap documents; this is our definition.
- **seq precision/recall/F1** (supplementary; `sequence_metrics`): longest common subsequence between predicted and
  reference arc sequences — order- and duplicate-aware.
- **length precision/recall**: set metrics weighted by arc length.
- **route continuity**: fraction of consecutive predicted arcs with to(a) = from(b).
- **Top-1 / macro F1 / log loss of classifiers**: against **derived pseudo-labels**, not ground truth.
- Limitations: `.route` is a reference, not verified truth; in some train trips it ends before the GPS does
  (trips 33, 54), inflating false positives there; small test sets (10 trips) give wide uncertainty.

## 8. Bugs, uncertainties, remaining risks
1. **Data-dir layout — FIXED post-freeze** (commit `0fcc9b8`, setup only; predictions verified identical): `GET /trips`
   previously returned 0 trips when `PATHSNAP_DATA_DIR` pointed directly at a split folder. Both layouts now work.
2. Empty `.track` file returns HTTP 404 "Empty file" (clean error, though 422 would be more precise).
3. The ML gain on evaluator_test is not significant (p = {dvg['wilcoxon_p']}); trips 15, 67, 87 are slightly worse than geometric.
4. Gap-fill arcs are the largest false-positive source; neither decoder addresses them.
5. Decoder params and emission weight were chosen on the same 64 trips used for classifier CV → tune-set numbers optimistic.
6. RF Bayesian optimisation incomplete (see §4).
7. Runtime: deployed mean {summ['variants']['rf80_hmm_deployed']['runtime_ms_server']['mean']} ms vs geometric
   {summ['variants']['geometric_hmm_tuned_v2']['runtime_ms_server']['mean']} ms per trip (Apple M5); model file 47 MB.
8. Folium maps need internet for map tiles.
9. Evaluator-trip route maps (`experiments/final_evaluation/maps/`) draw ground-truth geometry and are deliberately
   NOT committed (git-ignored); use them in the PDF only if the organisers permit showing evaluator reference routes.

## 9. Artifact paths and reproduction
```bash
python -m pytest -q                                   # automated tests
python -m scripts.train_classifiers --stage dataset   # ... cv, oof_hmm, emission_grid, fit_tune, final, student_test
python -m scripts.tune                                # decoder Optuna
python -m scripts.tune_classifier                     # RF Optuna (was stopped early)
python -m scripts.verify_labels
python -m scripts.final_evaluation                    # this evaluation (data root = repo root)
python -m scripts.final_eval_artifacts
python -m scripts.verify_post_freeze                  # post-freeze regression check (writes post_freeze_checks/)
python -m scripts.ablation                            # equal-data ablation (no training)
python -m scripts.make_figures && python -m scripts.verify_figures
python -m scripts.make_report_input
PATHSNAP_DATA_DIR=/path/to/evaluator_test uvicorn main:app --host 0.0.0.0 --port 8000   # demo (parent dir also works)
```
Environment: {man['python']} on {man['platform']}; exact packages in `experiments/final/pip_freeze.txt`.
"""
    (FE / "REPORT_INPUT.md").write_text(text)
    print("wrote REPORT_INPUT.md")


if __name__ == "__main__":
    main()
