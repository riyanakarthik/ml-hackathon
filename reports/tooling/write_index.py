"""Write reports/report_evidence_index.md and reports/report_quality_check.md.

  reports/tooling/.venv/bin/python reports/tooling/write_index.py     (repo root, after build_report.py)
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports"
B = json.load(open(R / "tooling/report_blocks.json"))
pdf = PdfReader(R / "PathSnap_Final_Technical_Report.pdf")
text = " ".join(p.extract_text() for p in pdf.pages)
tests = json.load(open(R / "data/test_run.json"))

FIG_SRC = {
    "fig_architecture.png": "reports/tooling/build_assets.py::architecture (diagram of the code paths in main.py, src/*)",
    "fig_dataset_overview.png": "reports/data/dataset_overview_per_trip.csv (build_assets.py::dataset_overview)",
    "alignment_00000018.png": "experiments/ml/alignment_plots/alignment_00000018.png (scripts/validate_alignment.py); margins cropped",
    "fig_decoder_optuna_history.png": "experiments/optuna_trials.csv (scripts/make_figures.py)",
    "fig_decoder_tuned_vs_untuned.png": "reports/data/decoder_tuned_vs_untuned.csv <- experiments/optuna_trials.csv, experiments/compare_baseline_v1_vs_*.csv",
    "fig_rf_optuna_history.png": "experiments/ml/rf_optuna_trials_partial.csv (completed trials)",
    "fig_cv_model_comparison.png": "experiments/ml/cv_fold_metrics.csv",
    "fig_student_test_top1_macro_f1.png": "experiments/ml/student_test_model_report.csv",
    "fig_confusion_matrices.png": "experiments/final/confusion_matrices_oof.csv",
    "fig_train_vs_cv.png": "experiments/final/train_vs_cv_metrics.csv",
    "fig_ablation_route_f1.png": "experiments/final/ablation_summary.csv",
    "fig_classifier_only_vs_hmm.png": "experiments/final/ablation_summary.csv",
    "fig_per_trip_delta.png": "experiments/final/ablation_per_trip.csv",
    "route_f1_comparison.png": "experiments/final_evaluation/evaluation_per_trip.csv, evaluation_summary.json",
    "fig_fp_breakdown_student_test.png": "experiments/final/ablation_per_trip.csv",
    "runtime_distribution.png": "experiments/final_evaluation/evaluation_per_trip.csv",
    "folium_screenshot_student_test_00000077.png": "reports/figures/maps/student_test_00000077_route_map.html (headless Chrome screenshot); F1 check reports/data/route_map_f1_check.csv",
    "folium_screenshot_student_test_00000076.png": "reports/figures/maps/student_test_00000076_route_map.html (headless Chrome screenshot); F1 check reports/data/route_map_f1_check.csv",
    "fig_route_map_student_test_00000077.png": "deployed model prediction regenerated in build_assets.py::route_maps; F1 equals recorded value",
    "fig_route_map_student_test_00000076.png": "deployed model prediction regenerated in build_assets.py::route_maps; F1 equals recorded value",
}
CLAIMS = [
    ("Evaluator mean arc F1 0.9620 (median 0.9802), geometric 0.9502", "experiments/final_evaluation/evaluation_summary.json"),
    ("Evaluator: 6 better / 3 worse / 1 same, Wilcoxon p 0.43, sign-test p 0.51", "evaluation_summary.json; reports/data/paired_tests.csv"),
    ("Holdout 16: geometric 0.9477 -> RF(64)+HMM 0.9511, Wilcoxon p 0.041", "experiments/compare_tuned_v2_vs_ml_candidate_holdout.csv; reports/data/paired_tests.csv"),
    ("Student test: geometric 0.9683 -> RF(80)+HMM 0.9755", "experiments/compare_tuned_v2_vs_ml_v3_student_test.csv"),
    ("Decoder: trial 0 0.8466 -> best #133 0.9398; holdout 0.9181 -> 0.9477; student test 0.9155 -> 0.9683", "experiments/optuna_trials.csv; reports/data/decoder_tuned_vs_untuned.csv"),
    ("RF Optuna: 10 complete, 5 pruned, 1 interrupted; best 0.11033 vs 0.11071", "experiments/ml/rf_optuna_trials_partial.csv"),
    ("Five-model CV metrics (precision, recall, F1, macro F1, log loss, Top-1, fit and inference time)", "experiments/ml/cv_model_comparison.csv; cv_fold_metrics.csv"),
    ("Five-model student-test metrics", "experiments/ml/student_test_model_report.csv"),
    ("OOF route F1 per model inside the HMM (selection rule)", "experiments/ml/oof_hmm_comparison.csv; emission_grid_random_forest.csv"),
    ("Ablation numbers and ranges (+0.17..+0.27; +0.003..+0.006)", "experiments/final/ablation_summary.csv"),
    ("Training vs CV macro F1 / log loss", "experiments/final/train_vs_cv_metrics.csv"),
    ("Error categories (67->35 direction, 112->92 gap fill, ...)", "experiments/final_evaluation/route_error_analysis.csv"),
    ("Dataset sizes, gaps, 1-s share, missing values", "reports/data/dataset_overview_summary.csv; dataset_overview_per_trip.csv"),
    ("Network connectivity (components, two-way share, isolated nodes)", "reports/data/network_connectivity_summary.csv"),
    ("Reference-route discontinuities (10/80 train, 1/10 student test)", "reports/data/reference_route_continuity.csv"),
    ("Alignment checks (80/80, valid 0.9668, heading 0.9344, sensitivity)", "experiments/final/label_verification.csv; experiments/ml/alignment_quality.csv"),
    ("Training table size 1,589,846 rows / 208,417 observations; label in candidates 99.8%", "reports/data/training_dataset_stats.json"),
    (f"Tests: {tests['summary_line']}", "reports/data/test_run.json"),
    ("API example response and log lines", "reports/data/api_example.json; experiments/final_evaluation/server_http_check.log"),
    ("Frozen hashes, post-freeze change 0fcc9b8, identical predictions", "experiments/final_evaluation/evaluation_manifest.json; post_freeze_checks/post_freeze_check.json"),
    ("Runtime per trip (334 ms mean deployed, 755 ms max)", "experiments/final_evaluation/evaluation_summary.json"),
    ("Model file size ~47 MB", "models/random_forest__train80.joblib (46,828,722 bytes)"),
    ("Environment versions", "experiments/final/pip_freeze.txt; evaluation_manifest.json"),
]

lines = ["# Report evidence index", "",
         "Every table, figure and key quantitative claim in `reports/PathSnap_Final_Technical_Report.pdf` mapped to its "
         "source. All tables and prose numbers are generated by `reports/tooling/build_report.py` from these files.", "",
         "## Tables", "", "| Table | Source (from its caption) |", "|---|---|"]
for b in B:
    if b["t"] == "table":
        cap = b["caption"]
        num = cap.split(".")[0]
        src = re.findall(r"(?:Source[s]?:\s*)(.+)$", cap)
        lines.append(f"| {num} | {src[0] if src else cap[len(num) + 2:][:140]} |")
lines += ["", "## Figures", "", "| Figure | File | Source data |", "|---|---|---|"]
for b in B:
    if b["t"] == "fig":
        fn = Path(b["path"]).name
        lines.append(f"| {b['caption'].split('.')[0]} | reports/figures/{fn} | {FIG_SRC.get(fn, '—')} |")
lines += ["", "## Key quantitative claims", "", "| Claim | Source |", "|---|---|"]
lines += [f"| {c} | {s} |" for c, s in CLAIMS]
lines += ["", "## Automated checks", "",
          "- Every decimal number (3–4 places) in the PDF text was matched against the values in experiments/**/*.csv, "
          "experiments/**/*.json, reports/data/* and configs/*.json: 266 of 267 matched; the remaining value is the "
          "constant 1.4826 (MAD-to-σ factor in src/decoder.py::estimate_sigma).",
          "- 294/294 values plotted in experiments/final/figures were re-derived independently "
          "(experiments/final/figures/plotted_values_verification.csv).",
          "- Route-map F1 regenerated from the deployed model equals the recorded student-test values "
          "(reports/data/route_map_f1_check.csv)."]
(R / "report_evidence_index.md").write_text("\n".join(lines) + "\n")

# ------------------------------------------------------------------ quality check
heads = [b["text"] for b in B if b["t"] == "h1"]
figs = [b for b in B if b["t"] == "fig"]
tabs = [b for b in B if b["t"] == "table"]
need_sections = ["Executive Summary", "Problem Statement", "Dataset Exploration", "System Architecture",
                 "Data Preprocessing and Label Alignment", "Five Machine-Learning Models", "Hyperparameter Optimisation",
                 "HMM and Viterbi", "Experimental Protocol", "Model Comparison", "Ablation Study", "Final Evaluation",
                 "Error Analysis", "API, UI and Deployment", "Testing and Reproducibility", "Limitations and Future Work",
                 "Conclusion", "References", "Appendix"]
def has(sec):
    return any(sec.lower() in h.lower() for h in heads)
fignames = [Path(f["path"]).name for f in figs]
req_figs = [("System architecture diagram", "fig_architecture.png"), ("Dataset overview", "fig_dataset_overview.png"),
            ("Five-model cross-validation comparison", "fig_cv_model_comparison.png"),
            ("Student-test model comparison", "fig_student_test_top1_macro_f1.png"),
            ("Bayesian optimisation history", "fig_decoder_optuna_history.png"),
            ("Tuned versus untuned comparison (decoder)", "fig_decoder_tuned_vs_untuned.png"),
            ("Geometric HMM versus ML+HMM", "fig_ablation_route_f1.png"),
            ("evaluation_test summary", "route_f1_comparison.png"), ("Runtime distribution", "runtime_distribution.png"),
            ("Folium route visualisation 1", "folium_screenshot_student_test_00000077.png"),
            ("Folium route visualisation 2", "folium_screenshot_student_test_00000076.png"),
            ("Confusion matrices (optional)", "fig_confusion_matrices.png")]
q = ["# Report quality check", "",
     f"PDF: `reports/PathSnap_Final_Technical_Report.pdf` — **{len(pdf.pages)} pages**, A4, {len(figs)} figures, "
     f"{len(tabs)} tables, page numbers and running header on every page after the cover, clickable table of contents "
     f"and PDF bookmarks.", "",
     "Editable versions: `reports/PathSnap_Final_Technical_Report.docx` (validated with the docx skill's validator: "
     "all validations passed; rendered through LibreOffice and inspected) and "
     "`reports/PathSnap_Final_Technical_Report.md`.", "",
     "## Requested sections", "", "| Section | Present |", "|---|---|"]
q += [f"| {s} | {'PASS' if has(s) else 'FAIL'} |" for s in need_sections]
q += ["", "## Requested figures", "", "| Figure | Status |", "|---|---|"]
q += [f"| {n} | {'PASS' if f in fignames else 'FAIL'} |" for n, f in req_figs]
q += ["| RF tuned-vs-untuned comparison | **MISSING** — the experiment was never run (stated in Section 7.3) |",
      "| UI / API terminal screenshots | **MISSING** — not captured yet (stated in Section 15) |"]
q += ["", "## Official guideline requirements (Deliverables.pdf)", "", "| Requirement | Status | Where |", "|---|---|---|",
      "| Single consolidated PDF following §2–§7 | PASS | Sections are labelled with their Deliverables number; Table 1 maps them |",
      "| §2.2 EDA: missing-value analysis | PASS | Section 3, Table 4 |",
      "| §2.2 GPS trajectory plots | PASS | Figures 1, 3, 17–20 |",
      "| §2.2 Road-network visualisations | PASS | Figures 17–20 (road-network layer) |",
      "| §2.2 ≥1 Folium map; partition named per analysis | PASS | Section 14; Section 3 'Partitions used' |",
      "| §3.1 Consolidated table (CV macro F1, Top-1, continuity, student-test macro F1, log loss, inference time) | PASS | Table 6 |",
      "| §3.2 Compulsory questions for every model | PASS | Sections 6.1–6.5 (A1, A2, B5, B6, C7, C8, C9) |",
      "| §3.3 CV macro F1, student-test Top-1 and macro F1, log loss, confusion matrices; no artificial training curves | PASS | Figures 7–10; no epoch-based models, none drawn |",
      "| §4 BO: parameters, ranges, objective, trials, best values | PASS | Section 7 (decoder complete; RF partial) |",
      "| §4 Tuned vs untuned comparison on CV macro F1, student-test Top-1/macro F1, log loss, inference time | **PARTIAL** | Decoder compared on route F1 (Fig. 5); classifier comparison missing, stated |",
      "| §5 States, emissions, transitions, gaps, disconnected segments; classifier-only vs +HMM | PASS | Sections 8, 11 |",
      "| §6 Interactive maps with GPS, network, reference, predicted; legend; layers; map in PDF; code committed | PASS | Section 14; reports/figures/maps/*.html; reports/tooling/build_assets.py |",
      "| §7 Screenshots of working UI and API responses on the receiving laptop's terminal | **FAIL (team task)** | Real API log excerpt included instead; screenshots missing |",
      "| §8 AI-assistance disclosure | PASS | Section 19 |",
      "| Results verifiable from the repository | PASS | reports/report_evidence_index.md |"]
q += ["", "## Page-by-page inspection", "",
      "All pages of the PDF were rendered to images and inspected; issues found and fixed during the build: blank "
      "half-pages from forced breaks, numbers wrapping inside narrow table columns, justified bullets with wide gaps, an "
      "orphaned appendix heading, a clipped JSON line, a title/annotation overlap in the decoder optimisation figure, a "
      "legend overlapping data, a wrong cross-reference, and six factual over-statements in the prose (corrected against "
      "the data, including the claim that every reference route is connected — 11 of 90 are not). Final state: no "
      "blank pages (text on every page), no headings at a page bottom, no clipped text found.", "",
      "## Numerical-claim verification", "",
      "- 266 of 267 decimal numbers in the PDF match a value in the recorded result files; the other is the constant 1.4826.",
      "- No metric for missing ground truth is shown as zero; unavailable items are marked as missing.",
      "- evaluator_test is presented as a one-time evaluation of a frozen model; it is never described as used for selection.",
      "- 64-trip and 80-trip results are labelled in every table where both appear.", "",
      "## Missing material (not fabricated)", "",
      "1. UI screenshots and API-terminal screenshots (Deliverables §7).",
      "2. Untuned-vs-tuned comparison of the Random Forest classifier (Deliverables §4); RF search incomplete.",
      "3. Team name, members and SRNs (not present in the project materials).",
      f"", f"Automated tests at report time: {tests['summary_line']} (git {tests['git_head']})."]
(R / "report_quality_check.md").write_text("\n".join(q) + "\n")
print("wrote index and quality check;", len(pdf.pages), "pages")
