"""Write reports/compliance_checklist.md, reports/report_evidence_index.md and reports/report_quality_check.md.

  reports/tooling/.venv/bin/python reports/tooling/write_index.py     (repo root, after build_report.py)
Page numbers are found by searching the final PDF's text, not typed in.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports"
B = json.load(open(R / "tooling/report_blocks.json"))
pdf = PdfReader(R / "PathSnap_Final_Technical_Report.pdf")
pages = [re.sub(r"\s+", " ", p.extract_text() or "") for p in pdf.pages]
tests = json.load(open(R / "data/test_run.json"))
NP = len(pages)


def pg(needle: str) -> str:
    """First page (1-based) whose text contains needle; skips the contents page."""
    n = re.sub(r"\s+", " ", needle)
    for i, t in enumerate(pages):
        if i < 2:
            continue
        if n in t:
            return f"p. {i + 1}"
    raise KeyError(needle)


def cap(prefix: str) -> str:
    return pg(prefix)


# ------------------------------------------------------------------ compliance checklist
C = [  # (requirement from Deliverables.pdf, status, where (section, page lookup), evidence)
    ("§1 Single consolidated PDF following the prescribed structure", "PASS",
     f"whole report, {NP} pages", "reports/PathSnap_Final_Technical_Report.pdf"),
    ("§1 Actual results, tables, graphs and geographical visualisations; results verifiable from the repository", "PASS",
     "all sections; App. A", "reports/report_evidence_index.md; 269 of 270 decimals match recorded values (the other is the constant 1.4826)"),
    ("§2.2 Q1 How the data was preprocessed", "PASS", f"§3, §5 ({pg('5 Preprocessing and Pseudo-label Alignment')})",
     "scripts/eda.py, src/data_loader.py, src/matcher.py, src/labels.py"),
    ("§2.2 Q2 Patterns observed; features extracted and why", "PASS", f"§3 ({pg('Sampling and gaps.')}), §5.3 ({pg('5.3 Candidate features')})",
     "src/features.py"),
    ("§2.2 Missing-value analysis", "PASS", f"§3, Table 2 ({cap('Table 2.')})", "scripts/eda.py -> reports/data/eda_summary.json; reports/data/dataset_overview_summary.csv"),
    ("§2.2 GPS trajectory plots", "PASS", f"Fig. 3 ({cap('Figure 3.')}), Fig. 15 ({cap('Figure 15.')})", "—"),
    ("§2.2 Road-network visualisations", "PASS", f"Fig. 15 road-network layer ({cap('Figure 15.')}); network statistics in §3",
     "reports/figures/maps/*.html"),
    ("§2.2 At least one Folium map", "PASS", f"§14, Fig. 15 ({cap('Figure 15.')})", "reports/figures/maps/student_test_*_route_map.html"),
    ("§2.2 Partition named for each analysis", "PASS", f"§3 'Partitions' ({pg('Partitions: the table and figure')})", "—"),
    ("§3 At least five models trained and evaluated; every model documented", "PASS", f"§6 ({pg('6 Candidate Classifiers')})",
     "experiments/ml/cv_model_comparison.csv"),
    ("§3.1 Consolidated table: CV macro F1, Top-1, route continuity, student-test macro F1, log loss, inference time", "PASS",
     f"Table 4 ({cap('Table 4.')})", "N/A columns: none needed"),
    ("§3.2 A1–A2, B5–B6, C7–C9 answered for every model", "PASS",
     f"§6.1–6.5 ({pg('6.1 Logistic Regression')}–{pg('6.5 HistGradientBoosting')})", "src/ml_train.py::make_models"),
    ("§3.3 CV macro F1 comparison across all models", "PASS", f"Fig. 7 ({cap('Figure 7.')})", "experiments/ml/cv_fold_metrics.csv"),
    ("§3.3 Student-test Top-1 accuracy and macro F1 comparison", "PASS", f"Fig. 8 ({cap('Figure 8.')}), Table 10", "experiments/ml/student_test_model_report.csv"),
    ("§3.3 Log loss comparison", "PASS", f"Fig. 7 right panel ({cap('Figure 7.')}), Tables 4, 9, 10", "—"),
    ("§3.3 Confusion matrices (optional)", "PASS", f"Fig. 9 ({cap('Figure 9.')})", "experiments/final/confusion_matrices_oof.csv"),
    ("§3.3 No artificial training curves; loss plots only for epoch-based models", "PASS",
     "none drawn (no epoch-based models); Fig. 10 shows train-vs-CV evidence instead", "—"),
    ("§3.3 Plots justifying model selection", "PASS", f"Fig. 11 ({cap('Figure 11.')}), §10.1 ({pg('10.1 Why Random Forest + HMM')})", "—"),
    ("§4 Q1–Q5 Hyperparameters searched, ranges, objective, number of trials, best values", "PASS",
     f"§7.1 Table 5 ({cap('Table 5.')}), §7.3 ({pg('7.3 Random Forest classifier (incomplete)')})",
     "experiments/optuna_trials.csv; experiments/ml/rf_optuna_trials_partial.csv"),
    ("§4 Q6 Did the tuned model outperform the untuned one?", "PARTIAL",
     f"decoder: yes, Fig. 5 ({cap('Figure 5.')}); classifier: not evaluated (§7.3)", "—"),
    ("§4 Untuned vs tuned comparison of the chosen model on CV macro F1, student-test Top-1 / macro F1, log loss, inference time",
     "**FAIL**", f"§7.3 note ({pg('Not done:')})",
     "The tuned Random Forest was never refitted or evaluated; the decoder comparison is on route F1, not these metrics"),
    ("§5 Q1–Q5 Hidden states, emissions, transitions, missing candidates and gaps, disconnected segments", "PASS",
     f"§8 ({pg('8 HMM and Viterbi Route Reconstruction')})", "src/decoder.py"),
    ("§5 Q6 Did HMM decoding improve the route?", "PASS", f"§11 ({pg('11 Ablation: Classifier-only vs Classifier + HMM')})",
     "experiments/final/ablation_summary.csv"),
    ("§5 Classifier-only vs classifier + HMM comparison; cases where HMM did not help", "PASS",
     f"Table 11, Figs. 11–12 ({cap('Figure 12.')})", "experiments/final/ablation_per_trip.csv"),
    ("§6 Interactive maps from the actual implementation: GPS, road network, reference route, predicted route", "PASS",
     f"§14, Fig. 15 ({cap('Figure 15.')})", "reports/figures/maps/*.html"),
    ("§6 Distinguishable colours/markers, legend, sensible extent, easy comparison, layer controls", "PASS",
     f"Fig. 15 ({cap('Figure 15.')})", "legend box and LayerControl in the HTML maps"),
    ("§6 Map in the PDF as a screenshot; HTML or generation code committed", "PASS (after commit)",
     f"Fig. 15 ({cap('Figure 15.')})", "reports/tooling/build_assets.py; reports/figures/maps/ (to be committed)"),
    ("§7 Screenshots of the working UI", "PASS", f"Fig. 16 ({cap('Figure 16.')})", "reports/figures/screenshots/ui_dashboard_trip_00000091.png"),
    ("§7 Screenshots of successful API responses on the receiving laptop's terminal", "PASS (single-machine capture)",
     f"Fig. 17 ({cap('Figure 17.')}), Fig. 18 ({cap('Figure 18.')})",
     "captured on localhost with student-test trips; the two-laptop run is not shown in a screenshot"),
    ("§7 Submitted code matches the demonstrated implementation", "PASS",
     f"§15 ({pg('15 API, UI and Deployment')})", "terminal F1 values equal recorded results"),
    ("§8 AI-assistance disclosure as required by event policy", "PASS",
     f"§16 ({pg('AI assistance (Deliverables §8).')})", "Deliverables §8 rule 5"),
    ("§8 Every member can explain the work", "TEAM", "—", "viva preparation"),
]
lines = ["# Deliverables.pdf compliance checklist", "",
         f"Report: `reports/PathSnap_Final_Technical_Report.pdf` ({NP} pages). Page numbers are found by searching "
         f"the PDF text (reports/tooling/write_index.py). Section and figure numbers refer to the revised report.", "",
         "| Requirement (Deliverables.pdf) | Status | Where in the report | Evidence / note |", "|---|---|---|---|"]
lines += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in C]
lines += ["", "## Remaining unmet or partial requirements", "",
          "1. **§4 tuned-vs-untuned comparison of the chosen classifier (FAIL).** The Random Forest Bayesian search was "
          "stopped after 10 completed trials, and the best configuration was never refitted or evaluated. The report states "
          "this (§7.3). Closing it means running `python -m scripts.tune_classifier --stage evaluate`, which uses training "
          "and student-test data only, and adding its results.",
          "2. **§7 screenshots show a single machine (localhost:8000).** They prove a working UI, API and terminal log, but "
          "not the two-laptop setup. A screenshot from the actual Laptop 1 terminal during the two-laptop run would fully "
          "match the wording of §7.",
          "3. **Commit and push** the new `reports/` files (maps, screenshots, report) so the PDF's references resolve in "
          "the public repository.", "",
          f"Automated tests at report time: {tests['summary_line']} (git {tests['git_head']})."]
(R / "compliance_checklist.md").write_text("\n".join(lines) + "\n")

# ------------------------------------------------------------------ evidence index
FIG_SRC = {
    "fig_dataset_overview.png": "reports/data/dataset_overview_per_trip.csv",
    "fig_architecture.png": "diagram of main.py and src/* (reports/tooling/build_assets.py)",
    "alignment_00000018.png": "experiments/ml/alignment_plots/alignment_00000018.png (scripts/validate_alignment.py)",
    "fig_decoder_optuna_history.png": "experiments/optuna_trials.csv",
    "fig_decoder_tuned_vs_untuned.png": "reports/data/decoder_tuned_vs_untuned.csv",
    "fig_rf_optuna_history.png": "experiments/ml/rf_optuna_trials_partial.csv",
    "fig_cv_model_comparison.png": "experiments/ml/cv_fold_metrics.csv",
    "fig_student_test_top1_macro_f1.png": "experiments/ml/student_test_model_report.csv",
    "fig_confusion_matrices.png": "experiments/final/confusion_matrices_oof.csv",
    "fig_train_vs_cv.png": "experiments/final/train_vs_cv_metrics.csv",
    "fig_classifier_only_vs_hmm.png": "experiments/final/ablation_summary.csv",
    "fig_per_trip_delta.png": "experiments/final/ablation_per_trip.csv",
    "route_f1_comparison.png": "experiments/final_evaluation/evaluation_per_trip.csv",
    "runtime_distribution.png": "experiments/final_evaluation/evaluation_per_trip.csv",
    "ui_dashboard_trip_00000091.png": "team screenshot of frontend/app.py connected to the API",
    "api_server_terminal_log.png": "team screenshot of the uvicorn terminal (F1 values match experiments/compare_tuned_v2_vs_ml_v3_student_test.csv)",
    "fastapi_openapi_docs.png": "team screenshot of /docs generated from main.py",
}
ev = ["# Report evidence index", "", "Source for every table and figure in the revised report. All tables and prose "
      "numbers are generated by reports/tooling/build_report.py (content in report_content.py) from these files.", "",
      "| Item | Page | Source |", "|---|---|---|"]
for b in B:
    if b["t"] == "table":
        num = b["caption"].split(".")[0]
        src = re.findall(r"Sources?:\s*(.+)$", b["caption"])
        ev.append(f"| {num} | {cap(num + '.')} | {src[0] if src else b['caption'][len(num) + 2:][:120]} |")
    elif b["t"] == "fig":
        num = b["caption"].split(".")[0]
        ev.append(f"| {num} ({Path(b['path']).name}) | {cap(num + '.')} | {FIG_SRC.get(Path(b['path']).name, '—')} |")
    elif b["t"] == "figrow":
        num = b["caption"].split(".")[0]
        ev.append(f"| {num} (two Folium screenshots) | {cap(num + '.')} | reports/figures/maps/*.html; F1 check reports/data/route_map_f1_check.csv |")
ev += ["", "Checks: 269 of 270 decimal numbers in the PDF match recorded values (the other is the constant 1.4826); derived "
       "differences in the prose (+0.172..+0.271, +0.003..+0.006, -0.0003/-0.0013, 3 of 16 / 2 of 10 trips worse, 3x training "
       "speed) were recomputed from the source CSVs; 294/294 plotted values verified "
       "(experiments/final/figures/plotted_values_verification.csv)."]
(R / "report_evidence_index.md").write_text("\n".join(ev) + "\n")

# ------------------------------------------------------------------ quality check (short)
q = ["# Report quality check (revision 2)", "",
     f"- PDF: {NP} pages, A4, {sum(b['t'] == 'fig' for b in B) + sum(b['t'] == 'figrow' for b in B)} figures, "
     f"{sum(b['t'] == 'table' for b in B)} tables, one-page contents, page numbers and running header after the cover.",
     "- Every page was rendered and inspected: no clipping, overlaps, missing images or orphaned headings; body pages are "
     "86–100% filled.",
     "- DOCX validated (docx skill validator: all validations passed) and rendered through LibreOffice for inspection.",
     "- Screenshots inserted at their original aspect ratio, uncropped.",
     f"- Tests: {tests['summary_line']}.", "", "See compliance_checklist.md for requirement-by-requirement status."]
(R / "report_quality_check.md").write_text("\n".join(q) + "\n")
print("written; pages:", NP)
