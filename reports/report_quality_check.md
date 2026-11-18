# Report quality check

PDF: `reports/PathSnap_Final_Technical_Report.pdf` — **29 pages**, A4, 20 figures, 20 tables, page numbers and running header on every page after the cover, clickable table of contents and PDF bookmarks.

Editable versions: `reports/PathSnap_Final_Technical_Report.docx` (validated with the docx skill's validator: all validations passed; rendered through LibreOffice and inspected) and `reports/PathSnap_Final_Technical_Report.md`.

## Requested sections

| Section | Present |
|---|---|
| Executive Summary | PASS |
| Problem Statement | PASS |
| Dataset Exploration | PASS |
| System Architecture | PASS |
| Data Preprocessing and Label Alignment | PASS |
| Five Machine-Learning Models | PASS |
| Hyperparameter Optimisation | PASS |
| HMM and Viterbi | PASS |
| Experimental Protocol | PASS |
| Model Comparison | PASS |
| Ablation Study | PASS |
| Final Evaluation | PASS |
| Error Analysis | PASS |
| API, UI and Deployment | PASS |
| Testing and Reproducibility | PASS |
| Limitations and Future Work | PASS |
| Conclusion | PASS |
| References | PASS |
| Appendix | PASS |

## Requested figures

| Figure | Status |
|---|---|
| System architecture diagram | PASS |
| Dataset overview | PASS |
| Five-model cross-validation comparison | PASS |
| Student-test model comparison | PASS |
| Bayesian optimisation history | PASS |
| Tuned versus untuned comparison (decoder) | PASS |
| Geometric HMM versus ML+HMM | PASS |
| evaluation_test summary | PASS |
| Runtime distribution | PASS |
| Folium route visualisation 1 | PASS |
| Folium route visualisation 2 | PASS |
| Confusion matrices (optional) | PASS |
| RF tuned-vs-untuned comparison | **MISSING** — the experiment was never run (stated in Section 7.3) |
| UI / API terminal screenshots | **MISSING** — not captured yet (stated in Section 15) |

## Official guideline requirements (Deliverables.pdf)

| Requirement | Status | Where |
|---|---|---|
| Single consolidated PDF following §2–§7 | PASS | Sections are labelled with their Deliverables number; Table 1 maps them |
| §2.2 EDA: missing-value analysis | PASS | Section 3, Table 4 |
| §2.2 GPS trajectory plots | PASS | Figures 1, 3, 17–20 |
| §2.2 Road-network visualisations | PASS | Figures 17–20 (road-network layer) |
| §2.2 ≥1 Folium map; partition named per analysis | PASS | Section 14; Section 3 'Partitions used' |
| §3.1 Consolidated table (CV macro F1, Top-1, continuity, student-test macro F1, log loss, inference time) | PASS | Table 6 |
| §3.2 Compulsory questions for every model | PASS | Sections 6.1–6.5 (A1, A2, B5, B6, C7, C8, C9) |
| §3.3 CV macro F1, student-test Top-1 and macro F1, log loss, confusion matrices; no artificial training curves | PASS | Figures 7–10; no epoch-based models, none drawn |
| §4 BO: parameters, ranges, objective, trials, best values | PASS | Section 7 (decoder complete; RF partial) |
| §4 Tuned vs untuned comparison on CV macro F1, student-test Top-1/macro F1, log loss, inference time | **PARTIAL** | Decoder compared on route F1 (Fig. 5); classifier comparison missing, stated |
| §5 States, emissions, transitions, gaps, disconnected segments; classifier-only vs +HMM | PASS | Sections 8, 11 |
| §6 Interactive maps with GPS, network, reference, predicted; legend; layers; map in PDF; code committed | PASS | Section 14; reports/figures/maps/*.html; reports/tooling/build_assets.py |
| §7 Screenshots of working UI and API responses on the receiving laptop's terminal | **FAIL (team task)** | Real API log excerpt included instead; screenshots missing |
| §8 AI-assistance disclosure | PASS | Section 19 |
| Results verifiable from the repository | PASS | reports/report_evidence_index.md |

## Page-by-page inspection

All pages of the PDF were rendered to images and inspected; issues found and fixed during the build: blank half-pages from forced breaks, numbers wrapping inside narrow table columns, justified bullets with wide gaps, an orphaned appendix heading, a clipped JSON line, a title/annotation overlap in the decoder optimisation figure, a legend overlapping data, a wrong cross-reference, and six factual over-statements in the prose (corrected against the data, including the claim that every reference route is connected — 11 of 90 are not). Final state: no blank pages (text on every page), no headings at a page bottom, no clipped text found.

## Numerical-claim verification

- 266 of 267 decimal numbers in the PDF match a value in the recorded result files; the other is the constant 1.4826.
- No metric for missing ground truth is shown as zero; unavailable items are marked as missing.
- evaluator_test is presented as a one-time evaluation of a frozen model; it is never described as used for selection.
- 64-trip and 80-trip results are labelled in every table where both appear.

## Missing material (not fabricated)

1. UI screenshots and API-terminal screenshots (Deliverables §7).
2. Untuned-vs-tuned comparison of the Random Forest classifier (Deliverables §4); RF search incomplete.
3. Team name, members and SRNs (not present in the project materials).

Automated tests at report time: 65 passed in 2.51s (git 627c448).
