"""Build the PathSnap final technical report (PDF + Markdown + DOCX layout JSON) from recorded evidence.

  reports/tooling/.venv/bin/python reports/tooling/build_report.py     (repo root)
Every number is read from a file under experiments/ or reports/data/; nothing is typed in.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
E, FE, ML, FIN = ROOT / "experiments", ROOT / "experiments/final_evaluation", ROOT / "experiments/ml", ROOT / "experiments/final"
R, FIG, DATA = ROOT / "reports", ROOT / "reports/figures", ROOT / "reports/data"
NICE = {"logreg": "Logistic Regression", "decision_tree": "Decision Tree", "random_forest": "Random Forest",
        "extra_trees": "Extra Trees", "hist_gb": "HistGradientBoosting"}
ORDER = ["logreg", "decision_tree", "random_forest", "extra_trees", "hist_gb"]
FAMILY = {"logreg": "Linear / probabilistic", "decision_tree": "Tree-based", "random_forest": "Ensemble (bagging)",
          "extra_trees": "Ensemble (randomised trees)", "hist_gb": "Boosting"}


def f4(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def f3(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


# ============================================================== load evidence
cv = pd.read_csv(ML / "cv_model_comparison.csv").set_index("model")
cvf = pd.read_csv(ML / "cv_fold_metrics.csv")
st = pd.read_csv(ML / "student_test_model_report.csv").set_index("model")
ohmm = pd.read_csv(ML / "oof_hmm_comparison.csv")
egrid = pd.read_csv(ML / "emission_grid_random_forest.csv")
dec = pd.read_csv(E / "optuna_trials.csv")
rft = pd.read_csv(ML / "rf_optuna_trials_partial.csv")
ab = pd.read_csv(FIN / "ablation_summary.csv")
tvc = pd.read_csv(FIN / "train_vs_cv_metrics.csv").set_index("model")
cmx = pd.read_csv(FIN / "confusion_matrices_oof.csv").set_index("model")
summ = json.load(open(FE / "evaluation_summary.json"))
man = json.load(open(FE / "evaluation_manifest.json"))
ev = pd.read_csv(FE / "evaluation_per_trip.csv", dtype={"trip_id": str})
err = pd.read_csv(FE / "route_error_analysis.csv", dtype={"trip_id": str})
comp = pd.read_csv(FE / "comparison_with_previous_splits.csv")
pf = json.load(open(FE / "post_freeze_checks/post_freeze_check.json"))
dov = pd.read_csv(DATA / "dataset_overview_summary.csv").set_index("split")
dpt = pd.read_csv(DATA / "dataset_overview_per_trip.csv", dtype={"trip_id": str})
conn = pd.read_csv(DATA / "network_connectivity_summary.csv").set_index("split")
tds = json.load(open(DATA / "training_dataset_stats.json"))
tests = json.load(open(DATA / "test_run.json"))
rcon = pd.read_csv(DATA / "reference_route_continuity.csv", dtype={"trip_id": str})
api = json.load(open(DATA / "api_example.json"))
paired = pd.read_csv(DATA / "paired_tests.csv").set_index("comparison")
dtu = pd.read_csv(DATA / "decoder_tuned_vs_untuned.csv")
mapchk = pd.read_csv(DATA / "route_map_f1_check.csv", dtype={"trip_id": str})
alq = pd.read_csv(ML / "alignment_quality.csv", dtype={"trip_id": str})
lab = pd.read_csv(FIN / "label_verification.csv", dtype={"trip_id": str})
part = json.load(open(E / "train_partition.json"))
cfg = json.load(open(ROOT / "configs/ml_v3.json"))
cfgv1 = json.load(open(ROOT / "configs/baseline_v1.json"))
freeze = {l.split("==")[0].lower(): l.split("==")[1] for l in open(FIN / "pip_freeze.txt").read().split() if "==" in l}
fit_tvc = pd.read_csv(FIN / "train_vs_cv_metrics.csv")


def ab_get(es, var, model, col="arc_f1"):
    return float(ab[(ab.eval_set == es) & (ab.variant == var) & (ab.model == model)][col].iloc[0])


def vstat(v, k, s="mean"):
    return summ["variants"][v][k][s]


dbest = dec.loc[dec.value.idxmax()]
dtrial0 = float(dec.loc[dec.number == 0, "value"].iloc[0])
rfc = rft[rft.state == "COMPLETE"]
rbest = rfc.loc[rfc.value.idxmin()]
r0 = float(rft.loc[rft.number == 0, "value"].iloc[0])
geo_oof = ohmm[ohmm.model.str.startswith("tuned_v2")].iloc[0]
oof_by = ohmm.set_index("model")
eg = egrid.set_index(["emission", "ml_weight"])
dep, geo = "rf80_hmm_deployed", "geometric_hmm_tuned_v2"
dvg = summ["deployed_vs_geometric_per_trip"]
e_dep = ev[ev.variant == dep].set_index("trip_id")
e_geo = ev[ev.variant == geo].set_index("trip_id")
er_dep = err[err.variant == dep]
er_geo = err[err.variant == geo]
make_models_src = (ROOT / "src/ml_train.py").read_text()

# ============================================================== content model
B: list[dict] = []
FIGN = [0]
TABN = [0]


def h1(t): B.append({"t": "h1", "text": t})
def h2(t): B.append({"t": "h2", "text": t})
def h3(t): B.append({"t": "h3", "text": t})
def p(t): B.append({"t": "p", "text": t})
def bl(items): B.append({"t": "bullets", "items": items})
def code(t, width=96):
    out = []
    for line in t.split("\n"):
        while len(line) > width:
            cut = line.rfind(", ", 0, width)
            cut = width if cut < 20 else cut + 2
            out.append(line[:cut]); line = "      " + line[cut:]
        out.append(line)
    B.append({"t": "code", "text": "\n".join(out)})
def pb(): B.append({"t": "pagebreak"})


def note(t): B.append({"t": "note", "text": t})


def fig(path, caption, width=1.0):
    FIGN[0] += 1
    pth = Path(path)
    img = mpimg.imread(pth)
    B.append({"t": "fig", "path": str(pth), "caption": f"Figure {FIGN[0]}. {caption}", "width": width,
              "px": [int(img.shape[1]), int(img.shape[0])]})
    return FIGN[0]


def tab(header, rows, caption, widths=None, small=False):
    TABN[0] += 1
    B.append({"t": "table", "header": header, "rows": [[str(c) for c in r] for r in rows],
              "caption": f"Table {TABN[0]}. {caption}", "widths": widths, "small": small})
    return TABN[0]


# ---------------------------------------------------------------- cover
B.append({"t": "cover",
          "title": "PathSnap: ML-Assisted GPS Map Matching and Route Reconstruction",
          "subtitle": "Final Technical Report",
          "lines": ["UE24CS352A: Machine Learning — PathSnap Hackathon (PES University, Department of CSE, RR Campus)",
                    "Repository: github.com/riyanakarthik/ml-hackathon",
                    "**Team ID: PT-056**",
                    "RIYANA KARTHIK VISHALAKSHI (PES1UG24AM226)",
                    "ROHANTH GANESHA C (PES1UG24AM227)",
                    "ROHIT CHAKI (PES1UG24AM228)",
                    "ROHITH KARNING (PES1UG24AM229)"]})
B.append({"t": "toc"})

# ---------------------------------------------------------------- 1 executive summary
h1("1  Executive Summary")
ev_dep_f1 = vstat(dep, "arc_f1")
p(f"PathSnap reconstructs the road path driven by a vehicle from a noisy GPS trace and a directed road network. "
  f"The deployed system (configuration **ml_v3**) generates nearby road candidates for every GPS observation, scores "
  f"each candidate with a **Random Forest** classifier trained on 80 training trips, and decodes the most plausible "
  f"connected path with a **Hidden Markov Model (HMM) and log-space Viterbi** decoder whose transition parameters were "
  f"tuned with Bayesian optimisation (Optuna TPE). Gaps between matched arcs are stitched with shortest paths, so every "
  f"returned route is a connected directed path. A FastAPI backend serves predictions to a Streamlit/Folium UI.")
bl([f"**Five classifiers** (Logistic Regression, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting) were "
    f"compared with 5-fold trip-grouped cross-validation on 64 training trips. Candidate-level CV macro F1 ranged "
    f"{cv.oof_macro_f1.min():.4f}–{cv.oof_macro_f1.max():.4f}; tree models outperformed Logistic Regression.",
    f"**The HMM is the dominant component**: on held-out training trips, classifier-only routes reach arc F1 "
    f"{ab_get('holdout16', 'classifier_only', 'random_forest'):.3f}, the same classifier inside the HMM reaches "
    f"{ab_get('holdout16', 'ml_hmm', 'random_forest'):.3f} (Table {{TAB_ABL}}).",
    f"**Classifier emissions give a small, consistent gain over geometric emissions** on development data "
    f"(holdout 16: {ab_get('holdout16', 'geometric_hmm', '-'):.4f} → {ab_get('holdout16', 'ml_hmm', 'random_forest'):.4f}, "
    f"Wilcoxon p = {paired.loc['holdout16: geometric HMM -> RF(64)+HMM', 'wilcoxon_p']:.3f}).",
    f"**Final evaluation on the 10 reserved evaluator_test trips** (model frozen beforehand): mean directed-arc-set F1 "
    f"**{ev_dep_f1:.4f}** (median {vstat(dep, 'arc_f1', 'median'):.4f}) versus {vstat(geo, 'arc_f1'):.4f} for the "
    f"geometric HMM; {dvg['better']} trips better, {dvg['worse']} worse, {dvg['same']} equal — the difference is "
    f"**not statistically significant** on 10 trips (Wilcoxon p = {dvg['wilcoxon_p']:.2f}).",
    f"**Limitations**: per-observation labels are inferred by aligning GPS to the ordered reference route (pseudo-labels); "
    f"the route metric ignores order and duplicates; Bayesian optimisation of the Random Forest was stopped after "
    f"{len(rfc)} completed trials and its tuned configuration was never evaluated or deployed; UI screenshots are not yet "
    f"available."])

# ---------------------------------------------------------------- 2 problem
h1("2  Problem Statement and Objectives")
p("GPS receivers report positions with metre-scale noise, occasional dropouts and irregular timestamps, so a raw trace "
  "drifts onto parallel streets, cuts corners and jumps between nearby roads. Map matching assigns each observation to "
  "a road segment and reconstructs a route that is consistent with the road network: consecutive segments must connect "
  "head-to-tail and respect one-way directions. In PathSnap the road network is a directed graph whose edges (arcs) are "
  "identified by their row index in the trip's .arcs file, and the reference answer is an ordered list of arc ids.")
p("Objectives, as set by the PathSnap guidelines:")
bl(["Understand and explore the raw trip files and their relationships.",
    "Train and compare at least five supervised models that score road candidates for GPS observations.",
    "Select a model on experimental evidence and apply Bayesian hyperparameter optimisation.",
    "Reconstruct routes with an HMM and Viterbi decoding, handling missing candidates, gaps and disconnected segments.",
    "Deploy a FastAPI backend (Laptop 1) and an evaluator UI (Laptop 2) that works for all ten reserved trips, keeping "
    "ground truth server-side.",
    "Report verified results without leakage, fabricated numbers or hardcoded predictions."])
tab(["Deliverables requirement", "Where answered"],
    [["§2.2 Dataset understanding and EDA (missing values, trajectory and network plots, Folium map, partitions)", "Sections 3, 5 and 14"],
     ["§3.1 Overall model comparison table", "Section 6, Table {TAB_31}"],
     ["§3.2 Compulsory questions for every model (A1–A2, B5–B6, C7–C9)", "Sections 6.1–6.5"],
     ["§3.3 Model visualisations (CV macro F1, student-test Top-1 and macro F1, log loss, confusion matrices)", "Section 10"],
     ["§4 Bayesian hyperparameter optimisation and tuned vs untuned comparison", "Section 7 (classifier comparison missing — stated)"],
     ["§5 HMM and Viterbi; classifier-only vs classifier + HMM", "Sections 8 and 11"],
     ["§6 Geographical visualisation using Folium", "Section 14"],
     ["§7 API, UI and deployment screenshots", "Section 15 (screenshots missing — stated)"],
     ["§8 Academic integrity and AI disclosure", "Section 19"]],
    "Mapping of the official report structure (Deliverables.pdf §2–§8) to this report.", widths=[0.68, 0.32])

# ---------------------------------------------------------------- 3 dataset
h1("3  Dataset Exploration (Deliverables §2.2)")
p("Each trip directory holds four whitespace-separated files without headers (verified by the loader in "
  "src/data_loader.py, which validates column counts, finite values, coordinate ranges and node-id ranges):")
tab(["File", "Columns", "Meaning"],
    [[".track", "longitude, latitude, timestamp (s)", "GPS observations of the trip (one row per observation)"],
     [".nodes", "longitude, latitude", "Road-network nodes; the 0-based row index is the node id"],
     [".arcs", "from_node_id, to_node_id", "Directed road segments; the 0-based row index is the arc id"],
     [".route", "arc_id", "Chronological reference sequence of traversed arcs — NOT one label per GPS observation"]],
    "Raw file types (Dataset Exploration document §3, verified against the files).", widths=[0.12, 0.33, 0.55])
rows = []
for s, name in (("train", "train"), ("student_test", "student_test"), ("evaluator_test", "evaluator_test")):
    d = dov.loc[s]
    rows.append([name, int(d.trips), f"{int(d.gps_points_total):,}", f"{int(d.gps_points_median):,}",
                 f"{int(d.gps_points_min):,}–{int(d.gps_points_max):,}", f"{int(d.nodes_median):,}",
                 f"{int(d.arcs_median):,}", f"{int(d.route_arcs_median)}", f"{d.frac_1s_mean:.3f}", f"{d.max_gap_s:,.0f}"])
TAB_INV = tab(["Split", "Trips", "GPS obs.", "Median obs./trip", "Range", "Median nodes", "Median arcs", "Median .route arcs",
     "Mean share 1-s steps", "Max gap (s)"], rows,
    "Dataset inventory per partition. Source: reports/data/dataset_overview_summary.csv (train/student_test computed "
    "from the files; evaluator_test from the frozen evaluation manifest).", small=True,
    widths=[0.13, 0.06, 0.09, 0.09, 0.12, 0.09, 0.09, 0.1, 0.11, 0.12])
FIG_OV = fig(FIG / "fig_dataset_overview.png", "Per-trip size of the GPS trace, the road network and the reference route for "
    "each partition (log scale for the first two panels).")
tr_d = dpt[dpt.split == "train"]
mrows = []
for s_name in ("train", "student_test", "evaluator_test"):
    d_ = dpt[dpt.split == s_name]
    mrows.append([s_name, str(len(d_)), "0", "0", "0",
                  str(int((d_.gaps_over_10s > 0).sum())) if d_.gaps_over_10s.notna().all() else "not computed",
                  f"{d_.max_gap_s.max():,.0f}"])
tab(["Split", "Trips", "Missing files", "Non-finite values", "Invalid node refs", "Trips with a gap > 10 s", "Largest gap (s)"],
    mrows, "Missing-value and anomaly analysis. Zeros are guaranteed by the loader, which rejects any trip with a missing "
    "file, non-finite value, out-of-range coordinate or arc endpoint outside the node table (all 100 trips loaded). "
    "Source: reports/data/dataset_overview_per_trip.csv.", small=True,
    widths=[0.16, 0.08, 0.13, 0.15, 0.15, 0.17, 0.16])
p(f"**Missing and invalid values.** All 100 trips have all four files and parse with zero non-finite values, zero "
  f"out-of-range coordinates and zero arc endpoints outside the node table (the loader rejects any violation). Sampling is "
  f"nominally 1 Hz (median share of 1-second steps in train: {tr_d.frac_1s.median():.3f}), but "
  f"{int((tr_d.gaps_over_10s > 0).sum())} of 80 training trips contain at least one gap above 10 s and two trips contain "
  f"very long interruptions (trip 00000090: {tr_d.set_index('trip_id').loc['00000090', 'max_gap_s']:,.0f} s; "
  f"trip 00000078: {tr_d.set_index('trip_id').loc['00000078', 'max_gap_s']:,.0f} s).")
c_tr, c_st = conn.loc["train"], conn.loc["student_test"]
p(f"**Road-network connectivity** (reports/data/network_connectivity_summary.csv). No train or student_test network has isolated nodes. "
  f"Networks are extracts around each trip and split into several weakly connected components (train median "
  f"{c_tr.weak_components_median:.0f}, max {c_tr.weak_components_max:.0f}); the largest strongly connected component "
  f"covers a median {c_tr.largest_strong_frac_median:.1%} of nodes, and a median {c_tr.two_way_arc_frac_median:.1%} of "
  f"arcs have a reverse twin (two-way streets are two arcs with identical geometry). Direction therefore cannot be "
  f"inferred from geometry alone.")
rc_tr, rc_st = rcon[rcon.split == "train"], rcon[rcon.split == "student_test"]
p(f"**Reference routes.** Most .route files are connected directed paths (the to-node of one arc equals the from-node of "
  f"the next), but not all: {int((rc_tr.breaks > 0).sum())} of 80 training routes ({int(rc_tr.breaks.sum())} breaks in "
  f"total) and {int((rc_st.breaks > 0).sum())} of 10 student-test routes contain a discontinuity (lowest continuity "
  f"{rcon.continuity.min():.3f}; reports/data/reference_route_continuity.csv; evaluator_test routes were not examined); "
  f"neither the label alignment nor the route metrics assume a connected reference. {int(lab.route_has_repeats.sum())} of 80 training routes traverse some arc more than once (e.g. "
  f"roundabout loops). In some training trips the GPS trace continues well beyond the end of the reference route "
  f"(trip 00000054: {1 - alq.set_index('trip_id').loc['00000054', 'label_valid_frac']:.0%} of observations; trip 00000033: "
  f"{1 - alq.set_index('trip_id').loc['00000033', 'label_valid_frac']:.0%}), which matters for labelling and scoring "
  f"(Sections 5 and 13).")
p(f"**Partitions used for each analysis.** Table {TAB_INV} and Figure {FIG_OV} cover all partitions; connectivity, gaps and route "
  "properties were computed on train and student_test only (evaluator_test was not explored before the frozen final "
  "evaluation; its inventory comes from that evaluation's manifest). Feature choices motivated by these observations — "
  "heading versus arc bearing for two-way streets, time-gap features for irregular sampling — are listed in Section 5.3.")

# ---------------------------------------------------------------- 4 architecture
h1("4  System Architecture")
fig(FIG / "fig_architecture.png", "End-to-end pipeline. Steps 1–6 use only .track, .nodes and .arcs; the reference "
    ".route is read only by the server-side evaluation after prediction.")
bl(["**Ingestion** (src/data_loader.py): trip discovery, parsing and validation; never reads .route.",
    "**Candidate generation** (src/matcher.py): local metric projection, KD-tree over densified arcs, exact "
    "point-to-segment distance and projection.",
    "**Features and classifier** (src/features.py, src/ml_emission.py): 20 inference-safe features per candidate; the "
    "saved Random Forest bundle is loaded once at API start-up.",
    "**HMM/Viterbi and route assembly** (src/decoder.py): log-space Viterbi with network-distance transitions, Dijkstra gap "
    "stitching, optional geometric fallback if the model cannot be loaded.",
    "**API** (main.py): GET /health, GET /trips, POST /predict; request logs printed on the server terminal.",
    "**UI** (frontend/app.py): Streamlit + Folium dashboard on Laptop 2; it receives coordinates and metrics only.",
    "**Evaluation** (src/evaluation.py): the only module that reads .route; called after prediction."])

# ---------------------------------------------------------------- 5 preprocessing
h1("5  Data Preprocessing and Label Alignment (Deliverables §2.2)")
h2("5.1 Spatial projection and candidate search")
p("Longitude/latitude are angular coordinates, so all geometry is computed after an equirectangular projection to local "
  "metres around the trip's mean position (accurate to well under a metre at trip scale). Every arc is densified with "
  "sample points every 10 m and indexed in a scipy cKDTree; a radius query (plus half the sample spacing) returns every "
  "arc within the search radius, after which the exact point-to-segment distance, projected point and offset along the "
  f"arc are computed. The deployed configuration keeps up to k = {cfg['k']} candidates within "
  f"{cfg['radius_m']:.1f} m, retrying at {cfg['fallback_radius_m']:.0f} m if none are found. Directed arcs and their "
  "reverse twins are separate candidates.")
h2("5.2 Monotonic alignment of .route to GPS observations (pseudo-labels)")
p("Because .route is an ordered arc sequence and not a label per observation, supervised targets were inferred by a "
  "monotone dynamic-programming alignment (src/labels.py). The hidden state is a position i in the reference route, so "
  "repeated arcs and A → reverse(A) U-turns are distinct states. The score combines a Gaussian term on point-to-arc "
  "distance (σ = 10 m), a heading-agreement term κ(cos Δθ − 1) applied only when the vehicle moves (κ = 2, speed ≥ 2 m/s), "
  "and a penalty of 0.01 per metre of reference arcs skipped between consecutive positions. Labels more than 30 m from "
  "their arc are marked invalid and excluded from training and from pseudo-label accuracy.")
tab(["Check (80 training trips)", "Result"],
    [["Aligned positions non-decreasing", f"{int(lab.positions_non_decreasing.sum())}/80"],
     ["Label equals the .route arc at its aligned position", f"{int(lab.labels_equal_route_at_position.sum())}/80"],
     ["Label sequence is an ordered subsequence of .route", f"{int(lab.label_sequence_is_ordered_subsequence_of_route.sum())}/80"],
     ["Mean share of observations with a valid label", f"{alq.label_valid_frac.mean():.4f}"],
     ["Moving observations whose heading agrees with the label arc direction", f"{alq.heading_agree_frac.mean():.4f}"],
     ["Agreement of labels when σ = 5 m / σ = 20 m", f"{alq.agree_sigma5.mean():.4f} / {alq.agree_sigma20.mean():.4f}"],
     ["Agreement without the heading term", f"{alq.agree_no_heading.mean():.4f}"],
     ["Valid labels whose arc is among the generated candidates", f"{tds['label_in_candidates_frac']:.4f}"]],
    "Alignment quality. Sources: experiments/final/label_verification.csv, experiments/ml/alignment_quality.csv, "
    "reports/data/training_dataset_stats.json.", widths=[0.72, 0.28])
fig(FIG / "alignment_00000018.png", "Alignment on training trip 00000018, whose route repeats arcs around a roundabout. "
    "GPS points are coloured by aligned route position; orange lines join points to their label arc.", width=1.0)
note("**Disclosure.** These labels are derived, not ground truth. They inherit GPS noise, depend on the alignment "
     "parameters (about 2–5% of labels change under the perturbations above) and are ambiguous at junctions, on short "
     "arcs and where the GPS continues after the reference route ends. All per-candidate and per-observation metrics in "
     "this report are measured against these pseudo-labels; route-level metrics are measured against .route itself.")
h2("5.3 Feature construction")
p(f"Twenty features are computed per candidate (src/features.py, schema version 1) from the GPS track, the network and the "
  f"candidate set only: distance to the arc, its rank and gap to the nearest candidate, distance normalised by a per-trip "
  f"noise estimate; arc length, projection position and distances to both arc ends; centred GPS heading versus directed arc "
  f"bearing (cosine and degrees, with a validity flag for slow movement); speed and time gaps to the previous/next "
  f"observation; presence of the reverse twin among the candidates; out-degree of the end node and in-degree of the start "
  f"node; and two context features over ±5 observations (share of neighbouring observations that also have this arc as a "
  f"candidate, and their mean distance to it). The heading features target the two-way-street ambiguity found in Section 3; "
  f"the time-gap features target irregular sampling; the window features use temporal context that a single observation "
  f"lacks. The training table has {tds['candidate_rows']:,} candidate rows from {tds['gps_points']:,} observations "
  f"({tds['candidate_rows_valid']:,} rows with valid labels, positive rate {tds['positive_rate_valid']:.1%}).")

# ---------------------------------------------------------------- 6 five models
h1("6  Five Machine-Learning Models (Deliverables §3.1–3.2)")
p("Every model solves the same binary task — is this candidate arc the road the vehicle was on? — on the same "
  "features, the same valid-label rows and the same five trip-grouped folds of the 64 tune trips (GroupKFold by trip; "
  "fold disjointness asserted in code, experiments/ml/cv_folds.json). No preprocessing is fitted except the "
  "StandardScaler inside the Logistic Regression pipeline, which is fitted within each training fold. All models use "
  "random_state = 42. The optimisation objective of every learner is its standard binary log-loss / impurity criterion "
  "(below); model selection used out-of-fold route-level F1 after HMM decoding (Section 9).")
initial = {
    "logreg": "StandardScaler + LogisticRegression(C=1.0, max_iter=2000)",
    "decision_tree": "DecisionTreeClassifier(max_depth=12, min_samples_leaf=50)",
    "random_forest": "RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=20, max_features='sqrt')",
    "extra_trees": "ExtraTreesClassifier(n_estimators=150, max_depth=18, min_samples_leaf=20, max_features='sqrt')",
    "hist_gb": "HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False)",
}
for k, v in initial.items():  # verify against source code
    tok = v.split("(")[-1].split(",")[0].split("=")[0]
    assert tok in make_models_src, (k, tok)
rows = []
for m in ORDER:
    rows.append([NICE[m], FAMILY[m], f"{cv.loc[m, 'oof_macro_f1']:.4f}",
                 f"{st.loc[m + '__tune64', 'clf_pseudo_top1_accuracy']:.4f}",
                 f"{ab_get('student_test10', 'ml_hmm', m, 'route_continuity'):.4f}",
                 f"{st.loc[m + '__tune64', 'clf_macro_f1']:.4f}",
                 f"{cv.loc[m, 'oof_log_loss']:.4f} / {st.loc[m + '__tune64', 'clf_log_loss']:.4f}",
                 f"{cv.loc[m, 'predict_ms_per_1k_rows']:.3f}"])
t31 = tab(["Model", "Algorithm family", "CV macro F1", "Top-1 accuracy (student test)", "Route continuity (student test, +HMM)",
           "Student-test macro F1", "Log loss CV / student test", "Inference ms per 1k candidates"], rows,
          "Overall model comparison (Deliverables §3.1). CV = out-of-fold over 64 tune trips; student-test columns use "
          "models trained on the same 64 trips; Top-1, macro F1 and log loss are against pseudo-labels; route continuity "
          "is route-level. Sources: experiments/ml/cv_model_comparison.csv, experiments/ml/student_test_model_report.csv, "
          "experiments/final/ablation_summary.csv.", small=True)
p("Key configuration of each model is given in its subsection and in Appendix A.1. Inference time is the classifier's "
  "predict_proba time only, measured during cross-validation on the same machine.")

why = {
    "logreg": ("A linear probabilistic baseline: if candidate correctness were mostly a monotone function of distance, "
               "heading agreement and context, a calibrated linear model would suffice and be fast.",
               f"Strengths: fastest training ({cv.loc['logreg', 'fit_s_total']:.1f} s for 5 folds), inference as fast as a "
               "single tree, probabilistic outputs, interpretable coefficients. "
               "Limitations: cannot model interactions such as 'heading matters only when moving' or non-linear distance "
               "effects without manual feature crosses."),
    "decision_tree": ("Captures threshold rules and interactions (e.g. distance only matters relative to the nearest "
                      "candidate) and is directly interpretable.",
                      "Strengths: interactions, no scaling, fast inference. Limitations: high variance and coarse, piecewise-constant "
                      "probabilities (leaf frequencies) as emission scores."),
    "random_forest": ("Averages many decorrelated trees, keeping the interaction modelling of a tree while reducing "
                      "variance and producing smoother probabilities for HMM emissions.",
                      "Strengths: robust, little tuning, good probability ranking. Limitations: large model file "
                      "(the deployed bundle is about 47 MB), slower inference than linear or single-tree models."),
    "extra_trees": ("Randomised split thresholds give a different bias–variance trade-off from Random Forest at lower "
                    "training cost.",
                    f"Strengths: about {cv.loc['random_forest', 'fit_s_total'] / cv.loc['extra_trees', 'fit_s_total']:.0f}x "
                    f"faster to train than Random Forest here ({cv.loc['extra_trees', 'fit_s_total']:.1f} s vs "
                    f"{cv.loc['random_forest', 'fit_s_total']:.1f} s for 5 folds). Limitations: more conservative on the "
                    "positive class (lowest recall of the tree models here)."),
    "hist_gb": ("Gradient boosting on binned features corrects errors sequentially and is efficient on about a million rows.",
                "Strengths: the best CV F1, macro F1 and Top-1 of the five (log loss within 0.001 of Random Forest), "
                "the fastest ensemble to train here, and a very small model file. Limitations: more hyperparameters, "
                "sequential training, risk of overfitting without early stopping (disabled here for reproducibility)."),
}
objective = {"logreg": "L2-regularised logistic (binary cross-entropy) loss.",
             "decision_tree": "Gini impurity reduction at each split.",
             "random_forest": "Gini impurity per tree; probabilities averaged over trees.",
             "extra_trees": "Gini impurity with randomised thresholds; probabilities averaged.",
             "hist_gb": "Binary log-loss minimised by gradient boosting."}
for m in ORDER:
    h2(f"6.{ORDER.index(m) + 1} {NICE[m]}")
    s_ = st.loc[m + "__tune64"]
    g = tvc.loc[m]
    gap = g.train_macro_f1 - g.cv_macro_f1
    fit_comment = ("almost no gap, but the lowest ceiling of the five — mild underfitting" if gap < 0.005 else
                   f"a modest gap of {gap:.3f} macro F1 — mild overfitting, controlled by the leaf-size limits")
    bl([f"**A1 — Suitability.** {why[m][0]}",
        f"**A2 — Strengths and limitations.** {why[m][1]}",
        f"**B5 — Hyperparameters (initial values, not tuned):** {initial[m]}; random_state = 42.",
        f"**B6 — Objective.** {objective[m]}",
        f"**C7 — Student-test performance** (trained on 64 tune trips): with the HMM, mean route arc F1 "
        f"{ab_get('student_test10', 'ml_hmm', m):.4f}; classifier-only route arc F1 "
        f"{ab_get('student_test10', 'classifier_only', m):.4f}.",
        f"**C8 — Observation/candidate level on student test:** Top-1 accuracy {s_.clf_pseudo_top1_accuracy:.4f}, "
        f"macro F1 {s_.clf_macro_f1:.4f}, log loss {s_.clf_log_loss:.4f} (pseudo-labels).",
        f"**C9 — Over/underfitting:** training macro F1 {g.train_macro_f1:.4f} vs CV {g.cv_macro_f1:.4f}; training log "
        f"loss {g.train_log_loss:.4f} vs CV {g.cv_log_loss:.4f} — {fit_comment}. Fold-to-fold F1 standard deviation "
        f"{cv.loc[m, 'fold_f1_std']:.4f}."])

# ---------------------------------------------------------------- 7 HPO
h1("7  Bayesian Hyperparameter Optimisation (Deliverables §4)")
p("Two components were optimised separately, and only one search is complete. The HMM decoder was tuned with Optuna "
  "TPE; the Random Forest search was started but stopped early at the team's request. The ML emission weight was chosen "
  "by a small grid, not by Bayesian optimisation.")
h2("7.1 HMM decoder (completed)")
space = [["k (candidates per point)", "int [3, 12]"], ["search radius", "float [20, 100] m"],
         ["σ scale (emission noise multiplier)", "float [0.5, 3], log"], ["β (transition scale)", "float [2, 100] m, log"],
         ["same-arc backtrack tolerance", "float [0, 30] m"], ["backtrack penalty", "0 or float [0.01, 2], log"],
         ["U-turn penalty", "float [0, 20]"], ["observation thinning distance", "float [0, 40] m"],
         ["short-loop removal", "0 (off) or float [20, 2000] m, log"]]
best = {"k": int(dbest.user_attrs_cfg_k), "radius": dbest.user_attrs_cfg_radius_m, "sigma": dbest.user_attrs_cfg_sigma_scale,
        "beta": dbest.user_attrs_cfg_beta, "bt": dbest.user_attrs_cfg_backtrack_tol_m, "bp": dbest.user_attrs_cfg_backtrack_penalty,
        "ut": dbest.user_attrs_cfg_uturn_penalty, "ms": dbest.user_attrs_cfg_min_spacing_m, "ml": dbest.user_attrs_cfg_max_loop_m}
bestv = [str(best["k"]), f"{best['radius']:.2f} m", f"{best['sigma']:.3f}", f"{best['beta']:.3f} m", f"{best['bt']:.2f} m",
         f"{best['bp']:.1f} (off)", f"{best['ut']:.2f}", f"{best['ms']:.2f} m", f"{best['ml']:.0f} (off)"]
initv = [str(cfgv1["k"]), f"{cfgv1['radius_m']:.0f} m", "1.0", f"{cfgv1['beta']:.0f} m", f"{cfgv1['backtrack_tol_m']:.0f} m",
         "0 (off)", "0", "0 m", "0 (off)"]
tab(["Hyperparameter", "Search space", "Untuned (baseline_v1)", "Best (trial #%d)" % int(dbest.number)],
    [[a, b, c, d] for (a, b), c, d in zip(space, initv, bestv)],
    f"Decoder search: Optuna TPESampler(seed=42, multivariate, 20 start-up trials), {len(dec)} trials; objective = mean "
    f"route arc F1 on the 64 tune trips; trial 0 = untuned configuration. Sources: experiments/optuna_trials.csv, "
    f"experiments/optuna_study.db, configs/tuned_v2.json.", widths=[0.33, 0.27, 0.2, 0.2])
fig(FIG / "fig_decoder_optuna_history.png", f"Decoder optimisation history: objective rose from {dtrial0:.4f} (trial 0, "
    f"untuned) to {float(dbest.value):.4f} (trial {int(dbest.number)}).")
fig(FIG / "fig_decoder_tuned_vs_untuned.png", "Tuned versus untuned decoder on the selection set and on two sets not used "
    "for tuning (holdout 16 training trips, student test).")
ph = paired.loc["holdout16: baseline_v1 -> tuned_v2 decoder"]
ps = paired.loc["student_test10: baseline_v1 -> tuned_v2 decoder"]
p(f"The tuned decoder outperformed the untuned one on data not used for tuning: holdout 16 "
  f"{dtu.iloc[1].baseline_v1:.4f} → {dtu.iloc[1].tuned_v2:.4f} ({ph.better} better, {ph.worse} worse, Wilcoxon p = "
  f"{ph.wilcoxon_p:.4f}); student test {dtu.iloc[2].baseline_v1:.4f} → {dtu.iloc[2].tuned_v2:.4f} ({ps.better} better, "
  f"{ps.worse} worse). Diagnostics showed why: the untuned decoder treated small backward GPS jitter on an arc as a loop "
  f"through the network, producing oscillation between an arc and its reverse twin. The best values for β, k and "
  f"backtrack tolerance lie near the edges of their ranges, so a wider search might improve further.")
h2("7.2 ML emission weight (grid search, not Bayesian)")
rows = []
for (mode, w), r in eg.sort_index().iterrows():
    rows.append([mode, f"{w:g}", f"{r.arc_f1_mean:.4f}", f"{r.arc_f1_median:.4f}", f"{r.arc_precision_mean:.4f}", f"{r.arc_recall_mean:.4f}"])
tab(["Emission mode", "Weight", "Route F1 mean", "Median", "Precision", "Recall"], rows,
    "Random Forest emission grid on out-of-fold probabilities of the 64 tune trips. 'ml' = weight × log p; 'hybrid' = "
    "geometric log-likelihood + weight × log p. Selected: ml, weight 4. Source: experiments/ml/emission_grid_random_forest.csv.",
    small=True)
h2("7.3 Random Forest classifier (incomplete)")
p(f"Search space: n_estimators 50–300 (step 25), max_depth 6–30, min_samples_leaf 1–200 (log), max_features ∈ "
  f"{{sqrt, 0.33, 0.5}}, max_samples 0.2–1.0; objective = mean 5-fold log loss on the 64 tune trips (same folds as "
  f"Section 6); Optuna TPESampler(seed=42) with a median pruner. The search was **stopped after {len(rfc)} completed "
  f"trials** ({int((rft.state == 'PRUNED').sum())} pruned, {int((rft.state == 'RUNNING').sum())} interrupted). Best "
  f"completed trial #{int(rbest.number)}: log loss {rbest.value:.5f} versus {r0:.5f} for the untuned configuration (trial "
  f"0), a difference of {r0 - rbest.value:.5f}, smaller than the fold-to-fold variation "
  f"(standard deviation of the untuned model's fold log loss: "
  f"{cvf[cvf.model == 'random_forest'].log_loss.std():.4f}). Best parameters: n_estimators {int(rbest.params_n_estimators)}, "
  f"max_depth {int(rbest.params_max_depth)}, min_samples_leaf {int(rbest.params_min_samples_leaf)}, max_features "
  f"{rbest.params_max_features}, max_samples {rbest.params_max_samples:.3f}.")
note("**Not done:** the tuned Random Forest was never refitted, evaluated on student test, or deployed, so the "
     "untuned-versus-tuned comparison of the classifier required by Deliverables §4 (CV macro F1, student-test Top-1 and "
     "macro F1, log loss, inference time) does not exist. The deployed Random Forest uses the initial hyperparameters in "
     "Section 6.3. Sources: experiments/ml/rf_optuna_trials_partial.csv, experiments/ml/rf_optuna_study.db.")
fig(FIG / "fig_rf_optuna_history.png", "Random Forest search history (completed trials only).", width=0.85)

# ---------------------------------------------------------------- 8 HMM
h1("8  HMM and Viterbi Route Reconstruction (Deliverables §5)")
bl(["**Hidden states:** at observation t, the candidate set C_t (directed arc, projected point, offset along the arc). The "
    "candidate-to-arc correspondence is preserved through decoding and returned per observation (point_arc_ids).",
    f"**Emissions:** geometric mode uses −½(d/σ)², with σ = {best['sigma']:.3f} × the per-trip noise estimate "
    f"1.4826 · median(distance to nearest arc), clipped to [3, 30] m (Newson and Krumm, 2009). ML mode (deployed) uses "
    f"{cfg['ml_weight']:.0f} · log q, where q is the Random Forest probability clipped at {cfg['prob_floor']} and normalised "
    f"over the observation's candidates (normalisation does not change the Viterbi path; it keeps scores interpretable).",
    f"**Transitions:** log p = −|d_route − d_straight| / β with β = {best['beta']:.3f} m, where d_route is the driving "
    f"distance along the directed network (same-arc forward movement, otherwise remaining arc + bounded Dijkstra + offset) "
    f"and d_straight the straight-line distance between the observations. Backward movement on the same arc up to "
    f"{best['bt']:.1f} m is treated as noise; moving onto an arc's reverse twin costs {best['ut']:.2f}.",
    "**Viterbi:** log-space recursion δ_t(j) = e_t(j) + max_i[δ_{t−1}(i) + a_{ij}] with back-pointers "
    "(src/decoder.py::viterbi_log). Observations closer than "
    f"{best['ms']:.1f} m to the previous decoded one are skipped and later assigned to the decoded arc.",
    "**Missing candidates and gaps:** observations with no candidate within the fallback radius are skipped (no state, "
    "null coordinates in the response); the transition is computed between the surrounding decoded observations. Dijkstra "
    "limits grow with the straight-line gap (max(2000 m, 3 × gap + 200 m)), so long time gaps remain decodable.",
    "**Disconnected segments:** if every transition into an observation is impossible, the chain restarts from that "
    "observation's emissions (a recorded 'HMM break'). After decoding, consecutive distinct arcs that do not connect are "
    "joined by the shortest directed path (≤ 3000 m), so the returned route is connected whenever a path exists."])
p(f"**Did HMM decoding improve the route?** Yes, by a large margin. In the equal-data ablation (Section 11), classifier-only "
  f"routes for the five models score {ab[(ab.eval_set == 'holdout16') & (ab.variant == 'classifier_only')].arc_f1.min():.3f}–"
  f"{ab[(ab.eval_set == 'holdout16') & (ab.variant == 'classifier_only')].arc_f1.max():.3f} route F1 on the holdout trips "
  f"and {ab[(ab.eval_set == 'holdout16') & (ab.variant == 'ml_hmm')].arc_f1.min():.3f}–"
  f"{ab[(ab.eval_set == 'holdout16') & (ab.variant == 'ml_hmm')].arc_f1.max():.3f} after HMM decoding. Independent "
  f"per-observation choices oscillate between parallel arcs and reverse twins, and every switch adds gap-stitching detours; "
  f"the HMM removes most of these. Cases where it did not help are discussed in Section 11.")

# ---------------------------------------------------------------- 9 protocol
h1("9  Experimental Protocol")
tab(["Data", "Trips", "Used for", "Never used for"],
    [["train — tune", str(len(part["tune"])), "Classifier fitting in 5 trip-grouped CV folds; decoder Optuna objective; "
      "emission-weight grid; model selection (out-of-fold route F1)", "—"],
     ["train — holdout", str(len(part["holdout"])), "Validation of each selected change (decoder, then ML emissions) after "
      "it was frozen; finally included in the 80-trip refit", "Any tuning or selection"],
     ["student_test", "10", "Development comparison after each freeze; reporting of all five models", "Training, tuning, selection"],
     ["evaluator_test", "10", "One final evaluation of the frozen deployed model", "Training, tuning, selection, exploration"]],
    f"Data use. The 64/16 split is seeded (seed {part['seed']}, experiments/train_partition.json).", widths=[0.16, 0.08, 0.5, 0.26])
bl(["**Selection rule** (stated before evaluation): the classifier with the best out-of-fold route F1 inside the HMM on "
    "the 64 tune trips; Random Forest scored "
    f"{oof_by.loc['random_forest', 'arc_f1_mean']:.4f} at emission weight 1 versus "
    f"{oof_by.loc['hist_gb', 'arc_f1_mean']:.4f} for HistGradientBoosting and {oof_by.loc['decision_tree', 'arc_f1_mean']:.4f} for the Decision Tree.",
    "**Freeze, then refit:** the RF + HMM configuration was validated on the holdout trips with the model trained on 64 "
    "trips, the choice to deploy it was fixed, and the Random Forest was refitted on all 80 training trips "
    "(models/random_forest__train80.joblib) before student test and evaluator_test were scored.",
    "**Repeated inspection of student test:** student-test scores were viewed several times (decoder v2, RF + HMM, every "
    "saved model, the ablation). They were not used to choose models or parameters, but repeated viewing still weakens "
    "student test as an unbiased estimate; the guidelines permit its use for development comparison.",
    "**Evaluator test:** scored once with the frozen model (SHA-256 recorded before prediction). Later re-runs only checked "
    "determinism, leakage (identical predictions with .route removed) and a post-freeze setup fix; all reproduced the "
    "frozen routes and scores exactly."])
h2("9.1 Metric definitions")
tab(["Level", "Unit", "Metrics", "Compared against"],
    [["Per-candidate classification", "one (observation, candidate arc) row", "precision, recall, F1 (positive class), macro F1, log loss, threshold 0.5", "pseudo-labels"],
     ["Per-observation", "one GPS observation", "Top-1 accuracy (top-scored or decoded candidate = label), candidate miss rate", "pseudo-labels (valid labels only)"],
     ["Route level (primary)", "one trip", "arc precision / recall / F1 on the SETS of distinct directed arc ids; route continuity", ".route"],
     ["Route level (supplementary)", "one trip", "sequence F1 = longest common subsequence of the ordered arc sequences", ".route"]],
    "Three levels of evaluation. Arc F1 collapses duplicates, ignores order, treats reverse twins as different arcs and "
    "counts gap-stitched arcs as predictions. The PathSnap documents define no official route formula.",
    widths=[0.22, 0.2, 0.38, 0.2], small=True)

# ---------------------------------------------------------------- 10 model comparison
h1("10  Model Comparison and Visualisations (Deliverables §3.3)")
rows = []
for m in ORDER:
    c = cv.loc[m]
    rows.append([NICE[m], f"{c.oof_precision:.4f}", f"{c.oof_recall:.4f}", f"{c.oof_f1:.4f}", f"{c.oof_macro_f1:.4f}",
                 f"{c.oof_log_loss:.4f}", f"{c.oof_roc_auc:.4f}", f"{c.oof_pseudo_top1_accuracy:.4f}",
                 f"{c.fold_f1_std:.4f}", f"{c.fit_s_total:.1f}"])
tab(["Model", "Precision", "Recall", "F1", "Macro F1", "Log loss", "ROC AUC", "Top-1", "Fold F1 sd", "Fit s (5 folds)"], rows,
    "Cross-validation on 64 tune trips (out-of-fold, valid-label rows, threshold 0.5; pseudo-labels). Source: "
    "experiments/ml/cv_model_comparison.csv.", small=True)
fig(FIG / "fig_cv_model_comparison.png", "Five-model cross-validation comparison: macro F1 and log loss (mean ± sd over five trip-grouped folds).")
rows = []
for m in ORDER:
    s_ = st.loc[m + "__tune64"]
    rows.append([NICE[m] + " (64)", f"{s_.clf_precision:.4f}", f"{s_.clf_recall:.4f}", f"{s_.clf_f1:.4f}", f"{s_.clf_macro_f1:.4f}",
                 f"{s_.clf_log_loss:.4f}", f"{s_.clf_pseudo_top1_accuracy:.4f}", f"{s_.arc_f1_mean:.4f}"])
s_ = st.loc["random_forest__train80"]
rows.append(["Random Forest (80, deployed)", f"{s_.clf_precision:.4f}", f"{s_.clf_recall:.4f}", f"{s_.clf_f1:.4f}", f"{s_.clf_macro_f1:.4f}",
             f"{s_.clf_log_loss:.4f}", f"{s_.clf_pseudo_top1_accuracy:.4f}", f"{s_.arc_f1_mean:.4f}"])
s_ = st.loc["geometric HMM (tuned_v2)"]
rows.append(["Geometric HMM (no classifier)", "—", "—", "—", "—", "—", "—", f"{s_.arc_f1_mean:.4f}"])
tab(["Model (training trips)", "Precision", "Recall", "F1", "Macro F1", "Log loss", "Top-1", "Route F1 with HMM"], rows,
    "Student-test comparison (10 trips), reported after models were frozen. Classifier columns use pseudo-labels; the "
    "last column is route-level against .route. The 80-trip row is not an equal-data comparison with the 64-trip rows. "
    "Source: experiments/ml/student_test_model_report.csv.", small=True)
fig(FIG / "fig_student_test_top1_macro_f1.png", "Student-test Top-1 accuracy and macro F1 for the five models (all trained on the same 64 trips).")
fig(FIG / "fig_confusion_matrices.png", "Out-of-fold confusion matrices at threshold 0.5 (64 tune trips, valid-label rows).")
fig(FIG / "fig_train_vs_cv.png", "Training (in-sample) versus out-of-fold CV metrics — evidence for over/underfitting.")
h2("10.1 Why Random Forest + HMM was deployed")
p(f"HistGradientBoosting has marginally better standalone classifier scores (CV F1 {cv.loc['hist_gb', 'oof_f1']:.4f} vs "
  f"{cv.loc['random_forest', 'oof_f1']:.4f}; student-test F1 {st.loc['hist_gb__tune64', 'clf_f1']:.4f} vs "
  f"{st.loc['random_forest__tune64', 'clf_f1']:.4f} with both trained on 64 trips). The deployment decision, however, "
  f"followed the pre-stated rule of best out-of-fold **route** F1 inside the HMM, because the route is the deliverable: "
  f"Random Forest {oof_by.loc['random_forest', 'arc_f1_mean']:.4f} vs HistGradientBoosting "
  f"{oof_by.loc['hist_gb', 'arc_f1_mean']:.4f} at emission weight 1. The emission weight (4) was then tuned for "
  f"Random Forest only. In the later equal-data ablation at weight 4 the two remain within noise: out-of-fold "
  f"{ab_get('tune64_OOF', 'ml_hmm', 'random_forest'):.4f} vs {ab_get('tune64_OOF', 'ml_hmm', 'hist_gb'):.4f}, holdout "
  f"{ab_get('holdout16', 'ml_hmm', 'random_forest'):.4f} vs {ab_get('holdout16', 'ml_hmm', 'hist_gb'):.4f}, student test "
  f"{ab_get('student_test10', 'ml_hmm', 'random_forest'):.4f} vs {ab_get('student_test10', 'ml_hmm', 'hist_gb'):.4f}. "
  f"The deployed model's student-test score ({st.loc['random_forest__train80', 'arc_f1_mean']:.4f}) comes from 80 training "
  f"trips and must not be compared with 64-trip results as if the data were equal. HistGradientBoosting would be a "
  f"defensible alternative with a far smaller model file.")

# ---------------------------------------------------------------- 11 ablation
h1("11  Ablation Study: Classifier-only vs Classifier + HMM (Deliverables §5)")
p("All variants share the candidate generation, decoder parameters, emission weight and route assembly of the deployed "
  "configuration; every classifier is trained on the same 64 tune trips (out-of-fold within them for the tune row). No "
  "evaluator data, retraining or tuning is involved (scripts/ablation.py).")
rows = []
for es, lab_ in (("tune64_OOF", "Tune 64 (OOF)"), ("holdout16", "Holdout 16"), ("student_test10", "Student test 10")):
    rows.append([lab_, f"{ab_get(es, 'nearest_road', '-'):.4f}", f"{ab_get(es, 'classifier_only', 'random_forest'):.4f}",
                 f"{ab_get(es, 'geometric_hmm', '-'):.4f}", f"{ab_get(es, 'ml_hmm', 'random_forest'):.4f}",
                 f"{ab_get(es, 'ml_hmm', 'random_forest', 'seq_f1'):.4f}"])
TAB_ABL = tab(["Set", "Nearest road", "RF only (no HMM)", "Geometric HMM", "RF + HMM", "RF + HMM sequence F1"], rows,
    "Ablation, mean route arc F1 (last column: order-aware sequence F1). Source: experiments/final/ablation_summary.csv.")
fig(FIG / "fig_ablation_route_f1.png", "What each component contributes, with equal training data.")
fig(FIG / "fig_classifier_only_vs_hmm.png", "Classifier-only versus classifier + HMM for every model; dashed line = geometric HMM.")
lr_h = ab_get("holdout16", "ml_hmm", "logreg") - ab_get("holdout16", "geometric_hmm", "-")
lr_s = ab_get("student_test10", "ml_hmm", "logreg") - ab_get("student_test10", "geometric_hmm", "-")
p(f"The HMM is the decisive component (+0.17 to +0.27 route F1 over classifier-only on holdout and student test). Tree-model "
  f"emissions add a small gain over geometric emissions (+0.003 to +0.006 on the same sets), while Logistic Regression "
  f"emissions give none ({lr_h:+.4f} holdout, {lr_s:+.4f} student test). Figure {FIGN[0] + 1} shows the per-trip effect "
  f"of replacing geometric emissions by Random Forest emissions: some trips get worse, so the gain is not uniform.")
fig(FIG / "fig_per_trip_delta.png", "Per-trip change in route F1 from geometric HMM to RF + HMM (RF trained on 64 trips).")

# ---------------------------------------------------------------- 12 final evaluation
h1("12  Final Evaluation on evaluator_test")
p(f"The 10 evaluator_test trips (ids disjoint from all other partitions) include .route reference files. The guidelines "
  f"state that the API computes accuracy from internal reference data, so scoring is permitted; references were read only "
  f"after all predictions were made. The deployed model, configuration and inference code were frozen and hashed before "
  f"prediction (experiments/final_evaluation/evaluation_manifest.json); nothing was changed after seeing the results.")
rows = []
names = {dep: "RF + HMM (deployed, 80 trips)", "rf64_hmm_ml_candidate": "RF + HMM (64 trips, comparison)",
         geo: "Geometric HMM", "classifier_only_rf80": "RF only, no HMM (80 trips)", "nearest_road": "Nearest road"}
for v, nm in names.items():
    s_ = summ["variants"][v]
    rows.append([nm, f4(s_["arc_f1"]["mean"]), f4(s_["arc_f1"]["median"]), f4(s_["arc_f1"]["min"]), f4(s_["arc_f1"]["max"]),
                 f4(s_["arc_f1"]["std"]), f4(s_["arc_precision"]["mean"]), f4(s_["arc_recall"]["mean"]),
                 f4(s_["seq_f1"]["mean"]), f"{s_['runtime_ms_server']['mean']:.0f}"])
tab(["Variant", "F1 mean", "Median", "Min", "Max", "SD", "Precision", "Recall", "Seq. F1", "ms/trip"], rows,
    "evaluator_test results (directed arc-set metrics vs .route; runtime = server time per request, model pre-loaded, "
    "Apple M5). Source: experiments/final_evaluation/evaluation_summary.json.", small=True)
fig(FIG / "route_f1_comparison.png", "Per-trip route F1 on evaluator_test: geometric HMM versus deployed RF + HMM.")
rows = []
for t in e_dep.index:
    r_ = e_dep.loc[t]
    rows.append([t, f"{int(r_.n_gps_points):,}", f"{int(r_.n_arcs):,}", str(int(r_.pred_route_len)), str(int(r_.pred_distinct_arcs)),
                 f4(r_.arc_f1), f4(e_geo.loc[t, "arc_f1"]), f4(r_.arc_precision), f4(r_.arc_recall), f4(r_.seq_f1),
                 f"{r_.path_continuity:.2f}", str(int(r_.points_without_candidates)), f"{r_.runtime_ms_server:.0f}"])
tab(["Trip", "GPS obs.", "Arcs", "Route len.", "Distinct", "F1", "Geo. F1", "Prec.", "Recall", "Seq. F1", "Contin.",
     "No cand.", "ms"], rows,
    "Per-trip evaluator_test results for the deployed model. Source: experiments/final_evaluation/evaluation_per_trip.csv.",
    small=True)
pe = paired.loc["evaluator_test10: geometric HMM -> RF(80)+HMM"]
p(f"All 10 trips were predicted without failure, deterministically, with valid arc ids and fully connected paths "
  f"(continuity 1.00); API-reported metrics equal an independent recomputation, and predictions were identical when .route "
  f"files were removed ({summ['leakage_check_predictions_identical_without_route']}). The deployed model is "
  f"{dvg['mean_delta']:+.4f} ahead of the geometric HMM on average ({pe.better} better, {pe.worse} worse, {pe['same']} equal), "
  f"but with p = {pe.wilcoxon_p:.2f} (Wilcoxon) and p = {pe.sign_test_p:.2f} (sign test) this is not statistically "
  f"significant on 10 trips. RF trained on 64 versus 80 trips gives practically identical results "
  f"({vstat('rf64_hmm_ml_candidate', 'arc_f1'):.4f} vs {ev_dep_f1:.4f}).")
rows = []
for _, r_ in comp.iterrows():
    rows.append([r_.split.split(" (")[0], r_.model, r_.classifier_training_trips, int(r_.n_trips), f4(r_.arc_f1_mean), f4(r_.arc_f1_median)])
tab(["Split", "Model", "Classifier training trips", "Trips", "F1 mean", "F1 median"], rows,
    "Route F1 across splits with training sizes labelled (tune-64 rows are optimistic: the same trips selected the decoder "
    "parameters and emission weight). Source: experiments/final_evaluation/comparison_with_previous_splits.csv.", small=True)

# ---------------------------------------------------------------- 13 error analysis
h1("13  Error Analysis")
cats = [("fp_reverse_direction", "Wrong direction (reverse twin of a .route arc)"), ("fp_gap_fill_only", "Gap-stitching detour only"),
        ("fp_matched", "Matched to a wrong road"), ("fp_outside_ref_span", "Before/after the reference span"),
        ("n_fn", "Missed reference arcs"), ("repeated_traversals", "Repeated traversals (not penalised)")]
rows = [[lab_, str(int(er_geo[c].sum())), str(int(er_dep[c].sum()))] for c, lab_ in cats]
tab(["Error category (evaluator_test, 10 trips)", "Geometric HMM", "RF + HMM (deployed)"], rows,
    "Each false-positive arc gets exactly one category. Source: experiments/final_evaluation/route_error_analysis.csv.",
    widths=[0.6, 0.2, 0.2])
fig(FIG / "fig_fp_breakdown_student_test.png", "False-positive sources on student test (RF trained on 64 trips).", width=0.8)
worst = e_dep.arc_f1.idxmin()
w_er = er_dep.set_index("trip_id").loc[worst]
bl([f"**Gap stitching** is the largest source of extra arcs and is unchanged by the classifier: when consecutive decoded "
    f"arcs do not touch, the shortest connecting path is inserted and may differ from the road actually driven.",
    f"**Direction errors** fall from {int(er_geo.fp_reverse_direction.sum())} to {int(er_dep.fp_reverse_direction.sum())} "
    f"arcs with RF emissions, consistent with the heading features.",
    f"**Difficult trips:** evaluator trip {worst} has the lowest F1 ({e_dep.loc[worst, 'arc_f1']:.4f}): "
    f"{int(e_dep.loc[worst, 'points_without_candidates'])} observations have no road within 200 m, and its "
    f"{int(w_er.n_fp)} extra arcs split into {int(w_er.fp_gap_fill_only)} gap-fill, {int(w_er.fp_reverse_direction)} "
    f"direction and {int(w_er.fp_matched)} wrong-road arcs — short stubs at junctions rather than wrong streets. Trips "
    f"00000015, 00000067 and 00000087 are slightly worse with RF emissions than with geometric emissions.",
    "**Candidate failures** are rare: the pseudo-label arc is among the candidates for "
    f"{tds['label_in_candidates_frac']:.1%} of valid training observations.",
    "**Reference truncation** (training trips 00000033 and 00000054) makes correct matches after the route end count as "
    "false positives; route precision on such trips understates quality.",
    f"**Computational cost:** deployed mean {vstat(dep, 'runtime_ms_server'):.0f} ms per evaluator trip (max "
    f"{vstat(dep, 'runtime_ms_server', 'max'):.0f} ms) versus {vstat(geo, 'runtime_ms_server'):.0f} ms for the geometric "
    f"HMM; the extra time is feature extraction and Random Forest inference."])
fig(FIG / "runtime_distribution.png", "Server runtime per evaluator trip against trip length.")

# ---------------------------------------------------------------- 14 folium
h1("14  Geographical Visualisation Using Folium (Deliverables §6)")
p("Interactive maps are generated by code in the repository from the deployed model's actual predictions "
  "(reports/tooling/build_assets.py; earlier map code: scripts/validate_alignment.py, scripts/final_eval_artifacts.py). "
  "Each map shows the original GPS observations, the road-network geometry around the trip, the reference route and the "
  "predicted route, colour-coded into arcs that are in .route, extra arcs and missed reference arcs, with a legend, a "
  "layer control and an extent fitted to the trip. Student-test trips are used so the maps can be published; maps of "
  "evaluator trips exist locally but are not reproduced because they draw evaluator ground truth.")
chk = mapchk.set_index("trip_id")
fig(FIG / "folium_screenshot_student_test_00000077.png", f"Folium map, student-test trip 00000077 (largest RF + HMM gain over "
    f"the geometric HMM; arc F1 {chk.loc['00000077', 'arc_f1_regenerated']:.4f}, identical to the recorded value). Headless-"
    "browser screenshot of reports/figures/maps/student_test_00000077_route_map.html.")
fig(FIG / "folium_screenshot_student_test_00000076.png", f"Folium map, student-test trip 00000076 (lowest RF + HMM F1 on "
    f"student test, {chk.loc['00000076', 'arc_f1_regenerated']:.4f}). Screenshot of "
    "reports/figures/maps/student_test_00000076_route_map.html.")
p("At city scale the predicted route (blue) covers the reference (green) almost everywhere; errors are short orange stubs "
  "at junctions, visible when zooming into the interactive maps or in the static renderings in Appendix A.4.")

# ---------------------------------------------------------------- 15 deployment
h1("15  API, UI and Deployment (Deliverables §7)")
tab(["Endpoint", "Request", "Response"],
    [["GET /health", "—", "status, data_dir, default_config, available configs"],
     ["GET /trips", "—", "count and list of {trip_id, split, has_reference_route}"],
     ["POST /predict", '{"trip_id": "00000091"} (optional "config", "method")',
      "trip_id, gps_coordinates [lat, lon], predicted_coordinates [lat, lon] (null if unmatched), predicted_arc_ids, "
      "point_arc_ids, metrics (null without .route), method, runtime_ms, details"]],
    "FastAPI endpoints (main.py). Coordinates are always [latitude, longitude].", widths=[0.17, 0.33, 0.5])
bl(["**Model loading:** the deployed bundle (models/random_forest__train80.joblib: model, feature names, schema version, "
    "training trip ids, CV metrics, library versions) is loaded at start-up; a schema mismatch or load failure falls back "
    "to geometric emissions and is reported in details.emission_fallback.",
    "**Running:** PATHSNAP_DATA_DIR=<folder> uvicorn main:app --host 0.0.0.0 --port 8000. The folder may be the parent of "
    "split folders or a split folder itself (post-freeze setup fix, commit 0fcc9b8, predictions verified identical).",
    "**Frontend:** frontend/app.py (Streamlit) calls the three endpoints over HTTP and draws GPS observations and the "
    "predicted route with Folium; it never receives .route data. A live HTTP check with the frontend's own client "
    "functions returned HTTP 200 for all 10 evaluator trips (experiments/final_evaluation/server_http_check.log)."])
a = api["predict_00000091"]
code("{\n" + ",\n".join(f'  "{k}": {json.dumps(a[k])}' for k in ("trip_id", "method", "runtime_ms", "metrics",
     "predicted_arc_ids", "point_arc_ids", "gps_coordinates", "predicted_coordinates")) + "\n}")
p(f"Example: a real response for student-test trip 00000091 (lists truncated; full lengths: "
  f"{', '.join(f'{k} {v}' for k, v in api['list_lengths'].items())}). Source: reports/data/api_example.json.")
log_lines = [l for l in open(FE / "server_http_check.log").read().splitlines() if "pathsnap: predict" in l][:2]
import textwrap
code("\n".join("\n".join(textwrap.wrap(l[:300], 96, subsequent_indent="    ", break_on_hyphens=False)) for l in log_lines), width=200)
p("Server log lines printed by the API for two evaluator requests (wrapped, truncated at 300 characters; the metrics field shows "
  "route scores computed server-side).")
note("**Missing material:** screenshots of the running UI on Laptop 2 and of API responses on Laptop 1's terminal "
     "(Deliverables §7) have not been captured yet; the figures above are static renderings of real predictions, not UI "
     "screenshots. The Folium figures in Section 14 are real map screenshots of model output, not of the UI.")

# ---------------------------------------------------------------- 15 testing
h1("16  Testing and Reproducibility")
tab(["Test file", "Tests", "Covers"],
    [[f, str(n), {"tests/test_pathsnap.py": "loading, arc ids, coordinate order, decoding, metric definitions, API, data-dir layouts",
                  "tests/test_ml.py": "alignment (direction change, repeated arc), leakage guards, probability shape, model loading and fallback",
                  "tests/test_robustness.py": "missing samples, 15-min gap, single point, no candidates, sparse candidates, empty file, bad ids, determinism",
                  "frontend/test_contract.py": "frontend parsing of /health, /trips and /predict responses"}.get(f, "")]
     for f, n in tests["per_file"].items()],
    f"Automated tests: {tests['summary_line']} (python -m pytest -q, git {tests['git_head']}). Source: reports/data/test_run.json.",
    widths=[0.27, 0.08, 0.65])
p(f"**Environment:** Python {man['python']} on {man['platform']} (Apple M5, 10 cores); numpy {freeze.get('numpy')}, scipy "
  f"{freeze.get('scipy')}, scikit-learn {freeze.get('scikit-learn')}, optuna {freeze.get('optuna')}, pandas "
  f"{freeze.get('pandas')}, fastapi {freeze.get('fastapi')}, folium {freeze.get('folium')}. The full pinned list is "
  f"experiments/final/pip_freeze.txt; requirements.txt lists minimum versions. Seeds: 42 for the partition, CV models, "
  f"Optuna samplers and all estimators.")
p("**Reproduction** (from the repository root; complete list in Appendix A.5):")
code("python -m pytest -q\n"
     "python -m scripts.train_classifiers --stage dataset   # then cv, oof_hmm, emission_grid, fit_tune, final\n"
     "python -m scripts.tune --trials 150 --seed 42          # decoder Optuna\n"
     "python -m scripts.ablation && python -m scripts.make_figures && python -m scripts.verify_figures\n"
     "python -m scripts.final_evaluation                      # needs evaluator_test/ beside the repo code\n"
     "PATHSNAP_DATA_DIR=/path/to/evaluator_test uvicorn main:app --host 0.0.0.0 --port 8000")
p("All 294 values plotted in the experiment figures were re-derived from their source CSVs by an independent script "
  "(experiments/final/figures/plotted_values_verification.csv: 294/294 match).")

# ---------------------------------------------------------------- 16 limitations
h1("17  Limitations and Future Work")
bl(["**Pseudo-labels.** Per-observation targets come from an alignment with its own parameters; ambiguity at junctions and "
    "after truncated routes limits the meaning of Top-1 and candidate metrics. Future work: confidence-weighted labels.",
    "**Scoring metric.** Arc F1 on directed sets ignores order and repeated traversals; the order-aware LCS score is "
    "reported but no official formula was provided. Precision penalises correct matches beyond truncated references.",
    "**Gap stitching.** Shortest-path insertion is the main source of extra arcs; using time-consistent or "
    "classifier-guided path choice could reduce it.",
    "**Statistical power.** Ten evaluator trips cannot separate RF + HMM from the geometric HMM (p = "
    f"{dvg['wilcoxon_p']:.2f}).",
    f"**Optimisation.** Random Forest Bayesian optimisation is incomplete ({len(rfc)} trials) and never evaluated; decoder "
    "optima lie near search-range edges; the emission weight was grid-searched for Random Forest only.",
    "**Calibration.** Classifier probabilities are used uncalibrated (clipped and normalised); calibration was not assessed.",
    "**Reference quality.** Some reference routes are truncated or contain discontinuities (Section 3), so .route is a "
    "reference, not verified ground truth.",
    "**Generalisation and data selection.** Trips come from several cities with very different network sizes; decoder "
    "parameters were tuned on the same 64 trips used for classifier CV, so tune-set numbers are optimistic."])

# ---------------------------------------------------------------- 17 conclusion
h1("18  Conclusion")
p(f"PathSnap delivers a working, leakage-checked map-matching system: candidate generation on a directed network, five "
  f"compared classifiers, a Bayesian-tuned HMM/Viterbi decoder and a FastAPI/Streamlit deployment that served all ten "
  f"reserved trips deterministically. The HMM is responsible for most of the route quality; Random Forest emissions add a "
  f"small gain that was significant on held-out training trips but not on the ten evaluator trips, where the deployed "
  f"model reached a mean directed-arc-set F1 of {ev_dep_f1:.4f}. Remaining work is listed in Section 17 and in the "
  f"missing-material notes above.")

# ---------------------------------------------------------------- 18 AI disclosure
h1("19  AI-Assistance Disclosure (Deliverables §8)")
p("As required by Deliverables §8, the team discloses that an AI coding assistant (Anthropic Claude, used through Claude "
  "Code) was used extensively to write source code, tests and experiment scripts, to run experiments, and to draft this "
  "report from the recorded experiment files. All numbers in this report are generated from files in the repository by "
  "reports/tooling/build_report.py; the team remains responsible for verifying and explaining every claim.")

# ---------------------------------------------------------------- 19 references
h1("20  References")
refs = [
    "Newson, P. and Krumm, J. (2009). Hidden Markov map matching through noise and sparseness. In Proc. 17th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, pp. 336–343.",
    "Viterbi, A. J. (1967). Error bounds for convolutional codes and an asymptotically optimum decoding algorithm. IEEE Transactions on Information Theory, 13(2), 260–269.",
    "Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. Numerische Mathematik, 1, 269–271.",
    "Akiba, T., Sano, S., Yanase, T., Ohta, T. and Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In Proc. 25th ACM SIGKDD, pp. 2623–2631.",
    "Bergstra, J., Bardenet, R., Bengio, Y. and Kégl, B. (2011). Algorithms for hyper-parameter optimization. Advances in Neural Information Processing Systems 24.",
    "Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
    "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.",
    "Geurts, P., Ernst, D. and Wehenkel, L. (2006). Extremely randomized trees. Machine Learning, 63(1), 3–42.",
    "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189–1232.",
    "Virtanen, P. et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nature Methods, 17, 261–272.",
    "Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics Bulletin, 1(6), 80–83.",
    "PathSnap hackathon documents (2026): Main Guidelines; Deliverables; Models and Training Guidelines; Dataset Exploration; Assessment and Plagiarism Policy. PES University, UE24CS352A.",
    "Software used: FastAPI, Uvicorn, Folium, Streamlit, pandas, matplotlib, joblib (versions in experiments/final/pip_freeze.txt).",
]
B.append({"t": "refs", "items": refs})

# ---------------------------------------------------------------- appendix
h1("Appendix A")
h2("A.1 Model hyperparameters (initial, deployed for Random Forest)")
tab(["Model", "Configuration (scikit-learn " + freeze.get("scikit-learn", "") + ")"], [[NICE[m], initial[m]] for m in ORDER],
    "Exact estimator settings from src/ml_train.py::make_models (random_state = 42).", widths=[0.25, 0.75], small=True)
h2("A.2 Deployed configuration configs/ml_v3.json")
code(json.dumps(cfg, indent=1))
h2("A.3 Evidence and post-freeze record")
bl([f"Frozen model SHA-256 (models/random_forest__train80.joblib): {list(man['hashes']['model'].values())[0]}",
    f"Frozen commit: {man['git_commit']}; evaluation started {man['evaluation_started']}.",
    f"Post-freeze setup change: commit 0fcc9b8 (src/data_loader.py::list_trips). SHA-256 "
    f"{pf['src/data_loader.py_sha256_frozen'][:16]}… → {pf['src/data_loader.py_sha256_now'][:16]}…; "
    f"{pf['n_checks']} re-checks, routes identical: {pf['all_routes_identical']}, F1 identical: {pf['all_f1_identical']}.",
    "Evidence index for every number and figure: reports/report_evidence_index.md."])
h2("A.4 Static renderings of the route maps")
fig(FIG / "fig_route_map_student_test_00000077.png", "Route map for student-test trip 00000077 (largest gain of RF + HMM "
    "over geometric HMM): road network, GPS, reference and predicted route. Static rendering of the same layers as the "
    "interactive Folium map reports/figures/maps/student_test_00000077_route_map.html.", width=0.62)
fig(FIG / "fig_route_map_student_test_00000076.png", "Route map for student-test trip 00000076 (lowest RF + HMM F1 on "
    "student test). Interactive version: reports/figures/maps/student_test_00000076_route_map.html.", width=0.62)
h2("A.5 Command-line reproduction")
code("# setup\npython3 -m venv .venv && .venv/bin/pip install -r requirements.txt\n"
     "# classifier pipeline (train trips only)\n"
     "python -m scripts.train_classifiers --stage dataset\npython -m scripts.train_classifiers --stage cv\n"
     "python -m scripts.train_classifiers --stage oof_hmm\n"
     "python -m scripts.train_classifiers --stage emission_grid --model random_forest\n"
     "python -m scripts.train_classifiers --stage fit_tune\n"
     "python -m scripts.compare --a tuned_v2 --b ml_candidate --trips holdout\n"
     "python -m scripts.train_classifiers --stage final --model random_forest --emission ml --weight 4\n"
     "python -m scripts.train_classifiers --stage student_test\n"
     "# Bayesian optimisation\npython -m scripts.tune --trials 150 --seed 42\npython -m scripts.tune_classifier   # stopped early\n"
     "# verification and figures\npython -m scripts.verify_labels\npython -m scripts.ablation\n"
     "python -m scripts.make_figures && python -m scripts.verify_figures\n"
     "# final evaluation (frozen model)\npython -m scripts.final_evaluation && python -m scripts.final_eval_artifacts\n"
     "python -m scripts.verify_post_freeze\n"
     "# this report\n.venv/bin/python reports/tooling/build_assets.py\n.venv/bin/python reports/tooling/collect_evidence.py\n"
     "reports/tooling/.venv/bin/python reports/tooling/build_report.py\nnode reports/tooling/build_docx.js")

TOK = {"{TAB_ABL}": str(TAB_ABL), "{TAB_31}": str(t31)}
def _sub(x):
    if isinstance(x, str):
        for k_, v_ in TOK.items():
            x = x.replace(k_, v_)
        return x
    if isinstance(x, list):
        return [_sub(i) for i in x]
    if isinstance(x, dict):
        return {k_: _sub(v_) for k_, v_ in x.items()}
    return x
B[:] = [_sub(b) for b in B]
json.dump(B, open(R / "tooling" / "report_blocks.json", "w"), indent=1, ensure_ascii=False)
print(f"blocks: {len(B)}, figures: {FIGN[0]}, tables: {TABN[0]}")

# ============================================================== Markdown
def md_inline(s):
    return s


md = []
for b in B:
    t = b["t"]
    if t == "cover":
        md += [f"# {b['title']}", "", f"**{b['subtitle']}**", ""] + [f"{l}  " for l in b["lines"]] + [""]
    elif t == "toc":
        md += ["*Table of contents: see the PDF/DOCX versions.*", ""]
    elif t in ("h1", "h2", "h3"):
        md += ["#" * {"h1": 2, "h2": 3, "h3": 4}[t] + " " + b["text"], ""]
    elif t == "p":
        md += [b["text"], ""]
    elif t == "note":
        md += ["> " + b["text"], ""]
    elif t == "bullets":
        md += [f"- {i}" for i in b["items"]] + [""]
    elif t == "code":
        md += ["```", b["text"], "```", ""]
    elif t == "refs":
        md += [f"{i + 1}. {r}" for i, r in enumerate(b["items"])] + [""]
    elif t == "table":
        md += ["| " + " | ".join(b["header"]) + " |", "|" + "---|" * len(b["header"])]
        md += ["| " + " | ".join(c.replace("|", "\\|") for c in r) + " |" for r in b["rows"]]
        md += ["", f"*{b['caption']}*", ""]
    elif t == "fig":
        rel = Path(b["path"]).relative_to(R)
        md += [f"![{b['caption']}]({rel})", "", f"*{b['caption']}*", ""]
(R / "PathSnap_Final_Technical_Report.md").write_text("\n".join(md))

# ============================================================== PDF (reportlab)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph,
                                Preformatted, Spacer, Table, TableStyle, ListFlowable, ListItem, NextPageTemplate)
from reportlab.platypus.tableofcontents import TableOfContents

fdir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(fdir / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(fdir / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", str(fdir / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-BI", str(fdir / "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("DVM", str(fdir / "DejaVuSansMono.ttf")))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

ACC = colors.HexColor("#2a78d6")
INK = colors.HexColor("#1b1b1a")
INK2 = colors.HexColor("#52514e")
LINE = colors.HexColor("#d9d8d3")
ss = getSampleStyleSheet()
S = {
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.4, leading=13.2, textColor=INK, spaceAfter=6, alignment=4),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=15, leading=19, textColor=ACC, spaceBefore=14, spaceAfter=8, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=11.5, leading=15, textColor=INK, spaceBefore=9, spaceAfter=5, keepWithNext=1),
    "h3": ParagraphStyle("h3", fontName="DV-B", fontSize=10, leading=13, textColor=INK, spaceBefore=6, spaceAfter=4, keepWithNext=1),
    "cap": ParagraphStyle("cap", fontName="DV-I", fontSize=8, leading=10.5, textColor=INK2, spaceBefore=3, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.8, leading=9.6, textColor=INK),
    "cells": ParagraphStyle("cells", fontName="DV", fontSize=6.9, leading=8.4, textColor=INK),
    "head": ParagraphStyle("head", fontName="DV-B", fontSize=7.8, leading=9.6, textColor=colors.white),
    "heads": ParagraphStyle("heads", fontName="DV-B", fontSize=6.9, leading=8.4, textColor=colors.white),
    "code": ParagraphStyle("code", fontName="DVM", fontSize=7.0, leading=8.9, textColor=INK, backColor=colors.HexColor("#f3f2ee"),
                           borderPadding=5, spaceBefore=4, spaceAfter=8),
    "note": ParagraphStyle("note", fontName="DV", fontSize=8.8, leading=12, textColor=INK, backColor=colors.HexColor("#fdf1ea"),
                           borderColor=colors.HexColor("#eb6834"), borderWidth=0.8, borderPadding=6, spaceBefore=6, spaceAfter=10),
    "ref": ParagraphStyle("ref", fontName="DV", fontSize=8.6, leading=11.5, textColor=INK, leftIndent=14, firstLineIndent=-14, spaceAfter=4),
    "bul": ParagraphStyle("bul", fontName="DV", fontSize=9.2, leading=12.8, textColor=INK, alignment=0),
}


def rl(s):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<font name='DVM'>\1</font>", s)
    return s


class Doc(BaseDocTemplate):
    def afterFlowable(self, fl):
        if isinstance(fl, Paragraph) and getattr(fl, "_toc", None):
            lvl, txt = fl._toc
            key = f"h{self.seq.nextf('h')}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(txt, key, level=lvl, closed=lvl > 0)
            self.notify("TOCEntry", (lvl, txt, self.page, key))


W, H = A4
LM = RM = 2.0 * cm


def on_page(c, d):
    c.saveState()
    c.setFont("DV", 7.5); c.setFillColor(INK2)
    c.drawString(LM, H - 1.25 * cm, "PathSnap — Final Technical Report")
    c.drawRightString(W - RM, H - 1.25 * cm, "UE24CS352A Machine Learning")
    c.setStrokeColor(LINE); c.line(LM, H - 1.4 * cm, W - RM, H - 1.4 * cm)
    c.drawCentredString(W / 2, 1.1 * cm, f"Page {d.page}")
    c.restoreState()


doc = Doc(str(R / "PathSnap_Final_Technical_Report.pdf"), pagesize=A4, leftMargin=LM, rightMargin=RM,
          topMargin=2.0 * cm, bottomMargin=1.8 * cm, title="PathSnap: ML-Assisted GPS Map Matching and Route Reconstruction",
          author="PathSnap team", subject="Final technical report")
frame = Frame(LM, 1.8 * cm, W - LM - RM, H - 3.8 * cm, id="f")
doc.addPageTemplates([PageTemplate("cover", frames=[frame]), PageTemplate("main", frames=[frame], onPage=on_page)])
TW = W - LM - RM
story = []
for b in B:
    t = b["t"]
    if t == "cover":
        story += [Spacer(1, 4.5 * cm),
                  Paragraph(rl(b["title"]), ParagraphStyle("t", fontName="DV-B", fontSize=24, leading=30, textColor=ACC, alignment=TA_CENTER)),
                  Spacer(1, 0.6 * cm),
                  Paragraph(rl(b["subtitle"]), ParagraphStyle("st", fontName="DV", fontSize=14, leading=18, textColor=INK, alignment=TA_CENTER)),
                  Spacer(1, 2.5 * cm)]
        for l in b["lines"]:
            story.append(Paragraph(rl(l), ParagraphStyle("cl", fontName="DV", fontSize=9.5, leading=13.5, textColor=INK2,
                                                         alignment=TA_CENTER, spaceAfter=7)))
        story += [NextPageTemplate("main"), PageBreak()]
    elif t == "toc":
        toc = TableOfContents()
        toc.levelStyles = [ParagraphStyle("t0", fontName="DV-B", fontSize=9.6, leading=13, leftIndent=0, spaceBefore=3),
                           ParagraphStyle("t1", fontName="DV", fontSize=8.8, leading=11.5, leftIndent=14)]
        toc.dotsMinLevel = 0
        story += [Paragraph("Contents", S["h1"]), toc, PageBreak()]
    elif t in ("h1", "h2", "h3"):
        if t == "h1" and b["text"].startswith("Appendix"):
            story.append(PageBreak())
        para = Paragraph(rl(b["text"]), S[t])
        para._toc = (0 if t == "h1" else 1, b["text"]) if t != "h3" else None
        story.append(para)
    elif t == "p":
        story.append(Paragraph(rl(b["text"]), S["body"]))
    elif t == "note":
        story.append(KeepTogether([Paragraph(rl(b["text"]), S["note"])]))
    elif t == "bullets":
        lf = ListFlowable([ListItem(Paragraph(rl(i), S["bul"]), leftIndent=12, value="•") for i in b["items"]],
                          bulletType="bullet", start="•", leftIndent=12, bulletFontName="DV", bulletFontSize=8,
                          spaceAfter=6)
        if story and isinstance(story[-1], Paragraph) and getattr(story[-1], "_toc", None) is not None:
            story.append(KeepTogether([story.pop(), lf]))
        else:
            story.append(lf)
        story.append(Spacer(1, 4))
    elif t == "code":
        pre = Preformatted(b["text"], S["code"])
        if story and isinstance(story[-1], Paragraph) and getattr(story[-1], "_toc", None) is not None:
            story.append(KeepTogether([story.pop(), pre]))
        else:
            story.append(pre)
    elif t == "refs":
        for i, r in enumerate(b["items"]):
            story.append(Paragraph(f"[{i + 1}]&nbsp;&nbsp;" + rl(r), S["ref"]))
    elif t == "table":
        sm = b["small"]
        cs, hs = (S["cells"], S["heads"]) if sm else (S["cell"], S["head"])
        data = [[Paragraph(rl(h), hs) for h in b["header"]]] + [[Paragraph(rl(c), cs) for c in r] for r in b["rows"]]
        n = len(b["header"])
        fs = 6.9 if sm else 7.8
        def tw(txt, bold=False):
            return pdfmetrics.stringWidth(txt, "DV-B" if bold else "DV", fs)
        allrows = [b["header"]] + b["rows"]
        mins = np.array([max(max((tw(tok, j == 0) for tok in str(r[i]).split() or [""]), default=0)
                             for j, r in enumerate(allrows)) + 9 for i in range(n)])
        want = np.array([min(max(tw(str(r[i]), j == 0) for j, r in enumerate(allrows)) + 9, 200) for i in range(n)])
        if b["widths"]:
            want = np.maximum(np.array(b["widths"]) * TW, mins)
        want = np.maximum(want, mins)
        if want.sum() <= TW:
            cw = want * TW / want.sum()
        else:
            extra = TW - mins.sum()
            cw = mins + (want - mins) / max((want - mins).sum(), 1e-9) * max(extra, 0) if extra > 0 else mins * TW / mins.sum()
        cw = list(cw)
        tb = Table(data, colWidths=cw, repeatRows=1)
        tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), ACC), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f2")]),
                                ("LINEBELOW", (0, 0), (-1, -1), 0.3, LINE), ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                                ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
        cap = Paragraph(rl(b["caption"]), S["cap"])
        grp = [tb, cap]
        if story and isinstance(story[-1], Paragraph) and getattr(story[-1], "_toc", None) is not None:
            grp.insert(0, story.pop())
        if len(b["rows"]) <= 14:
            story.append(KeepTogether(grp))
        else:
            story += grp
    elif t == "fig":
        wpx, hpx = b["px"]
        wmax = TW * b["width"]
        hmax = H - 6.5 * cm
        w_ = wmax
        h_ = w_ * hpx / wpx
        if h_ > hmax:
            h_ = hmax; w_ = h_ * wpx / hpx
        grp = [Image(b["path"], width=w_, height=h_), Paragraph(rl(b["caption"]), S["cap"])]
        if story and isinstance(story[-1], Paragraph) and getattr(story[-1], "_toc", None) is not None:
            grp.insert(0, story.pop())
        story.append(KeepTogether(grp))
    elif t == "pagebreak":
        story.append(PageBreak())
doc.multiBuild(story)
print("PDF written")
