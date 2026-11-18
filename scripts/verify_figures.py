"""Independently recompute every plotted value from its source CSV and compare.

  python -m scripts.verify_figures   (after scripts.make_figures)
Uses the csv/statistics stdlib (not the pandas code that drew the figures).
Writes experiments/final/figures/plotted_values_verification.csv.
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "experiments" / "final" / "figures"


def rows(path):
    with open(ROOT / path) as f:
        return list(csv.DictReader(f))


def expected(r: dict) -> float:
    fig, series, label, src = r["figure"], r["series"], r["label"], r["source"]
    if fig in ("fig_decoder_optuna_history.png", "fig_rf_optuna_history.png"):
        return float(next(x["value"] for x in rows(src) if int(x["number"]) == int(label)))
    if fig == "fig_cv_model_comparison.png":
        col = "macro_f1" if "macro" in series else "log_loss"
        vals = [float(x[col]) for x in rows(src) if x["model"] == label]
        return statistics.mean(vals) if series.endswith("mean") else statistics.stdev(vals)
    if fig == "fig_student_test_top1_macro_f1.png":
        col = "clf_pseudo_top1_accuracy" if series == "top1" else "clf_macro_f1"
        return float(next(x[col] for x in rows(src) if x["model"] == label))
    if fig in ("fig_ablation_route_f1.png", "fig_classifier_only_vs_hmm.png"):
        variant, model = label.split("|")
        return float(next(x["arc_f1"] for x in rows(src) if x["eval_set"] == series
                          and x["variant"] == variant and x["model"] == model))
    if fig == "fig_per_trip_delta.png":
        f1 = {(x["variant"], x["model"]): float(x["arc_f1"]) for x in rows(src)
              if x["eval_set"] == series and int(x["trip_id"]) == int(label)}
        return f1[("ml_hmm", "random_forest")] - f1[("geometric_hmm", "-")]
    if fig == "fig_fp_breakdown_student_test.png":
        variant, model = series.split("|")
        return float(sum(int(float(x[label])) for x in rows(src) if x["eval_set"] == "student_test10"
                         and x["variant"] == variant and x["model"] == model))
    if fig == "fig_confusion_matrices.png":
        return float(next(x[label] for x in rows(src) if x["model"] == series))
    if fig == "fig_train_vs_cv.png":
        return float(next(x[series] for x in rows(src) if x["model"] == label))
    raise KeyError(fig)


def main() -> None:
    out, by_fig = [], defaultdict(lambda: [0, 0])
    for r in rows("experiments/final/figures/plotted_values.csv"):
        e = expected(r)
        ok = abs(float(r["value"]) - e) < 1e-9
        out.append({**r, "expected_from_source": e, "match": ok})
        by_fig[r["figure"]][0] += ok
        by_fig[r["figure"]][1] += 1
    with open(FIG / "plotted_values_verification.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    for fig, (k, n) in sorted(by_fig.items()):
        print(f"  {fig:40s} {k}/{n} match")
    total = sum(k for k, _ in by_fig.values()), sum(n for _, n in by_fig.values())
    print(f"ALL: {total[0]}/{total[1]} plotted values match their source")


if __name__ == "__main__":
    main()
