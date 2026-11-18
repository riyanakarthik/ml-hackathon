"""Compare two configs trip-by-trip on a trip set.

  python -m scripts.compare --a baseline_v1 --b tuned_candidate --trips holdout
  python -m scripts.compare --a baseline_v1 --b tuned --trips student_test   # final, once

Writes experiments/compare_<a>_vs_<b>_<trips>.csv (per trip, both configs + deltas)
and appends a summary row to experiments/comparisons_summary.csv.
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from scripts.diagnose import FP_COLS, trip_ids_for
from src.config import get_config
from src.experiment import EXPERIMENT_DIR, evaluate_config, summarize

KEYS = ["arc_f1", "arc_precision", "arc_recall", "length_precision", "length_recall",
        "route_continuity", "obs_top1_accuracy", "obs_macro_f1", "candidate_miss_rate",
        "runtime_ms"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="baseline_v1")
    ap.add_argument("--b", default="tuned_candidate")
    ap.add_argument("--trips", default="holdout")
    args = ap.parse_args()
    ids = trip_ids_for(args.trips)
    ca, cb = get_config(args.a), get_config(args.b)
    ra = pd.DataFrame(evaluate_config(ca, ids)).set_index("trip_id").sort_index()
    rb = pd.DataFrame(evaluate_config(cb, ids)).set_index("trip_id").sort_index()
    # second pass for runtime only: first pass includes per-worker cache warm-up
    ra["runtime_ms"] = pd.DataFrame(evaluate_config(ca, ids)).set_index("trip_id").sort_index()["runtime_ms"]
    rb["runtime_ms"] = pd.DataFrame(evaluate_config(cb, ids)).set_index("trip_id").sort_index()["runtime_ms"]

    cols = KEYS + ["n_fp"] + FP_COLS + ["repeated_traversals", "n_fn"]
    per = pd.concat({args.a: ra[cols], args.b: rb[cols]}, axis=1)
    per.columns = [f"{c}__{n}" for n, c in per.columns]
    for c in KEYS:
        per[f"{c}__delta"] = rb[c] - ra[c]
    out = EXPERIMENT_DIR / f"compare_{args.a}_vs_{args.b}_{args.trips}.csv"
    per.to_csv(out)

    sa, sb = summarize(ra.reset_index().to_dict("records")), summarize(rb.reset_index().to_dict("records"))
    print(f"trips={args.trips} (n={len(ids)})  -> {out}\n")
    print(f"{'metric':22s} {args.a + ' mean':>18s} {args.b + ' mean':>22s} "
          f"{args.a + ' med':>16s} {args.b + ' med':>20s}")
    for k in KEYS:
        print(f"{k:22s} {sa[k + '_mean']:18.4f} {sb[k + '_mean']:22.4f} "
              f"{sa[k + '_median']:16.4f} {sb[k + '_median']:20.4f}")
    d = per["arc_f1__delta"]
    print(f"\nper-trip arc_f1 delta: improved {int((d > 1e-9).sum())}, worse {int((d < -1e-9).sum())}, "
          f"same {int((d.abs() <= 1e-9).sum())}; min {d.min():+.4f} max {d.max():+.4f}")
    print("\nper trip (arc_f1 a -> b, delta, precision a -> b, recall a -> b):")
    for t in per.index:
        print(f"  {t}  {ra.at[t, 'arc_f1']:.4f} -> {rb.at[t, 'arc_f1']:.4f} ({d[t]:+.4f})   "
              f"P {ra.at[t, 'arc_precision']:.3f} -> {rb.at[t, 'arc_precision']:.3f}   "
              f"R {ra.at[t, 'arc_recall']:.3f} -> {rb.at[t, 'arc_recall']:.3f}")
    print("\nFP breakdown totals (a -> b):")
    for c in FP_COLS + ["repeated_traversals", "n_fn"]:
        print(f"  {c:24s} {int(ra[c].sum()):6d} -> {int(rb[c].sum()):6d}")

    summ = EXPERIMENT_DIR / "comparisons_summary.csv"
    rec = []
    for name, s in ((args.a, sa), (args.b, sb)):
        rec.append({"timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                    "trips": args.trips, "config": name, **s})
    pd.DataFrame(rec).to_csv(summ, mode="a", header=not summ.exists(), index=False)


if __name__ == "__main__":
    main()
