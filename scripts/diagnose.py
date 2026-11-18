"""Per-trip metrics + error diagnostics for one config on a set of trips.

  python -m scripts.diagnose --config baseline_v1 --trips holdout
  (--trips: tune | holdout | train_all | student_test)
Writes experiments/diagnostics_<config>_<trips>.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.config import get_config
from src.data_loader import list_trips
from src.experiment import EXPERIMENT_DIR, evaluate_config, load_partition, summarize

FP_COLS = ["fp_reverse_direction", "fp_outside_ref_span", "fp_gap_fill_only", "fp_matched"]


def trip_ids_for(which: str) -> list[str]:
    part = load_partition()
    if which in ("tune", "holdout"):
        return part[which]
    if which == "train_all":
        return part["tune"] + part["holdout"]
    if which == "student_test":
        return sorted(t["trip_id"] for t in list_trips() if t["split"] == "student_test")
    raise SystemExit(f"unknown trip set {which}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baseline_v1")
    ap.add_argument("--trips", default="train_all")
    ap.add_argument("--method", default="hmm")
    args = ap.parse_args()

    cfg = get_config(args.config)
    rows = evaluate_config(cfg, trip_ids_for(args.trips), args.method)
    df = pd.DataFrame(rows).sort_values("trip_id")
    EXPERIMENT_DIR.mkdir(exist_ok=True)
    out = EXPERIMENT_DIR / f"diagnostics_{args.config}_{args.method}_{args.trips}.csv"
    df.to_csv(out, index=False)

    print(f"config={args.config} method={args.method} trips={args.trips} -> {out}")
    for k, v in summarize(rows).items():
        print(f"  {k:28s} {v}")
    tot = df[["n_fp"] + FP_COLS + ["repeated_traversals", "n_fn", "fn_never_candidate",
                                    "fn_candidate_not_chosen"]].sum()
    print("\nfalse positives (distinct arcs, summed over trips):")
    for c in FP_COLS:
        print(f"  {c:24s} {int(tot[c]):5d} arcs  {df[c + '_len_m'].sum():9.0f} m"
              f"  ({tot[c] / max(tot['n_fp'], 1):.1%})")
    print(f"  {'total':24s} {int(tot['n_fp']):5d}")
    print(f"repeated traversals (not penalised by set metric): {int(tot['repeated_traversals'])}")
    print(f"false negatives: {int(tot['n_fn'])}  never a candidate: {int(tot['fn_never_candidate'])}"
          f"  candidate but not chosen: {int(tot['fn_candidate_not_chosen'])}")


if __name__ == "__main__":
    main()
