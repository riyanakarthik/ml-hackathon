"""Equal-data ablation (see src/ablation.py for what is held fixed).

  python -m scripts.ablation
Writes experiments/final/ablation_per_trip.csv and ablation_summary.csv.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.ablation import MODELS, run_variant
from src.data_loader import list_trips
from src.experiment import EXPERIMENT_DIR, load_partition

OUT = EXPERIMENT_DIR / "final"
METRICS = ["arc_f1", "arc_precision", "arc_recall", "seq_f1", "route_continuity",
           "pseudo_top1_accuracy", "fp_reverse_direction", "fp_gap_fill_only", "fp_matched",
           "runtime_ms_excl_features"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    part = load_partition()
    sets = {
        "tune64_OOF": sorted(part["tune"]),
        "holdout16": sorted(part["holdout"]),
        "student_test10": sorted(t["trip_id"] for t in list_trips() if t["split"] == "student_test"),
    }
    jobs = []
    for set_name, ids in sets.items():
        for t in ids:
            jobs.append((set_name, (t, "nearest_road", None, None)))
            jobs.append((set_name, (t, "geometric_hmm", None, None)))
            for m in MODELS:
                oof = str(EXPERIMENT_DIR / "ml" / f"oof_{m}.joblib") if set_name == "tune64_OOF" else None
                jobs.append((set_name, (t, "classifier_only", m, oof)))
                jobs.append((set_name, (t, "ml_hmm", m, oof)))
    # group jobs by trip so each worker reuses its per-trip cache
    jobs.sort(key=lambda j: j[1][0])
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(run_variant, [j[1] for j in jobs], chunksize=12))
    for (set_name, _), r in zip(jobs, rows):
        r["eval_set"] = set_name
        r["classifier_training"] = ("5-fold OOF within 64 tune trips" if set_name == "tune64_OOF"
                                    else ("64 tune trips" if r["model"] != "-" else "n/a"))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ablation_per_trip.csv", index=False)

    g = df.groupby(["eval_set", "variant", "model"])
    summ = g[METRICS].mean().round(4)
    summ["arc_f1_median"] = g["arc_f1"].median().round(4)
    summ["n_trips"] = g.size()
    summ = summ.reset_index()
    order = {"nearest_road": 0, "classifier_only": 1, "geometric_hmm": 2, "ml_hmm": 3}
    summ["o"] = summ["variant"].map(order)
    summ = summ.sort_values(["eval_set", "o", "model"]).drop(columns="o")
    summ.to_csv(OUT / "ablation_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ[["eval_set", "variant", "model", "n_trips", "arc_f1", "arc_f1_median", "arc_precision",
                "arc_recall", "seq_f1", "route_continuity", "pseudo_top1_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
