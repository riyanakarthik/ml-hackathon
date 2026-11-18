"""Formal checks that pseudo-labels follow the chronological .route order.

  python -m scripts.verify_labels      -> experiments/final/label_verification.csv
For each train trip: aligned positions are non-decreasing; the label sequence
(consecutive duplicates collapsed) is an in-order subsequence of .route; every
label is a route arc at the claimed position; repeated route arcs are matched
to distinct positions in order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import list_trips, load_trip
from src.evaluation import load_reference_route
from src.experiment import EXPERIMENT_DIR
from src.labels import align_track_to_route
from src.pipeline import build_network


def is_subsequence(seq, ref) -> bool:
    it = iter(ref)
    return all(any(x == y for y in it) for x in seq)


def main() -> None:
    rows = []
    for item in list_trips():
        if item["split"] != "train":
            continue
        tid = item["trip_id"]
        trip = load_trip(tid)
        net, xy = build_network(trip)
        ref = load_reference_route(tid)
        al = align_track_to_route(xy, trip.timestamps, ref, net)
        collapsed = [int(a) for i, a in enumerate(al.labels) if i == 0 or a != al.labels[i - 1]]
        rows.append({
            "trip_id": tid, "n_points": len(xy), "n_route_arcs": len(ref),
            "route_has_repeats": len(ref) != len(set(ref.tolist())),
            "positions_non_decreasing": bool(np.all(np.diff(al.position) >= 0)),
            "labels_equal_route_at_position": bool(np.all(ref[al.position] == al.labels)),
            "label_sequence_is_ordered_subsequence_of_route": is_subsequence(collapsed, ref.tolist()),
            "label_valid_frac": round(float(al.valid.mean()), 4),
        })
    df = pd.DataFrame(rows)
    out = EXPERIMENT_DIR / "final" / "label_verification.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    checks = ["positions_non_decreasing", "labels_equal_route_at_position",
              "label_sequence_is_ordered_subsequence_of_route"]
    print(f"{len(df)} train trips; trips with repeated route arcs: {int(df.route_has_repeats.sum())}")
    for c in checks:
        print(f"  {c}: {int(df[c].sum())}/{len(df)}")
    print(f"  mean valid-label fraction {df.label_valid_frac.mean():.4f}; "
          f"trips < 0.5 valid: {df[df.label_valid_frac < 0.5].trip_id.tolist()}")


if __name__ == "__main__":
    main()
