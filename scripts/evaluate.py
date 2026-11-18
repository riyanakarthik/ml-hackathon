"""Evaluate a method over a split: python -m scripts.evaluate --split train --method hmm"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from src.data_loader import list_trips, load_trip
from src.evaluation import load_reference_route, route_metrics
from src.pipeline import METHODS, predict_trip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--method", default="hmm", choices=METHODS)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    rows = []
    for item in list_trips(args.data_dir):
        if item["split"] != args.split or not item["has_reference_route"]:
            continue
        trip = load_trip(item["trip_id"], args.data_dir)
        t0 = time.perf_counter()
        res, net, _ = predict_trip(trip, args.method)
        ms = (time.perf_counter() - t0) * 1000
        ref = load_reference_route(trip.trip_id, args.data_dir)
        m = route_metrics(res.route_arc_ids, ref, trip.arcs, net.arc_len)
        rows.append((m["arc_precision"], m["arc_recall"], m["arc_f1"],
                     m["length_precision"], m["length_recall"], m["route_continuity"] or 0, ms))
        print(f"{trip.trip_id}  P={m['arc_precision']:.3f} R={m['arc_recall']:.3f} "
              f"F1={m['arc_f1']:.3f} lenP={m['length_precision']:.3f} lenR={m['length_recall']:.3f} "
              f"cont={m['route_continuity']}  {ms:.0f}ms")
    if rows:
        a = np.array(rows)
        print(f"\n{args.split} / {args.method}: {len(rows)} trips  mean "
              f"P={a[:,0].mean():.4f} R={a[:,1].mean():.4f} F1={a[:,2].mean():.4f} "
              f"lenP={a[:,3].mean():.4f} lenR={a[:,4].mean():.4f} cont={a[:,5].mean():.4f} "
              f"runtime={a[:,6].mean():.0f}ms (route-level metrics, not per-point accuracy)")


if __name__ == "__main__":
    main()
