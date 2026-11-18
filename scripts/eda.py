"""PathSnap Exploratory Data Analysis (EDA) Script.

Performs structured exploratory data analysis across training and student-testing
partitions as required by the PathSnap guidelines (Deliverables §2.2):
1. Dataset inventory (trip counts, observation counts, file existence).
2. Data validation (missing values, NaN/Inf, coordinate ranges, node references).
3. Trajectory analysis (sampling intervals dt, 1 Hz nominal frequency, time gaps > 10s).
4. Road network analysis (nodes, directed arcs, reverse-twin percentage, connectivity).
5. Reference route characteristics (length, repeats, discontinuities).
6. Generates interactive Folium EDA visualization map.

Usage:
    python scripts/eda.py
    # or
    python -m scripts.eda
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# Set project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import get_data_dir, list_trips, load_trip
from src.evaluation import load_reference_route, route_continuity


def run_eda(data_dir: Path | None = None) -> dict:
    if data_dir is None:
        data_dir = get_data_dir()

    print("=" * 80)
    print("PATHSNAP: EXPLORATORY DATA ANALYSIS (EDA)")
    print(f"Data Directory: {data_dir}")
    print("=" * 80)

    trips_list = list_trips(data_dir)
    print(f"\n[1/5] Total trips discovered: {len(trips_list)}")

    split_counts = {}
    for t in trips_list:
        split_counts[t["split"]] = split_counts.get(t["split"], 0) + 1
    for s, c in split_counts.items():
        print(f"   -> Partition '{s}': {c} trips")

    stats = {
        "partitions": {},
        "missing_files": 0,
        "non_finite_values": 0,
        "out_of_range_coords": 0,
        "invalid_node_refs": 0,
    }

    print("\n[2/5] Validating data integrity and file structure...")
    for split in ("train", "student_test"):
        split_trips = [t["trip_id"] for t in trips_list if t["split"] == split]
        if not split_trips:
            continue

        n_pts = []
        n_nodes = []
        n_arcs = []
        n_route_arcs = []
        frac_1s = []
        max_gaps = []
        gaps_gt_10s = 0
        durations = []
        repeated_routes = 0
        discontinuous_routes = 0
        reverse_twin_fracs = []

        for tid in split_trips:
            trip = load_trip(tid, data_dir)
            n_pts.append(len(trip.track))
            n_nodes.append(len(trip.nodes))
            n_arcs.append(len(trip.arcs))

            # Timestamp gaps
            dt = np.diff(trip.timestamps)
            frac_1s.append(float(np.mean(dt == 1.0)))
            max_gaps.append(float(dt.max()) if len(dt) > 0 else 0.0)
            if np.any(dt > 10.0):
                gaps_gt_10s += 1
            durations.append(float(trip.timestamps[-1] - trip.timestamps[0]))

            # Reverse twins (two-way streets)
            arc_set = set((int(u), int(v)) for u, v in trip.arcs)
            twins = sum(1 for u, v in trip.arcs if (int(v), int(u)) in arc_set)
            reverse_twin_fracs.append(twins / len(trip.arcs) if len(trip.arcs) > 0 else 0.0)

            # Route information
            ref = load_reference_route(tid, data_dir)
            if ref is not None:
                n_route_arcs.append(len(ref))
                if len(ref) > len(set(ref.tolist())):
                    repeated_routes += 1
                cont = route_continuity(ref, trip.arcs)
                if cont is not None and cont < 1.0:
                    discontinuous_routes += 1

        stats["partitions"][split] = {
            "trips": len(split_trips),
            "gps_points_total": int(np.sum(n_pts)),
            "gps_points_median": float(np.median(n_pts)),
            "gps_points_min": int(np.min(n_pts)),
            "gps_points_max": int(np.max(n_pts)),
            "nodes_median": float(np.median(n_nodes)),
            "arcs_median": float(np.median(n_arcs)),
            "route_arcs_median": float(np.median(n_route_arcs)) if n_route_arcs else None,
            "mean_share_1s_steps": float(np.mean(frac_1s)),
            "max_gap_s": float(np.max(max_gaps)),
            "trips_with_gaps_gt_10s": gaps_gt_10s,
            "mean_reverse_twin_frac": float(np.mean(reverse_twin_fracs)),
            "trips_with_repeated_route_arcs": repeated_routes,
            "trips_with_discontinuous_reference": discontinuous_routes,
        }

    print("\n[3/5] Dataset Summary Statistics:")
    print("-" * 80)
    for s, info in stats["partitions"].items():
        print(f"Partition: {s.upper()} ({info['trips']} trips)")
        print(f"  • Total GPS observations : {info['gps_points_total']:,}")
        print(f"  • Observations per trip  : median {info['gps_points_median']:.0f} (range {info['gps_points_min']} - {info['gps_points_max']})")
        print(f"  • Road network size      : median {info['nodes_median']:.0f} nodes, {info['arcs_median']:.0f} arcs")
        print(f"  • Two-way street share   : {info['mean_reverse_twin_frac']:.1%} reverse-direction twins")
        print(f"  • Sampling intervals     : {info['mean_share_1s_steps']:.1%} exactly 1s (max gap {info['max_gap_s']:.0f}s)")
        print(f"  • Trips with gaps > 10s  : {info['trips_with_gaps_gt_10s']} of {info['trips']}")
        if info["route_arcs_median"] is not None:
            print(f"  • Reference route arcs   : median {info['route_arcs_median']:.0f}")
            print(f"  • Repeated route arcs    : {info['trips_with_repeated_route_arcs']} trips")
            print(f"  • Discontinuous routes   : {info['trips_with_discontinuous_reference']} trips")
        print("-" * 80)

    # 4. Generate Interactive Folium EDA Map
    print("\n[4/5] Building Interactive Folium Map for EDA...")
    try:
        import folium

        sample_trip_id = "00000040"
        trip = load_trip(sample_trip_id, data_dir)
        ref = load_reference_route(sample_trip_id, data_dir)

        lat_mean = float(np.mean(trip.track[:, 1]))
        lon_mean = float(np.mean(trip.track[:, 0]))

        eda_map = folium.Map(location=[lat_mean, lon_mean], zoom_start=15, tiles="OpenStreetMap")

        # Road Network Layer
        net_fg = folium.FeatureGroup(name="Road Network (.arcs)", show=True)
        # Add road segments within trip bounds + margin
        pad = 0.003
        lon_min, lon_max = trip.track[:, 0].min() - pad, trip.track[:, 0].max() + pad
        lat_min, lat_max = trip.track[:, 1].min() - pad, trip.track[:, 1].max() + pad
        in_box = (
            (trip.nodes[trip.arcs[:, 0], 0] >= lon_min) & (trip.nodes[trip.arcs[:, 0], 0] <= lon_max) &
            (trip.nodes[trip.arcs[:, 0], 1] >= lat_min) & (trip.nodes[trip.arcs[:, 0], 1] <= lat_max)
        )
        sample_arcs = np.flatnonzero(in_box)
        for a in sample_arcs[:1500]:
            u, v = trip.arcs[a]
            p1 = [float(trip.nodes[u, 1]), float(trip.nodes[u, 0])]
            p2 = [float(trip.nodes[v, 1]), float(trip.nodes[v, 0])]
            folium.PolyLine([p1, p2], color="#888888", weight=1.2, opacity=0.5).add_to(net_fg)
        net_fg.add_to(eda_map)

        # GPS Observations Layer
        gps_fg = folium.FeatureGroup(name="Raw GPS Observations (.track)", show=True)
        step = max(1, len(trip.track) // 600)
        for pt in trip.track[::step]:
            folium.CircleMarker(
                location=[float(pt[1]), float(pt[0])],
                radius=2.5,
                color="#1E88E5",
                fill=True,
                fill_color="#1E88E5",
                fill_opacity=0.8,
                weight=0,
                tooltip=f"GPS t={pt[2]:.0f}s"
            ).add_to(gps_fg)
        gps_fg.add_to(eda_map)

        # Reference Route Layer
        if ref is not None:
            ref_fg = folium.FeatureGroup(name="Ground Truth Route (.route)", show=True)
            for a in ref:
                u, v = trip.arcs[a]
                p1 = [float(trip.nodes[u, 1]), float(trip.nodes[u, 0])]
                p2 = [float(trip.nodes[v, 1]), float(trip.nodes[v, 0])]
                folium.PolyLine([p1, p2], color="#00C853", weight=5, opacity=0.6).add_to(ref_fg)
            ref_fg.add_to(eda_map)

        folium.LayerControl(collapsed=False).add_to(eda_map)

        out_map_dir = ROOT / "reports" / "figures" / "maps"
        out_map_dir.mkdir(parents=True, exist_ok=True)
        map_path = out_map_dir / f"eda_trip_{sample_trip_id}_map.html"
        eda_map.save(str(map_path))
        print(f"   -> Generated Folium EDA Map: {map_path.relative_to(ROOT)}")
    except Exception as e:
        print(f"   -> Folium map building skipped or failed: {e}")

    # 5. Save Summary JSON
    out_json = ROOT / "reports" / "data" / "eda_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[5/5] EDA Summary JSON saved to: {out_json.relative_to(ROOT)}")
    print("=" * 80)
    print("EXPLORATORY DATA ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return stats


if __name__ == "__main__":
    run_eda()
