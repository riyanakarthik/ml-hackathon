"""Loading and validation of PathSnap trip files.

Each trip directory ``<data_dir>/<split>/<trip_id>/`` holds whitespace-separated,
header-less files:

* ``<id>.track``  longitude latitude timestamp   (GPS observations)
* ``<id>.nodes``  longitude latitude             (row index = node id)
* ``<id>.arcs``   from_node_id to_node_id        (row index = arc id)
* ``<id>.route``  reference arc ids (chronological, NOT one per GPS point)

IMPORTANT: this module deliberately never reads ``.route``. Reference routes
are only loaded by ``src.evaluation`` so they cannot leak into prediction.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "Student folder"
KNOWN_SPLITS = ("train", "student_test")


class TripDataError(ValueError):
    """Raised when a trip is missing or its files fail validation."""


def get_data_dir(data_dir: str | os.PathLike | None = None) -> Path:
    """Resolve the data root: explicit arg > $PATHSNAP_DATA_DIR > default."""
    if data_dir is None:
        data_dir = os.environ.get("PATHSNAP_DATA_DIR", DEFAULT_DATA_DIR)
    return Path(data_dir).expanduser().resolve()


def normalize_trip_id(trip_id: str | int) -> str:
    """Accept 38, "38" or "00000038" and return the 8-digit form."""
    s = str(trip_id).strip()
    if not s.isdigit():
        raise TripDataError(f"Trip id must be numeric, got {trip_id!r}")
    return s.zfill(8)


@dataclass(frozen=True)
class Trip:
    trip_id: str
    split: str
    path: Path
    track: np.ndarray  # (T, 3): lon, lat, timestamp
    nodes: np.ndarray  # (N, 2): lon, lat
    arcs: np.ndarray   # (A, 2): from_node, to_node (int64)

    @property
    def track_lonlat(self) -> np.ndarray:
        return self.track[:, :2]

    @property
    def timestamps(self) -> np.ndarray:
        return self.track[:, 2]


def list_trips(data_dir: str | os.PathLike | None = None) -> list[dict]:
    """List trips found under every split directory of the data root.

    Also accepts a data root that IS a split directory (e.g. PATHSNAP_DATA_DIR
    pointing straight at evaluator_test/): trip folders found directly under
    the root are listed with split = the root's name. Mirrors the direct-folder
    fallback that find_trip_dir already uses for /predict."""
    root = get_data_dir(data_dir)
    out = []
    if not root.is_dir():
        return out
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if (split_dir / f"{split_dir.name}.track").exists():   # root is itself a split dir
            out.append({
                "trip_id": split_dir.name,
                "split": root.name,
                "has_reference_route": (split_dir / f"{split_dir.name}.route").exists(),
            })
            continue
        for trip_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            tid = trip_dir.name
            if (trip_dir / f"{tid}.track").exists():
                out.append({
                    "trip_id": tid,
                    "split": split_dir.name,
                    "has_reference_route": (trip_dir / f"{tid}.route").exists(),
                })
    return out


def find_trip_dir(trip_id: str | int, data_dir: str | os.PathLike | None = None) -> tuple[Path, str]:
    root = get_data_dir(data_dir)
    tid = normalize_trip_id(trip_id)
    # Prefer known splits, then any other sub-directory (e.g. evaluator data).
    splits = [s for s in KNOWN_SPLITS if (root / s).is_dir()]
    if root.is_dir():
        splits += sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in splits)
    for split in splits:
        d = root / split / tid
        if d.is_dir():
            return d, split
    # Also allow data_dir pointing directly at a split or flat directory.
    if (root / tid).is_dir():
        return root / tid, root.name
    raise TripDataError(f"Trip {tid} not found under {root}")


def _load_matrix(path: Path, ncols: int, dtype) -> np.ndarray:
    if not path.exists():
        raise TripDataError(f"Missing file: {path}")
    try:
        arr = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except ValueError as e:
        raise TripDataError(f"Could not parse {path}: {e}") from e
    if arr.size == 0:
        raise TripDataError(f"Empty file: {path}")
    if arr.shape[1] != ncols:
        raise TripDataError(f"{path.name}: expected {ncols} columns, got {arr.shape[1]}")
    if not np.all(np.isfinite(arr)):
        raise TripDataError(f"{path.name}: contains non-finite values")
    if np.issubdtype(dtype, np.integer):
        if not np.all(arr == np.round(arr)):
            raise TripDataError(f"{path.name}: expected integer values")
        return arr.astype(dtype)
    return arr


def _validate_lonlat(arr: np.ndarray, name: str) -> None:
    lon, lat = arr[:, 0], arr[:, 1]
    if np.any(np.abs(lon) > 180) or np.any(np.abs(lat) > 90):
        raise TripDataError(f"{name}: coordinates out of lon/lat range")


def load_trip(trip_id: str | int, data_dir: str | os.PathLike | None = None) -> Trip:
    """Load and validate track, nodes and arcs. Never touches ``.route``."""
    trip_dir, split = find_trip_dir(trip_id, data_dir)
    tid = trip_dir.name
    track = _load_matrix(trip_dir / f"{tid}.track", 3, np.float64)
    nodes = _load_matrix(trip_dir / f"{tid}.nodes", 2, np.float64)
    arcs = _load_matrix(trip_dir / f"{tid}.arcs", 2, np.int64)

    _validate_lonlat(track, f"{tid}.track")
    _validate_lonlat(nodes, f"{tid}.nodes")
    if arcs.min() < 0 or arcs.max() >= len(nodes):
        raise TripDataError(
            f"{tid}.arcs references node ids outside [0, {len(nodes) - 1}]")

    ts = track[:, 2]
    if np.any(np.diff(ts) < 0):
        log.warning("Trip %s: timestamps not monotonic; sorting track by time", tid)
        track = track[np.argsort(ts, kind="stable")]

    log.info("Loaded trip %s (%s): %d GPS points, %d nodes, %d arcs",
             tid, split, len(track), len(nodes), len(arcs))
    return Trip(trip_id=tid, split=split, path=trip_dir, track=track, nodes=nodes, arcs=arcs)
