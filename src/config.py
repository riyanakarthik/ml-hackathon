"""Matcher / HMM configuration.

``BASELINE_V1`` reproduces the original hand-set decoder exactly and is kept
for rollback. Other configs are JSON files under ``configs/``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@dataclass(frozen=True)
class MatcherConfig:
    name: str = "baseline_v1"
    # candidate generation
    k: int = 8                        # max candidate arcs per GPS point
    radius_m: float = 50.0            # candidate search radius
    fallback_radius_m: float = 200.0  # retry radius when nothing within radius_m
    # emissions
    sigma_m: float | None = None      # GPS noise std; None = estimate per trip (MAD)
    sigma_scale: float = 1.0          # multiplier on the (estimated or fixed) sigma
    # transitions
    beta: float = 15.0                # scale of |d_route - d_straight| (metres)
    backtrack_tol_m: float = 5.0      # same-arc backwards movement treated as noise
    backtrack_penalty: float = 0.0    # extra log-penalty per metre of same-arc backtrack beyond tol
                                      #   (0 => v1 behaviour: such moves must loop via the graph)
    uturn_penalty: float = 0.0        # log-penalty for moving onto an arc's reverse twin
    dijkstra_limit_m: float = 2000.0
    # observation thinning (Newson & Krumm): skip points < min_spacing_m from last kept
    min_spacing_m: float = 0.0
    # route assembly
    fill_max_gap_m: float = 3000.0
    max_loop_m: float = 0.0           # remove route cycles shorter than this (0 = off).
                                      #   Reference routes DO sometimes repeat arcs
                                      #   (9/80 train trips), so this is not free.

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_dict(cls, d: dict) -> "MatcherConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def load(cls, path: str | Path) -> "MatcherConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def with_(self, **kw) -> "MatcherConfig":
        return replace(self, **kw)


BASELINE_V1 = MatcherConfig()


def get_config(name: str) -> MatcherConfig:
    """'baseline_v1' (built in) or the stem of a JSON file in configs/."""
    if name == BASELINE_V1.name:
        return BASELINE_V1
    if name not in available_configs():  # also blocks path tricks like "../x"
        raise KeyError(f"Unknown config {name!r}; available: {available_configs()}")
    return MatcherConfig.load(CONFIG_DIR / f"{name}.json")


def available_configs() -> list[str]:
    names = [BASELINE_V1.name]
    if CONFIG_DIR.is_dir():
        names += sorted(p.stem for p in CONFIG_DIR.glob("*.json") if p.stem != BASELINE_V1.name)
    return names
