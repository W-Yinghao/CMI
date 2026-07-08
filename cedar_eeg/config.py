"""Configuration constants for the CEDAR-EEG P0 gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ProbeConfig:
    """Settings for light-weight source-side probe audits."""

    probe: str = "linear"
    n_splits: int = 3
    seed: int = 0
    max_iter: int = 500
    hidden: tuple[int, ...] = (64,)


@dataclass(frozen=True)
class P0Thresholds:
    """Frozen P0 go/no-go thresholds.

    Accuracy margins are absolute balanced-accuracy points, so 0.01 means one
    point on a 0-1 scale.
    """

    min_leakage_drop_frac: float = 0.30
    max_source_bacc_drop: float = 0.01
    # Diagnostic-only when target labels are present. This must not affect mask
    # selection.
    max_target_bacc_drop: float = 0.01
    max_r3_delta: float = 0.0
    max_random_control_drop_frac: float = 0.05
    min_stability: float = 0.60


@dataclass(frozen=True)
class DatasetBackboneCell:
    dataset: str
    backbone: str


DEFAULT_PROBE = ProbeConfig()
DEFAULT_P0_THRESHOLDS = P0Thresholds()

MVP_CELLS: tuple[DatasetBackboneCell, ...] = (
    DatasetBackboneCell("BNCI2014_001", "EEGNet"),
    DatasetBackboneCell("BNCI2014_001", "EEGConformer"),
    DatasetBackboneCell("BNCI2015_001", "EEGNet"),
    DatasetBackboneCell("BNCI2015_001", "EEGConformer"),
)

DEFAULT_DROP_FRACTIONS: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
CEDAR_01_DROP_KS: tuple[int, ...] = (1, 2, 4)


def parse_drop_fractions(text: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(text, str):
        vals = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    else:
        vals = tuple(float(x) for x in text)
    if not vals:
        raise ValueError("at least one drop fraction is required")
    for val in vals:
        if val <= 0.0 or val >= 1.0:
            raise ValueError(f"drop fractions must be in (0, 1), got {val}")
    return vals


def parse_drop_ks(text: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(text, str):
        vals = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    else:
        vals = tuple(int(x) for x in text)
    if not vals:
        raise ValueError("at least one drop k is required")
    for val in vals:
        if val <= 0:
            raise ValueError(f"drop ks must be positive, got {val}")
    return vals
