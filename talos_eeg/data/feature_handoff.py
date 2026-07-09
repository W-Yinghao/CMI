"""TALOS handoff boundary checks.

TALOS_00A is an implementation/red-team preflight. It must not run the real
CEDAR_01F BNCI2014 frozen-feature readout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CEDAR_01F_FEATURE_ROOT = Path("results/cedar/feature_supply/cedar01f_bnci2014_001_seed0")
CEDAR_01F_HANDOFF_MANIFEST = CEDAR_01F_FEATURE_ROOT / "CEDAR_01F_HANDOFF_MANIFEST.json"


@dataclass(frozen=True)
class TalosHandoffBoundary:
    phase: str
    real_eeg_readout_run: bool
    approved_real_handoff_input: str
    source_state_mode: str
    source_free_deployment_claim: bool
    allowed_for_talos00a: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def talos00a_handoff_boundary() -> TalosHandoffBoundary:
    return TalosHandoffBoundary(
        phase="TALOS_00A_adapter_implementation_red_team_preflight",
        real_eeg_readout_run=False,
        approved_real_handoff_input=str(CEDAR_01F_HANDOFF_MANIFEST),
        source_state_mode="constructed_from_frozen_source_features_for_P0_replay",
        source_free_deployment_claim=False,
        allowed_for_talos00a=True,
    )


def assert_no_real_feature_readout(paths: Iterable[str | Path] = ()) -> None:
    """Reject accidental TALOS_00A reads of real CEDAR feature artifacts."""

    for raw in paths:
        path = Path(raw)
        if path.suffix == ".npz":
            raise ValueError(f"TALOS_00A must not read real feature artifact: {path}")
        if path.name == CEDAR_01F_HANDOFF_MANIFEST.name:
            raise ValueError(f"TALOS_00A must not validate real handoff manifest as a readout: {path}")
