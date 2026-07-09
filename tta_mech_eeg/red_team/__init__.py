"""Red-team checks for TTA-MECH-EEG."""

from .baseline_universe_freeze import BaselineUniverseFailure, validate_baseline_universe
from .no_new_method_guard import NoNewMethodFailure, validate_no_new_method
from .replay_determinism import ReplayDeterminismFailure, validate_replay_determinism
from .target_label_quarantine import TargetLabelContractFailure, validate_target_label_contract

__all__ = [
    "BaselineUniverseFailure",
    "NoNewMethodFailure",
    "ReplayDeterminismFailure",
    "TargetLabelContractFailure",
    "validate_baseline_universe",
    "validate_no_new_method",
    "validate_replay_determinism",
    "validate_target_label_contract",
]
