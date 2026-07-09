"""Red-team checks for TTA-MECH-EEG."""

from .baseline_universe_freeze import BaselineUniverseFailure, validate_baseline_universe
from .bn_condition_freeze import BnConditionFreezeFailure, validate_bn_condition_freeze
from .bn_target_label_quarantine import BnTargetLabelQuarantineFailure, validate_bn_target_label_quarantine
from .no_new_method_guard import NoNewMethodFailure, validate_no_new_method
from .no_weight_update_guard import NoWeightUpdateFailure, validate_no_weight_update_guard
from .replay_determinism import ReplayDeterminismFailure, validate_replay_determinism
from .target_label_quarantine import TargetLabelContractFailure, validate_target_label_contract

__all__ = [
    "BaselineUniverseFailure",
    "BnConditionFreezeFailure",
    "BnTargetLabelQuarantineFailure",
    "NoNewMethodFailure",
    "NoWeightUpdateFailure",
    "ReplayDeterminismFailure",
    "TargetLabelContractFailure",
    "validate_baseline_universe",
    "validate_bn_condition_freeze",
    "validate_bn_target_label_quarantine",
    "validate_no_new_method",
    "validate_no_weight_update_guard",
    "validate_replay_determinism",
    "validate_target_label_contract",
]
