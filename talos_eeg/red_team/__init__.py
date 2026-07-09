"""Red-team checks for TALOS-EEG."""

from .adapter_determinism import AdapterDeterminismFailure, validate_adapter_determinism
from .target_label_quarantine import TargetLabelQuarantineFailure, validate_target_label_quarantine
from .variant_freeze import VariantFreezeFailure, validate_variant_freeze

__all__ = [
    "AdapterDeterminismFailure",
    "TargetLabelQuarantineFailure",
    "VariantFreezeFailure",
    "validate_adapter_determinism",
    "validate_target_label_quarantine",
    "validate_variant_freeze",
]
