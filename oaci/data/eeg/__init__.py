"""OACI real-EEG data contract (offline-only). No runtime import of `cmi`/`h2cmi`."""
from __future__ import annotations

from .schema import EEGBundle, tensor_content_hash
from .units import aggregate_mean_prob, base_mass, cell_mass, eligibility_counts
from .splits import SplitPlan, apply_missing_cell_mask, make_loso_split
from .audit import canonical_hash, split_manifest_hash, tensor_hash, validate_prediction_bundle
from .cache import atomic_write_bytes, cache_key
from .preprocess import PreprocessSpec, apply_normalization, assert_fit_excludes_target, fit_normalization
from .registry import REGISTRY, DatasetEntry, OfflineDownloadError, ensure_offline_available, get_entry

__all__ = [
    "EEGBundle", "tensor_content_hash",
    "aggregate_mean_prob", "base_mass", "cell_mass", "eligibility_counts",
    "SplitPlan", "apply_missing_cell_mask", "make_loso_split",
    "canonical_hash", "split_manifest_hash", "tensor_hash", "validate_prediction_bundle",
    "atomic_write_bytes", "cache_key",
    "PreprocessSpec", "apply_normalization", "assert_fit_excludes_target", "fit_normalization",
    "REGISTRY", "DatasetEntry", "OfflineDownloadError", "ensure_offline_available", "get_entry",
]
