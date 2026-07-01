"""ACAR V5 Stage-1B REAL substrate trainer. Constructed by its factory with a gate-issued Stage1BExecutionContext, AFTER the gate.
NO heavy import at module level (torch is lazy inside train_fold). It reads signal ONLY via dataset_view.read_windows and labels
ONLY via dataset_view.read_label (the FIT training view) — never raw cohort roots, never the reader directly, never a filesystem
scan. Training under training_config + file-artifact emission (into context.output_root) is the remaining seam wired at the real run.
"""
from __future__ import annotations


class RealTrainerError(RuntimeError):
    pass


class RealSubstrateTrainer:
    def __init__(self, context):
        if context is None:
            raise RealTrainerError("RealSubstrateTrainer requires a gate-issued Stage1BExecutionContext")
        self._ctx = context

    def train_fold(self, disease, fold, seed, train_subject_keys, val_subject_keys, dataset_view):
        import torch  # noqa: F401  (lazy)
        # signal + labels are FIT-only, via the view — no raw roots, no filesystem scan, no direct reader calls
        for key in list(train_subject_keys) + list(val_subject_keys):
            dataset_view.read_windows(key)
            dataset_view.read_label(key)
        raise NotImplementedError("real EEGNet + source-state training (training_config) + file-artifact emission wired at the Stage-1B run")


def make_real_trainer(context):
    """Factory — construct AFTER the full-build gate, bound to the run's execution context (uses context.output_root)."""
    return RealSubstrateTrainer(context)
