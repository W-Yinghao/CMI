"""ACAR V5 Stage-1B REAL substrate trainer. NO heavy import at module level (torch/braindecode imported lazily inside train_fold).
It reads signal ONLY via dataset_view.read_windows(subject_key) — it never receives or scans raw cohort roots, and never calls the
reader directly. The EEGNet/source-state training + file-artifact emission is the remaining seam wired ONLY at an authorized
Stage-1B real run (train_fold raises until then). Constructed via its factory AFTER the gate.
"""
from __future__ import annotations


class RealTrainerError(RuntimeError):
    pass


class RealSubstrateTrainer:
    """Trains ONE fold-contained substrate on FIT subjects, reading exclusively through the authorized FIT dataset view."""

    def __init__(self, output_dir):
        self._output_dir = output_dir                          # where the real run would write encoder/source-state/feat files

    def train_fold(self, disease, fold, seed, train_subject_keys, val_subject_keys, dataset_view):
        import torch  # noqa: F401  (lazy; not imported at module load)
        # signal access is view-only — no raw cohort roots, no filesystem scan
        for key in list(train_subject_keys) + list(val_subject_keys):
            dataset_view.read_windows(key)
        raise NotImplementedError("real EEGNet + source-state training and file-artifact emission are wired at the Stage-1B real run")


def make_real_trainer(output_dir):
    """Factory — construct AFTER the full-build gate (see run_stage1b_real_build)."""
    return RealSubstrateTrainer(output_dir)
