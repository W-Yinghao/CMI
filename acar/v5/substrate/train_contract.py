"""ACAR V5 Stage-1B trainer CONTRACT (pure/stdlib; importing this reads/trains NOTHING). The REAL EEGNet encoder + source-state
trainer is a separate, later-authorized patch (it will import torch/braindecode lazily). Here we define the interface + a
fail-closed unwired default. A trainer must expose:
    train_fold(disease, fold, seed, train_subjects, val_subjects, cohort_paths) -> artifact manifest dict
The trainer is handed ONLY FIT (train/val) subjects by the build orchestrator — CAL/EVAL subjects are never passed to it.
"""
from __future__ import annotations


class TrainerNotWiredError(RuntimeError):
    """Raised when a real Stage-1B substrate train is attempted without an authorized, wired trainer."""


class UnwiredTrainer:
    def train_fold(self, disease, fold, seed, train_subjects, val_subjects, cohort_paths):
        raise TrainerNotWiredError("real Stage-1B trainer not wired (Stage-1B2)")


def require_trainer(trainer):
    if trainer is None:
        raise TrainerNotWiredError("Stage-1B execute requires an authorized trainer (none supplied)")
    if not callable(getattr(trainer, "train_fold", None)):
        raise TrainerNotWiredError("trainer is missing required method train_fold()")
    return trainer
