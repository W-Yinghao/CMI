"""ACAR V5 Stage-1B trainer CONTRACT (pure/stdlib; importing this reads/trains NOTHING). The REAL EEGNet encoder + source-state
trainer is a separate, later-authorized patch (it will import torch/braindecode lazily, inside a factory, only after the gate).
A trainer must expose:
    train_fold(disease, fold, seed, train_subject_keys, val_subject_keys, dataset_view) -> raw build output dict
The trainer is handed ONLY the FIT (train∪val) canonical subject keys + an AuthorizedFitDatasetView — it can read a subject's
windows ONLY via dataset_view.read_windows(key), which refuses any CAL/EVAL/unknown key. It never receives raw cohort roots. The
raw build output carries bytes payloads (see stage1b_artifact_writer.HASH_SOURCE); the artifact writer computes the registry
hashes from those bytes (trainer-reported hashes are ignored).
"""
from __future__ import annotations


class TrainerNotWiredError(RuntimeError):
    """Raised when a real Stage-1B substrate train is attempted without an authorized, wired trainer."""


class UnwiredTrainer:
    def train_fold(self, disease, fold, seed, train_subject_keys, val_subject_keys, dataset_view):
        raise TrainerNotWiredError("real Stage-1B trainer not wired (Stage-1B3)")


def require_trainer(trainer):
    if trainer is None:
        raise TrainerNotWiredError("Stage-1B execute requires an authorized trainer (none supplied)")
    if not callable(getattr(trainer, "train_fold", None)):
        raise TrainerNotWiredError("trainer is missing required method train_fold()")
    return trainer
