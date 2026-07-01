"""ACAR V5 Stage-1B authorized FIT dataset view (pure/stdlib). The trainer is handed ONLY this view — NOT raw cohort root paths —
so it can read a subject's windows ONLY if that subject is in the allowed FIT (train∪val) set. A read of any CAL/EVAL (or unknown)
subject fails closed. This turns "CAL/EVAL never passed" into "CAL/EVAL cannot be read" (strong isolation, not a weak promise).
"""
from __future__ import annotations


class DatasetViewAccessError(RuntimeError):
    """Raised when the trainer tries to read a subject outside its authorized FIT set (e.g. a CAL/EVAL subject)."""


class AuthorizedFitDatasetView:
    """Exposes ONLY read_windows(subject_key) for the allowed FIT keys of ONE fold. Holds the reader + index + cohort paths
    internally; never exposes the raw cohort roots to the trainer."""

    def __init__(self, index, allowed_subject_keys, reader, cohort_paths):
        self._index = index
        self._allowed = frozenset(allowed_subject_keys)
        self._reader = reader
        self._cohort_paths = dict(cohort_paths)
        self.reads = []                                       # audit: subject_keys actually read through this view

    @property
    def allowed_subject_keys(self):
        return self._allowed

    def read_windows(self, subject_key):
        if subject_key not in self._allowed:
            raise DatasetViewAccessError(f"subject {subject_key} is not in the authorized FIT set (CAL/EVAL/unknown → refused)")
        cohort = self._index.cohort_of(subject_key)          # (raises if unknown)
        raw = self._index.raw_of(subject_key)
        path = self._cohort_paths[cohort]
        self.reads.append(subject_key)
        return self._reader.read_subject_windows(self._index.disease, cohort, raw, path)
