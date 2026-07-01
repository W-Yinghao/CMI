"""ACAR V5 Stage-1B authorized FIT dataset view (pure/stdlib). The trainer is handed ONLY this view — NOT raw cohort root paths —
so it can read a subject's windows ONLY if that subject is in the allowed FIT (train∪val) set; a CAL/EVAL/unknown read fails
closed. The reader + cohort paths are held in a CLOSURE, so the view object exposes NO `_reader` / `_cohort_paths` attribute for a
trainer to introspect (accidental raw-root escape prevented). Public surface = read_windows + allowed_subject_keys + reads.
"""
from __future__ import annotations


class DatasetViewAccessError(RuntimeError):
    """Raised when the trainer tries to read a subject outside its authorized FIT set (e.g. a CAL/EVAL subject)."""


class AuthorizedFitDatasetView:
    def __init__(self, index, allowed_subject_keys, reader, cohort_paths):
        allowed = frozenset(allowed_subject_keys)
        disease = index.disease
        cps = dict(cohort_paths)
        reads = []

        def _read(subject_key):
            if subject_key not in allowed:
                raise DatasetViewAccessError(f"subject {subject_key} not in the authorized FIT set (CAL/EVAL/unknown → refused)")
            cohort = index.cohort_of(subject_key)             # raises if unknown
            raw = index.raw_of(subject_key)
            reads.append(subject_key)
            return reader.read_subject_windows(disease, cohort, raw, cps[cohort])

        def _read_label(subject_key):
            # labels are readable ONLY for FIT subjects and ONLY via this training view (never the embedding-dump view)
            if subject_key not in allowed:
                raise DatasetViewAccessError(f"label for {subject_key} refused (not in the authorized FIT set)")
            cohort = index.cohort_of(subject_key)
            raw = index.raw_of(subject_key)
            return reader.read_subject_label(disease, cohort, raw, cps[cohort])

        # only these are set as attributes; reader/cohort_paths live in the closure above (not on the object)
        self.read_windows = _read
        self.read_label = _read_label
        self.allowed_subject_keys = allowed
        self._reads = reads

    @property
    def reads(self):
        return list(self._reads)
