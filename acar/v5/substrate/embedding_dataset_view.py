"""ACAR V5 Stage-1B LABEL-FREE embedding dataset view (pure/stdlib). Used AFTER the encoder/source-state are frozen, to dump
routing features/embeddings for ALL fold subjects (TRAIN/VAL/CAL/EVAL). It exposes read_windows ONLY — there is NO read_label
method and no raw-root attribute — so the embedding/feature-dump code physically cannot read labels. Closure-backed like the FIT
view.
"""
from __future__ import annotations
from acar.v5.substrate.fit_dataset_view import DatasetViewAccessError


class AuthorizedEmbeddingDatasetView:
    def __init__(self, index, allowed_subject_keys, reader, cohort_paths):
        allowed = frozenset(allowed_subject_keys)
        disease = index.disease
        cps = dict(cohort_paths)
        reads = []

        def _read(subject_key):
            if subject_key not in allowed:
                raise DatasetViewAccessError(f"subject {subject_key} not in this fold's embedding set")
            cohort = index.cohort_of(subject_key)
            raw = index.raw_of(subject_key)
            reads.append(subject_key)
            return reader.read_subject_windows(disease, cohort, raw, cps[cohort])   # SIGNAL ONLY — no label read exists

        self.read_windows = _read
        self.allowed_subject_keys = allowed
        self._reads = reads
        # deliberately NO read_label attribute — label access is impossible through this view.

    @property
    def reads(self):
        return list(self._reads)
