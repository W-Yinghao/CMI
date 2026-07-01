"""ACAR V5 Stage-1B LABEL-FREE embedding dataset view (pure/stdlib). Used AFTER the encoder/source-state are frozen, to dump
routing features/embeddings for ALL fold subjects (TRAIN/VAL/CAL/EVAL). It exposes read_windows ONLY — there is NO read_label
method and no raw-root attribute — so the embedding/feature-dump code physically cannot read labels. Closure-backed like the FIT
view.

Defense-in-depth (label firewall): the reader handed to this view MUST be a WINDOWS-ONLY reader — it must NOT itself expose a
read_subject_label method. Because the view is closure-backed, an introspecting caller could reach the captured reader via
`read_windows.__closure__`; requiring a label-incapable reader (built via reader.windows_only()) means even that path cannot reach
labels. Construction FAILS CLOSED if a label-capable reader is passed.
"""
from __future__ import annotations
from acar.v5.substrate.fit_dataset_view import DatasetViewAccessError


class AuthorizedEmbeddingDatasetView:
    def __init__(self, index, allowed_subject_keys, reader, cohort_paths):
        if hasattr(reader, "read_subject_label"):
            raise DatasetViewAccessError("embedding view requires a WINDOWS-ONLY reader (no read_subject_label); "
                                         "build it via reader.windows_only()")
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
