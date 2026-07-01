"""ACAR V5 Stage-1B REAL DEV reader (BIDS). Constructed by its factory with a gate-issued Stage1BExecutionContext, AFTER the gate.
NO heavy import at module level (mne is lazy inside the signal read). `list_subjects` is a plain directory listing returning RAW
subject ids; the mne signal read + label read are the remaining seams wired ONLY at an authorized Stage-1B run. Paths are validated
against the context's approved per-disease source paths.
"""
from __future__ import annotations
import os


class RealReaderError(RuntimeError):
    pass


class RealBidsDevReader:
    def __init__(self, context):
        if context is None:
            raise RealReaderError("RealBidsDevReader requires a gate-issued Stage1BExecutionContext")
        self._ctx = context

    def _check_approved(self, disease, cohort, path):
        approved = self._ctx.source_paths(disease)             # raises if disease not approved
        if approved.get(cohort) != path:
            raise RealReaderError(f"{disease}/{cohort}: path {path!r} is not the approved source for this run")

    def list_subjects(self, disease, cohort, path):
        self._check_approved(disease, cohort, path)
        if not path or not os.path.isdir(path):
            raise RealReaderError(f"{disease}/{cohort}: BIDS cohort dir not found: {path}")
        subs = sorted(d for d in os.listdir(path)
                      if d.startswith("sub-") and os.path.isdir(os.path.join(path, d)))
        if not subs:
            raise RealReaderError(f"{disease}/{cohort}: no sub-* directories under {path}")
        return subs                                            # RAW ids, e.g. "sub-001" (never namespaced)

    def read_subject_windows(self, disease, cohort, subject, path):
        self._check_approved(disease, cohort, path)
        import mne  # noqa: F401  (lazy)
        raise NotImplementedError("real signal read (mne DSP → SubjectWindows per preprocessing_config) wired at the Stage-1B run")

    def read_subject_label(self, disease, cohort, subject, path):
        self._check_approved(disease, cohort, path)
        raise NotImplementedError("real label read (FIT training only) wired at the Stage-1B run")


def make_real_dev_reader(context):
    """Factory — construct AFTER the full-build gate, bound to the run's execution context."""
    return RealBidsDevReader(context)
