"""ACAR V5 Stage-1B REAL DEV reader (BIDS). Constructed by its factory with a gate-issued Stage1BExecutionContext, AFTER the gate.
NO heavy import at module level (mne is lazy inside the signal read). `list_subjects` is a plain directory listing returning RAW
subject ids; the mne signal read + label read are the remaining seams wired ONLY at an authorized Stage-1B run. Paths are validated
against the context's approved per-disease source paths.
"""
from __future__ import annotations
import os


class RealReaderError(RuntimeError):
    pass


def _check_approved(ctx, disease, cohort, path):
    approved = ctx.source_paths(disease)                       # raises if disease not approved
    if approved.get(cohort) != path:
        raise RealReaderError(f"{disease}/{cohort}: path {path!r} is not the approved source for this run")


class WindowsOnlyReader:
    """A label-INCAPABLE reader facade for the embedding view. It holds ONLY the execution context (which has no label capability)
    and exposes read_subject_windows — there is no read_subject_label here and no reference to any object that has one, so even a
    closure-introspecting embedding dumper cannot reach labels through it."""

    def __init__(self, context):
        if context is None:
            raise RealReaderError("WindowsOnlyReader requires a gate-issued Stage1BExecutionContext")
        self._ctx = context

    def read_subject_windows(self, disease, cohort, subject, path):
        _check_approved(self._ctx, disease, cohort, path)
        from acar.v5.substrate import real_mne_reader as RMR   # RMR lazy-imports mne inside preprocess_subject
        subject_dir = os.path.join(path, subject)
        return RMR.preprocess_subject(disease, cohort, subject, subject_dir)   # SIGNAL ONLY → validated SubjectWindows


class RealBidsDevReader:
    def __init__(self, context):
        if context is None:
            raise RealReaderError("RealBidsDevReader requires a gate-issued Stage1BExecutionContext")
        self._ctx = context

    def _check_approved(self, disease, cohort, path):
        _check_approved(self._ctx, disease, cohort, path)

    def windows_only(self):
        """A label-incapable facade for the embedding view (bound only to the context, not to this label-capable reader)."""
        return WindowsOnlyReader(self._ctx)

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
        from acar.v5.substrate import real_mne_reader as RMR   # RMR lazy-imports mne inside preprocess_subject
        subject_dir = os.path.join(path, subject)
        return RMR.preprocess_subject(disease, cohort, subject, subject_dir)   # SIGNAL ONLY → validated SubjectWindows

    def read_subject_label(self, disease, cohort, subject, path):
        # reachable ONLY via AuthorizedFitDatasetView.read_label (FIT training only); pinned mapping, fail-closed
        self._check_approved(disease, cohort, path)
        from acar.v5.substrate import stage1b_label_source as LS
        participants_tsv = os.path.join(path, "participants.tsv")
        return LS.resolve_subject_label(participants_tsv, subject)


def make_real_dev_reader(context):
    """Factory — construct AFTER the full-build gate, bound to the run's execution context."""
    return RealBidsDevReader(context)
