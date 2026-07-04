"""ACAR V5 Stage-1B REAL DEV reader (BIDS). Constructed by its factory with a gate-issued Stage1BExecutionContext, AFTER the gate.
NO heavy import at module level (mne is lazy inside the signal read). `list_subjects` is a plain directory listing returning RAW
subject ids; the mne signal read + label read are the remaining seams wired ONLY at an authorized Stage-1B run. Paths are validated
against the context's approved per-disease source paths.

Stage-1B15: BOTH signal-read paths (RealBidsDevReader + the label-incapable WindowsOnlyReader facade) go through
`_read_windows_with_repair`, which drives `real_mne_reader.preprocess_subject(..., staging_dir=...)` — the reviewed Stage-1B12/1B13/
1B14 BrainVision header repair (marker synth / pointer rewrite / channels.tsv ordinal rename). The staging dir is a fresh PER-CALL
temp subdir under the gate-issued context's validated `repair_staging_root` (EPHEMERAL scratch, never a registered artifact); it is
fail-closed (no staging root → refuse to read) and auto-removed after each read. The label firewall is unchanged: WindowsOnlyReader
still carries only the (label-free) context — no read_subject_label and no reference to the label-capable reader.
"""
from __future__ import annotations
import os


class RealReaderError(RuntimeError):
    pass


def _check_approved(ctx, disease, cohort, path):
    approved = ctx.source_paths(disease)                       # raises if disease not approved
    if approved.get(cohort) != path:
        raise RealReaderError(f"{disease}/{cohort}: path {path!r} is not the approved source for this run")


def _read_windows_with_repair(ctx, disease, cohort, subject, path):
    """Read one subject's SIGNAL-ONLY SubjectWindows through the reviewed Stage-1B12/1B13/1B14 BrainVision read-repair. Fail-closed:
    the gate-issued context MUST carry a validated repair staging root; a fresh PER-CALL temp subdir is created under it (so repeated
    reads across folds/seeds/phases never collide), passed to preprocess_subject as `staging_dir`, and cleaned up after the read (the
    returned SubjectWindows holds the windows in memory — the ephemeral repaired headers are no longer needed)."""
    import tempfile
    from acar.v5.substrate import real_mne_reader as RMR       # RMR lazy-imports mne inside preprocess_subject
    staging_root = getattr(ctx, "repair_staging_root", "")
    if not (staging_root and os.path.isdir(staging_root)):
        raise RealReaderError("real read requires a validated repair staging root in the execution context (Stage-1B15) — none present")
    subject_dir = os.path.join(path, subject)
    with tempfile.TemporaryDirectory(dir=staging_root) as sdir:   # PER-CALL scratch; auto-removed after the read
        return RMR.preprocess_subject(disease, cohort, subject, subject_dir, staging_dir=sdir)


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
        return _read_windows_with_repair(self._ctx, disease, cohort, subject, path)   # SIGNAL ONLY → validated SubjectWindows


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
        return _read_windows_with_repair(self._ctx, disease, cohort, subject, path)   # SIGNAL ONLY → validated SubjectWindows

    def read_subject_label(self, disease, cohort, subject, path):
        # reachable ONLY via AuthorizedFitDatasetView.read_label (FIT training only); COHORT-EXACT mapping, fail-closed
        self._check_approved(disease, cohort, path)
        from acar.v5.substrate import cohort_label_spec as CLS
        return CLS.resolve_label(disease, cohort, subject, os.path.join(path, "participants.tsv"))

    def subject_label_resolvable(self, disease, cohort, subject, path):
        # eligibility check (build-time): returns a BOOLEAN only — the label VALUE never leaves this method (no leak into routing/dump)
        self._check_approved(disease, cohort, path)
        from acar.v5.substrate import cohort_label_spec as CLS
        return CLS.label_resolvable(disease, cohort, subject, os.path.join(path, "participants.tsv"))


def make_real_dev_reader(context):
    """Factory — construct AFTER the full-build gate, bound to the run's execution context."""
    return RealBidsDevReader(context)
