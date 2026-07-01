"""ACAR V5 Stage-1B REAL DEV reader (BIDS). NO heavy import at module level (mne is imported lazily, only inside the signal read).
`list_subjects` is a plain directory listing that returns RAW subject ids ("sub-001"); the signal read is the remaining seam wired
ONLY at an authorized Stage-1B real run. Constructed via its factory AFTER the gate. It performs NO real DEV read in Stage-1B4 —
`read_subject_windows` raises until the real run wires the mne DSP.
"""
from __future__ import annotations
import os


class RealReaderError(RuntimeError):
    pass


class RealBidsDevReader:
    """Returns RAW subject ids by listing sub-* dirs in a cohort BIDS root. The orchestrator builds the canonical SubjectKey."""

    def list_subjects(self, disease, cohort, path):
        if not path or not os.path.isdir(path):
            raise RealReaderError(f"{disease}/{cohort}: BIDS cohort dir not found: {path}")
        subs = sorted(d for d in os.listdir(path)
                      if d.startswith("sub-") and os.path.isdir(os.path.join(path, d)))
        if not subs:
            raise RealReaderError(f"{disease}/{cohort}: no sub-* directories under {path}")
        return subs                                            # RAW ids, e.g. "sub-001" (never namespaced)

    def read_subject_windows(self, disease, cohort, subject, path):
        # LAZY heavy import + real DSP is wired only at the authorized Stage-1B real run.
        import mne  # noqa: F401  (lazy; not imported at module load)
        raise NotImplementedError("real signal read (mne DSP → 19ch/128Hz/0.5-45Hz/4s windows) is wired at the Stage-1B real run")


def make_real_dev_reader():
    """Factory — construct AFTER the full-build gate (see run_stage1b_real_build)."""
    return RealBidsDevReader()
