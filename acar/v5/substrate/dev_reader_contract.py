"""ACAR V5 Stage-1B DEV reader CONTRACT (pure/stdlib; importing this module reads NOTHING). The REAL BIDS/mne DEV reader is a
separate, later-authorized patch; here we only define the interface + a fail-closed unwired default, so Stage-1B code cannot
accidentally read real data. A reader must expose:
    list_subjects(disease, cohort, path) -> list[str]     (metadata listing; namespaced subject ids)
    read_subject_windows(disease, cohort, subject, path)  (the actual signal read — only at an authorized real run)
"""
from __future__ import annotations


class DevReaderNotWiredError(RuntimeError):
    """Raised when a real Stage-1B DEV read is attempted without an authorized, wired reader."""


class UnwiredDevReader:
    """Default reader: every method fails closed. The CLI uses this, so `--execute` cannot read real data in Stage-1B2."""

    def list_subjects(self, disease, cohort, path):
        raise DevReaderNotWiredError("real DEV reader not wired (Stage-1B2): cannot list subjects")

    def read_subject_windows(self, disease, cohort, subject, path):
        raise DevReaderNotWiredError("real DEV reader not wired (Stage-1B2): cannot read windows")


def require_reader(dev_reader):
    if dev_reader is None:
        raise DevReaderNotWiredError("Stage-1B execute requires an authorized DEV reader (none supplied)")
    for m in ("list_subjects", "read_subject_windows"):
        if not callable(getattr(dev_reader, m, None)):
            raise DevReaderNotWiredError(f"dev_reader is missing required method {m}()")
    return dev_reader
