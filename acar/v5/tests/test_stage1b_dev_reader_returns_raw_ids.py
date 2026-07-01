"""Guard (Stage-1B4): the real DEV reader returns RAW subject ids (not namespaced), and the subject index rejects a namespaced
raw id. Synthetic only (temp BIDS-like dirs; no DEV)."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import real_dev_reader as RDR
from acar.v5.substrate import subject_index as SI
from acar.v5.tests._util import expect_raises, ok


def test_lists_raw_sub_ids():
    with tempfile.TemporaryDirectory() as d:
        for name in ("sub-001", "sub-002", "sub-hc7", "notasub", "dataset_description.json"):
            if name.endswith(".json"):
                open(os.path.join(d, name), "w").close()
            else:
                os.makedirs(os.path.join(d, name))
        subs = RDR.RealBidsDevReader().list_subjects("PD", "ds002778", d)
        assert subs == ["sub-001", "sub-002", "sub-hc7"], subs      # RAW ids only; non-sub entries ignored
        assert all("/" not in s for s in subs)
    ok("RealBidsDevReader.list_subjects returns RAW sub-* ids (no namespacing), ignoring non-subject entries")


def test_missing_or_empty_cohort_dir_rejected():
    expect_raises(RDR.RealReaderError, lambda: RDR.RealBidsDevReader().list_subjects("PD", "ds002778", "/no/such/dir"))
    with tempfile.TemporaryDirectory() as d:
        expect_raises(RDR.RealReaderError, lambda: RDR.RealBidsDevReader().list_subjects("PD", "ds002778", d))
    ok("missing cohort dir / no sub-* dirs → RealReaderError")


def test_index_rejects_namespaced_raw():
    expect_raises(SI.SubjectIndexError,
                  lambda: SI.build_subject_index("PD", {"ds002778": ["ds002778/sub-1"], "ds003490": ["sub-2"], "ds004584": ["sub-3"]}))
    ok("a namespaced raw id ('ds002778/sub-1') → SubjectIndexError (readers must return raw ids)")


def test_signal_read_is_seam():
    expect_raises((NotImplementedError, ModuleNotFoundError, ImportError),
                  lambda: RDR.RealBidsDevReader().read_subject_windows("PD", "ds002778", "sub-1", "/tmp/x"))
    ok("read_subject_windows is the remaining seam (raises; real mne DSP wired at the Stage-1B run)")


def main():
    print("ACAR v5 Stage-1B4 guard: dev reader returns raw ids")
    test_lists_raw_sub_ids()
    test_missing_or_empty_cohort_dir_rejected()
    test_index_rejects_namespaced_raw()
    test_signal_read_is_seam()
    print("ALL V5 STAGE1B-DEV-READER-RAW-IDS GUARDS PASS")


if __name__ == "__main__":
    main()
