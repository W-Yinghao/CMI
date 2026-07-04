"""Guard (Stage-1B15): RealBidsDevReader.read_subject_windows passes a NON-NULL staging_dir (under the context's repair_staging_root)
into real_mne_reader.preprocess_subject — the reviewed BrainVision repair is active in the production read path. Also fail-closed when
the context has no staging root. Synthetic (monkeypatched preprocess; no real read)."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import real_dev_reader as RDR
from acar.v5.tests._util import ok, expect_raises, stage1b_reader_ctx, capture_preprocess_staging

_PATH = "/approved/PD/ds002778"


def test_real_reader_passes_staging_dir_under_root():
    root = tempfile.mkdtemp()
    reader = RDR.make_real_dev_reader(stage1b_reader_ctx("PD", "ds002778", _PATH, root))
    cap, res = capture_preprocess_staging(lambda: reader.read_subject_windows("PD", "ds002778", "sub-1", _PATH))
    assert res == "WINDOWS:PD/ds002778/sub-1"
    assert cap and cap[0] and os.path.realpath(cap[0]).startswith(os.path.realpath(root) + os.sep)
    ok("RealBidsDevReader.read_subject_windows passes a non-null staging_dir under repair_staging_root to preprocess_subject")


def test_real_reader_fail_closed_without_staging_root():
    reader = RDR.make_real_dev_reader(stage1b_reader_ctx("PD", "ds002778", _PATH, ""))   # empty staging root
    expect_raises(RDR.RealReaderError, lambda: reader.read_subject_windows("PD", "ds002778", "sub-1", _PATH))
    ok("a real reader with no repair staging root refuses to read (fail-closed)")


def main():
    print("ACAR v5 Stage-1B15 guard: real reader passes staging_dir")
    test_real_reader_passes_staging_dir_under_root()
    test_real_reader_fail_closed_without_staging_root()
    print("ALL V5 STAGE1B15-REAL-READER-PASSES-STAGING GUARDS PASS")


if __name__ == "__main__":
    main()
