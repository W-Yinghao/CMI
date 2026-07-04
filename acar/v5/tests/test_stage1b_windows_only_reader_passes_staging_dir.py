"""Guard (Stage-1B15): the label-incapable WindowsOnlyReader facade (embedding view) ALSO passes a non-null staging_dir under the
context's repair_staging_root into preprocess_subject, and fail-closes without one. Synthetic (monkeypatched preprocess)."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import real_dev_reader as RDR
from acar.v5.tests._util import ok, expect_raises, stage1b_reader_ctx, capture_preprocess_staging

_PATH = "/approved/PD/ds002778"


def test_windows_only_reader_passes_staging_dir():
    root = tempfile.mkdtemp()
    wor = RDR.make_real_dev_reader(stage1b_reader_ctx("PD", "ds002778", _PATH, root)).windows_only()
    assert isinstance(wor, RDR.WindowsOnlyReader)
    cap, res = capture_preprocess_staging(lambda: wor.read_subject_windows("PD", "ds002778", "sub-9", _PATH))
    assert res == "WINDOWS:PD/ds002778/sub-9"
    assert cap and cap[0] and os.path.realpath(cap[0]).startswith(os.path.realpath(root) + os.sep)
    ok("WindowsOnlyReader.read_subject_windows passes a non-null staging_dir under repair_staging_root to preprocess_subject")


def test_windows_only_reader_fail_closed_without_staging_root():
    wor = RDR.make_real_dev_reader(stage1b_reader_ctx("PD", "ds002778", _PATH, "")).windows_only()
    expect_raises(RDR.RealReaderError, lambda: wor.read_subject_windows("PD", "ds002778", "sub-9", _PATH))
    ok("a windows-only reader with no repair staging root refuses to read (fail-closed)")


def main():
    print("ACAR v5 Stage-1B15 guard: windows-only reader passes staging_dir")
    test_windows_only_reader_passes_staging_dir()
    test_windows_only_reader_fail_closed_without_staging_root()
    print("ALL V5 STAGE1B15-WINDOWS-ONLY-READER-PASSES-STAGING GUARDS PASS")


if __name__ == "__main__":
    main()
