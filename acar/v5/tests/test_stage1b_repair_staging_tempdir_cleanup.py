"""Guard (Stage-1B15): the per-call staging subdir is EPHEMERAL — it is removed after each read (the returned SubjectWindows holds the
windows in memory; the repaired headers are scratch). Synthetic (monkeypatched preprocess captures the subdir path)."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import real_dev_reader as RDR
from acar.v5.tests._util import ok, stage1b_reader_ctx, capture_preprocess_staging

_PATH = "/approved/PD/ds002778"


def test_per_call_staging_subdir_removed_after_read():
    root = tempfile.mkdtemp()
    reader = RDR.make_real_dev_reader(stage1b_reader_ctx("PD", "ds002778", _PATH, root))
    cap, _ = capture_preprocess_staging(lambda: reader.read_subject_windows("PD", "ds002778", "sub-1", _PATH))
    assert cap and cap[0]
    assert not os.path.exists(cap[0])                           # the per-call TemporaryDirectory was cleaned up after the read
    assert os.path.isdir(root) and os.listdir(root) == []       # the repair staging root itself remains, now empty
    ok("the per-call staging subdir is removed after each read; the repair staging root remains empty scratch")


def main():
    print("ACAR v5 Stage-1B15 guard: repair staging tempdir cleanup")
    test_per_call_staging_subdir_removed_after_read()
    print("ALL V5 STAGE1B15-REPAIR-STAGING-TEMPDIR-CLEANUP GUARDS PASS")


if __name__ == "__main__":
    main()
