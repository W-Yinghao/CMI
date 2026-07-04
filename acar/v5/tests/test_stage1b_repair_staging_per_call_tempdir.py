"""Guard (Stage-1B15): each read gets a FRESH per-call staging subdir under the repair staging root (so the same subject read across
folds/seeds/phases never collides on repaired-header filenames). Synthetic (monkeypatched preprocess)."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import real_dev_reader as RDR
from acar.v5.tests._util import ok, stage1b_reader_ctx, capture_preprocess_staging

_PATH = "/approved/PD/ds002778"


def test_each_read_uses_a_distinct_subdir_under_root():
    root = tempfile.mkdtemp()
    reader = RDR.make_real_dev_reader(stage1b_reader_ctx("PD", "ds002778", _PATH, root))

    def _two_reads():
        reader.read_subject_windows("PD", "ds002778", "sub-1", _PATH)   # e.g. FIT read
        reader.read_subject_windows("PD", "ds002778", "sub-1", _PATH)   # e.g. later dump read of the SAME subject
        return None

    cap, _ = capture_preprocess_staging(_two_reads)
    assert len(cap) == 2 and cap[0] != cap[1]                            # distinct per-call subdirs (no collision)
    for c in cap:
        assert os.path.realpath(c).startswith(os.path.realpath(root) + os.sep)   # both under the repair staging root
    ok("each read uses a fresh distinct per-call staging subdir under the repair staging root (no cross-read collision)")


def main():
    print("ACAR v5 Stage-1B15 guard: repair staging per-call tempdir")
    test_each_read_uses_a_distinct_subdir_under_root()
    print("ALL V5 STAGE1B15-REPAIR-STAGING-PER-CALL-TEMPDIR GUARDS PASS")


if __name__ == "__main__":
    main()
