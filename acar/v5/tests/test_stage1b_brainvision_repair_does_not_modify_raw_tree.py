"""Guard (Stage-1B12): read-repair NEVER writes into the raw data tree and never modifies the original raw files. All repaired
artifacts live under the ephemeral staging dir; a staging dir that overlaps the raw recording dir is rejected fail-closed."""
from __future__ import annotations
import hashlib
import os
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, expect_raises, make_brainvision_triplet


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_raw_tree_untouched_and_artifacts_in_staging():
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-1448_task-Rest_eeg", ("Fp1", "Fp2"), with_marker=False)
    before = {f: _sha(os.path.join(raw_dir, f)) for f in sorted(os.listdir(raw_dir))}
    staging = tempfile.mkdtemp()
    repaired, man = BR.apply_repair(BR.plan_repair("SCZ", "ds003944", "sub-1448", vhdr), staging)
    after = {f: _sha(os.path.join(raw_dir, f)) for f in sorted(os.listdir(raw_dir))}
    assert before == after, "raw tree files changed or new files appeared in the raw dir"
    assert set(before) == {"sub-1448_task-Rest_eeg.eeg", "sub-1448_task-Rest_eeg.vhdr"}   # no .vmrk created in raw dir
    assert os.path.dirname(os.path.realpath(repaired)) == os.path.realpath(staging)
    assert os.path.dirname(os.path.realpath(man["marker_file_target"])) == os.path.realpath(staging)
    ok("repair writes ONLY into staging; the original .vhdr/.eeg are byte-identical and no .vmrk is created in the raw tree")


def test_staging_inside_raw_dir_is_rejected():
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-1448_task-Rest_eeg", ("Fp1", "Fp2"), with_marker=False)
    plan = BR.plan_repair("SCZ", "ds003944", "sub-1448", vhdr)
    expect_raises(BR.BrainvisionReadRepairError, lambda: BR.apply_repair(plan, raw_dir),
                  "staging == raw recording dir must be rejected")
    sub = os.path.join(raw_dir, "staging_child")
    os.makedirs(sub)
    expect_raises(BR.BrainvisionReadRepairError, lambda: BR.apply_repair(plan, sub),
                  "staging inside the raw recording dir must be rejected")
    ok("a staging dir that overlaps the raw recording directory is rejected fail-closed (no writing into the raw tree)")


def main():
    print("ACAR v5 Stage-1B12 guard: read-repair does not modify the raw tree")
    test_raw_tree_untouched_and_artifacts_in_staging()
    test_staging_inside_raw_dir_is_rejected()
    print("ALL V5 STAGE1B12-BV-RAW-TREE-UNTOUCHED GUARDS PASS")


if __name__ == "__main__":
    main()
