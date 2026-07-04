"""Guard (Stage-1B15): the repair staging root must not be a symlink (a symlink could redirect scratch into the raw tree or the
artifact package). Rejected both at validation and after creation."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import stage1b_repair_staging as RS
from acar.v5.tests._util import ok, expect_raises


def test_rejects_symlink_staging_root():
    target = tempfile.mkdtemp()                                # a real (empty) dir
    link = os.path.join(tempfile.mkdtemp(), "stg_link")
    os.symlink(target, link)
    assert os.path.islink(link)
    expect_raises(RS.RepairStagingError, lambda: RS.validate_repair_staging_root(
        link, output_root="/out", run_id="run-1", approved_source_paths=[]))
    ok("a symlinked repair_staging_root → RepairStagingError")


def test_create_rejects_symlink():
    target = tempfile.mkdtemp()
    link = os.path.join(tempfile.mkdtemp(), "stg_link")
    os.symlink(target, link)
    expect_raises(RS.RepairStagingError, lambda: RS.create_repair_staging_root(
        link, output_root="/out", run_id="run-1", approved_source_paths=[]))
    ok("create_repair_staging_root also refuses a symlink")


def main():
    print("ACAR v5 Stage-1B15 guard: repair staging root rejects symlink")
    test_rejects_symlink_staging_root()
    test_create_rejects_symlink()
    print("ALL V5 STAGE1B15-STAGING-REJECTS-SYMLINK GUARDS PASS")


if __name__ == "__main__":
    main()
