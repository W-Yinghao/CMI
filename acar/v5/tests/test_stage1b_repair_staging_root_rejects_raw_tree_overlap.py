"""Guard (Stage-1B15): the repair staging root must NEVER overlap an approved raw cohort source path (neither inside it nor
containing it) — scratch must never touch the raw DEV tree. Also rejects a non-absolute path."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import stage1b_repair_staging as RS
from acar.v5.tests._util import ok, expect_raises


def test_rejects_staging_inside_raw_source():
    raw = tempfile.mkdtemp()
    expect_raises(RS.RepairStagingError, lambda: RS.validate_repair_staging_root(
        os.path.join(raw, "stg"), output_root="/out", run_id="run-1", approved_source_paths=[raw]))
    ok("a staging root INSIDE an approved raw source path → RepairStagingError")


def test_rejects_staging_containing_raw_source():
    parent = tempfile.mkdtemp()
    raw = os.path.join(parent, "cohort")
    os.makedirs(raw)
    expect_raises(RS.RepairStagingError, lambda: RS.validate_repair_staging_root(
        parent, output_root="/out", run_id="run-1", approved_source_paths=[raw]))
    ok("a staging root that CONTAINS an approved raw source path → RepairStagingError")


def test_rejects_equal_to_raw_source():
    raw = tempfile.mkdtemp()
    expect_raises(RS.RepairStagingError, lambda: RS.validate_repair_staging_root(
        raw, output_root="/out", run_id="run-1", approved_source_paths=[raw]))
    ok("a staging root EQUAL to an approved raw source path → RepairStagingError")


def test_rejects_relative_path():
    expect_raises(RS.RepairStagingError, lambda: RS.validate_repair_staging_root(
        "relative/staging", output_root="/out", run_id="run-1", approved_source_paths=[]))
    ok("a non-absolute repair_staging_root → RepairStagingError")


def main():
    print("ACAR v5 Stage-1B15 guard: repair staging root rejects raw-tree overlap")
    test_rejects_staging_inside_raw_source()
    test_rejects_staging_containing_raw_source()
    test_rejects_equal_to_raw_source()
    test_rejects_relative_path()
    print("ALL V5 STAGE1B15-STAGING-REJECTS-RAW-OVERLAP GUARDS PASS")


if __name__ == "__main__":
    main()
