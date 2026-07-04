"""Guard (Stage-1B15): the repair staging root must NEVER overlap the run's hash-bound artifact root (output_root/run_id) — scratch
must never become part of the registered package. Also rejects a non-empty existing directory."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import stage1b_repair_staging as RS
from acar.v5.tests._util import ok, expect_raises


def test_rejects_inside_run_root():
    out = tempfile.mkdtemp()
    expect_raises(RS.RepairStagingError, lambda: RS.validate_repair_staging_root(
        os.path.join(out, "run-1", "stg"), output_root=out, run_id="run-1", approved_source_paths=[]))
    ok("a staging root inside output_root/run_id → RepairStagingError")


def test_rejects_equal_to_run_root():
    out = tempfile.mkdtemp()
    expect_raises(RS.RepairStagingError, lambda: RS.validate_repair_staging_root(
        os.path.join(out, "run-1"), output_root=out, run_id="run-1", approved_source_paths=[]))
    ok("a staging root equal to output_root/run_id → RepairStagingError")


def test_rejects_nonempty_existing_dir():
    d = tempfile.mkdtemp()
    open(os.path.join(d, "leftover"), "w").close()
    expect_raises(RS.RepairStagingError, lambda: RS.validate_repair_staging_root(
        d, output_root="/out", run_id="run-1", approved_source_paths=[]))
    ok("a non-empty existing staging root → RepairStagingError (must be absent or empty at launch)")


def test_accepts_absent_or_empty_disjoint_root():
    out = tempfile.mkdtemp()
    empty = tempfile.mkdtemp()                                 # existing empty dir, disjoint from out + raw
    assert RS.validate_repair_staging_root(empty, output_root=out, run_id="run-1", approved_source_paths=["/raw/PD/ds002778"]) is True
    absent = os.path.join(tempfile.mkdtemp(), "stg")           # absent child
    assert RS.validate_repair_staging_root(absent, output_root=out, run_id="run-1", approved_source_paths=[]) is True
    ok("an empty (or absent) absolute staging root disjoint from raw paths + the run root validates")


def main():
    print("ACAR v5 Stage-1B15 guard: repair staging root rejects the output/run artifact root")
    test_rejects_inside_run_root()
    test_rejects_equal_to_run_root()
    test_rejects_nonempty_existing_dir()
    test_accepts_absent_or_empty_disjoint_root()
    print("ALL V5 STAGE1B15-STAGING-REJECTS-RUN-ROOT GUARDS PASS")


if __name__ == "__main__":
    main()
