"""Guard (Stage-1B15 review fix): the real build ABORTS fail-closed (RepairStagingError) when the repair staging root is invalid, and
does so BEFORE any factory is instantiated / any read — the create/validate is wired into run_stage1b_build's post-gate factory branch.
Synthetic only."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import stage1b_build as B
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_repair_staging as RS
from acar.v5.tests._util import expect_raises, ok, stage1b_auth, stage1b_lock, stage1b_full_plan

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL


def _boom(ctx):
    raise AssertionError("factory must NOT be instantiated when repair-staging validation fails")


def test_build_aborts_when_staging_inside_run_root():
    with tempfile.TemporaryDirectory() as out:
        auth = stage1b_auth(protocol_tag_target_sha=FULL)
        bad = os.path.join(out, auth["run_id"], "stg")         # inside output_root/run_id (the artifact package) → rejected
        expect_raises(RS.RepairStagingError, lambda: B.run_stage1b_real_build(
            stage1b_full_plan(), auth, stage1b_lock(protocol_tag_target_sha=FULL, run_id=auth["run_id"]),
            output_root=out, repair_staging_root=bad, dev_reader_factory=_boom, trainer_factory=_boom, dumper_factory=_boom))
    ok("a real build with a staging root inside output_root/run_id → RepairStagingError before any factory instantiation")


def test_build_aborts_when_staging_is_symlink():
    with tempfile.TemporaryDirectory() as out:
        target = tempfile.mkdtemp()
        link = os.path.join(tempfile.mkdtemp(), "stg_link")
        os.symlink(target, link)
        auth = stage1b_auth(protocol_tag_target_sha=FULL)
        expect_raises(RS.RepairStagingError, lambda: B.run_stage1b_real_build(
            stage1b_full_plan(), auth, stage1b_lock(protocol_tag_target_sha=FULL, run_id=auth["run_id"]),
            output_root=out, repair_staging_root=link, dev_reader_factory=_boom, trainer_factory=_boom, dumper_factory=_boom))
    ok("a real build with a symlinked staging root → RepairStagingError before any factory instantiation")


def main():
    print("ACAR v5 Stage-1B15 guard: real build aborts on a bad repair staging root")
    test_build_aborts_when_staging_inside_run_root()
    test_build_aborts_when_staging_is_symlink()
    print("ALL V5 STAGE1B15-REAL-BUILD-ABORTS-BAD-STAGING GUARDS PASS")


if __name__ == "__main__":
    main()
