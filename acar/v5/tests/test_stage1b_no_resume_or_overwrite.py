"""Guard (Stage-1B9): Stage-1B never resumes/overwrites — re-running a build into a run root that already holds a prior (finalized)
run aborts fail-closed BEFORE any factory/read/train. Synthetic only."""
from __future__ import annotations
import tempfile
from acar.v5.substrate import stage1b_build as B
from acar.v5.tests._util import stage1b_repair_staging_root as _RSR
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_launch_guard as LG
from acar.v5.tests._util import expect_raises, ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeDevReader, FakeFileTrainer, FakeFileDumper

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL


def _run(d):
    return B.run_stage1b_real_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                                    stage1b_lock(protocol_tag_target_sha=FULL), output_root=d, repair_staging_root=_RSR(),
                                    dev_reader_factory=lambda ctx: FakeDevReader(),
                                    trainer_factory=lambda ctx: FakeFileTrainer(ctx.output_root, ctx.run_id),
                                    dumper_factory=lambda ctx: FakeFileDumper(ctx.output_root, ctx.run_id))


def test_second_run_into_same_root_aborts():
    with tempfile.TemporaryDirectory() as d:
        rep = _run(d)                                         # first run finalizes the run root
        assert rep["n_registered"] == 30
        # a factory that would explode if ever instantiated — proves the guard aborts BEFORE factory instantiation
        def _boom(ctx):
            raise AssertionError("factory must NOT be instantiated on a non-fresh run root")
        expect_raises(LG.Stage1bLaunchError,
                      lambda: B.run_stage1b_real_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                                                       stage1b_lock(protocol_tag_target_sha=FULL), output_root=d, repair_staging_root=_RSR(),
                                                       dev_reader_factory=_boom, trainer_factory=_boom, dumper_factory=_boom))
    ok("re-running into an already-finalized run root → Stage1bLaunchError before any factory instantiation (no resume/overwrite)")


def main():
    print("ACAR v5 Stage-1B9 guard: no resume or overwrite")
    test_second_run_into_same_root_aborts()
    print("ALL V5 STAGE1B-NO-RESUME-OVERWRITE GUARDS PASS")


if __name__ == "__main__":
    main()
