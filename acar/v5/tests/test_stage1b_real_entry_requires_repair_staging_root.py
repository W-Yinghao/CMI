"""Guard (Stage-1B15): the production real-run entry REQUIRES an explicit repair_staging_root (no silent default temp dir). Synthetic
only (fakes + temp files)."""
from __future__ import annotations
import inspect
import tempfile
from acar.v5.substrate import stage1b_build as B
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.tests._util import (expect_raises, ok, stage1b_auth, stage1b_lock, stage1b_full_plan, stage1b_repair_staging_root,
                                 FakeDevReader, FakeFileTrainer, FakeFileDumper)

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL


def test_signature_exposes_repair_staging_root():
    params = list(inspect.signature(B.run_stage1b_real_build).parameters)
    assert "repair_staging_root" in params and "output_root" in params, params
    ok("run_stage1b_real_build signature exposes repair_staging_root (explicit runtime parameter)")


def test_empty_repair_staging_root_rejected():
    with tempfile.TemporaryDirectory() as d:
        expect_raises(B.Stage1bBuildError, lambda: B.run_stage1b_real_build(
            stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL), stage1b_lock(protocol_tag_target_sha=FULL),
            output_root=d, repair_staging_root="", dev_reader_factory=lambda ctx: FakeDevReader(),
            trainer_factory=lambda ctx: FakeFileTrainer(ctx.output_root, ctx.run_id),
            dumper_factory=lambda ctx: FakeFileDumper(ctx.output_root, ctx.run_id)))
    ok("run_stage1b_real_build with an empty repair_staging_root → Stage1bBuildError (no silent default)")


def test_valid_repair_staging_root_builds_30():
    with tempfile.TemporaryDirectory() as d:
        rep = B.run_stage1b_real_build(
            stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL), stage1b_lock(protocol_tag_target_sha=FULL),
            output_root=d, repair_staging_root=stage1b_repair_staging_root(), dev_reader_factory=lambda ctx: FakeDevReader(),
            trainer_factory=lambda ctx: FakeFileTrainer(ctx.output_root, ctx.run_id),
            dumper_factory=lambda ctx: FakeFileDumper(ctx.output_root, ctx.run_id))
        assert rep["status"] == "STAGE1B_BUILT" and rep["n_registered"] == 30
    ok("run_stage1b_real_build with a valid repair_staging_root → 30 registered (fakes + temp files)")


def main():
    print("ACAR v5 Stage-1B15 guard: real entry requires repair staging root")
    test_signature_exposes_repair_staging_root()
    test_empty_repair_staging_root_rejected()
    test_valid_repair_staging_root_builds_30()
    print("ALL V5 STAGE1B15-REAL-ENTRY-REQUIRES-STAGING GUARDS PASS")


if __name__ == "__main__":
    main()
