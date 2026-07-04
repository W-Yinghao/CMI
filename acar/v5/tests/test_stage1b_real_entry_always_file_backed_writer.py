"""Guard (Stage-1B9b): the PRODUCTION real-run entry cannot be pointed at a custom artifact writer — it always uses the file-backed
writer, so a real run can only ever emit the file-backed, hash-bound package (registry.json + FINALIZED.json). Synthetic temp files
only."""
from __future__ import annotations
import inspect
import os
import tempfile
from acar.v5.substrate import stage1b_build as B
from acar.v5.tests._util import stage1b_repair_staging_root as _RSR
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_registry_io as RIO
from acar.v5.tests._util import expect_raises, ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeDevReader, FakeFileTrainer, FakeFileDumper

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL
RUN = "run-syn-0001"


def test_production_entry_has_no_artifact_writer_param():
    params = list(inspect.signature(B.run_stage1b_real_build).parameters)
    assert "artifact_writer" not in params, params        # no custom-writer override on the production entry
    assert "dev_reader" not in params and "trainer" not in params and "dumper" not in params
    assert all(p in params for p in ("dev_reader_factory", "trainer_factory", "dumper_factory", "output_root"))
    ok("run_stage1b_real_build exposes NO artifact_writer override (factories + output_root only)")


def test_custom_artifact_writer_rejected():
    with tempfile.TemporaryDirectory() as d:
        expect_raises(TypeError,
                      lambda: B.run_stage1b_real_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                                                       stage1b_lock(protocol_tag_target_sha=FULL), output_root=d, repair_staging_root=_RSR(),
                                                       dev_reader_factory=lambda ctx: FakeDevReader(),
                                                       trainer_factory=lambda ctx: FakeFileTrainer(ctx.output_root, ctx.run_id),
                                                       dumper_factory=lambda ctx: FakeFileDumper(ctx.output_root, ctx.run_id),
                                                       artifact_writer=lambda *a, **k: {}))   # not an accepted kwarg
    ok("passing a custom artifact_writer to the production entry → TypeError (cannot override the file-backed writer)")


def test_default_writes_file_backed_package():
    with tempfile.TemporaryDirectory() as d:
        rep = B.run_stage1b_real_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                                       stage1b_lock(protocol_tag_target_sha=FULL), output_root=d, repair_staging_root=_RSR(),
                                       dev_reader_factory=lambda ctx: FakeDevReader(),
                                       trainer_factory=lambda ctx: FakeFileTrainer(ctx.output_root, ctx.run_id),
                                       dumper_factory=lambda ctx: FakeFileDumper(ctx.output_root, ctx.run_id))
        assert rep["n_registered"] == 30
        run_root = os.path.join(d, RUN)
        assert os.path.isfile(os.path.join(run_root, RIO.REGISTRY_FILE))
        assert os.path.isfile(os.path.join(run_root, RIO.MARKER_FILE))
        assert len(RIO.admit_run(d, RUN)._entries) == 30     # the file-backed package is admissible
    ok("the production entry (default) writes the file-backed package: registry.json + FINALIZED.json, admissible via admit_run")


def main():
    print("ACAR v5 Stage-1B9b guard: real entry always file-backed writer")
    test_production_entry_has_no_artifact_writer_param()
    test_custom_artifact_writer_rejected()
    test_default_writes_file_backed_package()
    print("ALL V5 STAGE1B-REAL-ENTRY-FILE-BACKED GUARDS PASS")


if __name__ == "__main__":
    main()
