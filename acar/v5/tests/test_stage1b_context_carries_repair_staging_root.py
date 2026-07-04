"""Guard (Stage-1B15): the gate-issued execution context carries the repair_staging_root, and the production build validates + CREATES
it (after the gate) and threads it into the context handed to the factories. Synthetic only."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import stage1b_build as B
from acar.v5.substrate import stage1b_execution_context as EC
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.tests._util import (ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeDevReader, FakeFileTrainer, FakeFileDumper)

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL


def test_context_field_present_and_defaults_empty():
    plan = {"fold_contained_refs": [{"disease": "PD", "source_paths_by_cohort": {"ds002778": "/raw/PD/ds002778"}}]}
    ctx = EC.build_execution_context(stage1b_auth(protocol_tag_target_sha=FULL), {}, plan, output_root="/out",
                                     repair_staging_root="/scratch/acar/run-1/stg")
    assert ctx.repair_staging_root == "/scratch/acar/run-1/stg"
    ctx2 = EC.build_execution_context(stage1b_auth(protocol_tag_target_sha=FULL), {}, plan, output_root="/out")
    assert ctx2.repair_staging_root == ""                       # empty on the synthetic path (no staging)
    ok("Stage1BExecutionContext carries repair_staging_root (empty by default)")


def test_real_build_creates_and_threads_staging_root_into_context():
    captured = {}

    def rf(ctx):
        captured["root"] = ctx.repair_staging_root
        return FakeDevReader()

    with tempfile.TemporaryDirectory() as out:
        stg = os.path.join(tempfile.mkdtemp(), "stg")           # absent child → the build must create it after the gate
        B.run_stage1b_real_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                                 stage1b_lock(protocol_tag_target_sha=FULL), output_root=out, repair_staging_root=stg,
                                 dev_reader_factory=rf,
                                 trainer_factory=lambda c: FakeFileTrainer(c.output_root, c.run_id),
                                 dumper_factory=lambda c: FakeFileDumper(c.output_root, c.run_id))
        assert captured["root"] == os.path.abspath(stg) and os.path.isdir(captured["root"])
    ok("run_stage1b_real_build validates + CREATES the repair staging root (after the gate) and threads it into the context")


def main():
    print("ACAR v5 Stage-1B15 guard: context carries repair staging root")
    test_context_field_present_and_defaults_empty()
    test_real_build_creates_and_threads_staging_root_into_context()
    print("ALL V5 STAGE1B15-CONTEXT-CARRIES-STAGING GUARDS PASS")


if __name__ == "__main__":
    main()
