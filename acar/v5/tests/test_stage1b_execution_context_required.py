"""Guard (Stage-1B5): the real reader/trainer require a gate-issued execution context; the context pins run_id/output_root/the 30
approved refs/approved source paths; the reader refuses a non-approved source path. Synthetic only."""
from __future__ import annotations
from acar.v5.substrate import stage1b_execution_context as EC
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import real_dev_reader as RDR
from acar.v5.substrate import real_trainer as RT
from acar.v5.tests._util import expect_raises, ok, stage1b_auth, stage1b_full_plan

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL


def _ctx(output_root="/run/out"):
    return EC.build_execution_context(stage1b_auth(protocol_tag_target_sha=FULL), {}, stage1b_full_plan(), output_root=output_root)


def test_context_fields():
    c = _ctx()
    assert c.run_id == "run-syn-0001" and c.output_root == "/run/out"
    assert c.approved_fold_refs == frozenset(SA.CANONICAL_FOLD_REFS) and len(c.approved_fold_refs) == 30
    assert set(c.source_paths("PD")) == {"ds002778", "ds003490", "ds004584"}
    assert c.is_approved_ref("PD/fold0/seed20260711") and not c.is_approved_ref("PD/fold9/seed20260711")
    ok("execution context pins run_id/output_root/30 approved refs/per-disease source paths")


def test_empty_output_root_rejected():
    expect_raises(EC.Stage1bContextError, lambda: _ctx(output_root=""))
    ok("build_execution_context requires a non-empty output_root")


def test_real_reader_trainer_require_context():
    expect_raises(RDR.RealReaderError, lambda: RDR.RealBidsDevReader(None))
    expect_raises(RT.RealTrainerError, lambda: RT.RealSubstrateTrainer(None))
    assert isinstance(RDR.make_real_dev_reader(_ctx()), RDR.RealBidsDevReader)
    assert isinstance(RT.make_real_trainer(_ctx()), RT.RealSubstrateTrainer)
    ok("RealBidsDevReader / RealSubstrateTrainer require a context (None → error); factories build them from the context")


def test_reader_refuses_non_approved_path():
    reader = RDR.make_real_dev_reader(_ctx())
    expect_raises(RDR.RealReaderError, lambda: reader.list_subjects("PD", "ds002778", "/not/the/approved/path"))
    ok("the real reader refuses a source path not equal to the context-approved one")


def main():
    print("ACAR v5 Stage-1B5 guard: execution context required")
    test_context_fields()
    test_empty_output_root_rejected()
    test_real_reader_trainer_require_context()
    test_reader_refuses_non_approved_path()
    print("ALL V5 STAGE1B-EXECUTION-CONTEXT GUARDS PASS")


if __name__ == "__main__":
    main()
