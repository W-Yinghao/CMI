"""Guard (Stage-1B2): the build calls the full-build gate BEFORE any read — a failing gate means the reader is never touched, and
an unwired reader/trainer blocks execute. Synthetic only."""
from __future__ import annotations
from acar.v5.substrate import stage1b_build as B
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import dev_reader_contract as DR
from acar.v5.tests._util import expect_raises, ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeDevReader, FakeTrainer

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL


def test_bad_auth_no_read():
    reader = FakeDevReader()
    # prefix target sha → full-build gate rejects; reader must never be called
    expect_raises(SA.Stage1BuildNotAuthorizedError,
                  lambda: B.run_stage1b_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha="4278435"),
                                              stage1b_lock(protocol_tag_target_sha="4278435"), execute=True,
                                              dev_reader=reader, trainer=FakeTrainer()))
    assert reader.listed == [] and reader.read_calls == []
    ok("gate failure (prefix sha) → raises BEFORE any read (reader never called)")


def test_bad_lock_no_read():
    from acar.v5.substrate import stage1_runtime_lock as RL
    reader = FakeDevReader()
    expect_raises(RL.Stage1RuntimeLockError,
                  lambda: B.run_stage1b_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                                              stage1b_lock(protocol_tag_target_sha=FULL, status="PENDING"), execute=True,
                                              dev_reader=reader, trainer=FakeTrainer()))
    assert reader.listed == []
    ok("runtime-lock failure → raises before any read")


def test_execute_requires_wired_reader_and_trainer():
    expect_raises(DR.DevReaderNotWiredError,
                  lambda: B.run_stage1b_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                                              stage1b_lock(protocol_tag_target_sha=FULL), execute=True,
                                              dev_reader=None, trainer=None))
    ok("execute=True with no wired reader → DevReaderNotWiredError (CLI default is unwired → cannot read real data)")


def main():
    print("ACAR v5 Stage-1B2 guard: build requires full gate before read")
    test_bad_auth_no_read()
    test_bad_lock_no_read()
    test_execute_requires_wired_reader_and_trainer()
    print("ALL V5 STAGE1B-GATE-BEFORE-READ GUARDS PASS")


if __name__ == "__main__":
    main()
