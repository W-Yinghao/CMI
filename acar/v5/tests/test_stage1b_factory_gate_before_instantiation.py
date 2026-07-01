"""Guard (Stage-1B3): with factories, the real reader/trainer are instantiated ONLY after the gate passes — a failing gate never
constructs them (no pre-gate model init / GPU probe / BIDS scan). Synthetic only."""
from __future__ import annotations
from acar.v5.substrate import stage1b_build as B
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.tests._util import expect_raises, ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeDevReader, FakeTrainer, FakeDumper

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL


OUT = "/tmp/acar_v5_synth_ctx"


class RecFactory:
    def __init__(self, obj):
        self.calls = 0
        self._obj = obj

    def __call__(self, context):                              # factories now receive the gate-issued execution context
        self.calls += 1
        return self._obj


def test_gate_failure_never_instantiates():
    rf, tf, df = RecFactory(FakeDevReader()), RecFactory(FakeTrainer()), RecFactory(FakeDumper())
    # prefix target sha → full-build gate rejects BEFORE any factory call
    expect_raises(SA.Stage1BuildNotAuthorizedError,
                  lambda: B.run_stage1b_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha="4278435"),
                                              stage1b_lock(protocol_tag_target_sha="4278435"), execute=True,
                                              dev_reader_factory=rf, trainer_factory=tf, dumper_factory=df, output_root=OUT))
    assert rf.calls == 0 and tf.calls == 0 and df.calls == 0
    ok("gate failure → no factory is called (no pre-gate instantiation/import/probe)")


def test_gate_pass_instantiates_once():
    rf, tf, df = RecFactory(FakeDevReader()), RecFactory(FakeTrainer()), RecFactory(FakeDumper())
    rep = B.run_stage1b_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                              stage1b_lock(protocol_tag_target_sha=FULL), execute=True,
                              dev_reader_factory=rf, trainer_factory=tf, dumper_factory=df, output_root=OUT)
    assert rep["status"] == "STAGE1B_BUILT" and rf.calls == 1 and tf.calls == 1 and df.calls == 1
    ok("gate pass → each factory called exactly once (instantiation happens post-gate)")


def test_factory_pairing_and_exclusivity():
    expect_raises(B.Stage1bBuildError,
                  lambda: B.run_stage1b_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                                              stage1b_lock(protocol_tag_target_sha=FULL), execute=True,
                                              dev_reader_factory=RecFactory(FakeDevReader()), output_root=OUT))   # not all factories
    expect_raises(B.Stage1bBuildError,
                  lambda: B.run_stage1b_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                                              stage1b_lock(protocol_tag_target_sha=FULL), execute=True,
                                              dev_reader=FakeDevReader(), trainer=FakeTrainer(), dumper=FakeDumper(),
                                              dev_reader_factory=RecFactory(FakeDevReader()), trainer_factory=RecFactory(FakeTrainer()),
                                              dumper_factory=RecFactory(FakeDumper()), output_root=OUT))
    expect_raises(B.Stage1bBuildError,                        # factory path without output_root
                  lambda: B.run_stage1b_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                                              stage1b_lock(protocol_tag_target_sha=FULL), execute=True,
                                              dev_reader_factory=RecFactory(FakeDevReader()), trainer_factory=RecFactory(FakeTrainer()),
                                              dumper_factory=RecFactory(FakeDumper())))
    ok("all three factories required together; factories XOR objects; factory path requires output_root")


def main():
    print("ACAR v5 Stage-1B3 guard: factory gate before instantiation")
    test_gate_failure_never_instantiates()
    test_gate_pass_instantiates_once()
    test_factory_pairing_and_exclusivity()
    print("ALL V5 STAGE1B-FACTORY-GATE GUARDS PASS")


if __name__ == "__main__":
    main()
