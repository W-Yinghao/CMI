"""Guard (Stage-1B4): the production real-run entry accepts ONLY factories (no preconstructed objects), so the real reader/trainer
can never be instantiated before the gate. Synthetic only (fakes + temp files)."""
from __future__ import annotations
import inspect
import tempfile
from acar.v5.substrate import stage1b_build as B
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.tests._util import expect_raises, ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeDevReader, FakeFileTrainer

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL


def test_signature_has_no_object_params():
    params = list(inspect.signature(B.run_stage1b_real_build).parameters)
    assert "dev_reader" not in params and "trainer" not in params, params
    assert "dev_reader_factory" in params and "trainer_factory" in params
    ok("run_stage1b_real_build exposes ONLY factories (no dev_reader/trainer object params)")


def test_non_callable_factory_rejected():
    expect_raises(B.Stage1bBuildError,
                  lambda: B.run_stage1b_real_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                                                   stage1b_lock(protocol_tag_target_sha=FULL),
                                                   dev_reader_factory=FakeDevReader(), trainer_factory=lambda: None))
    ok("a non-callable dev_reader_factory → Stage1bBuildError")


def test_real_build_end_to_end_file_backed():
    with tempfile.TemporaryDirectory() as d:
        rep = B.run_stage1b_real_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                                       stage1b_lock(protocol_tag_target_sha=FULL),
                                       dev_reader_factory=lambda: FakeDevReader(), trainer_factory=lambda: FakeFileTrainer(d))
        assert rep["status"] == "STAGE1B_BUILT" and rep["n_artifacts"] == 30 and rep["n_registered"] == 30
    ok("run_stage1b_real_build (factories + FILE artifact writer) → 30 substrates registered (synthetic, temp files)")


def main():
    print("ACAR v5 Stage-1B4 guard: real run requires factories")
    test_signature_has_no_object_params()
    test_non_callable_factory_rejected()
    test_real_build_end_to_end_file_backed()
    print("ALL V5 STAGE1B-REAL-RUN-FACTORIES GUARDS PASS")


if __name__ == "__main__":
    main()
