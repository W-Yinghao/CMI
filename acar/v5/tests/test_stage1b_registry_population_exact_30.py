"""Guard (Stage-1B3): a build populates the substrate registry with EXACTLY the 30 canonical fold refs, once each, no overwrite.
Synthetic only."""
from __future__ import annotations
from acar.v5.substrate import stage1b_build as B
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_registry_populate as RP
from acar.v5.substrate.registry import substrate_ref
from acar.v5.tests._util import expect_raises, ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeDevReader, FakeTrainer, FakeDumper

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL


def test_build_populates_exactly_30():
    rep = B.run_stage1b_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                              stage1b_lock(protocol_tag_target_sha=FULL), execute=True,
                              dev_reader=FakeDevReader(), trainer=FakeTrainer(), dumper=FakeDumper())
    assert rep["n_registered"] == 30
    reg = rep["registry"]
    for ref in SA.CANONICAL_FOLD_REFS:
        d, rest = ref.split("/", 1)
        fold = int(rest.split("/")[0][4:])
        seed = int(rest.split("seed")[1])
        assert reg.is_registered(substrate_ref(d, fold, seed)), ref
    ok("build → registry populated with all 30 canonical fold refs (each is_registered)")


def test_no_silent_overwrite():
    rep = B.run_stage1b_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                              stage1b_lock(protocol_tag_target_sha=FULL), execute=True,
                              dev_reader=FakeDevReader(), trainer=FakeTrainer(), dumper=FakeDumper())
    # re-populating the SAME registry with the same refs must fail (no overwrite)
    expect_raises((ValueError, RP.Stage1bRegistryPopulateError),
                  lambda: RP.populate_registry(rep["registry"], rep["artifacts"], git_commit="0" * 40,
                                               env_lock_sha256="a" * 64, channel_montage="10-20-19",
                                               sampling_rate=128, windowing_config="4s/512"))
    ok("re-populating an already-populated registry → fail-closed (no silent overwrite)")


def test_partial_artifacts_rejected():
    # populate with fewer than 30 artifacts → Stage1bRegistryPopulateError
    from acar.v5.substrate.registry import SubstrateRegistry
    rep = B.run_stage1b_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                              stage1b_lock(protocol_tag_target_sha=FULL), execute=True,
                              dev_reader=FakeDevReader(), trainer=FakeTrainer(), dumper=FakeDumper())
    partial = dict(list(rep["artifacts"].items())[:29])
    expect_raises(RP.Stage1bRegistryPopulateError,
                  lambda: RP.populate_registry(SubstrateRegistry(), partial, git_commit="0" * 40, env_lock_sha256="a" * 64,
                                               channel_montage="10-20-19", sampling_rate=128, windowing_config="4s/512"))
    ok("populating with fewer than 30 canonical artifacts → Stage1bRegistryPopulateError")


def main():
    print("ACAR v5 Stage-1B3 guard: registry population exact 30")
    test_build_populates_exactly_30()
    test_no_silent_overwrite()
    test_partial_artifacts_rejected()
    print("ALL V5 STAGE1B-REGISTRY-POPULATION GUARDS PASS")


if __name__ == "__main__":
    main()
