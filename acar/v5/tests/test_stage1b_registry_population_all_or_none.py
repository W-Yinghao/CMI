"""Guard (Stage-1B5): registry population is ALL-OR-NONE. The exact-30 set check runs BEFORE any register(), so any wrong set
(fewer than 30, extra ref, mismatched ref) raises AND leaves the fresh registry with ZERO entries — no partial substrate table
can ever exist. Synthetic only."""
from __future__ import annotations
from acar.v5.substrate import stage1b_build as B
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_registry_populate as RP
from acar.v5.substrate.registry import SubstrateRegistry
from acar.v5.tests._util import expect_raises, ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeDevReader, FakeTrainer, FakeDumper

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL
_META = dict(git_commit="0" * 40, env_lock_sha256="a" * 64, channel_montage="10-20-19", sampling_rate=128, windowing_config="4s/512")


def _artifacts():
    rep = B.run_stage1b_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                              stage1b_lock(protocol_tag_target_sha=FULL), execute=True,
                              dev_reader=FakeDevReader(), trainer=FakeTrainer(), dumper=FakeDumper())
    return rep["artifacts"]


def _n(reg):
    return len(reg._entries)


def test_partial_leaves_registry_empty():
    partial = dict(list(_artifacts().items())[:29])
    reg = SubstrateRegistry()
    expect_raises(RP.Stage1bRegistryPopulateError, lambda: RP.populate_registry(reg, partial, **_META))
    assert _n(reg) == 0, f"partial populate must leave 0 entries, got {_n(reg)}"
    ok("29/30 artifacts → error AND fresh registry has 0 entries (count-check before any register)")


def test_extra_ref_leaves_registry_empty():
    arts = dict(_artifacts())
    arts["PD/fold9/seed99999"] = dict(next(iter(arts.values())))   # a 31st, non-canonical ref
    reg = SubstrateRegistry()
    expect_raises(RP.Stage1bRegistryPopulateError, lambda: RP.populate_registry(reg, arts, **_META))
    assert _n(reg) == 0, f"extra-ref populate must leave 0 entries, got {_n(reg)}"
    ok("31 artifacts (one non-canonical) → error AND fresh registry has 0 entries")


def test_full_set_registers_all_30():
    reg = SubstrateRegistry()
    n = RP.populate_registry(reg, _artifacts(), **_META)
    assert n == 30 and _n(reg) == 30
    ok("exactly the 30 canonical artifacts → all 30 registered")


def main():
    print("ACAR v5 Stage-1B5 guard: registry population all-or-none")
    test_partial_leaves_registry_empty()
    test_extra_ref_leaves_registry_empty()
    test_full_set_registers_all_30()
    print("ALL V5 STAGE1B-REGISTRY-ALL-OR-NONE GUARDS PASS")


if __name__ == "__main__":
    main()
