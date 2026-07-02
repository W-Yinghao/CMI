"""Guard (Stage-1B9): the substrate registry has a CANONICAL, deterministic file export (sorted refs + sorted keys, exactly 30 refs,
each ref's hashes+meta) that round-trips through load_registry. Synthetic only."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import stage1b_build as B
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_registry_io as RIO
from acar.v5.tests._util import ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeDevReader, FakeTrainer, FakeDumper

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL


def _registry():
    rep = B.run_stage1b_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                              stage1b_lock(protocol_tag_target_sha=FULL), execute=True,
                              dev_reader=FakeDevReader(), trainer=FakeTrainer(), dumper=FakeDumper())
    return rep["registry"]


def test_export_is_canonical_and_deterministic():
    reg = _registry()
    doc = RIO.export_registry(reg)
    assert doc["schema"] == RIO.SCHEMA and doc["n_refs"] == 30
    assert set(doc["entries"]) == set(SA.CANONICAL_FOLD_REFS)
    for ref, e in doc["entries"].items():
        assert set(e["hashes"]) and "meta" in e
    assert RIO.registry_sha256(reg) == RIO.registry_sha256(reg)   # deterministic
    ok("registry export: schema + exactly 30 canonical refs + hashes/meta per ref; sha256 deterministic")


def test_write_load_round_trips():
    reg = _registry()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "registry.json")
        sha = RIO.write_registry(reg, p)
        assert os.path.isfile(p) and not os.path.exists(p + ".tmp") and len(sha) == 64
        reloaded = RIO.load_registry(p)
        assert RIO.export_registry(reloaded) == RIO.export_registry(reg)   # round-trip identical
        assert RIO.registry_sha256(reloaded) == sha
    ok("write_registry (atomic) → load_registry round-trips to an identical canonical export + matching sha256")


def main():
    print("ACAR v5 Stage-1B9 guard: registry persisted canonical")
    test_export_is_canonical_and_deterministic()
    test_write_load_round_trips()
    print("ALL V5 STAGE1B-REGISTRY-PERSISTED-CANONICAL GUARDS PASS")


if __name__ == "__main__":
    main()
