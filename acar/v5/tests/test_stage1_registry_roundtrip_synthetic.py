"""Guard (Stage-1A): a fold ref from the plan round-trips through the registry using SYNTHETIC dummy hashes only (no real
artifact, no file). Synthetic only."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5.substrate import plan as PLAN
from acar.v5.substrate.registry import SubstrateRegistry, SubstrateHashMissingError
from acar.v5.tests._util import expect_raises, ok

DUMMY = {f: (chr(97 + i) * 64) for i, f in enumerate(P.REGISTRY_HASH_FIELDS)}   # distinct 64-hex dummies


def _meta(seed):
    return {"channel_montage": "10-20-19", "sampling_rate": 128, "windowing_config": "4s/512",
            "cohort_inclusion_list": "synthetic", "random_seed": seed, "git_commit": "0" * 40, "env_lock_sha256": "b" * 64}


def test_roundtrip_first_selection_ref():
    r = next(x for x in PLAN.fold_refs() if x["seed"] == P.SELECTION_SEED)
    reg = SubstrateRegistry()
    ref = reg.register(r["disease"], r["fold"], r["seed"], hashes=DUMMY, meta=_meta(r["seed"]))
    assert ref == r["ref"]
    assert reg.admit_embedding({"substrate_ref": ref, "hashes": DUMMY}) == ref
    expect_raises(SubstrateHashMissingError, lambda: reg.admit_embedding({"substrate_ref": ref}))   # still mandatory hashes
    ok("a plan fold ref registers + admits with synthetic dummy hashes (roundtrip); ref-only still rejected")


def test_all_selection_refs_registerable_synthetic():
    reg = SubstrateRegistry()
    for r in PLAN.selection_refs():
        reg.register(r["disease"], r["fold"], r["seed"], hashes=DUMMY, meta=_meta(r["seed"]))
    ok(f"all {len(PLAN.selection_refs())} selection refs register with synthetic hashes (no real artifact touched)")


def main():
    print("ACAR v5 Stage-1A guard: registry roundtrip (synthetic)")
    test_roundtrip_first_selection_ref()
    test_all_selection_refs_registerable_synthetic()
    print("ALL V5 STAGE1-REGISTRY-ROUNDTRIP GUARDS PASS")


if __name__ == "__main__":
    main()
