"""Guard: no DEV-selection embedding is admissible without a registered, complete substrate hash set (Stage-0). Synthetic only."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5.substrate.registry import SubstrateRegistry, SubstrateHashMissingError
from acar.v5.tests._util import expect_raises, ok

H = "a" * 64
GOOD_HASHES = {f: H for f in P.REGISTRY_HASH_FIELDS}
GOOD_META = {"channel_montage": "10-20-19", "sampling_rate": 128, "windowing_config": "4s/512",
             "cohort_inclusion_list": "PD:ds002778,ds003490,ds004584", "random_seed": P.SELECTION_SEED,
             "git_commit": "0" * 40, "env_lock_sha256": "b" * 64}


def _reg():
    return SubstrateRegistry()


def test_register_and_admit():
    r = _reg()
    ref = r.register("PD", 0, P.SELECTION_SEED, hashes=GOOD_HASHES, meta=GOOD_META)
    assert r.admit_embedding({"substrate_ref": ref}) == ref
    assert r.admit_embedding({"substrate_ref": ref, "hashes": GOOD_HASHES}) == ref
    ok("register a complete substrate → its embeddings are admissible")


def test_missing_ref_or_unregistered_rejected():
    r = _reg()
    expect_raises(SubstrateHashMissingError, lambda: r.admit_embedding({"foo": 1}), "no substrate_ref")
    expect_raises(SubstrateHashMissingError, lambda: r.admit_embedding({"substrate_ref": "PD/fold0/seed20260711"}), "unregistered")
    ok("embedding with no substrate_ref, or an unregistered ref → inadmissible (no hash ⇒ inadmissible)")


def test_incomplete_or_bad_hashes_rejected():
    r = _reg()
    incomplete = {k: H for k in list(P.REGISTRY_HASH_FIELDS)[:-1]}
    expect_raises(SubstrateHashMissingError, lambda: r.register("PD", 1, P.SELECTION_SEED, hashes=incomplete, meta=GOOD_META))
    badhex = dict(GOOD_HASHES, feat_dump_sha256="xyz")
    expect_raises(SubstrateHashMissingError, lambda: r.register("PD", 1, P.SELECTION_SEED, hashes=badhex, meta=GOOD_META))
    expect_raises(SubstrateHashMissingError, lambda: r.register("PD", 1, P.SELECTION_SEED, hashes=GOOD_HASHES,
                                                                meta={k: GOOD_META[k] for k in list(GOOD_META)[:-1]}))
    ok("incomplete hash set / non-hex hash / missing meta → registration fail-closed")


def test_embedding_hash_substitution_rejected():
    r = _reg()
    ref = r.register("SCZ", 2, P.SELECTION_SEED, hashes=GOOD_HASHES, meta=dict(GOOD_META, random_seed=P.SELECTION_SEED))
    tampered = dict(GOOD_HASHES, feat_dump_sha256="c" * 64)
    expect_raises(SubstrateHashMissingError, lambda: r.admit_embedding({"substrate_ref": ref, "hashes": tampered}))
    ok("embedding hashes disagreeing with the registered substrate → rejected (no substitution)")


def test_seed_and_fold_bounds():
    r = _reg()
    expect_raises(ValueError, lambda: r.register("PD", 99, P.SELECTION_SEED, hashes=GOOD_HASHES, meta=GOOD_META))
    expect_raises(ValueError, lambda: r.register("PD", 0, 12345, hashes=GOOD_HASHES, meta=GOOD_META))
    ok("out-of-range fold / non-pinned seed rejected")


def main():
    print("ACAR v5 guard: substrate hash required")
    test_register_and_admit()
    test_missing_ref_or_unregistered_rejected()
    test_incomplete_or_bad_hashes_rejected()
    test_embedding_hash_substitution_rejected()
    test_seed_and_fold_bounds()
    print("ALL V5 SUBSTRATE-HASH-REQUIRED GUARDS PASS")


if __name__ == "__main__":
    main()
