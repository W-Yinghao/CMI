"""Synthetic invariance/guard tests for acar/v3/set_features.py. TOY INPUTS ONLY — no DEV cohorts, no candidate /
width / AUROC / router metrics, no labels. Run: python -m acar.v3.tests.test_set_features

Covers the 12 pre-registered guards (review): label-free API; window-permutation invariance; action-order
invariance incl. exact ties; unavailable-geometry vs genuine-zero distinguishable; variable set cardinality;
uniform full-set duplication behavior; single-window duplication ≠ permutation; fallback <MIN_BATCH short-circuit;
serialization round-trip; canonical digest row-order-insensitive but content-sensitive; T3A geometry mask; NaN/Inf
rejection.
"""
import copy
import inspect
import pickle
import numpy as np

from cmi.eval.source_state import fit_source_state
from acar.config import MIN_BATCH, N_CLS
from acar.v3.set_features import (
    extract_action_set, build_action_sets, canonical_digest, canonical_tie_break,
    WindowActionSet, NON_IDENTITY, PER_WINDOW_FEATURES, CONTEXT_FEATURES, action_index)


def _toy(n=20, d=8, seed=0):
    rng = np.random.default_rng(seed)
    ytr = (rng.random(140) < 0.5).astype(int)
    ztr = rng.standard_normal((140, d)) + np.where(ytr[:, None] == 1, 0.8, -0.8)
    state = fit_source_state(ztr, ytr, N_CLS, rho=0.1)
    z = rng.standard_normal((n, d)) + 0.3
    keys = [f"w{i:03d}" for i in range(n)]
    return state, z, keys


def test_label_free_api():
    for fn in (extract_action_set, build_action_sets):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"y", "label", "labels", "y_target", "target"}), f"{fn.__name__} exposes a label arg"
    print("  [ok] set-extraction API is label-free")


def test_window_permutation_invariance():
    state, z, keys = _toy()
    sets = build_action_sets(state, z, keys)
    perm = np.random.default_rng(1).permutation(len(z))
    zp, kp = z[perm], [keys[i] for i in perm]
    setsp = build_action_sets(state, zp, kp)
    for a in NON_IDENTITY:
        assert canonical_digest(sets[a]) == canonical_digest(setsp[a]), f"{a}: digest changed under window permutation"
    print("  [ok] window-permutation invariant (rows sorted by key)")


def test_action_order_invariance_and_ties():
    state, z, keys = _toy()
    s1 = build_action_sets(state, z, keys, actions=("matched_coral", "spdim", "t3a"))
    s2 = build_action_sets(state, z, keys, actions=("t3a", "spdim", "matched_coral"))
    for a in NON_IDENTITY:
        assert canonical_digest(s1[a]) == canonical_digest(s2[a]), f"{a}: digest depends on action iteration order"
    # exact-tie break ignores input order -> lowest ACTION_VOCAB index
    assert canonical_tie_break(["t3a", "matched_coral"]) == "matched_coral"
    assert canonical_tie_break(["t3a", "spdim", "matched_coral"]) == "matched_coral"
    assert canonical_tie_break(["t3a", "spdim"]) == "spdim"
    print("  [ok] action-order invariant; ties broken by canonical vocabulary, not input order")


def test_unavailable_geometry_vs_genuine_zero():
    state, z, keys = _toy()
    was = build_action_sets(state, z, keys)["matched_coral"]
    d0 = canonical_digest(was)
    # genuine value 0 with mask 1  !=  structurally-missing 0 with mask 0
    j = PER_WINDOW_FEATURES.index("embed_disp")
    flipped = copy.deepcopy(was); flipped.availability_mask[0, j] = 0.0   # same value (whatever), mask now "missing"
    assert canonical_digest(flipped) != d0, "mask is not part of identity (missing-zero == genuine-zero)"
    print("  [ok] missing-zero (mask 0) distinguishable from genuine-zero (mask 1)")


def test_variable_cardinality():
    state, z, keys = _toy(n=24)
    digs = set()
    for n in (MIN_BATCH, MIN_BATCH + 1, 20):
        s = build_action_sets(state, z[:n], keys[:n])
        assert "__fallback__" not in s
        digs.add(canonical_digest(s["matched_coral"]))
    assert len(digs) == 3, "different cardinalities collided"
    print("  [ok] variable set cardinality handled; distinct digests")


def test_duplicate_key_rejected():
    state, z, keys = _toy(n=10)
    bad = keys[:]; bad[5] = bad[4]                                    # duplicate window key
    try:
        extract_action_set(state, z, bad, "matched_coral"); raise AssertionError("duplicate key not rejected")
    except ValueError:
        pass
    print("  [ok] duplicate window_keys rejected (window identity must be unique)")


def test_duplication_is_not_permutation():
    state, z, keys = _toy(n=12)
    base = canonical_digest(extract_action_set(state, z, keys, "matched_coral"))
    # single-window duplication with a NEW key -> n+1 -> different set (not a permutation)
    z1 = np.vstack([z, z[0:1]]); k1 = keys + ["wDUP"]
    assert canonical_digest(extract_action_set(state, z1, k1, "matched_coral")) != base, "single-window dup looked like permutation"
    # uniform full-set duplication with new keys -> 2n windows -> different set (pre-registered: distinct, not equal)
    z2 = np.vstack([z, z]); k2 = keys + [f"{k}_b" for k in keys]
    assert canonical_digest(extract_action_set(state, z2, k2, "matched_coral")) != base, "uniform dup collided with original"
    print("  [ok] duplication (new keys) yields a distinct set, never mistaken for permutation")


def test_fallback_short_circuit():
    state, z, keys = _toy(n=MIN_BATCH - 1)
    out = build_action_sets(state, z, keys)
    assert out == {"__fallback__": "identity"}, "small batch not short-circuited to identity"
    print(f"  [ok] len(B)<MIN_BATCH ({MIN_BATCH}) short-circuits to identity before any adapter/extractor")


def test_serialization_roundtrip():
    state, z, keys = _toy()
    was = build_action_sets(state, z, keys)["spdim"]
    was2 = pickle.loads(pickle.dumps(was))
    assert canonical_digest(was) == canonical_digest(was2), "digest changed across serialize round-trip"
    print("  [ok] serialization round-trip preserves canonical digest")


def test_digest_sensitivity():
    state, z, keys = _toy()
    was = build_action_sets(state, z, keys)["matched_coral"]; d0 = canonical_digest(was)
    v = copy.deepcopy(was); v.values[3, 0] += 0.01
    m = copy.deepcopy(was); m.availability_mask[2, 0] = 0.0
    k = copy.deepcopy(was); k.window_keys = ("ZZZ",) + was.window_keys[1:]
    c = copy.deepcopy(was); c.context_values[CONTEXT_FEATURES.index("n_eff")] += 0.01
    assert canonical_digest(v) != d0 and canonical_digest(m) != d0 and canonical_digest(k) != d0 and canonical_digest(c) != d0
    print("  [ok] digest sensitive to any value / mask / window-key / context change")


def test_t3a_geometry_masked():
    state, z, keys = _toy()
    sets = build_action_sets(state, z, keys)
    t3a, mc = sets["t3a"], sets["matched_coral"]
    j = PER_WINDOW_FEATURES.index("embed_disp")
    assert (t3a.availability_mask[:, j] == 0).all(), "t3a embed_disp should be unavailable"
    assert (mc.availability_mask[:, j] == 1).all(), "matched_coral embed_disp should be available"
    geom_ctx = ("bures", "post_sep", "s_support", "s_sep", "pr_cmi_proxy")
    for f in geom_ctx:
        assert t3a.context_mask[CONTEXT_FEATURES.index(f)] == 0, f"t3a {f} should be unavailable"
        assert mc.context_mask[CONTEXT_FEATURES.index(f)] == 1, f"matched_coral {f} should be available"
    for f in ("n_eff", "g_unc"):
        assert t3a.context_mask[CONTEXT_FEATURES.index(f)] == 1, f"t3a {f} should be available (prob-based)"
    print("  [ok] T3A geometry features masked unavailable; prob-based features available")


def test_nan_inf_rejection():
    state, z, keys = _toy()
    zbad = z.copy(); zbad[0, 0] = np.inf
    for caller in (lambda: build_action_sets(state, zbad, keys),
                   lambda: extract_action_set(state, zbad, keys, "matched_coral")):
        try:
            caller(); raise AssertionError("NaN/Inf input not rejected")
        except ValueError:
            pass
    print("  [ok] NaN/Inf inputs rejected")


def main():
    print("ACAR v3 set_features synthetic guards:")
    test_label_free_api()
    test_window_permutation_invariance()
    test_action_order_invariance_and_ties()
    test_unavailable_geometry_vs_genuine_zero()
    test_variable_cardinality()
    test_duplicate_key_rejected()
    test_duplication_is_not_permutation()
    test_fallback_short_circuit()
    test_serialization_roundtrip()
    test_digest_sensitivity()
    test_t3a_geometry_masked()
    test_nan_inf_rejection()
    print("ALL V3 SET-FEATURE GUARDS PASS")


if __name__ == "__main__":
    main()
