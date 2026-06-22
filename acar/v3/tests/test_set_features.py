"""Synthetic invariance/guard tests for acar/v3/set_features.py (hardened per the 93f417c review).
TOY INPUTS ONLY — uses toy SOURCE labels to fit a frozen SourceState, so the precise claim is **no target labels,
no DEV labels** (not "no labels"). No DEV cohorts, no candidate/width/AUROC/router metrics.

Run: python -m acar.v3.tests.test_set_features
"""
import inspect
import pickle
import numpy as np

from cmi.eval.source_state import fit_source_state
from acar.config import MIN_BATCH, N_CLS
import acar.v3.set_features as sf
from dataclasses import FrozenInstanceError
from acar.v3.set_features import (
    extract_action_set, build_action_sets, canonical_digest, canonical_tie_break,
    WindowActionSet, WindowKey, FallbackBatchRecord, canon_key, action_index, _validate_proba,
    NON_IDENTITY, ACTION_VOCAB, ACTION_GEOMETRY, PER_WINDOW_FEATURES, CONTEXT_FEATURES)


def _toy(n=20, d=8, seed=0):
    rng = np.random.default_rng(seed)
    ytr = (rng.random(140) < 0.5).astype(int)
    ztr = rng.standard_normal((140, d)) + np.where(ytr[:, None] == 1, 0.8, -0.8)
    state = fit_source_state(ztr, ytr, N_CLS, rho=0.1)
    z = rng.standard_normal((n, d)) + 0.3
    keys = [f"w{i:03d}" for i in range(n)]
    return state, z, keys


def _rebuild(was, values=None, mask=None, cvals=None, cmask=None, name=None, idx=None, keys=None):
    return WindowActionSet(
        was.values.copy() if values is None else values,
        was.availability_mask.copy() if mask is None else mask,
        was.context_values.copy() if cvals is None else cvals,
        was.context_mask.copy() if cmask is None else cmask,
        was.action_name if name is None else name,
        was.action_index if idx is None else idx,
        was.window_keys if keys is None else keys)


def _expect(exc, fn, *a, needle=None):
    try:
        fn(*a)
    except exc as e:
        if needle and needle not in str(e):
            raise AssertionError(f"wrong message: {e!r} (want {needle!r})")
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_label_free_api():
    for fn in (extract_action_set, build_action_sets):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"y", "label", "labels", "y_target", "target"}), f"{fn.__name__} exposes a label arg"
    print("  [ok] set-extraction API is label-free")


def test_window_permutation_path_invariance():
    state, z, keys = _toy()
    sets = build_action_sets(state, z, keys)
    perm = np.random.default_rng(1).permutation(len(z))
    setsp = build_action_sets(state, z[perm], [keys[i] for i in perm])
    for a in NON_IDENTITY:
        assert np.array_equal(sets[a].values, setsp[a].values), f"{a}: values differ under permutation (path)"
        assert np.array_equal(sets[a].availability_mask, setsp[a].availability_mask)
        assert np.array_equal(sets[a].context_values, setsp[a].context_values)
        assert canonical_digest(sets[a]) == canonical_digest(setsp[a])
    print("  [ok] window permutation: byte-identical values/masks/context (canonical order before adapters)")


def test_action_order_invariance_and_canonical_keys():
    state, z, keys = _toy()
    s1 = build_action_sets(state, z, keys, actions=("t3a", "matched_coral", "spdim"))
    assert tuple(s1.keys()) == NON_IDENTITY, "dict not in canonical action order"
    s2 = build_action_sets(state, z, keys, actions=("spdim", "t3a", "matched_coral"))
    for a in NON_IDENTITY:
        assert canonical_digest(s1[a]) == canonical_digest(s2[a])
    assert canonical_tie_break(["t3a", "matched_coral"]) == "matched_coral"
    assert canonical_tie_break(["t3a", "spdim", "matched_coral"]) == "matched_coral"
    # action-selection validation
    _expect(ValueError, lambda: build_action_sets(state, z, keys, actions=("bogus",)), needle="unknown")
    _expect(ValueError, lambda: build_action_sets(state, z, keys, actions=("spdim", "spdim")), needle="duplicate")
    _expect(ValueError, lambda: build_action_sets(state, z, keys, actions=("identity",)), needle="identity")
    _expect(ValueError, lambda: build_action_sets(state, z, keys, actions=()), needle="empty")
    print("  [ok] canonical action execution order; selection validated")


def test_missing_zero_vs_genuine_zero():
    state, z, keys = _toy()
    was = build_action_sets(state, z, keys)["matched_coral"]
    j = PER_WINDOW_FEATURES.index("embed_disp")
    v = was.values.copy(); v[:, j] = 0.0                       # value identically zero in both objects
    genuine = _rebuild(was, values=v, mask=was.availability_mask.copy())          # mask=1 -> genuine zero
    m = was.availability_mask.copy(); m[:, j] = 0
    missing = _rebuild(was, values=v, mask=m)                                      # mask=0 -> structurally missing
    assert canonical_digest(genuine) != canonical_digest(missing), "missing-zero == genuine-zero"
    print("  [ok] identical zero value, different availability -> different digest")


def test_variable_cardinality():
    state, z, keys = _toy(n=24)
    digs = {canonical_digest(build_action_sets(state, z[:n], keys[:n])["matched_coral"])
            for n in (MIN_BATCH, MIN_BATCH + 1, 20)}
    assert len(digs) == 3
    print("  [ok] variable cardinality -> distinct digests")


def test_duplicate_and_duplication():
    state, z, keys = _toy(n=12)
    bad = keys[:]; bad[5] = bad[4]
    _expect(ValueError, lambda: extract_action_set(state, z, bad, "matched_coral"), needle="duplicate")
    base = canonical_digest(extract_action_set(state, z, keys, "matched_coral"))
    z1 = np.vstack([z, z[0:1]]); k1 = keys + ["wDUP"]
    assert canonical_digest(extract_action_set(state, z1, k1, "matched_coral")) != base
    z2 = np.vstack([z, z]); k2 = keys + [f"{k}_b" for k in keys]
    assert canonical_digest(extract_action_set(state, z2, k2, "matched_coral")) != base
    print("  [ok] duplicate-key rejected; duplication (new keys) != permutation")


def test_fallback_record_no_adapter():
    state, z, keys = _toy(n=MIN_BATCH - 1)
    orig = sf.apply_action
    sf.apply_action = lambda *a, **k: (_ for _ in ()).throw(AssertionError("adapter called on fallback path"))
    try:
        out = build_action_sets(state, z, keys)
    finally:
        sf.apply_action = orig
    assert isinstance(out, FallbackBatchRecord) and out.forced_identity and out.n_windows == len(z)
    assert out.window_keys == tuple(keys) and len(out.canonical_input_digest) == 64
    print(f"  [ok] len(B)<MIN_BATCH ({MIN_BATCH}) -> immutable FallbackBatchRecord, NO adapter call, full input digest")


def test_frozen_object_rebind():
    state, z, keys = _toy()
    was = build_action_sets(state, z, keys)["matched_coral"]
    for assign in (lambda: setattr(was, "action_name", "spdim"),
                   lambda: setattr(was, "values", was.values)):
        try:
            assign(); raise AssertionError("field rebind allowed on frozen WindowActionSet")
        except FrozenInstanceError:
            pass
    print("  [ok] WindowActionSet is object-level immutable (FrozenInstanceError on field rebind)")


def test_identity_computed_once():
    state, z, keys = _toy()
    orig = sf.apply_action; calls = []
    def counting(name, st, zz):
        calls.append(name); return orig(name, st, zz)
    sf.apply_action = counting
    try:
        sf.build_action_sets(state, z, keys)
    finally:
        sf.apply_action = orig
    assert calls.count("identity") == 1, calls
    for a in NON_IDENTITY:
        assert calls.count(a) == 1, calls
    print("  [ok] identity reference computed exactly once; each requested action exactly once")


def test_window_key_encoding():
    assert canon_key(WindowKey("ds", "s1", "r1", 5)).startswith("WK")
    assert canon_key("w001") == "Sw001"
    assert canon_key(WindowKey("ds", "s1", "r1", 5)) != canon_key(WindowKey("ds", "s1", "r1", 6))
    _expect(TypeError, lambda: canon_key(123))
    print("  [ok] disambiguated window-key encoding (WK structured / S string); non-key rejected")


def test_serialization_roundtrip():
    state, z, keys = _toy()
    was = build_action_sets(state, z, keys)["spdim"]
    assert canonical_digest(was) == canonical_digest(pickle.loads(pickle.dumps(was)))
    print("  [ok] serialization round-trip preserves digest")


def test_digest_sensitivity_single_ulp_and_more():
    state, z, keys = _toy()
    was = build_action_sets(state, z, keys)["matched_coral"]; d0 = canonical_digest(was)
    v = was.values.copy(); v[3, 0] = np.nextafter(v[3, 0], np.inf)        # single ULP
    assert canonical_digest(_rebuild(was, values=v)) != d0, "single-ULP change collided"
    k = ("ZZZ",) + tuple(was.window_keys[1:])
    assert canonical_digest(_rebuild(was, keys=k)) != d0, "window-key change collided"
    cv = was.context_values.copy(); ci = CONTEXT_FEATURES.index("n_eff")
    cv[ci] = np.nextafter(cv[ci], np.inf)
    assert canonical_digest(_rebuild(was, cvals=cv)) != d0, "context single-ULP collided"
    # context-mask flip (zero that slot to keep the object legal)
    cv2 = was.context_values.copy(); cm2 = was.context_mask.copy()
    bi = CONTEXT_FEATURES.index("bures"); cv2[bi] = 0.0; cm2[bi] = 0
    assert canonical_digest(_rebuild(was, cvals=cv2, cmask=cm2)) != d0, "context-mask flip collided"
    # action identity in header: same arrays, different action_name
    assert canonical_digest(_rebuild(was, name="spdim", idx=action_index("spdim"))) != d0, "action header not in digest"
    print("  [ok] digest sensitive to single-ULP value, key, context value, context-mask, action header")


def test_t3a_geometry_masked():
    state, z, keys = _toy()
    sets = build_action_sets(state, z, keys)
    t3a, mc = sets["t3a"], sets["matched_coral"]
    j = PER_WINDOW_FEATURES.index("embed_disp")
    assert (t3a.availability_mask[:, j] == 0).all() and (mc.availability_mask[:, j] == 1).all()
    for f in ("bures", "post_sep", "s_support", "s_sep", "pr_cmi_proxy"):
        assert t3a.context_mask[CONTEXT_FEATURES.index(f)] == 0 and mc.context_mask[CONTEXT_FEATURES.index(f)] == 1
    for f in ("n_eff", "g_unc"):
        assert t3a.context_mask[CONTEXT_FEATURES.index(f)] == 1
    print("  [ok] T3A geometry masked; prob-based features available")


def test_action_capability_contract():
    state, z, keys = _toy()
    from acar.actions import apply_action
    for a in NON_IDENTITY:
        _, za = apply_action(a, state, z)
        assert (za is not None) == ACTION_GEOMETRY[a], f"{a} geometry capability drifted"
    print("  [ok] action capability map matches adapter behavior (no drift)")


def test_nan_inf_rejected():
    state, z, keys = _toy()
    zbad = z.copy(); zbad[0, 0] = np.inf
    _expect(ValueError, lambda: build_action_sets(state, zbad, keys), needle="non-finite")
    _expect(ValueError, lambda: extract_action_set(state, zbad, keys, "matched_coral"), needle="non-finite")
    print("  [ok] NaN/Inf inputs rejected")


def test_proba_validation():
    _expect(ValueError, lambda: _validate_proba(np.zeros((5, 3)), 5, 2, "p"), needle="shape")
    _expect(ValueError, lambda: _validate_proba(np.full((5, 2), 0.4), 5, 2, "p"), needle="sum")
    bad = np.full((5, 2), 0.5); bad[0, 0] = -0.1; bad[0, 1] = 1.1
    _expect(ValueError, lambda: _validate_proba(bad, 5, 2, "p"), needle="negative")
    _validate_proba(np.full((5, 2), 0.5), 5, 2, "p")          # valid passes
    print("  [ok] probability validation (shape / sum / negativity)")


def test_contract_validation_and_immutability():
    state, z, keys = _toy()
    was = build_action_sets(state, z, keys)["matched_coral"]
    # strong immutability: read-only AND writeable cannot be re-enabled (bytes-backed)
    assert not was.values.flags.writeable
    try:
        was.values[0, 0] = 1.0; raise AssertionError("values writable after freeze")
    except ValueError:
        pass
    try:
        was.values.flags.writeable = True; raise AssertionError("re-enabled writeable flag")
    except ValueError:
        pass
    # illegal constructions (n=MIN_BATCH rows; WAS requires n in [MIN_BATCH,B])
    F = len(PER_WINDOW_FEATURES); n = MIN_BATCH; K = tuple(f"k{i}" for i in range(n))
    good = (np.zeros((n, F)), np.ones((n, F), np.uint8), np.zeros(len(CONTEXT_FEATURES)),
            np.ones(len(CONTEXT_FEATURES), np.uint8), "matched_coral", 1, K)
    WindowActionSet(*good)                                                            # valid
    _expect(ValueError, lambda: WindowActionSet(np.zeros((n, F + 1)), np.ones((n, F + 1), np.uint8), *good[2:]), needle="features")
    _expect(ValueError, lambda: WindowActionSet(np.zeros((n, F)), np.ones((n - 1, F), np.uint8), *good[2:]), needle="same shape")
    _expect(ValueError, lambda: WindowActionSet(np.zeros((MIN_BATCH - 1, F)), np.ones((MIN_BATCH - 1, F), np.uint8),
                                                *good[2:5], 1, tuple(f"k{i}" for i in range(MIN_BATCH - 1))), needle="n_windows")
    badmask = np.full((n, F), 2, np.uint8)
    _expect(ValueError, lambda: WindowActionSet(good[0], badmask, *good[2:]), needle="binary")
    nz = np.zeros((n, F)); nz[0, 0] = 1.0; mm = np.ones((n, F), np.uint8); mm[0, 0] = 0
    _expect(ValueError, lambda: WindowActionSet(nz, mm, *good[2:]), needle="exactly 0")
    _expect(ValueError, lambda: WindowActionSet(*good[:5], 2, good[6]), needle="inconsistent")
    _expect(ValueError, lambda: WindowActionSet(*good[:6], ("a",) * n), needle="duplicate")
    nf = np.zeros((n, F)); nf[0, 0] = np.inf
    _expect(ValueError, lambda: WindowActionSet(nf, *good[1:]), needle="non-finite")
    # WindowKey + FallbackBatchRecord validation
    _expect(ValueError, lambda: WindowKey("ds", "", "r", 0))                          # empty id
    _expect(ValueError, lambda: WindowKey("ds", "s", "r", -1))                        # negative index
    _expect(ValueError, lambda: FallbackBatchRecord(False, "x", (), "a" * 64, 0))     # forced_identity must be True
    _expect(ValueError, lambda: FallbackBatchRecord(True, "x", ("k",), "zz", 1))      # bad digest
    print("  [ok] WindowActionSet validates shape/mask/zero/action/keys/finite + n∈[MIN_BATCH,B]; WindowKey + Fallback validated; immutable")


def test_structured_window_key():
    state, z, keys = _toy(n=10)
    wk = [WindowKey("dsX", f"s{i%3}", f"r{i}", i) for i in range(10)]
    was = extract_action_set(state, z, wk, "matched_coral")
    assert canonical_digest(was) == canonical_digest(pickle.loads(pickle.dumps(was)))
    dup = wk[:]; dup[3] = dup[2]
    _expect(ValueError, lambda: extract_action_set(state, z, dup, "matched_coral"), needle="duplicate")
    print("  [ok] structured WindowKey supported; canon_key serialization stable")


def main():
    print("ACAR v3 set_features synthetic guards (hardened):")
    for t in (test_label_free_api, test_window_permutation_path_invariance, test_action_order_invariance_and_canonical_keys,
              test_missing_zero_vs_genuine_zero, test_variable_cardinality, test_duplicate_and_duplication,
              test_fallback_record_no_adapter, test_frozen_object_rebind, test_identity_computed_once,
              test_window_key_encoding, test_serialization_roundtrip,
              test_digest_sensitivity_single_ulp_and_more, test_t3a_geometry_masked, test_action_capability_contract,
              test_nan_inf_rejected, test_proba_validation, test_contract_validation_and_immutability,
              test_structured_window_key):
        t()
    st, zz, kk = _toy()
    dlen = len(canonical_digest(build_action_sets(st, zz, kk)["matched_coral"]))
    assert dlen == 64, f"digest not full SHA-256 ({dlen})"
    print(f"ALL V3 SET-FEATURE GUARDS PASS (full 64-char digest, len={dlen})")


if __name__ == "__main__":
    main()
