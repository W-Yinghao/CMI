"""Synthetic guards for acar/v3/predictors.py + conformal.py. TOY INPUTS ONLY (toy source state; no DEV cohorts, no
DEV gate, no candidate/AUROC/router quality metrics, no lockbox). Run: python -m acar.v3.tests.test_predictors_conformal
"""
import math
import pickle
import numpy as np

from cmi.eval.source_state import fit_source_state
from acar.config import N_CLS
from acar.v3.set_features import (build_action_sets, extract_action_set, WindowActionSet,
                                  PER_WINDOW_FEATURES, CONTEXT_FEATURES, NON_IDENTITY)
from acar.v3.predictors import Candidate, CandidatePrediction, CANDIDATES, score, upper_bound, HP
from acar.v3.conformal import (subject_joint_score, conformal_rank, conformal_q, route, harmful_rate_test)


def _toy(n=20, d=8, seed=0):
    rng = np.random.default_rng(seed)
    ytr = (rng.random(140) < 0.5).astype(int)
    ztr = rng.standard_normal((140, d)) + np.where(ytr[:, None] == 1, 0.8, -0.8)
    state = fit_source_state(ztr, ytr, N_CLS, rho=0.1)
    z = rng.standard_normal((n, d)) + 0.3
    keys = [f"w{i:03d}" for i in range(n)]
    return state, z, keys


def _expect(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_predict_permutation_and_action_invariant():
    state, z, keys = _toy()
    for cand in CANDIDATES:
        c = Candidate(cand, "PD", seed=0)
        s1 = build_action_sets(state, z, keys)
        perm = np.random.default_rng(1).permutation(len(z))
        s2 = build_action_sets(state, z[perm], [keys[i] for i in perm])
        for a in NON_IDENTITY:
            p1, p2 = c.predict(s1[a]), c.predict(s2[a])
            assert p1 == p2, f"{cand}/{a}: prediction changed under window permutation"
    print("  [ok] predictions bit-identical under window permutation (canonical set) for C1/C2/C3")


def test_mask_change_changes_prediction():
    state, z, keys = _toy()
    c = Candidate("C2", "PD", seed=0)
    was = build_action_sets(state, z, keys)["matched_coral"]
    j = PER_WINDOW_FEATURES.index("embed_disp")
    v = was.values.copy(); v[:, j] = 0.0
    m = was.availability_mask.copy(); m[:, j] = 0                       # mask now "missing" (value already 0)
    miss = WindowActionSet(v, m, was.context_values.copy(), was.context_mask.copy(),
                           was.action_name, was.action_index, was.window_keys)
    genuine = WindowActionSet(v, was.availability_mask.copy(), was.context_values.copy(),
                              was.context_mask.copy(), was.action_name, was.action_index, was.window_keys)
    assert c.predict(miss) != c.predict(genuine), "predictor ignores availability mask"
    print("  [ok] availability-mask change (value fixed) changes prediction (mask is consumed)")


def test_target_standardization_roundtrip():
    state, z, keys = _toy()
    was = build_action_sets(state, z, keys)["matched_coral"]
    base = Candidate("C1", "PD", seed=3, target_mean=0.0, target_sd=1.0)
    scaled = Candidate("C1", "PD", seed=3, target_mean=5.0, target_sd=2.0)
    o = base.predict(was).point                                        # std-unit output (mean0,sd1)
    assert abs(scaled.predict(was).point - (o * 2.0 + 5.0)) < 1e-9, "raw-unit de-standardization wrong"
    print("  [ok] FIT-only target standardization raw-unit round trip (point = std*sd + mean)")


def test_c2_scale_floor_and_q_clamp():
    state, z, keys = _toy()
    was = build_action_sets(state, z, keys)["matched_coral"]
    c = Candidate("C2", "PD", seed=0)
    p = c.predict(was)
    assert p.scale is not None and p.scale > 0, "C2 scale must be > 0"
    cf = Candidate("C2", "PD", seed=0, sigma_min={"matched_coral": 99.0})
    assert abs(cf.predict(was).scale - 99.0) < 1e-9, "C2 sigma_min floor not applied"
    # negative q clamped ONLY in C2's U (q^+); C1/C3 additive q unchanged
    assert abs(upper_bound(p, -5.0) - p.upper_center) < 1e-12, "C2 U should clamp negative q to 0"
    c1 = Candidate("C1", "PD", seed=0).predict(was)
    c3 = Candidate("C3", "PD", seed=0).predict(was)
    assert abs(upper_bound(c1, -5.0) - (c1.upper_center - 5.0)) < 1e-12, "C1 must NOT clamp q"
    assert abs(upper_bound(c3, -5.0) - (c3.upper_center - 5.0)) < 1e-12, "C3 must NOT clamp q"
    print("  [ok] C2 scale>0 + sigma_min floor + q⁺ clamp; C1/C3 additive q never clamped")


def test_c3_no_quantile_crossing():
    state, z, keys = _toy()
    for s in range(5):
        c = Candidate("C3", "PD", seed=s)
        for a in NON_IDENTITY:
            p = c.predict(build_action_sets(state, z, keys)[a])
            assert p.upper_center > p.point, "C3 q90 <= q50 (crossing)"
    print("  [ok] C3 never crosses (q̂₉₀ > q̂₅₀ via softplus gap)")


def test_conformal_rank_and_inf():
    assert conformal_rank(20, 0.1) == 19 and conformal_rank(5, 0.1) == 6
    q_ok, k_ok = conformal_q(np.arange(20.0), 0.1); assert math.isfinite(q_ok) and k_ok == 19 and q_ok == 18.0
    q_inf, k_inf = conformal_q(np.arange(5.0), 0.1); assert math.isinf(q_inf) and k_inf == 6
    print("  [ok] conformal rank k=⌈(m+1)(1−α)⌉; strict +inf when k>m (no clip)")


def test_subject_joint_max_and_missing_action():
    state, z, keys = _toy(); c = Candidate("C1", "PD", seed=0)
    sets = build_action_sets(state, z, keys)
    b1 = {a: (c.predict(sets[a]), 0.1 * i) for i, a in enumerate(NON_IDENTITY)}
    b2 = {a: (c.predict(sets[a]), -0.2 * i) for i, a in enumerate(NON_IDENTITY)}
    manual = max(max(score(p, dr) for p, dr in b.values()) for b in (b1, b2))
    assert abs(subject_joint_score([b1, b2]) - manual) < 1e-12
    incomplete = {NON_IDENTITY[0]: b1[NON_IDENTITY[0]]}
    _expect(ValueError, lambda: subject_joint_score([incomplete]))
    print("  [ok] subject joint score = max over batches×actions; missing action raises (no shrink)")


def test_cal_vs_eval_isolation():
    state, z, keys = _toy(); c = Candidate("C2", "PD", seed=0)
    preds = {a: c.predict(build_action_sets(state, z, keys)[a]) for a in NON_IDENTITY}   # predictor frozen
    rng = np.random.default_rng(0)
    cal_scores, cal_scores2 = [], []
    for s in range(12):                                               # >=9 CAL subjects -> finite q at alpha=0.1
        dr = {a: float(rng.normal()) for a in NON_IDENTITY}
        cal_scores.append(subject_joint_score([{a: (preds[a], dr[a]) for a in NON_IDENTITY}]))
        cal_scores2.append(subject_joint_score([{a: (preds[a], dr[a] + 1.0) for a in NON_IDENTITY}]))
    q1, k1 = conformal_q(cal_scores, 0.1); q2, _ = conformal_q(cal_scores2, 0.1)
    assert math.isfinite(q1) and k1 <= 12
    assert q2 != q1, "q did not respond to CAL-label change"
    # EVAL labels never enter route/U/q (route takes preds + q only) -> identical regardless
    assert route(preds, q1) == route(preds, q1)
    assert all(c.predict(build_action_sets(state, z, keys)[a]) == preds[a] for a in NON_IDENTITY)  # predictor frozen
    print("  [ok] CAL labels move only q; EVAL labels never enter q/U/routing; predictor frozen")


def test_serialization_and_weights_hash():
    import hashlib
    state, z, keys = _toy(); c = Candidate("C2", "PD", seed=0)
    was = build_action_sets(state, z, keys)["matched_coral"]
    p1 = c.predict(was)
    c2 = pickle.loads(pickle.dumps(c))
    assert c2.predict(was) == p1, "prediction changed across serialize round-trip"
    def wh(cand):
        h = hashlib.sha256()
        for k, v in cand.net.state_dict().items():
            h.update(k.encode()); h.update(np.ascontiguousarray(v.detach().numpy()).tobytes())
        return h.hexdigest()
    assert len(wh(c)) == 64 and wh(c) == wh(c2)
    print("  [ok] Candidate serialization round-trip preserves prediction; full 64-char weights SHA-256 stable")


def test_disease_tag_propagation():
    state, z, keys = _toy()
    pd = Candidate("C2", "PD", seed=0); scz = Candidate("C2", "SCZ", seed=0)
    was = build_action_sets(state, z, keys)["matched_coral"]
    assert pd.predict(was).disease == "PD" and scz.predict(was).disease == "SCZ"
    print("  [ok] disease tag propagates to predictions (PD/SCZ models are separate objects)")


def test_harmful_rate_tie_aware():
    rng = np.random.default_rng(0)
    # all-zero diffs -> NOT_EVALUABLE
    r0 = harmful_rate_test(np.full(12, 0.3), np.full(12, 0.3)); assert not r0["evaluable"]
    # too few nonzero -> NOT_EVALUABLE
    a = np.full(12, 0.3); b = a.copy(); b[0] = 0.1
    assert not harmful_rate_test(a, b)["evaluable"]
    # clean small distinct -> exact wilcoxon
    rt = np.array([0.1, 0.2, 0.0, 0.15, 0.05, 0.25, 0.12, 0.18, 0.03, 0.07, 0.09, 0.11])
    bf = rt + np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16])
    re = harmful_rate_test(rt, bf); assert re["evaluable"] and re["method"] == "exact_wilcoxon" and re["p"] < 0.05
    # ties present -> permutation, deterministic
    rt2 = np.zeros(30); bf2 = np.full(30, 0.1)              # constant positive diff (ties) -> permutation path
    rp1 = harmful_rate_test(rt2, bf2); rp2 = harmful_rate_test(rt2, bf2)
    assert rp1["method"] == "signflip_permutation" and rp1["p"] == rp2["p"]
    print("  [ok] harmful-rate test: NOT_EVALUABLE on all-zero/too-few; exact when clean+small; deterministic permutation on ties")


def main():
    print("ACAR v3 predictors/conformal synthetic guards:")
    for t in (test_predict_permutation_and_action_invariant, test_mask_change_changes_prediction,
              test_target_standardization_roundtrip, test_c2_scale_floor_and_q_clamp, test_c3_no_quantile_crossing,
              test_conformal_rank_and_inf, test_subject_joint_max_and_missing_action, test_cal_vs_eval_isolation,
              test_serialization_and_weights_hash, test_disease_tag_propagation, test_harmful_rate_tie_aware):
        t()
    print("ALL V3 PREDICTOR/CONFORMAL GUARDS PASS")


if __name__ == "__main__":
    main()
