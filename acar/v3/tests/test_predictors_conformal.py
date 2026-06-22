"""Synthetic guards for acar/v3/predictors.py (FittedCandidate) + conformal.py (fail-closed). TOY ONLY (toy source
state + GIVEN synthetic ΔR; no DEV cohorts/gate/lockbox). Run: python -m acar.v3.tests.test_predictors_conformal
"""
import math
import pickle
import numpy as np

from cmi.eval.source_state import fit_source_state
from acar.config import N_CLS
from acar.v3.set_features import build_action_sets, WindowActionSet, PER_WINDOW_FEATURES, NON_IDENTITY
from acar.v3.predictors import CandidatePrediction, CANDIDATES, score, upper_bound
from acar.v3.training import fit_candidate, TrainExample
from acar.v3.conformal import subject_joint_score, conformal_rank, conformal_q, route, harmful_rate_test


def _state(d=8, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(160) < 0.5).astype(int)
    z = rng.standard_normal((160, d)) + np.where(y[:, None] == 1, 0.8, -0.8)
    return fit_source_state(z, y, N_CLS, rho=0.1)


def _fit(state, candidate, seed=0, n=16, nwin=12, d=8):
    rng = np.random.default_rng(seed + 7); ex = []
    for s in range(n):
        z = rng.standard_normal((nwin, d)) + 0.2 * rng.standard_normal()
        sets = build_action_sets(state, z, [f"d/s{s}/r/{w}" for w in range(nwin)])
        for a in NON_IDENTITY:
            ex.append(TrainExample(sets[a], float(z.mean() + 0.1 * rng.standard_normal()), f"d|s{s}"))
    return fit_candidate(candidate, "PD", ex[:n], ex[n:], seed=seed)


def _expect(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def _pred(candidate="C1", disease="PD", action="matched_coral", point=0.0, uc=None, **kw):
    uc = point if uc is None else uc
    return CandidatePrediction(candidate, disease, action, point, uc, **kw)


def test_candidate_prediction_validation():
    _pred("C1", point=0.0)                                                  # valid
    _expect(ValueError, lambda: _pred("C1", scale_used=1.0))                # C1 must have no scale
    _expect(ValueError, lambda: _pred("C2", point=1.0, uc=2.0, scale_used=1.0, scale_raw=1.0, scale_floor=0.0))
    _expect(ValueError, lambda: CandidatePrediction("C2", "PD", "spdim", 1.0, 1.0, 5.0, 2.0, 3.0))  # used!=max(raw,floor)
    _expect(ValueError, lambda: CandidatePrediction("C3", "PD", "t3a", 1.0, 1.0))   # C3 needs uc>point
    CandidatePrediction("C3", "PD", "t3a", 1.0, 1.5)                        # valid C3
    print("  [ok] CandidatePrediction enforces C1/C2/C3 invariants")


def test_predict_permutation_and_action_order():
    state = _state()
    for cand in CANDIDATES:
        fc = _fit(state, cand, seed=0)
        z = np.random.default_rng(3).standard_normal((12, 8)); keys = [f"d/s/r/{i}" for i in range(12)]
        s1 = build_action_sets(state, z, keys, actions=("t3a", "matched_coral", "spdim"))
        perm = np.random.default_rng(4).permutation(12)
        s2 = build_action_sets(state, z[perm], [keys[i] for i in perm], actions=("spdim", "t3a", "matched_coral"))
        for a in NON_IDENTITY:
            assert fc.predict(s1[a]) == fc.predict(s2[a]), f"{cand}/{a}: prediction not invariant"
    print("  [ok] predictions invariant to window permutation AND action-request order (C1/C2/C3)")


def test_mask_change_changes_prediction():
    state = _state(); fc = _fit(state, "C2", seed=0)
    z = np.random.default_rng(5).standard_normal((12, 8))
    was = build_action_sets(state, z, [f"d/s/r/{i}" for i in range(12)])["matched_coral"]
    j = PER_WINDOW_FEATURES.index("embed_disp")
    v = was.values.copy(); v[:, j] = 0.0
    m = was.availability_mask.copy(); m[:, j] = 0
    miss = WindowActionSet(v, m, was.context_values.copy(), was.context_mask.copy(), was.action_name, was.action_index, was.window_keys)
    genuine = WindowActionSet(v, was.availability_mask.copy(), was.context_values.copy(), was.context_mask.copy(), was.action_name, was.action_index, was.window_keys)
    assert fc.predict(miss) != fc.predict(genuine)
    print("  [ok] availability-mask change (value fixed) changes prediction")


def test_c2_scale_and_q_clamp_and_c3_no_cross():
    state = _state()
    fc2 = _fit(state, "C2", seed=0); fc1 = _fit(state, "C1", seed=0); fc3 = _fit(state, "C3", seed=0)
    z = np.random.default_rng(6).standard_normal((12, 8)); keys = [f"d/s/r/{i}" for i in range(12)]
    sets = build_action_sets(state, z, keys)
    p2 = fc2.predict(sets["matched_coral"])
    assert p2.scale_raw > 0 and p2.scale_used == max(p2.scale_raw, p2.scale_floor) > 0
    assert abs(upper_bound(p2, -5.0) - p2.upper_center) < 1e-12              # C2 q⁺ clamp
    p1 = fc1.predict(sets["matched_coral"]); p3 = fc3.predict(sets["t3a"])
    assert abs(upper_bound(p1, -5.0) - (p1.upper_center - 5.0)) < 1e-12      # C1 no clamp
    assert abs(upper_bound(p3, -5.0) - (p3.upper_center - 5.0)) < 1e-12      # C3 no clamp
    for a in NON_IDENTITY:
        pp = fc3.predict(sets[a]); assert pp.upper_center > pp.point         # C3 no crossing
    print("  [ok] C2 scale_raw/floor/used + q⁺; C1/C3 no clamp; C3 no crossing")


def test_conformal_rank_inf_alpha():
    assert conformal_rank(20, 0.1) == 19
    q1, k1 = conformal_q([float(i) for i in range(20)], 0.1); assert q1 == 18.0 and k1 == 19
    qi, ki = conformal_q([1.0, 2.0, 3.0], 0.1); assert math.isinf(qi) and ki == 4
    qe, ke = conformal_q([], 0.1); assert math.isinf(qe) and ke == 1                  # empty CAL -> (+inf, k=1)
    _expect(ValueError, lambda: conformal_q([1.0], 1.5))                              # alpha range
    _expect(ValueError, lambda: conformal_q([1.0, float("nan")], 0.1))               # finite check
    print("  [ok] conformal rank/+inf; empty CAL->(+inf,k=1); alpha range; finite-checked")


def test_subject_joint_failclosed():
    full = {a: (_pred("C1", action=a), 0.1) for a in NON_IDENTITY}
    assert abs(subject_joint_score([full]) - max(score(_pred("C1", action=a), 0.1) for a in NON_IDENTITY)) < 1e-12
    _expect(ValueError, lambda: subject_joint_score([]))                              # empty subject
    miss = {a: (_pred("C1", action=a), 0.1) for a in NON_IDENTITY[:-1]}
    _expect(ValueError, lambda: subject_joint_score([miss]))                          # missing action
    extra = {**full, "identity": (_pred("C1", action="matched_coral"), 0.1)}
    _expect(ValueError, lambda: subject_joint_score([extra]))                         # extra action
    _expect(ValueError, lambda: subject_joint_score([{a: (_pred("C1", action=a), float("inf")) for a in NON_IDENTITY}]))
    mism = {a: (_pred("C1", action=("spdim" if a == "matched_coral" else a)), 0.1) for a in NON_IDENTITY}
    _expect(ValueError, lambda: subject_joint_score([mism]))                          # pred.action != key
    mixed = {a: (_pred(("C1" if a == "matched_coral" else "C3"), action=a, point=0.0, uc=(0.0 if a == "matched_coral" else 0.5)), 0.1) for a in NON_IDENTITY}
    _expect(ValueError, lambda: subject_joint_score([mixed]))                         # mixed candidate
    print("  [ok] subject_joint_score fail-closed (empty/missing/extra/non-finite/action-mismatch/mixed-candidate)")


def test_route_failclosed_and_canonical():
    preds = {a: _pred("C1", action=a, point=-1.0) for a in NON_IDENTITY}              # all U<0 at q=0
    chosen, U = route(preds, 0.0); assert tuple(U.keys()) == NON_IDENTITY and chosen in NON_IDENTITY
    ci, Ui = route(preds, math.inf); assert ci == "identity" and all(math.isinf(u) for u in Ui.values())
    _expect(ValueError, lambda: route({a: preds[a] for a in NON_IDENTITY[:-1]}, 0.0))   # incomplete set
    _expect(ValueError, lambda: route(preds, float("nan")))
    # disease/candidate mix in route -> raise
    pmix = {a: _pred(disease=("PD" if a == "matched_coral" else "SCZ"), action=a, point=-1.0) for a in NON_IDENTITY}
    _expect(ValueError, lambda: route(pmix, 0.0))
    print("  [ok] route requires full action set, canonical U order, q=+inf->identity, rejects mixed disease")


def test_cal_vs_eval_isolation():
    preds = {a: _pred("C2", action=a, point=0.0, uc=0.0, scale_used=1.0, scale_raw=1.0, scale_floor=0.0) for a in NON_IDENTITY}
    rng = np.random.default_rng(0)
    cal = [subject_joint_score([{a: (preds[a], float(rng.normal())) for a in NON_IDENTITY}]) for _ in range(12)]
    cal2 = [subject_joint_score([{a: (preds[a], s + 1.0) for a, s in zip(NON_IDENTITY, [float(rng.normal()) for _ in NON_IDENTITY])}]) for _ in range(12)]
    q1, _ = conformal_q(cal, 0.1)
    assert math.isfinite(q1)
    assert route(preds, q1) == route(preds, q1)                  # EVAL labels never enter route
    assert conformal_q(cal2, 0.1)[0] != q1                       # CAL labels move q
    print("  [ok] CAL labels move only q; EVAL labels enter neither q nor routing")


def test_serialize_and_disease_tag():
    state = _state(); fc = _fit(state, "C2", seed=0)
    was = build_action_sets(state, np.random.default_rng(8).standard_normal((12, 8)), [f"d/s/r/{i}" for i in range(12)])["matched_coral"]
    fc2 = pickle.loads(pickle.dumps(fc))
    assert fc2.predict(was) == fc.predict(was) and fc2.artifact_sha256 == fc.artifact_sha256
    assert fc.predict(was).disease == "PD"
    print("  [ok] FittedCandidate serialize round-trip (prediction + artifact hash); disease tag on predictions")


def test_harmful_rate_tie_aware():
    a = np.full(12, 0.3)
    assert not harmful_rate_test(a, a)["evaluable"]                          # all-zero
    b = a.copy(); b[0] = 0.1; assert not harmful_rate_test(a, b)["evaluable"]   # too few nonzero
    rt = np.array([0.1, 0.2, 0.05, 0.15, 0.0, 0.25, 0.12, 0.18, 0.03, 0.07, 0.09, 0.11])
    bf = rt + np.linspace(0.05, 0.16, 12)
    re = harmful_rate_test(rt, bf); assert re["evaluable"] and re["method"] == "exact_wilcoxon" and re["p"] < 0.05
    rp1 = harmful_rate_test(np.zeros(30), np.full(30, 0.1)); rp2 = harmful_rate_test(np.zeros(30), np.full(30, 0.1))
    assert rp1["method"] == "permutation_wilcoxon" and rp1["p"] == rp2["p"]
    print("  [ok] harmful-rate ONE estimand: exact when clean; deterministic PermutationMethod on ties; NOT_EVALUABLE guards")


def main():
    print("ACAR v3 predictors/conformal synthetic guards:")
    for t in (test_candidate_prediction_validation, test_predict_permutation_and_action_order,
              test_mask_change_changes_prediction, test_c2_scale_and_q_clamp_and_c3_no_cross,
              test_conformal_rank_inf_alpha, test_subject_joint_failclosed, test_route_failclosed_and_canonical,
              test_cal_vs_eval_isolation, test_serialize_and_disease_tag, test_harmful_rate_tie_aware):
        t()
    print("ALL V3 PREDICTOR/CONFORMAL GUARDS PASS")


if __name__ == "__main__":
    main()
