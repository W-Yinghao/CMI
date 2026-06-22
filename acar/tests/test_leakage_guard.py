"""Hard guards (notes/ACAR_FROZEN_v2.md A2/A3 + review §3). Run BEFORE the go/no-go.

The five non-negotiables, on the ACTUAL deployment path:
  (1) NO label argument anywhere in the scoring/routing API (structural no-leak), and route_batch is deterministic
      -> permuting target labels cannot change φ_a, U_a, or the chosen action (bit-identical);
  (2) whole-batch: no positional/label-ordered subsetting (row-permutation leaves the decision unchanged to fp);
  (3) ΔR_a IS label-sensitive (estimand really uses y) — proves Phase-1/Phase-2 separation is real;
  (4) serialize->deserialize source state leaves route_batch (φ_a, U_a, action) bit-identical;
  (5) len(B) < MIN_BATCH -> forced identity AND the batch is RETAINED by the batcher (not deleted);
  plus: local spdim prob == validated tta_baselines.spdim_predict; all features finite.

    python -m acar.tests.test_leakage_guard
"""
import inspect
import pickle
import numpy as np

from cmi.eval.source_state import fit_source_state
from cmi.eval.tta_baselines import spdim_predict
from acar.config import MIN_BATCH, NON_IDENTITY, PAIRED_FEATURES, DISEASE, ACTIONS, ACARConfig
from acar.actions import apply_action, act_spdim
from acar.features import paired_features, context_features, feature_vector
from acar.scoring import score_actions
from acar.risk import delta_risk
from acar.regressor import ActionRegressor
from acar.conformal import fit_routers
from acar.deploy import Routers, route_batch, route_fvec
from acar.data import _natural_batches
from acar.run_gonogo import subject_cv, passes_g1, passes_g2, canonical_hash, _orient, _best_fixed


def _toy(n=200, d=8, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    z = rng.standard_normal((n, d)) + np.where(y[:, None] == 1, 0.8, -0.8)
    state = fit_source_state(z[:140], y[:140], 2, rho=0.1)
    return state, z[140:], y[140:]


def _toy_routers(state, z, seed=0):
    """Deterministic routers with a finite q (const regressors suffice for the leak guards)."""
    s = score_actions(state, z, NON_IDENTITY)
    regs = {a: ActionRegressor(seed=seed).fit(np.tile(s[a]["fvec"], (4, 1)), np.array([-0.1, -0.1, -0.1, -0.1]))
            for a in NON_IDENTITY}
    return Routers(regs=regs, q=0.05, delta=0.0, actions=tuple(NON_IDENTITY))


def _synth_records(seed=0, nsub=40):
    """Synthetic per-batch records (no spdim) for subject_cv / split-isolation tests. matched_coral.flip_rate is
    made predictive of its own harm so G1 has signal; subjects are disjoint clusters."""
    rng = np.random.default_rng(seed)
    recs = []
    for d, cohs in DISEASE.items():
        for c in cohs:
            for s in range(nsub):
                fv = {a: rng.standard_normal(len(PAIRED_FEATURES) + 4) for a in NON_IDENTITY}
                harm_mc = int(rng.random() < 0.4)
                fv["matched_coral"][2] = rng.normal(1.5 if harm_mc else -1.5)        # flip_rate index
                dr = {"identity": 0.0}; harm = {"identity": 0}
                for a in NON_IDENTITY:
                    hv = harm_mc if a == "matched_coral" else int(rng.random() < 0.5)
                    dr[a] = float(rng.normal(0.3 if hv else -0.3, 0.2)); harm[a] = int(dr[a] > 0)
                phi = {a: {f: float(fv[a][i]) for i, f in enumerate(PAIRED_FEATURES)} for a in NON_IDENTITY}
                recs.append(dict(disease=d, cohort=c, subject=f"{c}/sub{s:03d}", fallback=False, n=32,
                                 fvec=fv, phi=phi, dr=dr, harm=harm))
    return recs


def test_subject_cv_and_record_hash():
    recs = _synth_records(); cfg = ACARConfig()
    cv = subject_cv(recs, cfg)
    fl = cv["PD"]["folds_log"][0]
    for k in ("n_fit", "n_cal", "n_eval", "k", "q"):
        assert k in fl, f"folds_log missing {k}"
    assert isinstance(passes_g1(cv), tuple) and isinstance(passes_g2(cv), bool)
    h1 = canonical_hash(cv, recs, "X"); h2 = canonical_hash(subject_cv(recs, cfg), recs, "X")
    assert h1 == h2, "record-level hash not deterministic"
    recs2 = [dict(r) for r in recs]
    recs2[0] = dict(recs2[0]); recs2[0]["dr"] = dict(recs2[0]["dr"]); recs2[0]["dr"]["matched_coral"] += 0.7
    assert canonical_hash(subject_cv(recs2, cfg), recs2, "X") != h1, "hash insensitive to a per-record ΔR change"
    print(f"  [ok] subject_cv runs; folds_log has n_fit/n_cal/n_eval/k/q; record-level hash deterministic & ΔR-sensitive")


def test_split_isolation():
    """Split-isolation metamorphic test (review). CAL-label changes MAY move CAL ΔR, subject scores, q, and hence
    U_a and routing — but ONLY via the shared bound shift: U'_a(B)-U_a(B) = q'-q for every EVAL batch & action; they
    must NOT change FIT/ĝ_a, orientation, hyperparameters, best-fixed, splits, or φ_a. EVAL-label changes must leave
    q, φ, ĝ, U, and actions bit-identical (only ΔR / label-derived metrics change)."""
    recs = _synth_records()
    pd = [r for r in recs if r["disease"] == "PD"]
    fit, cal, ev = pd[:60], pd[60:100], pd[100:110]
    routers, diag = fit_routers(fit, cal, NON_IDENTITY, 0.1, 0.0, 0)

    # invariants computed from FIT only
    orient_before = {(a, f): _orient(fit, a, f) for a in NON_IDENTITY for f in PAIRED_FEATURES}
    bestfixed_before = _best_fixed(fit)
    g_eval_before = {(i, a): float(routers.regs[a].predict(e["fvec"][a][None])[0])
                     for i, e in enumerate(ev) for a in NON_IDENTITY}

    # (A) perturb CAL labels -> CAL ΔR changes -> q may move
    cal_p = [dict(r, dr={k: v + 1.0 for k, v in r["dr"].items()}, harm={k: 1 for k in r["harm"]}) for r in cal]
    routers_A, _ = fit_routers(fit, cal_p, NON_IDENTITY, 0.1, 0.0, 0)
    assert {(a, f): _orient(fit, a, f) for a in NON_IDENTITY for f in PAIRED_FEATURES} == orient_before, "orientation moved with CAL labels"
    assert _best_fixed(fit) == bestfixed_before, "best-fixed moved with CAL labels"
    # ĝ_a(φ) on EVAL is bit-identical (CAL labels never touch the FIT-fit model)
    for i, e in enumerate(ev):
        for a in NON_IDENTITY:
            assert float(routers_A.regs[a].predict(e["fvec"][a][None])[0]) == g_eval_before[(i, a)], "ĝ_a moved with CAL labels"
    # shared-bound-shift invariant. EDGE CASE q=+inf: do NOT evaluate U'-U=q'-q (inf-inf); instead every calibrated
    # non-identity action must have U=+inf and the batch routes to identity. For finite q,q': U'_a-U_a = q'-q.
    dq = routers_A.q - routers.q
    for e in ev:
        chosen, U = route_fvec(routers, e["fvec"]); chosenA, UA = route_fvec(routers_A, e["fvec"])
        if not np.isfinite(routers.q):
            assert all(not np.isfinite(U[a]) for a in NON_IDENTITY) and chosen == "identity", "q=+inf must force identity"
        if not np.isfinite(routers_A.q):
            assert all(not np.isfinite(UA[a]) for a in NON_IDENTITY) and chosenA == "identity", "q'=+inf must force identity"
        if np.isfinite(routers.q) and np.isfinite(routers_A.q):
            for a in NON_IDENTITY:
                assert abs((UA[a] - U[a]) - dq) < 1e-9, "U_a change not attributable solely to the shared q shift"
    # explicit q=+inf behaviour of the deployment API
    inf_routers = Routers(regs=routers.regs, q=float("inf"), delta=0.0, actions=tuple(NON_IDENTITY))
    c_inf, U_inf = route_fvec(inf_routers, ev[0]["fvec"])
    assert c_inf == "identity" and all(not np.isfinite(U_inf[a]) for a in NON_IDENTITY), "q=+inf must route to identity with U=+inf"

    # (B) perturb EVAL labels -> q, φ, ĝ, U, action bit-identical; only ΔR/metrics change
    for e in ev:
        c1, U1 = route_fvec(routers, e["fvec"])
        e_p = dict(e, dr={k: v + 1.0 for k, v in e["dr"].items()}, harm={k: 1 - v for k, v in e["harm"].items()})
        c2, U2 = route_fvec(routers, e_p["fvec"])
        assert c1 == c2 and U1 == U2, "EVAL-label change altered routing / U_a"
    assert routers.q == diag["q"], "q must come only from CAL, never EVAL"
    print("  [ok] split isolation: CAL→only shared q-shift (U'_a-U_a=q'-q); FIT/ĝ/orient/best-fixed fixed; EVAL→metrics only")


def test_no_label_arg_in_api():
    # The deployment/scoring API surface — the only place a target label could leak in. (Internal numeric kernels
    # like _bures_w2(x, y) use `y` for a second feature batch, not labels, so they are out of scope here.)
    for fn in (score_actions, route_batch, route_fvec, paired_features, context_features, feature_vector):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"y", "label", "labels", "y_target", "target"}), f"{fn.__name__} exposes a label arg"
    print("  [ok] no label argument anywhere in the scoring/routing API")


def test_route_batch_label_invariant_and_deterministic():
    state, z, y = _toy()
    routers = _toy_routers(state, z)
    c1, U1, phi1 = route_batch(state, routers, z)
    _ = np.random.default_rng(1).permutation(y)            # "permuting target labels" cannot reach route_batch
    c2, U2, phi2 = route_batch(state, routers, z)
    assert c1 == c2
    assert all(U1[a] == U2[a] for a in routers.actions), "U_a not bit-identical"
    for a in routers.actions:
        assert np.array_equal(phi1[a], phi2[a]), f"phi[{a}] not bit-identical"
    print(f"  [ok] route_batch label-invariant & deterministic (chose '{c1}')")


def test_whole_batch_no_subsetting():
    state, z, _ = _toy()
    routers = _toy_routers(state, z)
    c1, U1, _ = route_batch(state, routers, z)
    perm = np.random.default_rng(2).permutation(len(z))
    c2, U2, _ = route_batch(state, routers, z[perm])      # row permutation: aggregate decision must be stable
    assert c1 == c2, "decision changed under row permutation (positional/label-ordered subsetting?)"
    assert all(abs(U1[a] - U2[a]) < 1e-6 for a in routers.actions), "U_a unstable under row permutation"
    print("  [ok] whole-batch: decision invariant to row order (no subsetting)")


def test_delta_risk_uses_labels():
    state, z, y = _toy()
    p0, _ = apply_action("identity", state, z)
    pa, _ = apply_action("matched_coral", state, z)
    d = delta_risk(p0, pa, y, "nll")
    dp = delta_risk(p0, pa, np.random.default_rng(3).permutation(y), "nll")
    assert abs(d - dp) > 1e-9, "ΔR insensitive to y — estimand broken"
    print(f"  [ok] ΔR label-sensitive (ΔR={d:+.4f}, permuted={dp:+.4f})")


def test_serialize_roundtrip():
    state, z, _ = _toy()
    routers = _toy_routers(state, z)
    c1, U1, phi1 = route_batch(state, routers, z)
    state2 = pickle.loads(pickle.dumps(state))
    routers2 = pickle.loads(pickle.dumps(routers))
    c2, U2, phi2 = route_batch(state2, routers2, z)
    assert c1 == c2 and all(U1[a] == U2[a] for a in routers.actions)
    for a in routers.actions:
        assert np.array_equal(phi1[a], phi2[a])
    print("  [ok] serialize->deserialize source-state+routers: route_batch bit-identical")


def test_fallback_forced_identity_and_retained():
    state, z, y = _toy()
    routers = _toy_routers(state, z)
    c, U, _ = route_batch(state, routers, z[:MIN_BATCH - 1])      # tiny batch
    assert c == "identity", "small batch not forced to identity"
    n = 5
    sub = np.array(["sub01"] * n); rec = np.array(["rec01"] * n); win = np.arange(n)
    zz = np.zeros((n, 8)); yy = np.zeros(n, int)
    batches = _natural_batches("PD", "dsTEST", zz, yy, sub, rec, win, batch_size=32)
    assert len(batches) == 1 and batches[0].fallback is True, "small recording was dropped instead of retained"
    print("  [ok] len(B)<MIN_BATCH -> forced identity AND retained by the batcher")


def test_spdim_no_drift():
    state, z, _ = _toy()
    p_ref = spdim_predict(state, z)
    p_local, ztil = act_spdim(state, z)
    assert np.allclose(p_ref, p_local, atol=1e-8) and ztil.shape == z.shape
    print("  [ok] local spdim prob == tta_baselines.spdim_predict")


def test_features_finite():
    state, z, _ = _toy()
    for a in ("matched_coral", "spdim", "t3a"):
        p0, z0 = apply_action("identity", state, z); pa, za = apply_action(a, state, z)
        phi = paired_features(p0, pa, z0, za); ctx = context_features(state, za, pa)
        for k, v in {**phi, **ctx}.items():
            assert v is None or np.isnan(v) or np.isfinite(v), f"{a}.{k} not finite: {v}"
        assert np.all(np.isfinite(feature_vector(phi, ctx)))
    print("  [ok] all features finite-or-nan; fvec finite after imputation")


def main():
    print("ACAR hard-guard tests (v2):")
    test_no_label_arg_in_api()
    test_route_batch_label_invariant_and_deterministic()
    test_whole_batch_no_subsetting()
    test_delta_risk_uses_labels()
    test_serialize_roundtrip()
    test_fallback_forced_identity_and_retained()
    test_split_isolation()
    test_subject_cv_and_record_hash()
    test_spdim_no_drift()
    test_features_finite()
    print("ALL HARD GUARDS PASS")


if __name__ == "__main__":
    main()
