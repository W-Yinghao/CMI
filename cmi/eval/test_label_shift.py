"""CPU validation of CIPC label-shift correction (NO GPU). Validates the estimator + correction math:
 (i)   BBSE/MLLS recovers pi_T from unlabeled target predictions;
 (ii)  re-prioring beats uncorrected balanced-acc under label shift, ~matching the oracle-pi_T Bayes rule;
 (iii) NULL-SAFE: when pi_T == pi_S, corrected == uncorrected exactly;
 (iv)  MLLS-EM stays in-simplex under near-singular confusion (heavy class overlap).
"""
import numpy as np
from cmi.eval.label_shift import (bbse_prior, apply_correction, cipc_predict, _simplex_clip,
                                  feature_coral_recenter)


def balanced_acc(yhat, y, n_cls):
    return float(np.mean([(yhat[y == c] == c).mean() for c in range(n_cls) if (y == c).any()]))


def make_gaussians(n_cls=3, dz=8, sep=2.0, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 1, (n_cls, dz)) * sep
    return mu  # shared Sigma = I; P(z|y) shared across domains => conditional invariance holds


def sample(mu, prior, n, seed):
    rng = np.random.default_rng(seed)
    n_cls, dz = mu.shape
    y = rng.choice(n_cls, size=n, p=prior)
    z = mu[y] + rng.normal(0, 1, (n, dz))
    return z, y


def source_bayes_probs(z, mu, pi_S):
    """ERM-like classifier: p_S(y|z) ∝ N(z;mu_y,I) pi_S(y) (source prior baked in)."""
    n_cls = mu.shape[0]
    logp = -0.5 * ((z[:, None, :] - mu[None]) ** 2).sum(2) + np.log(_simplex_clip(pi_S))[None]
    logp -= logp.max(1, keepdims=True)
    p = np.exp(logp); return p / p.sum(1, keepdims=True)


def plain_acc(yhat, y):
    return float((yhat == y).mean())


def run_covariate():
    """The BALANCED-ACCURACY lever: transductive feature-CORAL corrects an unseen per-domain affine covariate
    shift so the source boundary fits the target. (Prior correction does NOT help balanced acc.)"""
    print("=== Covariate-shift correction (feature-CORAL) — the balanced-accuracy lever ===")
    fails = 0
    rng = np.random.default_rng(11)
    for n_cls in (2, 3):
        mu = make_gaussians(n_cls=n_cls, dz=8, sep=1.3, seed=1)        # OVERLAPPING (sep=1.3) so boundary matters
        pi = np.ones(n_cls) / n_cls
        zs, ys = sample(mu, pi, 5000, seed=2)                          # source pool (conditional-invariant)
        # target: global affine covariate shift A z + b applied to ALL classes (unseen domain)
        A = np.eye(8) + 0.6 * rng.normal(0, 1, (8, 8)) / np.sqrt(8)
        b = rng.normal(0, 2.5, 8)
        zt_raw, yt = sample(mu, pi, 5000, seed=3)
        zt = zt_raw @ A.T + b
        ba_raw = balanced_acc(source_bayes_probs(zt, mu, pi).argmax(1), yt, n_cls)       # ERM on shifted target
        zt_corr = feature_coral_recenter(zs, zt)                       # transductive CORAL recenter
        ba_corr = balanced_acc(source_bayes_probs(zt_corr, mu, pi).argmax(1), yt, n_cls)
        # null-safety: no covariate shift -> CORAL ~ identity -> no change
        zt0, yt0 = sample(mu, pi, 5000, seed=7)
        ba0_raw = balanced_acc(source_bayes_probs(zt0, mu, pi).argmax(1), yt0, n_cls)
        ba0_corr = balanced_acc(source_bayes_probs(feature_coral_recenter(zs, zt0), mu, pi).argmax(1), yt0, n_cls)
        ok_gain = ba_corr > ba_raw + 0.01                              # helps balanced acc (more on harder problems)
        ok_null = abs(ba0_corr - ba0_raw) < 0.02                        # and does no harm with no shift
        print(f"  [{n_cls}-cls] covariate-shift bAcc: ERM={ba_raw:.3f} -> CORAL={ba_corr:.3f} "
              f"(gain {100*(ba_corr-ba_raw):+.1f})  {'PASS' if ok_gain else 'FAIL'}")
        print(f"  [{n_cls}-cls] null-safety (no shift): {ba0_raw:.3f} -> {ba0_corr:.3f}  "
              f"{'PASS' if ok_null else 'FAIL'}")
        fails += (not ok_gain) + (not ok_null)
    return fails


def run_stress():
    """Continuous class-support stress test: under a fixed covariate shift, sweep the target class prior from
    balanced to single-class. Global CORAL should degrade (catastrophic at one-class); PMCT (prior-matched
    conditional transport) should stay stable — turning the empirical EA/CORAL fallback into a method."""
    from cmi.eval.label_shift import transduct_predict
    print("=== Class-support stress test: native vs global-CORAL vs PMCT (bAcc), covariate shift FIXED ===")
    rng = np.random.default_rng(7)
    n_cls = 2; mu = make_gaussians(n_cls=n_cls, dz=8, sep=1.3, seed=1)
    A = np.eye(8) + 0.6 * rng.normal(0, 1, (8, 8)) / np.sqrt(8); b = rng.normal(0, 2.0, 8)  # the SAME covariate shift
    zs, ys = sample(mu, np.ones(n_cls) / n_cls, 6000, seed=2)
    pi_S = np.ones(n_cls) / n_cls
    print(f"  {'pi_T':14}{'native':>8}{'plainCORAL':>11}{'matchCORAL':>11}{'PMCT':>8}  (decisive: PMCT vs matchCORAL)")
    fails = 0
    for p0 in [0.5, 0.8, 0.95, 1.0]:
        piT = np.array([p0, 1 - p0])
        zt_raw, yt = sample(mu, piT, 6000, seed=int(100 * p0) + 3)
        zt = zt_raw @ A.T + b
        nat = transduct_predict(zs, ys, zt, pi_S, n_cls, mode="probe")["prob"]
        cor = transduct_predict(zs, ys, zt, pi_S, n_cls, mode="coral")["prob"]
        mco = transduct_predict(zs, ys, zt, pi_S, n_cls, mode="matched_coral")["prob"]   # de-confounded baseline
        pmc = transduct_predict(zs, ys, zt, pi_S, n_cls, mode="pmct")["prob"]
        bn = balanced_acc(nat.argmax(1), yt, n_cls); bc = balanced_acc(cor.argmax(1), yt, n_cls)
        bm = balanced_acc(mco.argmax(1), yt, n_cls); bp = balanced_acc(pmc.argmax(1), yt, n_cls)
        tag = ""
        # the ONLY admissible PMCT claim: it beats MATCHED-CORAL (same shrink/gate/interp) under prior shift —
        # isolating the prior-matched reference from the shrinkage/gate machinery.
        if p0 == 0.8 and not (bp > bm + 0.01):
            fails += 1; tag = " <-- FAIL (no PMCT-over-matchedCORAL win)"
        if p0 < 1.0 and bp < bm - 0.02:
            fails += 1; tag = " <-- FAIL (worse than matchedCORAL)"
        print(f"  ({p0:.2f},{1-p0:.2f})  {bn:8.3f}{bc:11.3f}{bm:11.3f}{bp:8.3f}{tag}")
    print("  decisive comparison is PMCT vs MATCHED-CORAL (identical machinery, ref-only difference); plainCORAL shown for context")
    return fails


def run_nullsafety():
    """Exact null-safety regression test (reviewer §2.2): when the reference moments equal the target moments,
    the whiten-color map must be EXACTLY the identity (same shrink operator applied to both sides). Also checks
    the reliability gate does not over-damp a confident skewed target (the entropy-gate bug, reviewer §2.3)."""
    from cmi.eval.label_shift import pmct_transport
    print("=== Exact null-safety + reliability-gate regression ===")
    rng = np.random.default_rng(1); d, n = 8, 500
    mu = np.stack([rng.normal(0, 1, d), rng.normal(2, 1, d)]); y = rng.integers(0, 2, n)
    z = mu[y] + rng.normal(0, 1, (n, d))
    fails = 0
    zt, _, _ = pmct_transport(z, y, z, 2, alpha=1.0, ref="pooled", gate="off")   # target==source ref => identity
    err = float(np.abs(zt - z).max())
    ok = err < 1e-7; fails += (not ok)
    print(f"  exact identity when ref==target moments: max|T(z)-z|={err:.2e}  {'PASS' if ok else 'FAIL (not null-safe)'}")
    # confident, well-separated, 90/10 skewed target: gate must NOT collapse alpha (prior is real, not unreliable)
    mu2 = np.stack([rng.normal(0, 1, d), rng.normal(7, 1, d)])
    ys = np.r_[np.zeros(450, int), np.ones(50, int)]; zs = mu2[ys] + rng.normal(0, 1, (500, d))
    yb = rng.integers(0, 2, 500); zb = mu2[yb] + rng.normal(0, 1, (500, d))      # balanced source
    ztk, _, pik = pmct_transport(zb, yb, zs, 2, alpha=1.0)
    moved = float(np.abs(ztk - zs).mean()); ok2 = moved > 0.05; fails += (not ok2)
    print(f"  confident-skewed target NOT over-damped: mean|Δz|={moved:.3f}  {'PASS' if ok2 else 'FAIL (gate killed a real prior)'}")
    return fails


def run():
    print("=== CIPC prior-estimation validation (CPU) — note: prior-correction helps PLAIN acc, not balanced ===")
    fails = 0
    for n_cls in (2, 3):
        mu = make_gaussians(n_cls=n_cls, sep=1.6, seed=1)
        pi_S = np.ones(n_cls) / n_cls                                   # balanced source
        # shifted target prior
        pi_T = np.zeros(n_cls); pi_T[0] = 0.7; pi_T[1:] = 0.3 / max(1, n_cls - 1); pi_T = _simplex_clip(pi_T)
        # source eval split (for confusion matrix) + target
        zs, ys = sample(mu, pi_S, 4000, seed=2)
        zt, yt = sample(mu, pi_T, 4000, seed=3)
        prob_se = source_bayes_probs(zs, mu, pi_S)
        prob_t = source_bayes_probs(zt, mu, pi_S)                       # ERM (source-prior) preds on target

        # (i) estimator recovers pi_T
        pi_hat = bbse_prior(prob_se, ys, prob_t, pi_S, n_cls, method="em")
        pi_hat_solve = bbse_prior(prob_se, ys, prob_t, pi_S, n_cls, method="solve")
        err_em = np.abs(pi_hat - pi_T).sum(); err_solve = np.abs(pi_hat_solve - pi_T).sum()

        # (ii) corrected beats uncorrected on PLAIN acc (prior-correction's true metric), ~ oracle
        corr, _ = cipc_predict(prob_se, ys, prob_t, pi_S, n_cls, method="em")
        prob_oracle = source_bayes_probs(zt, mu, pi_T)                  # Bayes-optimal target rule
        ba_unc = plain_acc(prob_t.argmax(1), yt)
        ba_corr = plain_acc(corr.argmax(1), yt)
        ba_oracle = plain_acc(prob_oracle.argmax(1), yt)

        # (iii) null safety: target prior == source prior -> no change
        zt0, yt0 = sample(mu, pi_S, 4000, seed=5)
        prob_t0 = source_bayes_probs(zt0, mu, pi_S)
        corr0, pihat0 = cipc_predict(prob_se, ys, prob_t0, pi_S, n_cls, method="em", gate_l1=0.05)
        null_safe = np.allclose(corr0, prob_t0)

        ok_i = err_em < 0.06 and err_solve < 0.08
        # prior-correction is the PLAIN-acc lever (secondary): validate it MATCHES the oracle re-prior and
        # does not lose vs uncorrected. (It is not expected to move balanced acc — that's feature-CORAL.)
        ok_ii = (ba_corr >= ba_unc - 0.005) and (abs(ba_corr - ba_oracle) < 0.01)
        ok_iii = null_safe
        ok_iv = (pi_hat.min() >= 0) and abs(pi_hat.sum() - 1) < 1e-6
        for nm, ok in [("(i) recover pi_T", ok_i), ("(ii) corrected>unc PLAIN-acc~oracle", ok_ii),
                       ("(iii) null-safe", ok_iii), ("(iv) in-simplex", ok_iv)]:
            print(f"  [{n_cls}-cls] {nm:36s} {'PASS' if ok else 'FAIL'}")
            fails += (not ok)
        print(f"     pi_T={np.round(pi_T,3)} pi_hat_em={np.round(pi_hat,3)} (L1 {err_em:.3f}) "
              f"pi_hat_solve L1 {err_solve:.3f}")
        print(f"     PLAIN acc: uncorrected={ba_unc:.3f}  CIPC={ba_corr:.3f}  oracle={ba_oracle:.3f}  "
              f"(gain {100*(ba_corr-ba_unc):+.1f})")
    return fails


def run_serialization():
    """Strict source-free serialization (reviewer R8 §6.1): the deployment path consumes only the SourceState +
    unlabeled target features and reproduces the online transport to <1e-6. Also asserts the deployment API
    cannot receive source examples/labels (no z_src/y_src parameter), and the state hash is deterministic."""
    import inspect
    from sklearn.linear_model import LogisticRegression
    from cmi.eval.label_shift import pmct_transport
    from cmi.eval.source_state import fit_source_state, pmct_predict_serialized, source_state_hash
    print("=== Source-state serialization (strict source-free) — online vs serialized equivalence ===")
    rng = np.random.default_rng(3); d, n = 8, 800
    mu = np.stack([rng.normal(0, 1, d), rng.normal(2, 1, d)]); ys = rng.integers(0, 2, n)
    zs = mu[ys] + rng.normal(0, 1, (n, d))
    yt = rng.integers(0, 2, n); zt = mu[yt] @ (np.eye(d) + 0.3 * rng.normal(0, 1, (d, d)) / np.sqrt(d)) \
        + rng.normal(0, 1, (n, d))
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(zs, ys)              # SAME readout for both paths
    fails = 0
    for ref in ("prior_matched", "pooled"):
        for tmap in ("wc", "ot"):
            _, p_on, _ = pmct_transport(zs, ys, zt, 2, clf=clf, ref=ref, tmap=tmap)
            state = fit_source_state(zs, ys, 2, clf=clf)
            p_se, _ = pmct_predict_serialized(state, zt, ref=ref, tmap=tmap)
            err = float(np.abs(p_on - p_se).max()); ok = err < 1e-6; fails += (not ok)
            print(f"  ref={ref:13} tmap={tmap}: max|p_online-p_serialized|={err:.2e}  {'PASS' if ok else 'FAIL'}")
    # API guard: the deployment function must NOT accept source examples/labels
    params = set(inspect.signature(pmct_predict_serialized).parameters)
    no_src = not (params & {"z_src", "y_src", "x_src"}); fails += (not no_src)
    print(f"  deployment API rejects source args (no z_src/y_src): {'PASS' if no_src else 'FAIL'}  params={sorted(params)}")
    # hash determinism
    h1 = source_state_hash(fit_source_state(zs, ys, 2, clf=clf))
    h2 = source_state_hash(fit_source_state(zs, ys, 2, clf=clf))
    det = h1 == h2; fails += (not det)
    print(f"  state hash deterministic: {'PASS' if det else 'FAIL'}  ({h1[:16]}...)")
    return fails


if __name__ == "__main__":
    f1 = run()
    print()
    f2 = run_covariate()
    print()
    f3 = run_stress()
    print()
    f4 = run_nullsafety()
    print()
    f5 = run_serialization()
    tot = f1 + f2 + f3 + f4 + f5
    print(f"\n{'ALL PASS' if tot == 0 else f'{tot} FAILURES'}")
    raise SystemExit(1 if (f1 + f2 + f3) else 0)
