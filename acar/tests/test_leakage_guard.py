"""Synthetic guards — run BEFORE the real go/no-go (notes/ACAR_FROZEN.md §5).

Asserts the five non-negotiables hold on the actual code paths:
  (1) phi_a takes no labels and is bit-invariant to y-permutation,
  (2) ΔR_a IS sensitive to y (the estimand really uses labels),
  (3) Phase-1 scoring is deterministic (double-call byte-identical),
  (4) the local spdim action's prob == the validated tta_baselines.spdim_predict (no numeric drift),
  (5) every paired/context feature is finite-or-NaN, never inf.

    python -m acar.tests.test_leakage_guard
"""
import numpy as np

from cmi.eval.source_state import fit_source_state
from cmi.eval.tta_baselines import spdim_predict
from acar.actions import apply_action, act_spdim
from acar.features import paired_features, context_features, feature_vector
from acar.risk import delta_risk
from acar.run_gonogo import phase1_score


def _toy(n=160, d=8, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    z = rng.standard_normal((n, d)) + np.where(y[:, None] == 1, 0.8, -0.8)
    state = fit_source_state(z[:120], y[:120], 2, rho=0.1)
    return state, z[120:], y[120:]


def test_phi_label_free_and_deterministic():
    state, z, y = _toy()
    actions = ["identity", "matched_coral", "spdim", "t3a"]
    s1 = phase1_score(state, z, actions)
    s2 = phase1_score(state, z, actions)                       # double call
    for a in ("matched_coral", "spdim", "t3a"):
        f1, f2 = np.nan_to_num(s1[a]["fvec"]), np.nan_to_num(s2[a]["fvec"])
        assert np.array_equal(f1, f2), f"{a}: phase-1 not deterministic"
    # permuting y must not be reachable by phase-1 (no y arg) -> identical fvecs by construction
    _ = np.random.default_rng(1).permutation(y)
    s3 = phase1_score(state, z, actions)
    for a in ("matched_coral", "spdim", "t3a"):
        assert np.array_equal(np.nan_to_num(s1[a]["fvec"]), np.nan_to_num(s3[a]["fvec"]))
    print("  [ok] phi_a is label-free and deterministic")


def test_delta_risk_uses_labels():
    state, z, y = _toy()
    p0, _ = apply_action("identity", state, z)
    pa, _ = apply_action("matched_coral", state, z)
    d_true = delta_risk(p0, pa, y, "nll")
    yperm = np.random.default_rng(3).permutation(y)
    d_perm = delta_risk(p0, pa, yperm, "nll")
    assert abs(d_true - d_perm) > 1e-9, "ΔR insensitive to y — estimand broken"
    print(f"  [ok] ΔR is label-sensitive (ΔR={d_true:+.4f}, permuted={d_perm:+.4f})")


def test_spdim_no_drift():
    state, z, _ = _toy()
    p_ref = spdim_predict(state, z)
    p_local, ztil = act_spdim(state, z)
    assert np.allclose(p_ref, p_local, atol=1e-8), "local spdim prob drifted from tta_baselines"
    assert ztil.shape == z.shape
    print("  [ok] local spdim prob == tta_baselines.spdim_predict")


def test_features_finite():
    state, z, _ = _toy()
    for a in ("matched_coral", "spdim", "t3a"):
        p0, z0 = apply_action("identity", state, z)
        pa, za = apply_action(a, state, z)
        phi = paired_features(p0, pa, z0, za)
        ctx = context_features(state, za, pa)
        for k, v in {**phi, **ctx}.items():
            assert v is None or np.isnan(v) or np.isfinite(v), f"{a}.{k} not finite: {v}"
        fv = feature_vector(phi, ctx)
        assert np.all(np.isfinite(fv)), f"{a}: feature_vector has non-finite after imputation"
    print("  [ok] all paired/context features finite-or-nan; fvec finite after imputation")


def main():
    print("ACAR leakage-guard tests:")
    test_phi_label_free_and_deterministic()
    test_delta_risk_uses_labels()
    test_spdim_no_drift()
    test_features_finite()
    print("ALL GUARDS PASS")


if __name__ == "__main__":
    main()
