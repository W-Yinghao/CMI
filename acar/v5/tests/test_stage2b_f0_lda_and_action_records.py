"""Guard (Stage-2B0): the f_0 = LDA readout of source_state (means, shared cov, priors) is correct + fail-closed, and the
action-record assembly is label-free with exactly the 7 protocol features per non-identity action. Synthetic only (torch-free)."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import stage2_action_records as AR
from acar.v5.tests._util import expect_raises, ok, stage2b_synthetic_source_state


def test_lda_readout_matches_pinned_formula():
    import numpy as np
    ss = stage2b_synthetic_source_state(D=6, seed=3)
    lda = AR.SourceLDA(ss)
    Z = np.random.RandomState(1).randn(5, 6)
    p = lda.predict_proba(Z)
    assert p.shape == (5, 2) and np.allclose(p.sum(1), 1.0)
    cov_inv = np.linalg.inv(ss["cov"])
    W = ss["means"] @ cov_inv
    b = np.log(ss["priors"]) - 0.5 * (W * ss["means"]).sum(1)    # b_k = log π_k − 0.5 μ_kᵀ Σ⁻¹ μ_k
    s = Z @ W.T + b
    man = np.exp(s - s.max(1, keepdims=True))
    man /= man.sum(1, keepdims=True)
    assert np.allclose(p, man)
    ok("f_0 = softmax(w_k·z + b_k), w_k=Σ⁻¹μ_k, b_k=logπ_k−0.5μ_kᵀΣ⁻¹μ_k (matches the pinned LDA readout)")


def test_lda_fail_closed():
    import numpy as np
    ss = stage2b_synthetic_source_state(D=4, seed=0)
    cases = [
        ("missing_cov", {k: v for k, v in ss.items() if k != "cov"}),
        ("singular_cov", {**ss, "cov": np.zeros((4, 4))}),
        ("wrong_class_order", {**ss, "classes": np.array([1, 0])}),
        ("nonpositive_prior", {**ss, "priors": np.array([0.0, 1.0])}),
        ("wrong_means_shape", {**ss, "means": np.zeros((3, 4))}),
        ("nonfinite", {**ss, "means": np.full((2, 4), np.nan)}),
    ]
    for name, bad in cases:
        expect_raises(AR.Stage2ActionError, lambda b=bad: AR.SourceLDA(b))
    ok("SourceLDA fails closed on missing μ/Σ/π, singular Σ, wrong class order, non-positive prior, bad shape, NaN")


def test_action_records_label_free_seven_features():
    import numpy as np
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=6, seed=2))
    Z = np.random.RandomState(0).randn(12, 6)
    batch = AR.build_subject_batch("PD/ds002778/sub-001", Z, lda, action_provider=AR.synthetic_action_provider)
    assert batch["batch_id"] == "PD/ds002778/sub-001"
    assert set(batch["features"]) == set(P.ACTIONS)             # the 3 non-identity actions
    for a in P.ACTIONS:
        assert set(batch["features"][a]) == set(P.FEATURES)     # exactly the 7 protocol features
    # identity via the production provider is the torch-free LDA path (== f_0)
    p_id, z_id = AR.production_action_provider("identity", lda, Z)
    assert np.allclose(p_id, lda.predict_proba(Z)) and np.allclose(z_id, Z)
    ok("action-record assembly is label-free with the 7 protocol features per action; production identity == f_0")


def main():
    print("ACAR v5 Stage-2B0 guard: f_0 LDA readout + label-free action records")
    test_lda_readout_matches_pinned_formula()
    test_lda_fail_closed()
    test_action_records_label_free_seven_features()
    print("ALL V5 STAGE2B0-F0-LDA-ACTION-RECORDS GUARDS PASS")


if __name__ == "__main__":
    main()
