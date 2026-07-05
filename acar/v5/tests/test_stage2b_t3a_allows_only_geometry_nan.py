"""Guard (Stage-2B1): t3a is probability-only (z_post=None) so its Bures/post_sep may be NaN, but its 5 routing features stay
finite; and NaN Bures/post_sep for matched_coral/spdim is REJECTED. Torch-free (t3a real transform + numpy). Synthetic."""
from __future__ import annotations
import numpy as np
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_action_provider_validation as V
from acar.v5 import stage2_real_action_provider as RAP
from acar.v5.tests._util import expect_raises, ok, stage2b_synthetic_source_state


def test_t3a_geometry_nan_allowed_routing_finite():
    import acar.features as AF
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=6, seed=7))
    Z = np.random.RandomState(5).randn(20, 6)
    p0, z0 = RAP.validated_real_action("identity", lda, Z)
    pa, z_post = RAP.validated_real_action("t3a", lda, Z)
    assert z_post is None                                          # t3a is probability-only
    feats = AR._to_protocol_features(AF.paired_features(p0, pa, z0, z_post))
    assert not np.isfinite(feats["Bures"]) and not np.isfinite(feats["post_sep"])   # geometry NaN (z_post=None)
    for f in ("d_entropy", "d_margin", "flip_rate", "JS", "n_eff"):
        assert np.isfinite(feats[f])
    assert V.validate_feature_finiteness("t3a", feats) is True     # NaN geometry allowed for t3a
    ok("t3a z_post=None → Bures/post_sep NaN allowed; the 5 routing features remain finite")


def test_nan_geometry_rejected_for_matched_coral_spdim():
    feats = {"d_entropy": 0.0, "d_margin": 0.1, "flip_rate": 0.0, "JS": 0.0, "n_eff": 1.0,
             "Bures": float("nan"), "post_sep": 0.5}
    expect_raises(V.Stage2ActionValidationError, lambda: V.validate_feature_finiteness("matched_coral", feats))
    expect_raises(V.Stage2ActionValidationError, lambda: V.validate_feature_finiteness("spdim", feats))
    # a non-finite ROUTING feature is rejected for every action, including t3a
    bad = {"d_entropy": 0.0, "d_margin": float("nan"), "flip_rate": 0.0, "JS": 0.0, "n_eff": 1.0,
           "Bures": float("nan"), "post_sep": float("nan")}
    expect_raises(V.Stage2ActionValidationError, lambda: V.validate_feature_finiteness("t3a", bad))
    ok("NaN Bures/post_sep rejected for matched_coral/spdim; NaN in any routing feature rejected for all actions")


def main():
    print("ACAR v5 Stage-2B1 guard: t3a allows only geometry NaN")
    test_t3a_geometry_nan_allowed_routing_finite()
    test_nan_geometry_rejected_for_matched_coral_spdim()
    print("ALL V5 STAGE2B1-T3A-GEOMETRY-NAN GUARDS PASS")


if __name__ == "__main__":
    main()
