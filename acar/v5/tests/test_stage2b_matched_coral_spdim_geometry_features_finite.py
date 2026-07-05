"""Guard (Stage-2B1): matched_coral and spdim are geometric (z_post != None) so their Bures/post_sep must be FINITE. matched_coral
runs on both Pythons; spdim needs torch (validated on py3.13, skipped on py3.9). Synthetic."""
from __future__ import annotations
import numpy as np
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_action_provider_validation as V
from acar.v5 import stage2_real_action_provider as RAP
from acar.v5.tests._util import ok, has_torch, stage2b_synthetic_source_state


def test_geometric_actions_have_finite_geometry():
    import acar.features as AF
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=6, seed=9))
    Z = np.random.RandomState(6).randn(24, 6)
    p0, z0 = RAP.validated_real_action("identity", lda, Z)
    actions = ("matched_coral",) + (("spdim",) if has_torch() else ())
    for a in actions:
        pa, z_post = RAP.validated_real_action(a, lda, Z)
        assert z_post is not None and np.isfinite(z_post).all()   # geometric action → real z_post
        feats = AR._to_protocol_features(AF.paired_features(p0, pa, z0, z_post))
        assert np.isfinite(feats["Bures"]) and np.isfinite(feats["post_sep"])
        assert V.validate_feature_finiteness(a, feats) is True
    if not has_torch():
        print("  [skip:no-torch] spdim geometry-finiteness check (validated on py3.13)")
    ok(f"geometric actions {actions} → finite Bures/post_sep")


def main():
    print("ACAR v5 Stage-2B1 guard: matched_coral/spdim geometry features finite")
    test_geometric_actions_have_finite_geometry()
    print("ALL V5 STAGE2B1-GEOMETRY-FINITE GUARDS PASS")


if __name__ == "__main__":
    main()
