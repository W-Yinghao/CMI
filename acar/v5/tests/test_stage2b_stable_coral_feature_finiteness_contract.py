"""Guard (Stage-2B2): the paired features from stable matched_coral satisfy the finiteness contract — d_entropy/d_margin/flip_rate/
JS/n_eff finite, and (since matched_coral is geometric, z_post != None) Bures/post_sep finite. Torch-free. Synthetic."""
from __future__ import annotations
import numpy as np
from acar.v5 import protocol as P
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_action_provider_validation as V
from acar.v5 import stage2_real_action_provider as RAP
from acar.v5.tests._util import ok, stage2b_synthetic_source_state, stage2b_rank_deficient_batch


def test_matched_coral_feature_finiteness():
    import acar.features as AF
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=256, seed=3))
    for rank, seed in ((2, 0), (8, 4), (31, 8)):
        Z = stage2b_rank_deficient_batch(n=32, D=256, rank=rank, seed=seed)
        p0, z0 = RAP.real_action_provider("identity", lda, Z)
        pa, zp = RAP.real_action_provider("matched_coral", lda, Z)
        assert zp is not None                                                 # geometric action
        feats = AR._to_protocol_features(AF.paired_features(p0, pa, z0, zp))
        for f in ("d_entropy", "d_margin", "flip_rate", "JS", "n_eff", "Bures", "post_sep"):
            assert np.isfinite(feats[f]), f"{f} not finite: {feats[f]}"
        assert V.validate_feature_finiteness("matched_coral", feats) is True   # matched_coral must have finite geometry
    ok("stable matched_coral paired features all finite (incl Bures/post_sep) on rank-deficient batches")


def main():
    print("ACAR v5 Stage-2B2 guard: stable CORAL feature finiteness")
    test_matched_coral_feature_finiteness()
    print("ALL V5 STAGE2B2-STABLE-CORAL-FEATURE-FINITENESS GUARDS PASS")


if __name__ == "__main__":
    main()
