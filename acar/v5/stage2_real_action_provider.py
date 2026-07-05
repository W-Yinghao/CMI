"""ACAR V5 Stage-2 REAL action-provider seam (validated). identity is the source-state LDA f_0.

Stage-2B2 amendment: `matched_coral` is routed through the V5-local `stage2_stable_coral.stable_matched_coral_v1` (bounded,
deterministic, rank-aware — NOT `cmi.eval.pmct_predict_serialized`), because the frozen CORAL was numerically ill-conditioned on
the 256-D / 32-window substrate and produced non-finite p_a (see the blocker note). `spdim` and `t3a` still route through the
FROZEN `acar.actions.apply_action` via the v5→old source-state adapter (torch/cmi.eval lazy — only spdim needs torch). Every
output is validated (shape / finite / row-sum / class-order / geometry). The seam NEVER reads labels (only source_state + z).
`validated_real_action` keeps the FROZEN path (for spdim/t3a and old-vs-new comparison).
"""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_action_provider_validation as V
from acar.v5 import stage2_stable_coral as SC


def validated_real_action(name, source_lda, Z):
    """Run one action through the FROZEN acar.actions seam (identity=LDA; matched_coral/spdim/t3a=acar.actions) and validate its
    output contract. Kept for spdim/t3a and for the old-vs-new (unstable-CORAL) comparison. Returns (p_a, z_post)."""
    import numpy as np
    Z = np.asarray(Z, float)
    pa, z_post = AR.production_action_provider(name, source_lda, Z)
    V.validate_action_output(name, pa, z_post, Z.shape[0])
    return np.asarray(pa, float), (None if z_post is None else np.asarray(z_post, float))


def real_action_provider(name, source_lda, Z):
    """The V5 Stage-2 (Stage-2B2-amended) action provider: identity=LDA f_0; matched_coral=stable_matched_coral_v1 (bounded,
    deterministic — NOT pmct); spdim/t3a=frozen acar.actions. Every output validated. Same signature as the synthetic provider;
    NEVER reads a label."""
    import numpy as np
    Z = np.asarray(Z, float)
    if name == "matched_coral":
        pa, z_post = SC.stable_matched_coral_v1(source_lda, Z)                 # V5-local bounded CORAL (no pmct)
        V.validate_action_output(name, pa, z_post, Z.shape[0])
        return np.asarray(pa, float), np.asarray(z_post, float)
    return validated_real_action(name, source_lda, Z)                         # identity / spdim / t3a (frozen)


def probe_real_actions(source_lda, Z, actions=("identity",) + P.ACTIONS):
    """Run + validate the requested Stage-2 actions on Z via the AMENDED provider (stable matched_coral). First validates the
    source-state adapter, then each action's output contract AND the per-action feature-finiteness of the assembled batch. Returns
    {action: {"z_post_none": bool}}. Raises on any failure. NOTE: spdim requires torch — the caller chooses which actions to probe."""
    V.validate_source_state_adapter(source_lda)
    import numpy as np
    import acar.features as AF
    Z = np.asarray(Z, float)
    outputs, out = {}, {}
    for a in actions:
        pa, z_post = real_action_provider(a, source_lda, Z)
        outputs[a] = (pa, z_post)
        out[a] = {"z_post_none": z_post is None}
    if "identity" in outputs:                                                # per-action feature finiteness (only probed actions)
        p0, z0 = outputs["identity"]
        for a in P.ACTIONS:
            if a in outputs:
                pa, z_post = outputs[a]
                V.validate_feature_finiteness(a, AR._to_protocol_features(AF.paired_features(p0, pa, z0, z_post)))
    return out
