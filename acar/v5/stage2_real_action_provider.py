"""ACAR V5 Stage-2 REAL action-provider seam (validated). identity is the source-state LDA f_0; matched_coral / spdim / t3a route
through the FROZEN acar.actions.apply_action via the v5→old source-state adapter (torch/cmi.eval loaded LAZILY inside
acar.actions — only spdim needs torch). Every output is validated (shape / finite / row-sum / class-order / geometry). The seam
NEVER reads labels (it consumes only source_state + the z batch). Used only in a real (authorized) Stage-2B run; Stage-2B1
validates it on SYNTHETIC embeddings + synthetic source_state.
"""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_action_provider_validation as V


def validated_real_action(name, source_lda, Z):
    """Run one real action through the frozen acar.actions seam and validate its output contract. Returns (p_a, z_post)."""
    import numpy as np
    Z = np.asarray(Z, float)
    pa, z_post = AR.production_action_provider(name, source_lda, Z)          # identity=LDA; others=acar.actions (lazy torch)
    V.validate_action_output(name, pa, z_post, Z.shape[0])
    return np.asarray(pa, float), (None if z_post is None else np.asarray(z_post, float))


def real_action_provider(name, source_lda, Z):
    """A drop-in validated real action provider (same signature as the synthetic provider) for a real Stage-2B run."""
    return validated_real_action(name, source_lda, Z)


def probe_real_actions(source_lda, Z, actions=("identity",) + P.ACTIONS):
    """Run + validate the requested real actions on synthetic Z. First validates the source-state adapter, then each action's
    output contract AND the per-action feature-finiteness of the assembled batch. Returns {action: {"z_post_none": bool}}. Raises
    Stage2ActionValidationError on any failure. NOTE: spdim requires torch — the caller chooses which actions to probe."""
    V.validate_source_state_adapter(source_lda)
    import numpy as np
    import acar.features as AF
    Z = np.asarray(Z, float)
    outputs, out = {}, {}
    for a in actions:
        pa, z_post = validated_real_action(a, source_lda, Z)
        outputs[a] = (pa, z_post)
        out[a] = {"z_post_none": z_post is None}
    if "identity" in outputs:                                                # per-action feature finiteness (only probed actions)
        p0, z0 = outputs["identity"]
        for a in P.ACTIONS:
            if a in outputs:
                pa, z_post = outputs[a]
                V.validate_feature_finiteness(a, AR._to_protocol_features(AF.paired_features(p0, pa, z0, z_post)))
    return out
