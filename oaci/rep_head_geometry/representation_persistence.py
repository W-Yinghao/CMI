"""C29 Stage-2 scaffold — target pre-classifier z re-persistence. Because the head is LINEAR (logit = W·z + b),
the projection W·z = (logit − b) already captures ALL offset-relevant representation information; any z-component
orthogonal to W's row space cannot change the logits and is offset-IRRELEVANT. So a full 800-d z re-persistence
is NOT needed for the offset-origin question -- it would add only offset-orthogonal descriptive geometry.
Availability-gated: documents the P0 replay-identity gate should the full-z descriptive geometry ever be wanted;
default status is NOT_NEEDED_OFFSET_ORTHOGONAL."""
from __future__ import annotations

import os

from . import schema

C29_Z_SIDECAR = "/projects/EEG-foundation-model/yinghao/oaci-c29-target-z.json"

_P0_GATES = ("G1 checkpoint_hash matches manifest", "G2 forwarded logits reproduce C26 byte-identically or within tol",
             "G3 z shape / sample order stable", "G4 repeat-forward deterministic",
             "G5 no target labels loaded in representation construction", "G6 no selected-checkpoint artifact")


def availability(z_sidecar=None) -> dict:
    ready = os.path.exists(z_sidecar or C29_Z_SIDECAR)
    return {"per_candidate_z_ready": ready,
            "status": "computed" if ready else "NOT_NEEDED_OFFSET_ORTHOGONAL",
            "reason": ("full 800-d target z is persisted" if ready else
                       "W·z=(logit−b) captures ALL offset-relevant representation info; z-components orthogonal to "
                       "W are offset-irrelevant, so a full-z re-persistence would add only offset-orthogonal "
                       "descriptive geometry (Stage-2 optional, P0-gated if ever wanted)"),
            "p0_gates": list(_P0_GATES)}
