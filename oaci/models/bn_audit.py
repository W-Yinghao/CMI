"""BatchNorm buffer audit: every Stage-2 checkpoint must keep the ERM running stats (Stage-2 freezes
BatchNorm at the ERM running stats; only the affine params train). The audit hashes just the BN buffers
(running_mean / running_var / num_batches_tracked) and checks each checkpoint against the shared ERM.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..train.checkpoint import state_hash

_BN_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


@dataclass(frozen=True)
class BNBufferAudit:
    level: int
    method: str
    checkpoint_hash: str
    erm_bn_hash: str
    checkpoint_bn_hash: str
    equal_to_erm: bool


def bn_buffer_state(state) -> dict:
    return {k: v for k, v in state.items() if k.rsplit(".", 1)[-1] in _BN_SUFFIXES}


def bn_buffer_hash(state) -> str:
    bn = bn_buffer_state(state)
    if not bn:
        raise ValueError("no BatchNorm buffers found in the checkpoint state")
    return state_hash(bn)


def audit_level_bn_buffers(level: int, level_run_result) -> tuple:
    """Every Stage-2 trajectory + selected checkpoint must share the ERM BN buffers."""
    erm_bn = bn_buffer_hash(level_run_result.erm_stage.checkpoint.model_state)
    out = []
    for name, m in level_run_result.method_items:
        seen = {}
        for c in list(m.train_result.trajectory) + [m.selection]:
            if c.model_hash in seen:
                continue
            seen[c.model_hash] = True
            ck_bn = bn_buffer_hash(c.model_state)
            out.append(BNBufferAudit(level=int(level), method=name, checkpoint_hash=c.model_hash,
                                     erm_bn_hash=erm_bn, checkpoint_bn_hash=ck_bn, equal_to_erm=ck_bn == erm_bn))
    return tuple(out)
