"""ACAR V5 Stage-1B runtime-lock CAPTURE helper. `build_lock` is PURE (no torch/GPU/IO) and produces a lock dict cross-bindable to
an authorization. `capture_runtime_lock` does the heavy env probe (torch/device) and is LAZY — it is NOT called at import and NOT
exercised by the synthetic tests; a real capture happens only at an authorized Stage-1B run on the training node.
"""
from __future__ import annotations
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1_runtime_lock as RL


def build_lock(*, protocol_tag_target_sha, implementation_base_sha, run_id, device_kind, status=RL.VERIFIED_STATUS):
    """Pure constructor for a Stage-1B runtime lock (no probe). Use for tests / dry specs."""
    return {"stage": "Stage-1B", "protocol_tag": SA.PROTOCOL_TAG,
            "protocol_tag_target_sha": protocol_tag_target_sha, "implementation_base_sha": implementation_base_sha,
            "run_id": run_id, "device_kind": device_kind, "status": status}


def capture_runtime_lock(*, protocol_tag_target_sha, implementation_base_sha, run_id, require_gpu=True):  # pragma: no cover
    """Heavy env probe (LAZY torch import). Returns a lock with status CAPTURED_AND_VERIFIED iff the training stack imports and
    (require_gpu ⇒ a CUDA device is available). Never called at import or by the synthetic tests; runs only at a real Stage-1B."""
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except Exception:  # noqa
            pass
        cuda = bool(torch.cuda.is_available())
        device_kind = "cuda" if cuda else "cpu"
        ok = (cuda or not require_gpu)
        status = RL.VERIFIED_STATUS if ok else "CAPTURE_FAILED"
    except Exception:  # noqa
        device_kind, status = "cpu", "CAPTURE_FAILED"
    return build_lock(protocol_tag_target_sha=protocol_tag_target_sha, implementation_base_sha=implementation_base_sha,
                      run_id=run_id, device_kind=device_kind, status=status)
