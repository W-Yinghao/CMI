"""ACAR v4 — regen runtime env-lock CAPTURE tool (B1a; environment introspection ONLY, NO training).

Run on the intended training node/device. It imports torch + import-PROBES the training stack (braindecode EEGNet, cmi
backbone) and records versions / device / determinism / threads into a regen env-lock JSON. It NEVER trains, NEVER reads
DEV/held-out raw, NEVER fits a model or generates embeddings — it only imports + reads versions/device. The lock's status:
  CAPTURED_AND_VERIFIED  iff the training stack imports AND (require_gpu ⇒ a CUDA device is available);
  CAPTURE_FAILED         otherwise (honest failure record; capture_note carries the reason).
The heavy imports happen ONLY inside capture(); this module is base-importable (so the pure schema is testable without torch).

Usage (in the training env, e.g. eeg2025):
    python -m acar.v4.capture_regen_envlock --output notes/ACAR_V4_REGEN_ENV_LOCK.json \
        --protocol-commit <40-hex> [--device-kind cuda|cpu] [--allow-cpu]
"""
from __future__ import annotations
import argparse
import json
import os

from acar.v4 import regen_envlock as EL
from acar.v4.regen_substrate import canonical_pipeline_config_sha256


def _probe():
    """Import-probe the training stack (NO training/data). Returns (info: dict, stack_ok: bool, note: str)."""
    notes = []
    info = {"python_version": "", "torch_version": "", "braindecode_version": "", "numpy_version": "",
            "scipy_version": "", "sklearn_version": "", "cuda_version": "", "cudnn_version": "",
            "device_kind": "cpu", "device_name": "", "driver_version": "",
            "torch_intraop_threads": 1, "torch_interop_threads": 1, "omp_num_threads": 1, "threadpool_backends": []}
    import sys
    info["python_version"] = sys.version.split()[0]
    try:
        import torch
    except Exception as e:
        return info, False, f"torch import FAILED: {e!r}"
    info["torch_version"] = torch.__version__
    info["cuda_version"] = torch.version.cuda or ""
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:                                    # noqa
        notes.append(f"use_deterministic_algorithms warn: {e}")
    cuda_ok = bool(torch.cuda.is_available())
    if cuda_ok:
        info["device_kind"] = "cuda"
        try:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cudnn_version"] = str(getattr(torch.backends.cudnn, "version", lambda: "")() or "")
            info["driver_version"] = ""                       # driver string is environment-specific; left for operator
        except Exception as e:                                # noqa
            notes.append(f"cuda introspection warn: {e}")
    else:
        info["device_name"] = "cpu"
        notes.append("no CUDA device available on this node (cuda.is_available()==False)")
    info["torch_intraop_threads"] = int(torch.get_num_threads())
    try:
        info["torch_interop_threads"] = int(torch.get_num_interop_threads())
    except Exception:                                         # noqa
        pass
    info["omp_num_threads"] = int(os.environ.get("OMP_NUM_THREADS", "1") or "1")
    for mod in ("numpy", "scipy", "sklearn", "braindecode"):
        try:
            info[mod + "_version" if mod != "sklearn" else "sklearn_version"] = __import__(mod).__version__
        except Exception as e:                                # noqa
            notes.append(f"{mod} import FAILED: {e!r}")
    eeg_ok = True
    try:
        from braindecode.models import EEGNetv4  # noqa: F401
    except Exception as e:
        eeg_ok = False; notes.append(f"braindecode EEGNetv4 import FAILED: {e!r}")
    bb_ok = True
    try:
        from cmi.models.backbones import build_backbone  # noqa: F401
    except Exception as e:
        bb_ok = False; notes.append(f"cmi build_backbone import FAILED: {e!r}")
    try:
        import threadpoolctl
        info["threadpool_backends"] = sorted({d.get("internal_api", "") for d in threadpoolctl.threadpool_info()})
    except Exception as e:                                    # noqa
        notes.append(f"threadpoolctl warn: {e}")
    stack_ok = eeg_ok and bb_ok
    return info, stack_ok, "; ".join(notes)


def capture(output_json, *, protocol_commit, require_gpu=True):
    info, stack_ok, note = _probe()
    captured = stack_ok and (info["device_kind"] == "cuda" or not require_gpu)
    lock = EL.schema_only_template(protocol_commit=protocol_commit,
                                   pipeline_config_sha256=canonical_pipeline_config_sha256(),
                                   device_kind=info["device_kind"])
    lock.update({k: info[k] for k in info})
    lock["status"] = "CAPTURED_AND_VERIFIED" if captured else "CAPTURE_FAILED"
    lock["capture_note"] = note if note else ("training stack import OK" if captured else "")
    EL.validate_regen_env_lock(lock)                          # schema-valid regardless of status
    tmp = output_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(lock, f, sort_keys=True, indent=2)
    os.replace(tmp, output_json)
    return lock


def main(argv=None):
    ap = argparse.ArgumentParser(description="ACAR v4 regen env-lock capture (env introspection only; NO training)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--protocol-commit", required=True)
    ap.add_argument("--allow-cpu", action="store_true", help="treat a CPU-only node as training-capable (default: GPU required)")
    args = ap.parse_args(argv)
    lock = capture(args.output, protocol_commit=args.protocol_commit, require_gpu=not args.allow_cpu)
    print(f"regen env-lock status={lock['status']} sha={EL.hash_regen_env_lock(lock)} note={lock['capture_note']!r}")
    return lock


if __name__ == "__main__":
    main()
