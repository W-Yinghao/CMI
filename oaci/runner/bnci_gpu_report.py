"""GPU-smoke report serialization (B2a; extended in B2b with the native thread-pool + thread-env
evidence). The CudaRuntimeEvidence is reported and hashed SEPARATELY -- it is transport/runtime
provenance and never enters the scientific identity (the artifact_scientific_hash /
artifact_pure_science_hash / fold_result_hash are independent of it).
"""
from __future__ import annotations

import dataclasses

from ..artifacts.canonical_json import canonical_json_hash
from ..runtime.cuda import CudaRuntimeEvidence


def runtime_evidence_report(ev: CudaRuntimeEvidence) -> dict:
    """JSON-safe view of the runtime evidence (nested NativeThreadPoolRecords -> dicts via asdict)."""
    d = dataclasses.asdict(ev)
    d["device_capability"] = list(ev.device_capability)
    return d


def gpu_smoke_report_hash(report: dict) -> str:
    """Canonical hash of the GPU-smoke report -- runtime/transport evidence, distinct from the
    artifact scientific hashes (used as BNCIGPUSmokeResult.report_hash)."""
    return canonical_json_hash(report)
