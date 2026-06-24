"""GPU-smoke report serialization (B2a). The CudaRuntimeEvidence is reported separately and does NOT
enter the artifact scientific hash (fold_result_hash / context_hash / artifact_scientific_hash are
unchanged); the report computes its own hash.
"""
from __future__ import annotations

import dataclasses

from ..artifacts.canonical_json import canonical_json_hash
from ..runtime.cuda import CudaRuntimeEvidence


def runtime_evidence_report(ev: CudaRuntimeEvidence) -> dict:
    d = dataclasses.asdict(ev)
    d["device_capability"] = list(ev.device_capability)
    return d


def gpu_smoke_report_hash(report: dict) -> str:
    """Hash of the GPU-smoke report -- distinct from the artifact scientific hash."""
    return canonical_json_hash(report)
