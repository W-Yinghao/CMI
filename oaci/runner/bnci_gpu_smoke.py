"""Two-order BNCI GPU smoke (B2b): configure determinism BEFORE CUDA, load the real fold ONCE, run the
canonical and reversed method orders, write/deep-verify/read/compare each, then bit-exactly compare the
two runs' scientific identities and audit every Stage-2 BN buffer against the ERM.
"""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass

import torch

from ..artifacts.canonical_json import canonical_json_hash
from ..models.bn_audit import audit_level_bn_buffers
from ..runtime.cuda import assert_cuda_runtime_unchanged, configure_cuda_determinism
from ..runtime.rng_state import assert_rng_unchanged, snapshot_rng_state
from .bnci_artifact import run_bnci_artifact_once
from .bnci_data import build_bnci_real_fold
from .bnci_gpu_compare import comparison_all_equal, compare_scientific_results
from .bnci_gpu_report import runtime_evidence_report

CANONICAL_ORDER = ("ERM", "OACI", "global_lpc", "uniform")
REVERSED_ORDER = ("uniform", "global_lpc", "OACI", "ERM")


@dataclass(frozen=True)
class BNCIGPUSmokeResult:
    runtime: object
    canonical: object
    reversed: object
    rng_before_hash: str
    rng_after_canonical_hash: str
    rng_after_reversed_hash: str
    bn_audits: tuple
    comparison: object
    report_hash: str


def run_bnci_gpu_smoke(manifest_path, artifact_root, *, datalake_root, repo_root, model_seed=0,
                       git_evidence=None) -> BNCIGPUSmokeResult:
    device, runtime = configure_cuda_determinism()
    fold = build_bnci_real_fold(manifest_path, datalake_root)
    fold.fold_data.assert_integrity()

    rng_before = snapshot_rng_state(device)
    torch.cuda.synchronize(device)
    canonical = run_bnci_artifact_once(fold, os.path.join(artifact_root, "canonical"), model_seed=model_seed,
                                       method_order=CANONICAL_ORDER, repo_root=repo_root, device=device,
                                       git_evidence=git_evidence)
    torch.cuda.synchronize(device)
    rng_after_c = snapshot_rng_state(device)
    assert_rng_unchanged(rng_before, rng_after_c, "canonical run")
    assert_cuda_runtime_unchanged(runtime)

    gc.collect(); torch.cuda.synchronize(device); torch.cuda.empty_cache()

    reversed_ = run_bnci_artifact_once(fold, os.path.join(artifact_root, "reversed"), model_seed=model_seed,
                                       method_order=REVERSED_ORDER, repo_root=repo_root, device=device,
                                       git_evidence=git_evidence)
    torch.cuda.synchronize(device)
    rng_after_r = snapshot_rng_state(device)
    assert_rng_unchanged(rng_before, rng_after_r, "reversed run")
    assert_cuda_runtime_unchanged(runtime)

    cmp = compare_scientific_results(canonical, reversed_)
    if not comparison_all_equal(cmp):
        raise RuntimeError(f"method order changed a scientific identity: first mismatch at "
                           f"{cmp.first_mismatch.path if cmp.first_mismatch else '?'}")
    bn_audits = tuple(a for lvl, lr in canonical.fold_result.level_items for a in audit_level_bn_buffers(lvl, lr))
    if not all(a.equal_to_erm for a in bn_audits):
        raise RuntimeError("a Stage-2 checkpoint BN buffer differs from the ERM running stats")

    report = {"runtime": runtime_evidence_report(runtime), "comparison_hash": cmp.comparison_hash,
              "rng_before": rng_before.snapshot_hash, "rng_after_canonical": rng_after_c.snapshot_hash,
              "rng_after_reversed": rng_after_r.snapshot_hash,
              "artifact_scientific_hash": canonical.write_result.artifact_scientific_hash,
              "fold_result_hash": canonical.fold_result.fold_result_hash,
              "bn_all_equal_to_erm": all(a.equal_to_erm for a in bn_audits)}
    return BNCIGPUSmokeResult(runtime=runtime, canonical=canonical, reversed=reversed_,
                              rng_before_hash=rng_before.snapshot_hash, rng_after_canonical_hash=rng_after_c.snapshot_hash,
                              rng_after_reversed_hash=rng_after_r.snapshot_hash, bn_audits=bn_audits, comparison=cmp,
                              report_hash=canonical_json_hash(report))
