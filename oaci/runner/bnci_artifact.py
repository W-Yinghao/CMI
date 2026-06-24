"""BNCI artifact closed loop (B2a): one in-memory two-level run -> atomic write -> deep verify ->
read summary -> item-by-item compare to memory. Reuses the SAME write/verify/compare path as the fake
closed loop; never reloads raw data, rebuilds the FoldScope, or mutates the FoldData.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..artifacts.summary import compare_artifact_summary_to_memory, read_completed_artifact
from ..artifacts.verify import verify_artifact_tree
from ..artifacts.writer import write_artifact_tree_atomic
from .bnci import DEFAULT_METHOD_ORDER, run_bnci_two_level_in_memory
from .fake_artifact import build_fold_artifact_context


@dataclass(frozen=True)
class BNCIArtifactResult:
    bnci_fold: object
    fold_result: object
    context: object
    write_result: object
    verification: object
    loaded_summary: object
    comparison_hash: str


def run_bnci_artifact_once(bnci_fold, output_root, *, model_seed, method_order=DEFAULT_METHOD_ORDER,
                           repo_root, device, git_evidence=None) -> BNCIArtifactResult:
    bnci_fold.fold_data.assert_integrity()
    fr = run_bnci_two_level_in_memory(bnci_fold, model_seed=model_seed, method_order=method_order, device=device)
    bnci_fold.fold_data.assert_integrity()
    context = build_fold_artifact_context(bnci_fold, fr, repo_root=repo_root, git_evidence=git_evidence)
    write = write_artifact_tree_atomic(fr, context, output_root)
    rep = verify_artifact_tree(write.artifact_dir, deep=True)
    if not rep.ok:
        raise RuntimeError(f"deep verification failed: {rep.errors[:3]}")
    summary = read_completed_artifact(write.artifact_dir, deep_verify=True)
    cmp = compare_artifact_summary_to_memory(summary, fr, context,
                                             artifact_scientific_hash=write.artifact_scientific_hash)
    if not cmp.ok:
        raise RuntimeError(f"on-disk summary disagrees with memory: {cmp.mismatches[:5]}")
    return BNCIArtifactResult(bnci_fold=bnci_fold, fold_result=fr, context=context, write_result=write,
                              verification=rep, loaded_summary=summary, comparison_hash=cmp.comparison_hash)
