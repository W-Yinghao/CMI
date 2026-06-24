"""Fake two-level artifact closed loop (A2b-2b-ii).

Run the in-memory fixture, collect live Git evidence, atomically write the artifact, deep-verify it,
read back the logical summary and compare it item-by-item to the in-memory result. Returns only after
exact equality. The artifact destination must be OUTSIDE the repository scientific tree.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..artifacts.summary import compare_artifact_summary_to_memory, read_completed_artifact
from ..artifacts.verify import verify_artifact_tree
from ..artifacts.writer import (artifact_scientific_hash, collect_git_evidence, context_from_git_evidence,
                                write_artifact_tree_atomic)
from ..protocol.manifest_v2 import manifest_logical_payload
from .config import execution_config_logical_payload, model_spec_logical_payload
from .fake import DEFAULT_METHOD_ORDER, run_fake_two_level_in_memory
from .fake_data import build_fake_fold


@dataclass(frozen=True)
class FakeRunArtifactResult:
    fake_fold: object
    fold_result: object
    context: object
    write_result: object
    verification: object
    loaded_summary: object
    comparison_hash: str


def build_fake_artifact_context(fake_fold, fold_result, *, repo_root, git_evidence=None):
    """Build the ArtifactContext. With no injected evidence it collects LIVE git and refuses a dirty
    scientific tree (the demo path); tests may inject a synthetic clean GitEvidence."""
    return build_fold_artifact_context(fake_fold, fold_result, repo_root=repo_root, git_evidence=git_evidence)


def build_fold_artifact_context(fold_object, fold_result, *, repo_root, git_evidence=None):
    """Shared context builder for ANY fold object exposing .manifest / .manifest_hash /
    .execution_config / .model_spec (FakeFold or BNCIRealFold). Collects live git unless an evidence is
    injected, refuses a dirty tree, and lists the per-level config/spec payloads."""
    git = git_evidence if git_evidence is not None else collect_git_evidence(repo_root)
    if not git.clean or git.status_entries:
        raise ValueError("refusing to build an artifact context from a dirty scientific tree")
    mpay = manifest_logical_payload(fold_object.manifest)
    levels = sorted(int(l) for l in dict(fold_result.level_items))
    if levels != [0, 1]:
        raise ValueError("the artifact context expects exactly levels {0, 1}")
    ecp = tuple((lvl, execution_config_logical_payload(fold_object.execution_config)) for lvl in levels)
    msp = tuple((lvl, model_spec_logical_payload(fold_object.model_spec)) for lvl in levels)
    return context_from_git_evidence(mpay, fold_object.manifest_hash, ecp, msp, git, repo_root)


def run_fake_two_level(manifest_path, output_root, *, model_seed, method_order=DEFAULT_METHOD_ORDER,
                       repo_root, git_evidence=None) -> FakeRunArtifactResult:
    fake = build_fake_fold(manifest_path)
    fr = run_fake_two_level_in_memory(fake, model_seed=model_seed, method_order=method_order)
    context = build_fake_artifact_context(fake, fr, repo_root=repo_root, git_evidence=git_evidence)
    write = write_artifact_tree_atomic(fr, context, output_root)
    rep = verify_artifact_tree(write.artifact_dir, deep=True)
    if not rep.ok:
        raise RuntimeError(f"deep verification failed: {rep.errors[:3]}")
    summary = read_completed_artifact(write.artifact_dir, deep_verify=True)
    cmp = compare_artifact_summary_to_memory(summary, fr, context,
                                             artifact_scientific_hash=write.artifact_scientific_hash)
    if not cmp.ok:
        raise RuntimeError(f"on-disk summary disagrees with memory: {cmp.mismatches[:5]}")
    return FakeRunArtifactResult(fake_fold=fake, fold_result=fr, context=context, write_result=write,
                                 verification=rep, loaded_summary=summary, comparison_hash=cmp.comparison_hash)
