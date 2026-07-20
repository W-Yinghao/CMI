"""Confirmatory one-fold driver (pipeline validation, NOT confirmatory efficacy).

Materialize a full-budget `pilot` manifest for ONE held-out target, load the real BNCI2014_001 fold ONCE,
and run the four-method two-level pipeline for each requested model seed through the full
write -> deep-verify -> read -> compare closed loop. The data is loaded once and reused across seeds (only
the model seed varies). The materialized manifest must be written OUTSIDE the repo scientific tree (the
artifact writer refuses a dirty `oaci/` tree and an in-repo destination).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from ..runner.bnci import DEFAULT_METHOD_ORDER
from ..runner.bnci_artifact import run_bnci_artifact_once
from ..runner.bnci_data import build_bnci_real_fold
from .materialize import VALIDATION_BOOTSTRAP, materialize_pilot_manifest
from .schema import load_confirmatory

DATASET = "BNCI2014_001"


@dataclass(frozen=True)
class OneFoldSeedRun:
    model_seed: int
    artifact: object                                    # BNCIArtifactResult


@dataclass(frozen=True)
class OneFoldResult:
    protocol_path: str
    manifest_path: str
    manifest_hash: str
    dataset: str
    target_subject: int
    model_seeds: tuple
    bootstrap_mode: str                                 # "full_budget" | "pipeline_validation_reduced"
    fold: object                                        # BNCIRealFold
    seed_runs: tuple                                    # (OneFoldSeedRun, ...)


def run_confirmatory_onefold(protocol_path, *, datalake_root, repo_root, output_root, manifest_out,
                             dataset_name=DATASET, target_subject=1, model_seeds=(0, 1, 2),
                             device="cuda:0", git_evidence=None, bootstrap_override=None) -> OneFoldResult:
    if os.path.abspath(manifest_out).startswith(os.path.abspath(repo_root) + os.sep):
        raise ValueError("manifest_out must be OUTSIDE the repo (it is run-output provenance, not source)")
    proto = load_confirmatory(protocol_path)
    manifest_path, manifest = materialize_pilot_manifest(
        proto, dataset_name, target_subject=int(target_subject), out_path=manifest_out,
        model_seeds=[int(s) for s in model_seeds], bootstrap_override=bootstrap_override)
    fold = build_bnci_real_fold(manifest_path, datalake_root)
    fold.fold_data.assert_integrity()

    runs = []
    for s in model_seeds:
        art = run_bnci_artifact_once(fold, os.path.join(output_root, f"seed-{int(s)}"), model_seed=int(s),
                                     method_order=DEFAULT_METHOD_ORDER, repo_root=repo_root, device=device,
                                     git_evidence=git_evidence)
        runs.append(OneFoldSeedRun(model_seed=int(s), artifact=art))
    return OneFoldResult(protocol_path=str(protocol_path), manifest_path=manifest_path,
                         manifest_hash=manifest.freeze()["sha256"], dataset=dataset_name,
                         target_subject=int(target_subject), model_seeds=tuple(int(s) for s in model_seeds),
                         bootstrap_mode=("pipeline_validation_reduced" if bootstrap_override else "full_budget"),
                         fold=fold, seed_runs=tuple(runs))
