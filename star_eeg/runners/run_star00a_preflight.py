#!/usr/bin/env python
"""Run STAR_00A read-only inventory and synthetic red-team preflight.

This entry point cannot launch real training and never loads FACED/TUEG arrays.
It hashes and strictly reloads declared checkpoints, builds toy manifests, and
writes only the approved small JSON/report artifacts.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Mapping

from star_eeg.config import DEPENDENCY_COMMIT, PROJECT_NAME, STAR01, STAR_BRANCH
from star_eeg.data.anchor_manifest import build_anchor_manifest, synthetic_preview_records
from star_eeg.data.checkpoint_registry import build_checkpoint_inventory
from star_eeg.data.faced_split_contract import canonical_hash, contract_payload
from star_eeg.data.shuffled_label_manifest import build_shuffled_manifest
from star_eeg.objectives.alternating_schedule import build_compute_match_contract
from star_eeg.objectives.task_anchor import synthetic_training_step_smoke
from star_eeg.red_team.checkpoint_immutability import evaluate_checkpoint_immutability
from star_eeg.red_team.compute_match import evaluate_compute_match
from star_eeg.red_team.dependency_boundary import evaluate_dependency_boundary
from star_eeg.red_team.no_forbidden_method_guard import evaluate_no_forbidden_method_guard
from star_eeg.red_team.shuffled_manifest_freeze import evaluate_shuffled_manifest
from star_eeg.red_team.target_label_quarantine import inspect_training_signatures


ARTIFACT_NAMES = (
    "run_manifest.json",
    "dependency_manifest.json",
    "checkpoint_inventory.json",
    "faced_split_contract.json",
    "anchor_manifest_preview.json",
    "shuffled_manifest_preview.json",
    "compute_match_contract.json",
    "target_label_quarantine.json",
    "no_forbidden_method_guard.json",
    "preflight_summary.json",
)

AUTHORITY_FILES = (
    "docs/S2P_16_ROUTE_B_TRAINING_PROTOCOL.md",
    "docs/S2P_17_ROUTE_B_FINAL_RESULTS.md",
    "docs/S2P_18_ROUTE_B_CLAIM_LEDGER.md",
    "docs/S2P_19_NEXT_STAGE_SCIENTIFIC_EXPLORATION_PLAN.md",
    "docs/S2P_20_H2000_IMMUTABLE_CLOSURE_PROTOCOL.md",
    "s2p/scripts/run_frontier_cbramod.py",
    "s2p/scripts/tueg_subject_loader.py",
    "s2p/scripts/route_b_33ch_loader.py",
    "s2p/scripts/route_b_train_cbramod.py",
    "s2p/scripts/route_b_faced_downstream_audit.py",
    "s2p/scripts/route_b_faced_final_verification.py",
    "results/s2p_route_b_33ch_b1_faced/faced_checkpoint_manifest.csv",
    "results/s2p_route_b_h2000_immutable_closure/h2000_immutable_checkpoint_manifest.json",
)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _dependency_manifest(repo_root: Path, boundary: Mapping[str, object]) -> Dict[str, object]:
    source_files = [
        {"path": relative, "sha256": _sha256_file(repo_root / relative)}
        for relative in AUTHORITY_FILES
    ]
    core = {
        "schema_version": 1,
        "dependency_commit": DEPENDENCY_COMMIT,
        "star_branch": STAR_BRANCH,
        "authority_files": source_files,
        "boundary": dict(boundary),
        "s2p_behavior_modified": False,
        "stable_s2p_utilities_import_only": True,
    }
    return {**core, "dependency_manifest_hash": canonical_hash(core)}


def _run_manifest(smoke: Mapping[str, object]) -> Dict[str, object]:
    core = {
        "schema_version": 1,
        "project": PROJECT_NAME,
        "phase": "STAR_00A",
        "scope": "design_artifact_inventory_implementation_scaffold_red_team_preflight",
        "dependency_commit": DEPENDENCY_COMMIT,
        "star_branch": STAR_BRANCH,
        "python_executable": sys.executable,
        "artifacts": list(ARTIFACT_NAMES),
        "checkpoint_access": "read_only_hash_and_strict_reload",
        "synthetic_training_step_smoke": dict(smoke),
        "real_training_run": False,
        "real_eeg_loaded": False,
        "faced_target_metric_computed": False,
        "checkpoint_selected_using_target_labels": False,
        "slurm_launched": False,
        "star01_approved": False,
    }
    return {**core, "run_manifest_hash": canonical_hash(core)}


def assemble_static_preflight(repo_root: Path) -> Dict[str, object]:
    """Build every non-checkpoint object; useful for deterministic tests."""
    split = contract_payload()
    dataset_manifest = repo_root / "results/s2p_route_b_33ch_b1_faced/faced_dataset_manifest.csv"
    anchor = build_anchor_manifest(
        synthetic_preview_records(),
        dataset_manifest_hash=_sha256_file(dataset_manifest),
        preview_only=True,
    )
    shuffled = build_shuffled_manifest(anchor, STAR01.permutation_seed)
    compute = build_compute_match_contract()
    quarantine = inspect_training_signatures()
    no_forbidden = evaluate_no_forbidden_method_guard()
    smoke = synthetic_training_step_smoke()
    return {
        "faced_split_contract.json": split,
        "anchor_manifest_preview.json": anchor,
        "shuffled_manifest_preview.json": shuffled,
        "compute_match_contract.json": compute,
        "target_label_quarantine.json": quarantine,
        "no_forbidden_method_guard.json": no_forbidden,
        "run_manifest.json": _run_manifest(smoke),
    }


def _readout_markdown(summary: Mapping[str, object]) -> str:
    fields = (
        "status",
        "dependency_commit",
        "star_branch",
        "s2p_files_modified",
        "h2cmi_files_modified",
        "oaci_files_modified",
        "h200_start_checkpoints_ready",
        "checkpoint_inventory_hash",
        "faced_split_hash",
        "anchor_manifest_preview_hash",
        "shuffled_manifest_preview_hash",
        "compute_match_hash",
        "target_label_quarantine",
        "preflight_determinism",
        "real_training_run",
        "target_metrics_computed",
        "star01_approved",
    )
    lines = [
        "# STAR_00A Preflight Readout",
        "",
        "This is a STAR project design and red-team preflight.",
        "No real STAR EEG training was run.",
        "No FACED target metric was computed.",
        "No checkpoint was selected using target labels.",
        "No CMI, adversary, pruning, surgery, TTA, CSP-init, or safety gate was introduced.",
        "S2P Phase B remains independent and unchanged.",
        "STAR_01 scientific training is not approved by this commit.",
        "",
        "## Summary",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for field in fields:
        value = json.dumps(summary.get(field), sort_keys=True)
        lines.append(f"| `{field}` | `{value}` |")
    lines.extend([
        "",
        "## Scope and interpretation",
        "",
        "The preflight performed read-only repeated SHA256 checks and strict CBraMod reloads, exact split and API firewall checks, deterministic schedule/compute matching, fixed within-subject shuffled-manifest checks, active-method registry checks, and a tiny synthetic loss/gradient smoke. It did not load real EEG arrays.",
        "",
        "H500, H1000, H2000, released, and random remain frozen descriptive references. They cannot train STAR, select a variant, tune the anchor schedule, or support an equivalence/reproduction/superiority claim.",
        "",
        "The only primary future checkpoint is optimizer step 3750. Source-val diagnostics cannot replace it. STAR_01 remains unapproved regardless of artifact readiness.",
    ])
    return "\n".join(lines) + "\n"


def run_preflight(repo_root: Path, out_dir: Path) -> Dict[str, object]:
    static_first = assemble_static_preflight(repo_root)
    static_second = assemble_static_preflight(repo_root)
    static_deterministic = canonical_hash(static_first) == canonical_hash(static_second)

    boundary = evaluate_dependency_boundary(repo_root)
    dependency = _dependency_manifest(repo_root, boundary)
    inventory = build_checkpoint_inventory(repo_root)
    immutability = evaluate_checkpoint_immutability(inventory)
    compute_red_team = evaluate_compute_match(static_first["compute_match_contract.json"])
    shuffle_red_team = evaluate_shuffled_manifest(
        static_first["anchor_manifest_preview.json"],
        static_first["shuffled_manifest_preview.json"],
        STAR01.permutation_seed,
    )
    smoke = static_first["run_manifest.json"]["synthetic_training_step_smoke"]
    red_team = {
        "dependency_boundary": boundary["status"],
        "checkpoint_immutability": immutability["status"],
        "faced_split_firewall": "PASS",
        "compute_match": compute_red_team["status"],
        "shuffled_control": shuffle_red_team["status"],
        "target_label_quarantine": static_first["target_label_quarantine.json"]["status"],
        "no_forbidden_method_guard": static_first["no_forbidden_method_guard.json"]["status"],
        "synthetic_training_step_smoke": smoke["status"],
    }
    infrastructure_pass = all(value == "PASS" for value in red_team.values()) and static_deterministic
    h200_ready = bool(immutability["h200_start_checkpoints_ready"])
    status = "PASS" if infrastructure_pass and h200_ready else (
        "STAR_01_BLOCKED_ARTIFACT_SUPPLY" if infrastructure_pass else "FAIL"
    )
    summary_core = {
        "status": status,
        "dependency_commit": DEPENDENCY_COMMIT,
        "star_branch": STAR_BRANCH,
        "s2p_files_modified": boundary["s2p_files_modified"],
        "h2cmi_files_modified": boundary["h2cmi_files_modified"],
        "oaci_files_modified": boundary["oaci_files_modified"],
        "h200_start_checkpoints_ready": h200_ready,
        "checkpoint_inventory_hash": inventory["checkpoint_inventory_hash"],
        "faced_split_hash": static_first["faced_split_contract.json"]["faced_split_hash"],
        "anchor_manifest_preview_hash": static_first["anchor_manifest_preview.json"]["anchor_manifest_hash"],
        "shuffled_manifest_preview_hash": static_first["shuffled_manifest_preview.json"]["shuffled_manifest_hash"],
        "compute_match_hash": static_first["compute_match_contract.json"]["compute_match_hash"],
        "target_label_quarantine": static_first["target_label_quarantine.json"]["status"],
        "preflight_determinism": "PASS" if static_deterministic else "FAIL",
        "real_training_run": False,
        "target_metrics_computed": False,
        "star01_approved": False,
        "dependency_manifest_hash": dependency["dependency_manifest_hash"],
        "red_team": red_team,
        "checkpoint_immutability": immutability,
        "compute_match_red_team": compute_red_team,
        "shuffled_manifest_red_team": shuffle_red_team,
        "infrastructure_pass": infrastructure_pass,
        "star01_artifact_status": immutability["star01_artifact_status"],
    }
    summary = {**summary_core, "preflight_summary_hash": canonical_hash(summary_core)}

    artifacts = {
        **static_first,
        "dependency_manifest.json": dependency,
        "checkpoint_inventory.json": inventory,
        "preflight_summary.json": summary,
    }
    if set(artifacts) != set(ARTIFACT_NAMES):
        raise AssertionError(f"artifact set mismatch: {sorted(artifacts)}")
    for name in ARTIFACT_NAMES:
        _write_json(out_dir / name, artifacts[name])
    readout = repo_root / "star_eeg/reports/STAR_00A_PREFLIGHT_READOUT.md"
    readout.write_text(_readout_markdown(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--out-dir", default="results/star/star00a_preflight")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    summary = run_preflight(repo_root, out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
