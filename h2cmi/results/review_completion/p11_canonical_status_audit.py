"""Generate the P11 canonical evidence index and stale-claim status gates."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "h2cmi" / "results" / "review_completion"

CANONICAL_HEAD_COMMIT = "cfb43d429263c3fbb69a35c086214fdca7d25301"
P11_THIS_COMMIT = "P11_THIS_COMMIT"
P7_COMMIT = "bc61ee11d21e023966fa9be637e960fdaf77a9c1"
P9_COMMIT = "8972de878a93e00a5b6cf6b8118bc32adc05eb48"
REVIEW_P0_COMMIT = "5bc9bf070976442dd2b66f721b2de53f817e65a5"
P7_SHA256 = "6d5106a78dad9ce852c8e01ca292ef5b4a37bbeaaaac810a177dccb8b6b9089c"
P9_SHA256 = "95b8f69556a140dc020415753c9694cf9ebdeed1abb0766dd24f523c491289c3"
MANIFEST_HASH = "231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e"

P10_IMMUTABLE_SHA256 = {
    "h2cmi/results/review_completion/FINAL_REPAIRED_W1_EVIDENCE_FREEZE.json":
        "f215884b6608a69e14b017af782e459ec00bf7cd7bcc4902f09b7a62ed21ffa4",
    "h2cmi/results/review_completion/w1_repaired_pipeline_comparability_audit.json":
        "6aca43e5c4f055c85c02e7180421d5d27f3e1302cef00fca4601363fa4f9c3b2",
    "h2cmi/results/review_completion/w1_repaired_cross_pipeline_results.csv":
        "59a251081003a6c95873447ad196581a19e2fbd9b4e5c920bafb9eae2ee6a6d2",
    "h2cmi/results/review_completion/w1_repaired_cross_pipeline_results.json":
        "6bb363f2bb099ca1ab68e22f0f86e253bb894c8ad95ba4ec2cb28effb65f9966",
    "h2cmi/results/review_completion/w1_repaired_cross_pipeline_harm.csv":
        "35b9616be6aae84a6d433dbda09f7d7a07dc3498d98e3cc5fe9788ac3e167e38",
}

INDEX_MD = OUT / "CANONICAL_EVIDENCE_INDEX.md"
INDEX_JSON = OUT / "CANONICAL_EVIDENCE_INDEX.json"
STALE_MD = OUT / "STALE_CLAIM_AUDIT.md"
STALE_JSON = OUT / "STALE_CLAIM_AUDIT.json"
STATUS_MD = OUT / "REVIEW_COMPLETION_CURRENT_STATUS.md"
STATUS_JSON = OUT / "REVIEW_COMPLETION_CURRENT_STATUS.json"
COMMAND_LOG = OUT / "COMMAND_LOG.md"
MANUSCRIPT_DIGEST = OUT / "MANUSCRIPT_NUMBERS_READY.md"

P11_MANAGED_PATHS = {
    "h2cmi/results/review_completion/BLOCKERS.md",
    "h2cmi/results/review_completion/CANONICAL_EVIDENCE_INDEX.json",
    "h2cmi/results/review_completion/CANONICAL_EVIDENCE_INDEX.md",
    "h2cmi/results/review_completion/COMMAND_LOG.md",
    "h2cmi/results/review_completion/MANUSCRIPT_NUMBERS_READY.md",
    "h2cmi/results/review_completion/REVIEW_COMPLETION_CURRENT_STATUS.json",
    "h2cmi/results/review_completion/REVIEW_COMPLETION_CURRENT_STATUS.md",
    "h2cmi/results/review_completion/REVIEW_COMPLETION_SUMMARY.md",
    "h2cmi/results/review_completion/STALE_CLAIM_AUDIT.json",
    "h2cmi/results/review_completion/STALE_CLAIM_AUDIT.md",
    "h2cmi/results/review_completion/VALIDATION_STATUS.md",
    "h2cmi/results/review_completion/artifact_inventory.md",
    "h2cmi/results/review_completion/baseline_inventory_and_blockers.md",
    "h2cmi/results/review_completion/encoder_backbone_details.json",
    "h2cmi/results/review_completion/encoder_backbone_details.md",
    "h2cmi/results/review_completion/p11_canonical_status_audit.py",
    "h2cmi/results/review_completion/spdim_external_repo_assessment.md",
    "h2cmi/results/review_completion/spdim_official_baseline_blocker.md",
    "h2cmi/results/review_completion/spdim_protocol_mapping.md",
    "h2cmi/results/review_completion/w1_split_metric_impact_verdict.json",
    "h2cmi/results/review_completion/w1_split_metric_impact_verdict.md",
}

STATUSES = {
    "canonical_current",
    "supporting_current",
    "superseded_legacy_split",
    "superseded_intermediate",
    "exploratory_only",
    "unresolved_blocker",
    "historical_execution_audit",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def git_origin_commit(path: Path) -> str | None:
    if not path.exists():
        return None
    relative = str(path.relative_to(ROOT))
    proc = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip() or None


def artifact_spec(
    path: str,
    status: str,
    *,
    superseded_by: str | None,
    allowed_use: str,
    prohibited_use: str,
    notes: str,
    result_commit: str | None = None,
    recorded_sha256: str | None = None,
    sha_applicable: bool = True,
) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"invalid artifact status: {status}")
    absolute = ROOT / path
    exists = absolute.exists()
    sha = sha256_file(absolute) if exists and sha_applicable else recorded_sha256
    commit = result_commit if result_commit is not None else git_origin_commit(absolute)
    return {
        "path": path,
        "status": status,
        "result_commit": commit,
        "sha256": sha,
        "superseded_by": superseded_by,
        "allowed_use": allowed_use,
        "prohibited_use": prohibited_use,
        "notes": notes,
        "exists_in_worktree": exists,
    }


def build_artifact_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    def add(path: str, status: str, superseded_by: str | None, allowed: str,
            prohibited: str, notes: str, commit: str | None = None,
            sha: str | None = None, sha_applicable: bool = True) -> None:
        specs.append(artifact_spec(
            path,
            status,
            superseded_by=superseded_by,
            allowed_use=allowed,
            prohibited_use=prohibited,
            notes=notes,
            result_commit=commit,
            recorded_sha256=sha,
            sha_applicable=sha_applicable,
        ))

    freeze = "h2cmi/results/review_completion/FINAL_REPAIRED_W1_EVIDENCE_FREEZE.json"
    p9 = "h2cmi/results/review_completion/spdim_w1_repaired_three_seed_results.csv"
    p7 = "h2cmi/results/review_completion/w1_repaired_h2cmi_results.csv"

    for path in (
        "h2cmi/results/review_completion/FINAL_REPAIRED_W1_EVIDENCE_FREEZE.md",
        freeze,
    ):
        add(path, "canonical_current", None, "canonical repaired-W1 verdict and claim gate",
            "legacy-split reinterpretation or adapter-only attribution",
            "P10 final scientific evidence freeze", CANONICAL_HEAD_COMMIT)
    add(p7, "canonical_current", None, "confirmatory repaired-split H2CMI W1 result",
        "legacy contiguous-split substitution", "P7 accepted result", P7_COMMIT)
    add(p9, "canonical_current", None, "official repaired-split three-seed SPDIM baseline",
        "seed-zero-only or adapter-only cross-pipeline claim", "P9 accepted result", P9_COMMIT)
    for path in (
        "h2cmi/results/review_completion/w1_repaired_cross_pipeline_results.csv",
        "h2cmi/results/review_completion/w1_repaired_cross_pipeline_results.md",
        "h2cmi/results/review_completion/w1_repaired_cross_pipeline_results.json",
        "h2cmi/results/review_completion/w1_repaired_cross_pipeline_harm.csv",
        "h2cmi/results/review_completion/w1_repaired_cross_pipeline_harm.md",
    ):
        add(path, "canonical_current", None, "same-split full-pipeline reporting under P10",
            "controlled adapter-only attribution", "P10 standardized evidence", CANONICAL_HEAD_COMMIT)
    for path in (
        "h2cmi/results/review_completion/w1_repaired_split_manifest.csv",
        "h2cmi/results/review_completion/w1_repaired_split_manifest.json",
    ):
        add(path, "canonical_current", None, "frozen repaired-W1 split definition",
            "split mutation or legacy-row admission", "Shared P7/P9 repaired manifest")
    for path in (
        "h2cmi/results/review_completion/MANUSCRIPT_NUMBERS_READY.md",
        "h2cmi/results/review_completion/REVIEW_COMPLETION_CURRENT_STATUS.md",
        "h2cmi/results/review_completion/REVIEW_COMPLETION_CURRENT_STATUS.json",
    ):
        add(path, "canonical_current", None, "current writer/status entry point",
            "use of superseded W1 headlines", "P11 canonical pointer", CANONICAL_HEAD_COMMIT)
    for path in (
        "h2cmi/results/review_completion/CANONICAL_EVIDENCE_INDEX.md",
        "h2cmi/results/review_completion/CANONICAL_EVIDENCE_INDEX.json",
    ):
        add(path, "canonical_current", None, "canonical artifact-status index",
            "using the index as a scientific result",
            "Self-referential checksum is intentionally omitted; Git records the containing blob.",
            P11_THIS_COMMIT, sha_applicable=False)

    supporting = (
        "h2cmi/results/review_completion/w1_repaired_h2cmi_summary.json",
        "h2cmi/results/review_completion/w1_repaired_h2cmi_audit.md",
        "h2cmi/results/review_completion/w1_repaired_h2cmi_four_branch_ci.csv",
        "h2cmi/results/review_completion/w1_repaired_h2cmi_four_branch_ci.json",
        "h2cmi/results/review_completion/w1_repaired_h2cmi_dataset_heterogeneity_ci.csv",
        "h2cmi/results/review_completion/w1_repaired_h2cmi_method_contrasts.csv",
        "h2cmi/results/review_completion/w1_repaired_h2cmi_method_contrasts.json",
        "h2cmi/results/review_completion/w1_repaired_h2cmi_protocol.md",
        "h2cmi/results/review_completion/w1_repaired_h2cmi_dryrun_audit.md",
        "h2cmi/results/review_completion/w1_repaired_h2cmi_dryrun_audit.json",
        "h2cmi/results/review_completion/w1_repaired_h2cmi_runner_dryrun_crosscheck.json",
        "h2cmi/results/review_completion/p7_training_cache_hygiene.md",
        "h2cmi/results/review_completion/p7_training_cache_hygiene.json",
        "h2cmi/results/review_completion/w1_repaired_split_manifest_audit.md",
        "h2cmi/results/review_completion/spdim_w1_repaired_three_seed_summary.json",
        "h2cmi/results/review_completion/spdim_w1_repaired_three_seed_result_digest.md",
        "h2cmi/results/review_completion/spdim_w1_repaired_three_seed_method_ci.csv",
        "h2cmi/results/review_completion/spdim_w1_repaired_three_seed_contrast_ci.csv",
        "h2cmi/results/review_completion/spdim_w1_repaired_three_seed_harm.csv",
        "h2cmi/results/review_completion/spdim_w1_repaired_three_seed_seed_stability.csv",
        "h2cmi/results/review_completion/spdim_w1_repaired_three_seed_red_team_review.md",
        "h2cmi/results/review_completion/spdim_w1_repaired_three_seed_protocol.md",
        "h2cmi/results/review_completion/spdim_w1_repaired_seeds12_results.csv",
        "h2cmi/results/review_completion/spdim_w1_repaired_seeds12_dryrun_audit.md",
        "h2cmi/results/review_completion/spdim_w1_repaired_seeds12_dryrun_audit.json",
        "h2cmi/results/review_completion/w1_repaired_pipeline_comparability_audit.md",
        "h2cmi/results/review_completion/w1_repaired_pipeline_comparability_audit.json",
        "h2cmi/results/review_completion/w1_repaired_cross_pipeline_red_team_review.md",
        "h2cmi/results/review_completion/sleep_replay_hash_audit.md",
        "h2cmi/results/review_completion/sleep_branch_confusion_matrices.json",
        "h2cmi/results/review_completion/sleep_per_stage_recall.csv",
        "h2cmi/results/review_completion/v2p_corrected_unit_key_audit.md",
        "h2cmi/results/review_completion/v2p_corrected_grid_summary.csv",
        "h2cmi/results/review_completion/v2p_corrected_method_summary.csv",
        "h2cmi/results/review_completion/v2p_corrected_paired_contrasts.csv",
        "h2cmi/results/review_completion/v2p_corrected_cluster_bootstrap.json",
        "h2cmi/results/review_completion/geometry_capacity_existing_ci.csv",
        "h2cmi/results/review_completion/geometry_capacity_offdiagonal_results.csv",
        "h2cmi/results/review_completion/geometry_capacity_stress_methods.md",
        "h2cmi/results/review_completion/offdiag_completion_audit.md",
        "h2cmi/results/review_completion/four_branch_ci_methods.md",
        "h2cmi/results/review_completion/encoder_backbone_details.md",
        "h2cmi/results/review_completion/encoder_backbone_details.json",
        "h2cmi/results/review_completion/spdim_external_repo_assessment.md",
        "h2cmi/results/review_completion/spdim_protocol_mapping.md",
        "h2cmi/results/review_completion/README.md",
        "h2cmi/results/review_completion/REVIEW_COMPLETION_SUMMARY.md",
        "h2cmi/results/review_completion/VALIDATION_STATUS.md",
        "h2cmi/results/review_completion/artifact_inventory.md",
        "h2cmi/results/review_completion/baseline_inventory_and_blockers.md",
        "h2cmi/results/review_completion/STALE_CLAIM_AUDIT.md",
        "h2cmi/results/review_completion/STALE_CLAIM_AUDIT.json",
        "h2cmi/results/review_completion/p11_canonical_status_audit.py",
    )
    for path in supporting:
        commit = CANONICAL_HEAD_COMMIT if any(token in path for token in (
            "cross_pipeline", "STALE_CLAIM", "p11_canonical", "REVIEW_COMPLETION_SUMMARY",
            "VALIDATION_STATUS", "artifact_inventory", "encoder_backbone_details",
            "spdim_external_repo_assessment", "spdim_protocol_mapping",
        )) else None
        add(path, "supporting_current", None, "support canonical results or implementation detail",
            "override of the P10 claim gate", "Current supporting artifact", commit)

    add("results/h2cmi/p0_w1_all.jsonl", "superseded_legacy_split", freeze,
        "legacy split diagnosis and provenance only", "current W1 headline or confirmatory claim",
        "Original REVIEW_P0 W1 raw artifact; stored outside this worktree",
        REVIEW_P0_COMMIT,
        "c1ef4a6f4bb52a0561e14b5c26d1b40290f5ac44c4a7a86876f97639f2112633")
    for path in (
        "h2cmi/results/review_completion/four_branch_complete_ci.csv",
        "h2cmi/results/review_completion/four_branch_complete_ci.json",
        "h2cmi/results/review_completion/mi_dataset_heterogeneity_complete_ci.csv",
    ):
        allowed = "Sleep rows or explicitly labeled legacy W1 diagnosis" if "four_branch" in path else "legacy W1 diagnosis only"
        add(path, "superseded_legacy_split", freeze, allowed,
            "current MI/W1 result", "Contains old contiguous-split MI/W1 evidence")
    for path in (
        "h2cmi/results/review_completion/spdim_w1_seed0_results.csv",
        "h2cmi/results/review_completion/spdim_w1_seed0_summary.json",
        "h2cmi/results/review_completion/spdim_w1_seed0_result_digest.md",
        "h2cmi/results/review_completion/spdim_w1_seed0_audit.md",
        "h2cmi/results/review_completion/spdim_w1_seed0_protocol.md",
    ):
        add(path, "superseded_legacy_split", p9, "P6 execution/split diagnosis only",
            "current baseline or repaired-split claim", "P6 contiguous-split seed-zero packet")

    for path in (
        "h2cmi/results/review_completion/spdim_w1_repaired_seed0_results.csv",
        "h2cmi/results/review_completion/spdim_w1_repaired_seed0_summary.json",
        "h2cmi/results/review_completion/spdim_w1_repaired_seed0_result_digest.md",
        "h2cmi/results/review_completion/spdim_w1_repaired_seed0_audit.md",
        "h2cmi/results/review_completion/spdim_w1_repaired_seed0_protocol.md",
        "h2cmi/results/review_completion/w1_valid_subset_four_branch.csv",
        "h2cmi/results/review_completion/w1_valid_subset_recompute.json",
        "h2cmi/results/review_completion/w1_valid_subset_recompute.md",
        "h2cmi/results/review_completion/spdim_official_baseline_results.csv",
        "h2cmi/results/review_completion/ARTIFACT_INDEX.json",
        "h2cmi/results/review_completion/spdim_bnci001_clean_audit.md",
        "h2cmi/results/review_completion/spdim_bnci001_clean_compare_to_exploratory.csv",
        "h2cmi/results/review_completion/spdim_bnci001_clean_protocol.md",
        "h2cmi/results/review_completion/spdim_bnci001_clean_results.csv",
        "h2cmi/results/review_completion/spdim_bnci001_clean_summary.json",
    ):
        add(path, "superseded_intermediate", p9 if "spdim" in path else freeze,
            "intermediate audit or execution trace", "final-baseline or current headline use",
            "Superseded by repaired full packet")

    for path in (
        "h2cmi/results/review_completion/spdim_probe_results.csv",
        "h2cmi/results/review_completion/spdim_probe_audit.md",
        "h2cmi/results/review_completion/spdim_probe_protocol.md",
        "h2cmi/results/review_completion/spdim_probe_integrity_audit.json",
        "h2cmi/results/review_completion/spdim_probe_integrity_audit.md",
        "h2cmi/results/review_completion/spdim_bnci001_audit.md",
        "h2cmi/results/review_completion/spdim_bnci001_protocol.md",
        "h2cmi/results/review_completion/spdim_bnci001_provenance_reconciliation.md",
        "h2cmi/results/review_completion/spdim_bnci001_results.csv",
        "h2cmi/results/review_completion/spdim_bnci001_summary.json",
        "h2cmi/results/review_completion/spdim_bnci001_provenance_reconciliation.json",
    ):
        add(path, "exploratory_only", p9, "bounded feasibility/provenance diagnosis",
            "official multi-dataset baseline claim", "Pre-P9 bounded probe or expansion")

    for path in (
        "h2cmi/results/review_completion/orthogonal_score_blockers.md",
        "h2cmi/results/review_completion/orthogonal_score_diagnostic_methods.md",
        "h2cmi/results/review_completion/orthogonal_score_controlled_results.csv",
        "h2cmi/results/review_completion/orthogonal_score_natural_results.csv",
        "h2cmi/results/review_completion/orthogonal_score_v2p_results.csv",
    ):
        add(path, "unresolved_blocker", None, "blocker definition or placeholder schema",
            "negative result or completed evaluation claim", "Orthogonal score remains unresolved")
    add("h2cmi/results/review_completion/geometry_capacity_blockers.md",
        "unresolved_blocker", None, "montage-remapping limitation statement",
        "claim of universal montage robustness", "Cross-montage remapping remains unresolved")
    add("h2cmi/results/review_completion/BLOCKERS.md", "unresolved_blocker", None,
        "current blocker pointer", "treating resolved SPDIM as blocked",
        "Only orthogonal score and montage remapping remain active", CANONICAL_HEAD_COMMIT)

    for path in (
        "h2cmi/results/review_completion/COMMAND_LOG.md",
        "h2cmi/results/review_completion/RUN_PROVENANCE.md",
        "h2cmi/results/review_completion/BRANCH_RECONCILIATION.md",
        "h2cmi/results/review_completion/SLURM_MONITORING_POLICY.md",
        "h2cmi/results/review_completion/SPDIM_CLEAN_RUN_POLICY.md",
        "h2cmi/results/review_completion/spdim_official_baseline_blocker.md",
        "h2cmi/results/review_completion/w1_split_metric_audit.md",
        "h2cmi/results/review_completion/w1_split_metric_audit.json",
        "h2cmi/results/review_completion/w1_split_metric_impact_verdict.md",
        "h2cmi/results/review_completion/w1_split_metric_impact_verdict.json",
        "h2cmi/results/review_completion/w1_legacy_split_quarantine.md",
        "h2cmi/results/review_completion/w1_legacy_split_quarantine.json",
        "h2cmi/results/review_completion/spdim_w1_repaired_seeds12_audit.md",
        "h2cmi/results/review_completion/spdim_w1_repaired_seeds12_summary.json",
        "h2cmi/results/review_completion/spdim_w1_repaired_seed0_4shard_resubmission.md",
        "h2cmi/results/review_completion/spdim_w1_repaired_seed0_4shard_resubmission.json",
        "h2cmi/results/review_completion/w1_alternative_split_protocol.md",
        "h2cmi/results/review_completion/w1_alternative_split_dryrun.csv",
        "h2cmi/results/review_completion/w1_alternative_split_dryrun.json",
        "h2cmi/results/review_completion/w1_alternative_split_rerun_feasibility.md",
        "h2cmi/results/review_completion/w1_alternative_split_rerun_feasibility.json",
        "h2cmi/results/review_completion/w1_split_repair_decision_gate.md",
        "h2cmi/results/review_completion/w1_split_repair_decision_gate.json",
        "h2cmi/results/review_completion/spdim_w1_seed0_dryrun_audit.md",
        "h2cmi/results/review_completion/spdim_w1_seed0_dryrun_audit.json",
        "h2cmi/results/review_completion/spdim_w1_repaired_seed0_legacy_compare.md",
        "h2cmi/results/review_completion/spdim_w1_repaired_seed0_postprocess.py",
        "h2cmi/results/review_completion/spdim_w1_repaired_seed0_dryrun_audit.md",
        "h2cmi/results/review_completion/spdim_w1_repaired_seed0_dryrun_audit.json",
        "h2cmi/results/review_completion/w1_balanced_accuracy_scorer_audit.md",
        "h2cmi/results/review_completion/w1_balanced_accuracy_scorer_audit.json",
        "h2cmi/results/review_completion/w1_repaired_h2cmi_legacy_compare.md",
        "h2cmi/results/review_completion/offdiag_slurm_submission.md",
        "h2cmi/results/review_completion/offdiag_watch_status.md",
    ):
        add(path, "historical_execution_audit", freeze,
            "execution, provenance, or supersession history", "current headline override",
            "Retained historical audit; status must be read with canonical index")

    add("h2cmi/results/review_completion/spdim_seed0_valid_subset_summary.csv",
        "superseded_legacy_split", freeze, "legacy valid-subset diagnosis only",
        "current baseline or repaired-split claim",
        "P6 contiguous-split valid-subset diagnostic")

    for item in specs:
        if item["path"] in P11_MANAGED_PATHS:
            item["result_commit"] = P11_THIS_COMMIT

    paths = [item["path"] for item in specs]
    if len(paths) != len(set(paths)):
        duplicates = [path for path, count in Counter(paths).items() if count > 1]
        raise ValueError(f"duplicate canonical-index paths: {duplicates}")
    return sorted(specs, key=lambda item: (item["status"], item["path"]))


STALE_PATTERNS = [
    ("legacy_G_0.0604", "G=+0.0604 / G = +0.0604", re.compile(r"G\s*=\s*\+0\.0604", re.I)),
    ("cho2017_drives", "Cho2017 drives", re.compile(r"Cho2017 drives", re.I)),
    ("no_same_split_official", "no same-split H2CMI official SPDIM", re.compile(r"no same-split H2CMI official SPDIM", re.I)),
    ("no_official_spdim", "no official SPDIM", re.compile(r"\bno official SPDIM\b", re.I)),
    ("do_not_claim_official", "Do not claim an official SPDIM comparison", re.compile(r"Do not claim an official SPDIM comparison", re.I)),
    ("seed_zero_only", "seed-0 only", re.compile(r"seed-0\s+only", re.I)),
    (
        "legacy_w1_confirmatory_false",
        "current_w1_results_can_be_used_as_confirmatory=false",
        re.compile(r"current_w1_results_can_be_used_as_confirmatory.*false", re.I),
    ),
    (
        "legacy_p6_spdim_false",
        "current_spdim_p6_can_be_used_as_seed0_baseline=false",
        re.compile(r"current_spdim_p6_can_be_used_as_seed0_baseline.*false", re.I),
    ),
]


def markdown_sections(lines: list[str]) -> dict[int, str]:
    section = "document_root"
    out = {}
    for number, line in enumerate(lines, start=1):
        if line.startswith("#"):
            section = line.lstrip("#").strip()
        out[number] = section
    return out


def classify_stale_hit(path: str, section: str, full_text: str) -> tuple[str, str]:
    if path.endswith("MANUSCRIPT_NUMBERS_READY.md") and section == "Superseded Legacy Contiguous-Split Diagnostic":
        return "correctly_labeled_legacy_history", "Hit is confined to the explicit superseded legacy section."
    if path.endswith("spdim_official_baseline_blocker.md"):
        return "resolved_blocker_history", "File begins with RESOLVED_BY_P9 metadata."
    historical_tokens = (
        "COMMAND_LOG.md",
        "w1_split_metric_impact_verdict",
        "w1_legacy_split_quarantine",
        "spdim_w1_seed0_",
        "spdim_w1_repaired_seed0_",
        "spdim_seed0_valid_subset",
    )
    if any(token in path for token in historical_tokens):
        return "correctly_labeled_legacy_history", "Artifact is an explicit legacy/intermediate execution record."
    prefix = full_text[:2000].lower()
    if any(marker in prefix for marker in (
        "historical", "superseded", "legacy", "quarantine", "resolved_by_p9",
    )):
        return "correctly_labeled_legacy_history", "Document explicitly marks its historical or superseded status."
    if "not seed-0 only" in full_text.lower() or "is_seed0_only\": false" in full_text.lower():
        return "current_valid_statement", "Current statement explicitly rejects the stale seed-zero-only status."
    return "active_stale_error", "No explicit superseded, resolved, or current-valid context was found."


def scan_stale_claims() -> dict[str, Any]:
    excluded = {STALE_MD.resolve(), STALE_JSON.resolve()}
    paths = sorted(
        path for path in OUT.rglob("*")
        if path.suffix in {".md", ".json"} and path.resolve() not in excluded
    )
    hits = []
    pattern_counts = Counter()
    for path in paths:
        text = path.read_text(errors="replace")
        lines = text.splitlines()
        sections = markdown_sections(lines) if path.suffix == ".md" else {
            number: "json_document" for number in range(1, len(lines) + 1)
        }
        relative = str(path.relative_to(ROOT))
        for number, line in enumerate(lines, start=1):
            for pattern_id, query, regex in STALE_PATTERNS:
                match = regex.search(line)
                if not match:
                    continue
                classification, reason = classify_stale_hit(
                    relative, sections[number], text
                )
                hits.append({
                    "path": relative,
                    "line": number,
                    "section": sections[number],
                    "pattern_id": pattern_id,
                    "query": query,
                    "match": match.group(0),
                    "line_text": line.strip(),
                    "classification": classification,
                    "reason": reason,
                })
                pattern_counts[pattern_id] += 1
    classification_counts = Counter(hit["classification"] for hit in hits)
    return {
        "status": "pass" if classification_counts["active_stale_error"] == 0 else "fail",
        "scan_root": str(OUT.relative_to(ROOT)),
        "scanned_extensions": [".md", ".json"],
        "scanned_file_count": len(paths),
        "excluded_self_referential_outputs": [
            str(path.relative_to(ROOT)) for path in sorted(excluded)
        ],
        "patterns": [
            {
                "pattern_id": pattern_id,
                "query": query,
                "hit_count": pattern_counts[pattern_id],
            }
            for pattern_id, query, _ in STALE_PATTERNS
        ],
        "hits": hits,
        "classification_counts": {
            name: classification_counts[name]
            for name in (
                "active_stale_error",
                "correctly_labeled_legacy_history",
                "resolved_blocker_history",
                "current_valid_statement",
            )
        },
        "active_stale_error_count": classification_counts["active_stale_error"],
    }


def write_stale_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# Stale Claim Audit",
        "",
        f"- status: `{payload['status']}`",
        f"- scanned Markdown/JSON files: `{payload['scanned_file_count']}`",
        f"- active_stale_error_count: `{payload['active_stale_error_count']}`",
        "- excluded self-referential outputs: `STALE_CLAIM_AUDIT.md/json` only",
        "",
        "## Pattern Coverage",
        "",
        "| pattern | query | hits |",
        "|---|---|---:|",
    ]
    for item in payload["patterns"]:
        lines.append(f"| `{item['pattern_id']}` | `{item['query']}` | {item['hit_count']} |")
    lines.extend([
        "",
        "## Classified Hits",
        "",
        "| path:line | section | pattern | classification | statement |",
        "|---|---|---|---|---|",
    ])
    if not payload["hits"]:
        lines.append("| none | none | none | current_valid_statement | No searched phrase remains. |")
    for hit in payload["hits"]:
        statement = hit["line_text"].replace("|", "\\|")
        lines.append(
            f"| `{hit['path']}:{hit['line']}` | {hit['section']} | "
            f"`{hit['pattern_id']}` | `{hit['classification']}` | {statement} |"
        )
    lines.extend([
        "",
        "Historical statements are retained only where the document or section explicitly marks them as legacy, superseded, quarantined, or resolved. No active writer-facing stale claim remains.",
    ])
    STALE_MD.write_text("\n".join(lines) + "\n")


def write_current_status(
    active_stale_error_count: int,
    canonical_freeze: dict[str, Any],
) -> dict[str, Any]:
    blocker_text = (OUT / "BLOCKERS.md").read_text()
    resolved_text = (OUT / "spdim_official_baseline_blocker.md").read_text()
    official_complete = canonical_freeze[
        "official_spdim_three_seed_same_split_baseline_complete"
    ]
    payload = {
        "status": "pass",
        "canonical_head_commit": CANONICAL_HEAD_COMMIT,
        "legacy_w1_split_quarantined": canonical_freeze["legacy_w1_split_quarantined"],
        "repaired_h2cmi_w1_confirmatory": canonical_freeze["repaired_h2cmi_w1_confirmatory"],
        "official_spdim_three_seed_baseline_complete": official_complete,
        "official_spdim_blocker_resolved": (
            official_complete and resolved_text.startswith("status: RESOLVED_BY_P9")
        ),
        "adapter_only_cross_pipeline_comparison_supported": canonical_freeze[
            "adapter_only_h2cmi_vs_spdim_comparison_supported"
        ],
        "full_pipeline_same_split_comparison_supported": canonical_freeze[
            "full_pipeline_same_split_comparison_supported"
        ],
        "orthogonal_score_blocked": "Orthogonal-score estimator/evaluation" in blocker_text,
        "montage_layout_remapping_stress_blocked": (
            "Montage-layout or cross-montage remapping stress" in blocker_text
        ),
        "additional_gpu_required": canonical_freeze["additional_gpu_required"],
        "active_stale_error_count": active_stale_error_count,
        "canonical_w1_freeze_path": str(
            (OUT / "FINAL_REPAIRED_W1_EVIDENCE_FREEZE.md").relative_to(ROOT)
        ),
        "canonical_all_numbers_digest_path": str(MANUSCRIPT_DIGEST.relative_to(ROOT)),
        "canonical_evidence_index_path": str(INDEX_MD.relative_to(ROOT)),
        "stale_claim_audit_path": str(STALE_MD.relative_to(ROOT)),
        "source_hashes": {
            "p7_h2cmi": P7_SHA256,
            "p9_spdim": P9_SHA256,
            "repaired_manifest": MANIFEST_HASH,
        },
    }
    if active_stale_error_count != 0:
        raise RuntimeError("refusing current-status pass with active stale errors")
    write_json(STATUS_JSON, payload)
    lines = [
        "# Review Completion Current Status",
        "",
        f"- canonical scientific head: `{CANONICAL_HEAD_COMMIT}`",
        "- legacy W1 split: quarantined",
        "- repaired H2CMI W1: confirmatory",
        "- official SPDIM repaired-split three-seed baseline: complete",
        "- official SPDIM blocker: resolved by P9",
        "- cross-pipeline interpretation: same-split full-pipeline only",
        "- adapter-only H2CMI versus SPDIM comparison: not supported",
        "- equivalence/noninferiority: not supported",
        "- additional GPU required: `false`",
        f"- active stale errors: `{active_stale_error_count}`",
        "",
        "## Canonical Entry Points",
        "",
        "- Scientific freeze: `FINAL_REPAIRED_W1_EVIDENCE_FREEZE.md/json`",
        "- All writer-facing numbers: `MANUSCRIPT_NUMBERS_READY.md`",
        "- Artifact status map: `CANONICAL_EVIDENCE_INDEX.md/json`",
        "- Stale-claim gate: `STALE_CLAIM_AUDIT.md/json`",
        "",
        "## Active Blockers",
        "",
        "1. Orthogonal-score estimator/evaluation.",
        "2. Montage-layout or cross-montage remapping stress.",
        "",
        "No experiment is required or authorized by this status pointer.",
    ]
    STATUS_MD.write_text("\n".join(lines) + "\n")
    return payload


def validate_index_coverage(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {item["path"] for item in artifacts}
    top_level = {
        str(path.relative_to(ROOT))
        for path in OUT.iterdir()
        if path.is_file() and path.suffix in {".md", ".json", ".csv"}
    }
    missing = sorted(top_level - indexed)
    if missing:
        raise RuntimeError(f"unclassified top-level review artifacts: {missing}")
    return {
        "top_level_md_json_csv_count": len(top_level),
        "unclassified_top_level_artifacts": missing,
    }


def validate_index_provenance(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    sentinel = 0
    absent = 0
    failures = []
    for item in artifacts:
        commit = item["result_commit"]
        path = item["path"]
        if commit == P11_THIS_COMMIT:
            sentinel += 1
            continue
        if not item["exists_in_worktree"]:
            absent += 1
            continue
        if not commit or not re.fullmatch(r"[0-9a-f]{40}", commit):
            failures.append({"path": path, "reason": "invalid result_commit", "value": commit})
            continue
        proc = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=ROOT,
            capture_output=True,
        )
        if proc.returncode != 0:
            failures.append({"path": path, "reason": "path absent at result_commit"})
            continue
        blob_sha = hashlib.sha256(proc.stdout).hexdigest()
        if blob_sha != item["sha256"]:
            failures.append({
                "path": path,
                "reason": "result_commit blob does not match indexed SHA-256",
                "commit_blob_sha256": blob_sha,
                "indexed_sha256": item["sha256"],
            })
            continue
        checked += 1
    if failures:
        raise RuntimeError(f"canonical-index provenance mismatches: {failures}")
    return {
        "commit_blob_pairs_verified": checked,
        "p11_containing_commit_sentinels": sentinel,
        "absent_external_artifacts": absent,
        "mismatches": failures,
    }


def write_index(
    artifacts: list[dict[str, Any]],
    coverage: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    counts = Counter(item["status"] for item in artifacts)
    payload = {
        "status": "pass",
        "canonical_scientific_head": CANONICAL_HEAD_COMMIT,
        "result_commit_sentinel": {
            "value": P11_THIS_COMMIT,
            "meaning": "The Git commit containing this index and the listed P11 artifact.",
        },
        "coverage_validation": coverage,
        "result_commit_provenance_validation": provenance,
        "status_definitions": {
            "canonical_current": "Current result or required canonical pointer.",
            "supporting_current": "Current supporting analysis, audit, or implementation detail.",
            "superseded_legacy_split": "Old contiguous-split evidence; diagnostic history only.",
            "superseded_intermediate": "Valid intermediate evidence superseded by a later complete packet.",
            "exploratory_only": "Bounded feasibility or exploratory evidence; not a final baseline.",
            "unresolved_blocker": "Missing implementation/evaluation or explicitly untested scope.",
            "historical_execution_audit": "Retained execution/provenance history; not a current result pointer.",
        },
        "artifact_count": len(artifacts),
        "status_counts": {status: counts[status] for status in sorted(STATUSES)},
        "artifacts": artifacts,
    }
    write_json(INDEX_JSON, payload)

    lines = [
        "# Canonical Evidence Index",
        "",
        f"Canonical scientific head: `{CANONICAL_HEAD_COMMIT}`.",
        "",
        f"`result_commit={P11_THIS_COMMIT}` is a non-hash sentinel meaning the Git commit containing this index. It avoids falsely attributing uncommitted P11 content to the P10 scientific head. The two self-referential index files omit SHA-256 because embedding their own digest would be circular; their Git blobs provide integrity.",
        "",
        "Use `canonical_current` artifacts for current W1 numbers. Historical and superseded artifacts remain available only under their listed allowed use.",
    ]
    for status in (
        "canonical_current",
        "supporting_current",
        "superseded_legacy_split",
        "superseded_intermediate",
        "exploratory_only",
        "unresolved_blocker",
        "historical_execution_audit",
    ):
        lines.extend([
            "",
            f"## {status}",
            "",
            "| path | result commit | SHA-256 | superseded by | allowed use | prohibited use |",
            "|---|---|---|---|---|---|",
        ])
        for item in artifacts:
            if item["status"] != status:
                continue
            commit = item["result_commit"] or "n/a"
            sha = item["sha256"] or "n/a"
            superseded = item["superseded_by"] or "n/a"
            lines.append(
                f"| `{item['path']}` | `{commit}` | `{sha}` | `{superseded}` | "
                f"{item['allowed_use']} | {item['prohibited_use']} |"
            )
    INDEX_MD.write_text("\n".join(lines) + "\n")
    return payload


def append_command_log() -> None:
    marker = "Per PM P11, canonicalized final repaired-W1 evidence status"
    text = COMMAND_LOG.read_text()
    if marker in text:
        return
    entry = f"""
- {marker}. Writer-facing W1 values now point only to the P10 freeze; old
  contiguous-split values are confined to an explicit legacy diagnostic.
  Official SPDIM is marked complete/resolved by P9. The canonical index
  classifies current, supporting, superseded, exploratory, blocker, and
  historical artifacts. The stale-claim scan covers all review-completion
  Markdown/JSON files and requires zero active errors. P7/P9/P10 result
  artifacts and the repaired manifest remained unchanged. No experiment was
  launched.
"""
    COMMAND_LOG.write_text(text.rstrip() + "\n\n" + entry.lstrip())


def validate_immutable_artifacts() -> dict[str, Any]:
    checks = {
        "p7_sha256": sha256_file(OUT / "w1_repaired_h2cmi_results.csv"),
        "p9_sha256": sha256_file(OUT / "spdim_w1_repaired_three_seed_results.csv"),
    }
    if checks["p7_sha256"] != P7_SHA256 or checks["p9_sha256"] != P9_SHA256:
        raise RuntimeError(f"P7/P9 immutable source mismatch: {checks}")
    manifest = json.loads((OUT / "w1_repaired_split_manifest.json").read_text())
    checks["manifest_hash"] = manifest.get("manifest_hash")
    if checks["manifest_hash"] != MANIFEST_HASH:
        raise RuntimeError("repaired manifest semantic hash changed")
    p10_checks = {}
    for relative, expected in P10_IMMUTABLE_SHA256.items():
        actual = sha256_file(ROOT / relative)
        p10_checks[relative] = {"expected": expected, "actual": actual, "matches": actual == expected}
    if not all(item["matches"] for item in p10_checks.values()):
        raise RuntimeError("P10 result CSV/JSON immutable hash gate failed")
    proc = subprocess.run(
        ["git", "diff", "--name-only", CANONICAL_HEAD_COMMIT, "--", *P10_IMMUTABLE_SHA256],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    if proc.stdout.strip():
        raise RuntimeError(f"P10 immutable files modified: {proc.stdout.strip()}")
    checks["p10_result_artifacts"] = p10_checks
    checks["p10_git_diff_empty"] = True

    accepted_result_paths = set()
    for commit in (P7_COMMIT, P9_COMMIT, CANONICAL_HEAD_COMMIT):
        proc = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        accepted_result_paths.update(
            path for path in proc.stdout.splitlines()
            if path.endswith((".csv", ".json"))
        )
    proc = subprocess.run(
        ["git", "diff", "--name-only", CANONICAL_HEAD_COMMIT, "--", *sorted(accepted_result_paths)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    modified_accepted_results = sorted(filter(None, proc.stdout.splitlines()))
    if modified_accepted_results:
        raise RuntimeError(
            f"P7/P9/P10 CSV/JSON artifacts modified: {modified_accepted_results}"
        )
    checks["accepted_result_csv_json_count"] = len(accepted_result_paths)
    checks["accepted_result_csv_json_git_diff_empty"] = True
    return checks


def validate_manuscript_digest() -> dict[str, Any]:
    text = MANUSCRIPT_DIGEST.read_text()
    required = [
        "FINAL_REPAIRED_W1_EVIDENCE_FREEZE.md",
        "0.6736216",
        "0.6639614",
        "0.6814493",
        "0.6770274",
        "+0.0078277",
        "[+0.0026504, +0.0132705]",
        "-0.0096602",
        "[-0.0136878, -0.0057326]",
        "+0.0052383",
        "[+0.0012029, +0.0093253]",
        "+0.0034058",
        "[-0.0021352, +0.0091774]",
        "0.5419807",
        "[0.5334460, 0.5508986]",
        "0.6471643",
        "[0.6304244, 0.6637672]",
        "0.6444235",
        "[0.6277308, 0.6610388]",
        "0.6431530",
        "[0.6264633, 0.6599215]",
        "+0.1051836",
        "[+0.0918437, +0.1189907]",
        "+0.1024428",
        "[+0.0894991, +0.1161096]",
        "+0.1011723",
        "[+0.0882946, +0.1148618]",
        "-0.0027407",
        "[-0.0046506, -0.0008260]",
        "-0.0040113",
        "[-0.0072577, -0.0007150]",
        "Superseded Legacy Contiguous-Split Diagnostic",
    ]
    missing = [value for value in required if value not in text]
    if missing:
        raise RuntimeError(f"canonical manuscript digest missing values: {missing}")
    current, separator, legacy = text.partition("## Superseded Legacy Contiguous-Split Diagnostic")
    if not separator:
        raise RuntimeError("legacy diagnostic section missing")
    if "+0.0604" in current or "+0.1227" in current:
        raise RuntimeError("old W1 value appears before legacy diagnostic section")
    if "G = +0.0604" not in legacy or "G = +0.1227" not in legacy:
        raise RuntimeError("legacy values are not retained in the labeled history section")
    return {"required_value_count": len(required), "missing": missing, "legacy_values_confined": True}


def parse_changed_structured_files() -> dict[str, Any]:
    json_paths = (
        OUT / "encoder_backbone_details.json",
        OUT / "w1_split_metric_impact_verdict.json",
        STALE_JSON,
        STATUS_JSON,
        INDEX_JSON,
    )
    for path in json_paths:
        json.loads(path.read_text())
    csv_paths = (
        OUT / "w1_repaired_cross_pipeline_results.csv",
        OUT / "w1_repaired_cross_pipeline_harm.csv",
    )
    counts = {}
    for path in csv_paths:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        counts[str(path.relative_to(ROOT))] = len(rows)
    return {"json_files": len(json_paths), "csv_rows": counts}


def main() -> None:
    immutable = validate_immutable_artifacts()
    digest = validate_manuscript_digest()
    append_command_log()

    canonical_freeze = json.loads(
        (OUT / "FINAL_REPAIRED_W1_EVIDENCE_FREEZE.json").read_text()
    )

    stale = scan_stale_claims()
    write_stale_markdown(stale)
    write_json(STALE_JSON, stale)
    if stale["active_stale_error_count"] != 0:
        raise SystemExit(2)
    current_status = write_current_status(
        stale["active_stale_error_count"], canonical_freeze
    )
    artifacts = build_artifact_specs()
    coverage = validate_index_coverage(artifacts)
    provenance = validate_index_provenance(artifacts)
    index = write_index(artifacts, coverage, provenance)

    # The index participates in the stale scan, so close the one-pass loop and
    # regenerate dependent outputs if its current content changes the result.
    post_index_stale = scan_stale_claims()
    if post_index_stale != stale:
        stale = post_index_stale
        write_stale_markdown(stale)
        write_json(STALE_JSON, stale)
        if stale["active_stale_error_count"] != 0:
            raise SystemExit(2)
        current_status = write_current_status(
            stale["active_stale_error_count"], canonical_freeze
        )
        artifacts = build_artifact_specs()
        coverage = validate_index_coverage(artifacts)
        provenance = validate_index_provenance(artifacts)
        index = write_index(artifacts, coverage, provenance)
    structured = parse_changed_structured_files()

    print(json.dumps({
        "status": "pass",
        "active_stale_error_count": stale["active_stale_error_count"],
        "stale_hit_count": len(stale["hits"]),
        "canonical_index_artifacts": index["artifact_count"],
        "canonical_status": current_status["status"],
        "index_coverage": coverage,
        "index_provenance": provenance,
        "immutable": immutable,
        "manuscript_digest": digest,
        "structured_parse": structured,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
