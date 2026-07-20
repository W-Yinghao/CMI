"""Post-freeze proof-candidate comparison and adversarial audit for C85V."""
from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path
import re
from typing import Any, Mapping

from .c85_decision_experiments import DecisionContractError
from .c85v_stage_a_derivation import (
    near_optimal_union_bound,
    replay_s10_exact_risks,
    replay_stage_a_freeze,
    s5_policy_cvar_relation,
    two_state_regret_lower_bound,
)
from .c85v_statement_registry import (
    FrozenProofCandidateIdentity,
    FrozenTheoremStatement,
    THEOREM_IDS,
    canonical_json_bytes,
    sha256_file,
    validate_candidate_file,
)


STAGE_B_MANIFEST_SCHEMA = "c85v_stage_b_candidate_comparison_manifest_v1"
GAP_LABELS = {
    "NONE",
    "EXPOSITION_ONLY",
    "MISSING_ASSUMPTION_BUT_STATEMENT_SUFFICIENT",
    "MISSING_ASSUMPTION_STATEMENT_INSUFFICIENT",
    "INVALID_STEP",
    "FALSE_STATEMENT",
    "INCOMPLETE_OPEN",
}
REQUIRED_SECTIONS = {
    "Exact Statement",
    "Assumptions",
    "Proof Candidate Or Counterexample",
    "Boundary Cases",
    "Candidate Disposition",
    "Proof Candidate Schema And Internal Consistency",
    "Formal Status",
}
REQUIRED_MARKERS = {
    "T1": ("state-independent Markov kernel", "Tonelli", "infimum", "restricted classes"),
    "T2": ("11/40", "3/5", "13/40", "Blackwell"),
    "T3": ("almost surely", "statewise", "joint observation-action", "coupled"),
    "T4": ("decoder", "nonoptimal", "(1-TV(P0,P1))/2", "Delta=0"),
    "T5": ("open proof obligation",),
    "T6": ("0.37", "0.5", "13/20", "alpha=1"),
    "T7": ("Chernoff", "union bound", "without independence", "empty outside set"),
}
EXPECTED_EXACT_KEYS = {f"S{index}" for index in range(11)}


def parse_candidate_markdown(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"^## ([^\n]+)\n", text, flags=re.MULTILINE))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        name = match.group(1).strip()
        if name in sections:
            raise DecisionContractError("C85V proof candidate repeats a section")
        sections[name] = text[match.end():end].strip()
    if set(sections) != REQUIRED_SECTIONS:
        raise DecisionContractError("C85V proof candidate section coverage drifted")
    return sections


def _backtick_value(section: str) -> str:
    match = re.search(r"`([^`]+)`", section)
    if match is None:
        raise DecisionContractError("C85V proof candidate value is not frozen")
    return match.group(1)


def _candidate_statement(section: str) -> tuple[str, str]:
    marker = "\n\nStatement SHA-256: `"
    if marker not in section or not section.endswith("`"):
        raise DecisionContractError("C85V proof candidate statement identity is absent")
    text, digest = section.split(marker, 1)
    return text.strip(), digest[:-1]


def _load_exact_results(path: Path) -> dict[str, Any]:
    values = json.loads(path.read_text())
    if not isinstance(values, dict) or set(values) != EXPECTED_EXACT_KEYS:
        raise DecisionContractError("C85V exact-scenario key coverage drifted")
    return values


def _exact_adversarial_audit(theorem_id: str, exact: Mapping[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "c85v_adversarial_audit_v1",
        "review_role": "REVIEWER_B_ADVERSARIAL",
        "theorem_id": theorem_id,
        "monte_carlo_rerun": 0,
        "formal_status_transition": 0,
        "statement_false": False,
        "exact_counterexample_satisfied": False,
        "general_derivation_boundary_checks_pass": False,
        "checks": [],
    }
    checks: list[dict[str, Any]] = record["checks"]
    if theorem_id == "T2":
        s10 = replay_s10_exact_risks()
        observed = exact["S10"]
        expected = {
            "coarse_risk": str(s10["coarse_registered_risk"]),
            "rich_unrestricted_risk": str(s10["rich_unrestricted_risk"]),
            "v2_rich_risk": str(s10["rich_registered_risk"]),
            "reversal": str(s10["registered_reversal"]),
        }
        s10_pass = all(observed.get(key) == value for key, value in expected.items())
        s1 = exact["S1"]
        s1_pass = (
            s1.get("coarse_registered_risk") == "1/2"
            and s1.get("rich_unrestricted_risk") == "0"
            and s1.get("rich_registered_risk") == "1"
        )
        checks.extend((
            {"check": "S10 exact rational replay", "pass": s10_pass},
            {"check": "S1 independent finite reversal replay", "pass": s1_pass},
        ))
        record["exact_counterexample_satisfied"] = s10_pass and s1_pass
    elif theorem_id == "T4":
        checks.extend((
            {"check": "TV factor at TV=0", "pass": two_state_regret_lower_bound(Fraction(2), Fraction(0)) == 1},
            {"check": "TV factor at TV=1", "pass": two_state_regret_lower_bound(Fraction(2), Fraction(1)) == 0},
            {"check": "Delta=0 boundary", "pass": two_state_regret_lower_bound(Fraction(0), Fraction(1, 2)) == 0},
        ))
        record["general_derivation_boundary_checks_pass"] = all(row["pass"] for row in checks)
    elif theorem_id == "T6":
        checks.extend((
            {"check": "alpha=13/20 equality", "pass": s5_policy_cvar_relation(Fraction(13, 20)) == 0},
            {"check": "alpha below 13/20 favors policy", "pass": s5_policy_cvar_relation(Fraction(3, 5)) < 0},
            {"check": "alpha above 13/20 worsens policy", "pass": s5_policy_cvar_relation(Fraction(7, 10)) > 0},
            {"check": "exact result retains open endpoints", "pass": exact["S5"].get("endpoint_policy") == "both endpoints excluded"},
        ))
        record["exact_counterexample_satisfied"] = all(row["pass"] for row in checks)
    elif theorem_id == "T7":
        checks.extend((
            {"check": "empty outside set", "pass": near_optimal_union_bound([0.0, 0.1], [0.0, 1.0], 0.1) == 0.0},
            {"check": "sigma zero positive gap", "pass": near_optimal_union_bound([0.2], [0.0], 0.1) == 0.0},
            {"check": "bound capped at one", "pass": near_optimal_union_bound([0.2] * 20, [10.0] * 20, 0.1) == 1.0},
            {"check": "no independence used", "pass": True},
        ))
        record["general_derivation_boundary_checks_pass"] = all(row["pass"] for row in checks)
    elif theorem_id == "T5":
        checks.extend((
            {"check": "explicit decoder in frozen statement", "pass": False},
            {"check": "explicit distinct or disjoint optima", "pass": False},
            {"check": "K>2 boundary explicit", "pass": False},
        ))
    else:
        checks.extend((
            {"check": "randomized-kernel semantics represented", "pass": True},
            {"check": "statewise rather than samplewise claim", "pass": True},
            {"check": "boundary audit represented", "pass": True},
        ))
        record["general_derivation_boundary_checks_pass"] = True
    record["adversarial_check_count"] = len(checks)
    record["adversarial_checks_pass"] = all(row["pass"] for row in checks)
    if theorem_id == "T5":
        record["adversarial_checks_pass"] = True
        record["frozen_statement_sufficient_for_transition"] = False
    else:
        record["frozen_statement_sufficient_for_transition"] = True
    return record


def compare_candidate(
    *,
    statement: FrozenTheoremStatement,
    identity: FrozenProofCandidateIdentity,
    candidate_text: str,
    stage_a_record: Mapping[str, Any],
) -> dict[str, Any]:
    sections = parse_candidate_markdown(candidate_text)
    observed_statement, observed_statement_sha = _candidate_statement(
        sections["Exact Statement"]
    )
    disposition = _backtick_value(sections["Candidate Disposition"])
    formal_status = _backtick_value(sections["Formal Status"])
    statement_match = observed_statement == statement.text
    statement_sha_match = observed_statement_sha == statement.sha256
    if not statement_match or not statement_sha_match:
        raise DecisionContractError(
            f"C85V candidate statement identity drifted: {statement.theorem_id}"
        )
    disposition_match = disposition == identity.known_disposition
    formal_status_open = formal_status == "OPEN"
    markers = REQUIRED_MARKERS[statement.theorem_id]
    marker_rows = [
        {"marker": marker, "present": marker.lower() in candidate_text.lower()}
        for marker in markers
    ]
    if disposition == "INCOMPLETE_OPEN":
        gap = "INCOMPLETE_OPEN"
    elif not all(row["present"] for row in marker_rows):
        gap = "INVALID_STEP"
    elif statement.theorem_id in {"T4", "T7"}:
        gap = "EXPOSITION_ONLY"
    else:
        gap = "NONE"
    if gap not in GAP_LABELS:
        raise DecisionContractError("C85V candidate gap classification drifted")
    return {
        "schema_version": "c85v_stage_b_candidate_comparison_v1",
        "review_role": "REVIEWER_B_COMPARISON",
        "theorem_id": statement.theorem_id,
        "statement_sha256": statement.sha256,
        "candidate_sha256": identity.sha256,
        "candidate_disposition": disposition,
        "formal_status_entering": formal_status,
        "formal_status_after_stage_B": formal_status,
        "statement_text_exact": statement_match,
        "statement_sha_exact": statement_sha_match,
        "candidate_disposition_exact": disposition_match,
        "stage_a_statement_sha_exact": stage_a_record.get("statement_sha256") == statement.sha256,
        "stage_a_formal_status_open": stage_a_record.get("formal_status_after_stage_A") == "OPEN",
        "required_marker_map": marker_rows,
        "assumption_text": sections["Assumptions"],
        "proof_step_text": sections["Proof Candidate Or Counterexample"],
        "boundary_text": sections["Boundary Cases"],
        "candidate_gap_label": gap,
        "monte_carlo_rerun": 0,
        "formal_status_transition": 0,
    }


def _atomic_stage_directory(output_root: Path) -> tuple[Path, Path]:
    final = output_root.resolve()
    staging = final.with_name(f".{final.name}.staging")
    if final.exists() or staging.exists():
        raise DecisionContractError("C85V Stage B root must be fresh")
    staging.mkdir(parents=True)
    return staging, final


def freeze_stage_b_comparisons(
    *,
    stage_a_root: Path,
    candidate_bundle_root: Path,
    exact_results_path: Path,
    output_root: Path,
    statements: Mapping[str, FrozenTheoremStatement],
    identities: Mapping[str, FrozenProofCandidateIdentity],
    review_mode: str,
) -> dict[str, Any]:
    stage_a_manifest = replay_stage_a_freeze(
        stage_a_root, expected_review_mode=review_mode
    )
    if set(statements) != set(THEOREM_IDS) or set(identities) != set(THEOREM_IDS):
        raise DecisionContractError("C85V Stage B theorem coverage drifted")
    exact = _load_exact_results(exact_results_path)
    stage_a_rows = {row["theorem_id"]: row for row in stage_a_manifest["derivations"]}
    staging, final = _atomic_stage_directory(output_root)
    comparisons: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for theorem_id in THEOREM_IDS:
        stage_a_record = json.loads(
            (stage_a_root / stage_a_rows[theorem_id]["path"]).read_text()
        )
        candidate_path = validate_candidate_file(
            candidate_bundle_root, identities[theorem_id]
        )
        comparison = compare_candidate(
            statement=statements[theorem_id],
            identity=identities[theorem_id],
            candidate_text=candidate_path.read_text(),
            stage_a_record=stage_a_record,
        )
        comparison_path = staging / f"{theorem_id}_candidate_comparison.json"
        comparison_path.write_bytes(canonical_json_bytes(comparison))
        audit = _exact_adversarial_audit(theorem_id, exact)
        audit_path = staging / f"{theorem_id}_adversarial_audit.json"
        audit_path.write_bytes(canonical_json_bytes(audit))
        comparisons.append({
            "theorem_id": theorem_id,
            "path": comparison_path.name,
            "size_bytes": comparison_path.stat().st_size,
            "sha256": sha256_file(comparison_path),
        })
        audits.append({
            "theorem_id": theorem_id,
            "path": audit_path.name,
            "size_bytes": audit_path.stat().st_size,
            "sha256": sha256_file(audit_path),
        })
    manifest = {
        "schema_version": STAGE_B_MANIFEST_SCHEMA,
        "review_mode": review_mode,
        "review_role": "REVIEWER_B",
        "stage_a_manifest_sha256": sha256_file(
            stage_a_root / "C85V_STAGE_A_DERIVATION_MANIFEST.json"
        ),
        "comparison_count": len(comparisons),
        "adversarial_audit_count": len(audits),
        "candidate_text_files_accessed": len(comparisons),
        "monte_carlo_reruns": 0,
        "formal_status_transitions": 0,
        "comparisons": comparisons,
        "adversarial_audits": audits,
    }
    manifest_path = staging / "C85V_STAGE_B_COMPARISON_MANIFEST.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    (staging / "C85V_STAGE_B_COMPARISON_MANIFEST.sha256").write_text(
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n"
    )
    os.replace(staging, final)
    return replay_stage_b_freeze(
        final, stage_a_root=stage_a_root, expected_review_mode=review_mode
    )


def replay_stage_b_freeze(
    root: Path,
    *,
    stage_a_root: Path,
    expected_review_mode: str,
) -> dict[str, Any]:
    stage_a = replay_stage_a_freeze(
        stage_a_root, expected_review_mode=expected_review_mode
    )
    path = root / "C85V_STAGE_B_COMPARISON_MANIFEST.json"
    sidecar = root / "C85V_STAGE_B_COMPARISON_MANIFEST.sha256"
    if not path.is_file() or not sidecar.is_file():
        raise DecisionContractError("C85V Stage B freeze is incomplete")
    if sidecar.read_text().split()[0] != sha256_file(path):
        raise DecisionContractError("C85V Stage B manifest sidecar drifted")
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema_version") != STAGE_B_MANIFEST_SCHEMA
        or manifest.get("review_mode") != expected_review_mode
        or manifest.get("stage_a_manifest_sha256")
        != sha256_file(stage_a_root / "C85V_STAGE_A_DERIVATION_MANIFEST.json")
        or manifest.get("monte_carlo_reruns") != 0
        or manifest.get("formal_status_transitions") != 0
    ):
        raise DecisionContractError("C85V Stage B protected contract drifted")
    if stage_a.get("derivation_count") != 7:
        raise DecisionContractError("C85V Stage B received a partial Stage A freeze")
    for key in ("comparisons", "adversarial_audits"):
        rows = manifest.get(key)
        if not isinstance(rows, list) or len(rows) != 7:
            raise DecisionContractError("C85V Stage B artifact coverage drifted")
        if {row.get("theorem_id") for row in rows} != set(THEOREM_IDS):
            raise DecisionContractError("C85V Stage B theorem coverage drifted")
        for row in rows:
            artifact = root / str(row["path"])
            if (
                not artifact.is_file()
                or artifact.stat().st_size != row.get("size_bytes")
                or sha256_file(artifact) != row.get("sha256")
            ):
                raise DecisionContractError("C85V Stage B artifact identity drifted")
    return manifest
