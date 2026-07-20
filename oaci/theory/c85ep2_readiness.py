"""Metadata assembly for C85EP2 frozen inputs, schemas, and readiness evidence."""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .c85e_result_manifest import REGISTERED_TABLES
from .c85e_runtime_guard import FORBIDDEN_PATH_TOKENS, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c85ep2_tables"
C85U_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c85u-candidate-utility-v2/"
    "c85u-v2-77382c16a593f7c2-91a428488a634268"
)
SELECTION_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84s-analysis-v5/stage_b_selection_freeze"
)
RESULT_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84s-analysis-v5/stage_c_scientific_result"
)
C85T_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c85t-synthetic-v3/"
    "c85t-v3-3ee51a994969ebaa-9ec012bedbf24f1f"
)
C85V_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c85v-proof-review-v1/"
    "c85v-35cd029ba9cf6859-c83191ae05834c5d"
)


class C85EP2ReadinessError(RuntimeError):
    """Raised when a frozen input or readiness schema drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C85EP2ReadinessError(message)


def _artifact_rows(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = manifest.get("artifacts")
    if isinstance(value, dict):
        return list(value.values())
    _require(isinstance(value, list), "artifact registry is malformed")
    return list(value)


def _row(object_id: str, path: Path, role: str, *, expected_sha: str | None = None,
         expected_size: int | None = None) -> dict[str, Any]:
    target = path.resolve()
    _require(target.is_file(), f"frozen C85E input absent: {object_id}")
    lowered = str(target).lower()
    _require(not any(token in lowered for token in FORBIDDEN_PATH_TOKENS),
             f"forbidden direct-data path in C85E registry: {target}")
    observed_sha = sha256_file(target)
    observed_size = target.stat().st_size
    if expected_sha is not None:
        _require(observed_sha == expected_sha, f"frozen C85E input SHA drift: {object_id}")
    if expected_size is not None:
        _require(observed_size == expected_size, f"frozen C85E input size drift: {object_id}")
    return {
        "object_id": object_id, "path": str(target), "size_bytes": observed_size,
        "sha256": observed_sha, "semantic_role": role, "runtime_access": "READ_ONLY",
    }


def collect_frozen_input_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    u1 = C85U_ROOT / "stage_u1_candidate_utility_v2"
    compatibility = json.loads((u1 / "C85U_CANDIDATE_UTILITY_MANIFEST.json").read_text())
    for name, role in (
        ("C85U_CANDIDATE_UTILITY_MANIFEST.json", "C85U_U1_COMPATIBILITY_MANIFEST"),
        ("C85U_CANDIDATE_UTILITY_MANIFEST_V2.json", "C85U_U1_ACCEPTED_MANIFEST"),
        ("C85U_STAGE_U1_HANDOFF.json", "C85U_U1_HANDOFF"),
        ("candidate_utility_index.csv", "C85U_CANDIDATE_UTILITY_INDEX"),
    ):
        rows.append(_row(f"C85U_U1_{name}", u1 / name, role))
    for item in compatibility["context_artifacts"]:
        rows.append(_row(
            f"C85U_UTILITY_CONTEXT_{item['context_id']}", u1 / item["path"],
            "C85U_HELD_EVALUATION_UTILITY_CONTEXT",
            expected_sha=str(item["sha256"]), expected_size=int(item["bytes"]),
        ))

    u2 = C85U_ROOT / "stage_u2_historical_replay_v2"
    for name, role in (
        ("C85U_HISTORICAL_DECISION_REPLAY_V2.json", "C85U_U2_ENDPOINT_REPLAY"),
        ("C85U_STAGE_U2_HANDOFF.json", "C85U_U2_HANDOFF"),
    ):
        rows.append(_row(f"C85U_U2_{name}", u2 / name, role))
    acceptance = C85U_ROOT / "final_acceptance_bundle"
    for path in sorted(acceptance.iterdir()):
        if path.is_file():
            rows.append(_row(
                f"C85U_ACCEPTANCE_{path.name}", path, "C85U_FINAL_ACCEPTANCE_CONTROL",
            ))

    for name, role in (
        ("C84S_SELECTION_FREEZE_MANIFEST_V3.json", "C84S_SELECTION_MANIFEST"),
        ("candidate_ranks.csv", "C84S_FROZEN_SCORE_METHOD_ACTIONS"),
        ("fixed_default_selections.csv", "C84S_FROZEN_FIXED_ACTIONS"),
        ("q0_selection_shard_index.csv", "C84S_FROZEN_Q0_SHARD_INDEX"),
    ):
        rows.append(_row(f"C84S_STAGE_B_{name}", SELECTION_ROOT / name, role))
    with (SELECTION_ROOT / "q0_selection_shard_index.csv").open(newline="", encoding="utf-8") as handle:
        q0_rows = list(csv.DictReader(handle))
    _require(len(q0_rows) == 944, "C85E Q0 shard registry coverage drift")
    for item in q0_rows:
        rows.append(_row(
            f"C84S_Q0_SHARD_{item['context_id']}", SELECTION_ROOT / item["path"],
            "C84S_FROZEN_Q0_ACTION_SHARD",
            expected_sha=item["sha256"], expected_size=int(item["bytes"]),
        ))

    result_manifest = json.loads(
        (RESULT_ROOT / "C84S_RESULT_ARTIFACT_MANIFEST.json").read_text(encoding="utf-8")
    )
    result_by_name = {Path(str(item["path"])).name: item for item in _artifact_rows(result_manifest)}
    compact_names = (
        "C84S_RESULT.json", "C84S_RESULT_ARTIFACT_MANIFEST.json",
        "method_context_decisions.csv", "target_level_method_effects.csv",
        "dataset_Q1_Q2.csv", "level_specific_Q1_Q2.csv", "panel_seed_stability.csv",
        "leave_one_target_out.csv", "label_budget_context.csv", "label_budget_frontier.csv",
        "measurement_vs_decision.csv", "selected_regime_distribution.csv",
        "q0_selected_regime_distribution.csv",
    )
    for name in compact_names:
        item = result_by_name.get(name)
        rows.append(_row(
            f"C84S_RESULT_{name}", RESULT_ROOT / name, "C84S_FROZEN_COMPACT_RESULT",
            expected_sha=None if item is None else str(item["sha256"]),
            expected_size=None,
        ))

    c84a_dir = REPORT_DIR / "c84a_tables"
    for path in sorted(c84a_dir.glob("*.csv")):
        rows.append(_row(f"C84A_COMPACT_{path.name}", path, "C84A_POST_SCIENTIFIC_COMPACT_TABLE"))
    for name in (
        "C84A_POST_SCIENTIFIC_HETEROGENEITY_AUDIT.json",
        "C84A_PM_REALIZED_POLICY_USE_ADDENDUM.json",
    ):
        rows.append(_row(f"C84A_REPORT_{name}", REPORT_DIR / name, "C84A_COMPACT_REPORT"))

    for name in (
        "C85T_RESULT.json", "C85T_RESULT_ARTIFACT_MANIFEST.json",
        "C85T_V3_COMPLETION_RECEIPT.json", "C85T_V3_SEMANTIC_REPLAY_RECEIPT.json",
    ):
        rows.append(_row(f"C85T_{name}", C85T_ROOT / name, "C85T_FROZEN_SYNTHETIC_RESULT"))
    for name in (
        "C85V_RESULT.json", "C85V_RESULT_ARTIFACT_MANIFEST.json",
        "C85V_COMPLETION_RECEIPT.json", "adjudication/formal_theorem_status_registry.csv",
    ):
        rows.append(_row(
            f"C85V_{name.replace('/', '_')}", C85V_ROOT / name,
            "C85V_FROZEN_THEOREM_STATUS_RESULT",
        ))

    for name, role in (
        ("C85E_FROZEN_FIELD_DECISION_THEORY_BRIDGE_PROTOCOL.json", "C85E_ANALYSIS_PROTOCOL"),
        ("C85EP2_EXECUTABLE_SEMANTICS_AND_INPUT_REPLAY_PROTOCOL.json", "C85EP2_EXECUTABLE_SEMANTICS"),
        ("C85EP2_C85U_INPUT_ACCEPTANCE_CERTIFICATE.json", "C85EP2_INPUT_ACCEPTANCE"),
    ):
        rows.append(_row(f"REPOSITORY_{name}", REPORT_DIR / name, role))
    paths = [row["path"] for row in rows]
    ids = [row["object_id"] for row in rows]
    _require(len(paths) == len(set(paths)) and len(ids) == len(set(ids)),
             "C85E frozen input registry duplicates")
    return rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    values = [dict(row) for row in rows]
    _require(bool(values), f"refusing empty readiness table: {path}")
    fields = list(values[0])
    _require(all(list(row) == fields for row in values), f"readiness table schema drift: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)
    return sha256_file(path)


def static_isolation_rows() -> list[dict[str, Any]]:
    module_names = (
        "c85e_policy_use.py", "c85e_action_geometry.py", "c85e_rank_topk_regret.py",
        "c85e_robust_risk.py", "c85e_theorem_bridge.py", "c85e_result_manifest.py",
        "c85e_runtime_guard.py", "c85e_execute.py",
    )
    forbidden_imports = (
        "torch", "mne", "moabb", "c84s_label_views", "c84s_q0_budget",
        "c84sr3_stage_c_evaluation", "c84s_inference", "c84s_selectors",
    )
    rows = []
    for name in module_names:
        path = REPO_ROOT / "oaci/theory" / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        )
        matches = sorted(value for value in imports if value.startswith(forbidden_imports))
        rows.append({
            "module": name, "forbidden_import_count": len(matches),
            "forbidden_imports": "|".join(matches),
            "status": "PASS" if not matches else "FAIL",
        })
    return rows


def write_readiness_tables(table_dir: Path = TABLE_DIR) -> dict[str, Any]:
    registry = collect_frozen_input_rows()
    policy = [{
        "semantic_role": role, "allowed": 1, "mode": "READ_ONLY",
        "notes": "Exact registry-bound frozen or compact object; no recomputation callable",
    } for role in sorted({row["semantic_role"] for row in registry})] + [
        {"semantic_role": role, "allowed": 0, "mode": "FORBIDDEN", "notes": "No runtime path may be registered"}
        for role in (
            "DIRECT_EVALUATION_LABEL_VIEW", "DIRECT_CONSTRUCTION_LABEL_VIEW",
            "TARGET_LOGITS", "EEG", "SOURCE_ARRAYS", "MODEL_CHECKPOINT",
        )
    ]
    result_registry = [{
        "table": name, "result_tag": "POST_C84S_EXPLORATORY",
        "publication": "ATOMIC_FINAL_BUNDLE_ONLY", "new_pvalues": 0,
    } for name in REGISTERED_TABLES]
    isolation = static_isolation_rows()
    hashes = {
        "c85e_frozen_input_registry.csv": _write_csv(table_dir / "c85e_frozen_input_registry.csv", registry),
        "c85e_runtime_file_open_policy.csv": _write_csv(table_dir / "c85e_runtime_file_open_policy.csv", policy),
        "result_table_registry.csv": _write_csv(table_dir / "result_table_registry.csv", result_registry),
        "implementation_static_isolation_audit.csv": _write_csv(
            table_dir / "implementation_static_isolation_audit.csv", isolation,
        ),
    }
    _require(all(row["status"] == "PASS" for row in isolation), "C85E static isolation failed")
    return {
        "rows": len(registry), "bytes": sum(int(row["size_bytes"]) for row in registry),
        "utility_context_rows": sum(row["semantic_role"] == "C85U_HELD_EVALUATION_UTILITY_CONTEXT" for row in registry),
        "Q0_shard_rows": sum(row["semantic_role"] == "C84S_FROZEN_Q0_ACTION_SHARD" for row in registry),
        "hashes": hashes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    summary = write_readiness_tables()
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "C85EP2ReadinessError", "collect_frozen_input_rows", "static_isolation_rows",
    "write_readiness_tables",
]
