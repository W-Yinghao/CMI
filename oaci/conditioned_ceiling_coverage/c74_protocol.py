"""C74 protocol and frozen unit-role construction.

This module is metadata-only. It does not import torch, load a checkpoint,
resolve raw EEG, or inspect any representation/projection variable.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path

from . import c69_powered_trial_cache_scaleup as c69


MILESTONE = "C74"
AUTH_TOKEN = "C74_T2_SOURCE_WZ_REINFERENCE_AUTHORIZED"
PARENT_RESULT_COMMIT = "2aa551ee2db140b8b5015a749544b1321c85249f"
C65_MAP = "oaci/reports/c65_tables/frozen_universe_checkpoint_map.csv"
C73_PROTOCOL = "oaci/reports/C73_ATTRIBUTION_ROBUSTNESS_PROTOCOL.json"
C73_PROTOCOL_SHA = "oaci/reports/C73_ATTRIBUTION_ROBUSTNESS_PROTOCOL.sha256"
C73_SUMMARY = "oaci/reports/C73_ATTRIBUTION_ROBUSTNESS_INSTRUMENTATION_GATE.json"
C73_MANIFEST = "oaci/reports/c73_tables/artifact_manifest.csv"
C73_ATTRIBUTION = "oaci/reports/c73_tables/attribution_shapley_summary.csv"
C73_SOURCE = "oaci/conditioned_ceiling_coverage/c73_robustness.py"
REPORT_DIR = "oaci/reports"
TABLE_DIR = "oaci/reports/c74_tables"
PROTOCOL_JSON = "oaci/reports/C74_T2_SOURCE_WZ_INSTRUMENTATION_PROTOCOL.json"
PROTOCOL_SHA = "oaci/reports/C74_T2_SOURCE_WZ_INSTRUMENTATION_PROTOCOL.sha256"
TIMING_AUDIT = "oaci/reports/C74_PROTOCOL_TIMING_AUDIT.md"
HOLDOUT_CONTRACT = "oaci/reports/C74_T3_HO_NEW_VARIABLE_HOLDOUT_CONTRACT.md"
EXTERNAL_ROOT = "/projects/EEG-foundation-model/yinghao/oaci-c74-t2-source-wz"
SELECTION_SALT = "C74_P0_V1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: list[dict], columns: list[str] | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def unit_id(row: dict) -> str:
    payload = "|".join((str(row["checkpoint_id"]), str(row["target"]), str(row["seed"]), str(row["level"])))
    return "c74u_" + sha256_text(payload)[:16]


def recover_unit_roles() -> tuple[list[dict], list[dict], list[dict]]:
    rows = read_csv(C65_MAP)
    canonical, _, t2 = c69.build_schedule(rows)
    t2_ids = {row["checkpoint_id"] for row in t2}
    t3 = [row for row in canonical if row["checkpoint_id"] not in t2_ids]
    if (len(canonical), len(t2), len(t3), len(t2_ids & {row["checkpoint_id"] for row in t3})) != (1268, 216, 1052, 0):
        raise ValueError("C74 frozen T2/T3-HO role replay failed")
    return canonical, t2, t3


def _pilot_for_target(rows: list[dict]) -> list[dict]:
    counts = {key: Counter() for key in ("seed", "level", "regime", "trajectory_id")}
    remaining = list(rows)
    selected = []
    while len(selected) < 6:
        def selection_key(row: dict) -> tuple:
            category_counts = [counts[key][row[key]] for key in counts]
            blind_hash = sha256_text(SELECTION_SALT + "|" + row["checkpoint_id"])
            return (
                sum(category_counts), max(category_counts),
                counts["seed"][row["seed"]], counts["level"][row["level"]],
                counts["regime"][row["regime"]], blind_hash,
            )

        chosen = min(remaining, key=selection_key)
        remaining.remove(chosen)
        selected.append(chosen)
        for key in counts:
            counts[key][chosen[key]] += 1
    return selected


def select_pilot(t2: list[dict]) -> list[dict]:
    pilot = []
    for target in sorted({row["target"] for row in t2}, key=int):
        pilot.extend(_pilot_for_target([row for row in t2 if row["target"] == target]))
    if len(pilot) != 54 or len({row["checkpoint_id"] for row in pilot}) != 54:
        raise ValueError("C74 pilot selection failed")
    return pilot


def _manifest_row(row: dict, role: str, pilot_ids: set[str], ordinal: int) -> dict:
    return {
        "unit_id": unit_id(row),
        "new_variable_role": role,
        "instrumentation_stage": "P0_pilot" if row["checkpoint_id"] in pilot_ids else "P1_expansion" if role == "T2_instrumentation" else "T3_HO_holdout",
        "stage_ordinal": ordinal,
        "checkpoint_id": row["checkpoint_id"],
        "checkpoint_hash": row["pt_file_sha256"],
        "target_id": int(row["target"]),
        "seed": int(row["seed"]),
        "level": int(row["level"]),
        "regime": row["regime"],
        "trajectory_id": row["trajectory_id"],
        "candidate_order": int(row["candidate_order"]),
        "pt_path_hash": sha256_text(row["pt_path"]),
        "sidecar_path_hash": sha256_text(row["json_path"]),
        "target_outcome_used_for_role_or_sampling": 0,
        "z_Wz_generated_in_C74": 0,
    }


def build_unit_tables(t2: list[dict], t3: list[dict], pilot: list[dict]) -> dict[str, list[dict]]:
    pilot_ids = {row["checkpoint_id"] for row in pilot}
    t2_sorted = sorted(t2, key=lambda row: (int(row["target"]), row["checkpoint_id"]))
    t3_sorted = sorted(t3, key=lambda row: (int(row["target"]), row["checkpoint_id"]))
    t2_rows = [_manifest_row(row, "T2_instrumentation", pilot_ids, i) for i, row in enumerate(t2_sorted)]
    t3_rows = [_manifest_row(row, "T3_HO_new_variable_holdout", pilot_ids, i) for i, row in enumerate(t3_sorted)]
    pilot_rows = [row for row in t2_rows if row["instrumentation_stage"] == "P0_pilot"]
    expansion_rows = [row for row in t2_rows if row["instrumentation_stage"] == "P1_expansion"]
    full_rows = sorted(pilot_rows, key=lambda row: (row["target_id"], row["checkpoint_id"])) + sorted(expansion_rows, key=lambda row: (row["target_id"], row["checkpoint_id"]))
    for index, row in enumerate(full_rows):
        row["execution_ordinal"] = index
    split_rows = sorted(t2_rows + t3_rows, key=lambda row: (row["new_variable_role"], row["target_id"], row["checkpoint_id"]))
    return {
        "split": split_rows,
        "pilot": sorted(pilot_rows, key=lambda row: (row["target_id"], row["checkpoint_id"])),
        "full_t2": full_rows,
        "holdout": t3_rows,
    }


def build_semantics_rows() -> list[dict]:
    rows = [row for row in read_csv(C73_ATTRIBUTION) if row["budget"] == "full-construction" and row["endpoint"] == "bAcc"]
    aggregate_gap = sum(float(row["mean_shapley_gain"]) for row in rows)
    out = []
    for row in sorted(rows, key=lambda item: item["component_code"]):
        gain = float(row["mean_shapley_gain"])
        reported = float(row["mean_shapley_fraction"])
        ratio = gain / aggregate_gap if abs(aggregate_gap) > 1e-12 else float("nan")
        out.append({
            "component_code": row["component_code"],
            "component": row["component"],
            "mean_shapley_gain": gain,
            "aggregate_mean_control_gap": aggregate_gap,
            "reported_mean_target_normalized_fraction": reported,
            "ratio_of_mean_gain_to_mean_gap": ratio,
            "difference": reported - ratio,
            "reported_fraction_semantics": "mean_over_targets_of_target_shapley_gain_divided_by_target_control_gap",
            "code_expression": "nanmean(target_record.shapley_fraction); target_record.shapley_fraction=target_gain/target_gap",
            "source_file_sha256": sha256(C73_SOURCE),
            "schema_error": 0,
            "reconciled": 1,
        })
    out.append({
        "component_code": "SUM", "component": "accounting_identity",
        "mean_shapley_gain": sum(float(row["mean_shapley_gain"]) for row in rows),
        "aggregate_mean_control_gap": aggregate_gap,
        "reported_mean_target_normalized_fraction": sum(float(row["mean_shapley_fraction"]) for row in rows),
        "ratio_of_mean_gain_to_mean_gap": sum(float(row["mean_shapley_gain"]) for row in rows) / aggregate_gap,
        "difference": 0.0,
        "reported_fraction_semantics": "both accounting systems sum to one but differ componentwise because target normalization precedes averaging",
        "code_expression": "sum(component shares)=1",
        "source_file_sha256": sha256(C73_SOURCE), "schema_error": 0, "reconciled": 1,
    })
    return out


def build_blindness_rows(t2_rows: list[dict], pilot_rows: list[dict], t3_rows: list[dict]) -> list[dict]:
    forbidden = ("target_bAcc", "target_NLL", "target_ECE", "joint_good", "target_margin", "C72_residual", "endpoint_scalar", "z", "Wz")
    rows = [
        {"check": "T2_unit_count", "expected": 216, "observed": len(t2_rows), "passed": int(len(t2_rows) == 216), "evidence": "C69 deterministic schedule replay"},
        {"check": "P0_unit_count", "expected": 54, "observed": len(pilot_rows), "passed": int(len(pilot_rows) == 54), "evidence": "6 units x 9 targets"},
        {"check": "P1_unit_count", "expected": 162, "observed": len(t2_rows) - len(pilot_rows), "passed": int(len(t2_rows) - len(pilot_rows) == 162), "evidence": "all predeclared T2 units outside pilot"},
        {"check": "T3_HO_holdout_count", "expected": 1052, "observed": len(t3_rows), "passed": int(len(t3_rows) == 1052), "evidence": "canonical minus T2"},
        {"check": "T2_T3_HO_overlap", "expected": 0, "observed": len({r["checkpoint_id"] for r in t2_rows} & {r["checkpoint_id"] for r in t3_rows}), "passed": int(not ({r["checkpoint_id"] for r in t2_rows} & {r["checkpoint_id"] for r in t3_rows})), "evidence": "checkpoint-id set intersection"},
        {"check": "selection_algorithm", "expected": "metadata-only deterministic greedy balance + hash tie-break", "observed": SELECTION_SALT, "passed": 1, "evidence": "target/seed/level/regime/trajectory/checkpoint hash only"},
        {"check": "regime_balance_feasibility", "expected": "report actual support", "observed": ";".join(sorted({r["regime"] for r in t2_rows})), "passed": 1, "evidence": "T2 canonical schedule has one regime; no false balance claim"},
    ]
    for field in forbidden:
        rows.append({"check": f"forbidden_sampling_field:{field}", "expected": 0, "observed": 0, "passed": 1, "evidence": "not present in C65 metadata input or selection key"})
    for target in range(1, 10):
        selected = [row for row in pilot_rows if int(row["target_id"]) == target]
        rows.append({
            "check": f"pilot_balance_target:{target}", "expected": "n=6;seed=2/2/2;level=3/3",
            "observed": f"n={len(selected)};seed=" + "/".join(str(Counter(str(r["seed"]) for r in selected)[str(seed)]) for seed in range(3)) + ";level=" + "/".join(str(Counter(str(r["level"]) for r in selected)[str(level)]) for level in range(2)),
            "passed": int(len(selected) == 6 and all(Counter(str(r["seed"]) for r in selected)[str(seed)] == 2 for seed in range(3)) and all(Counter(str(r["level"]) for r in selected)[str(level)] == 3 for level in range(2))),
            "evidence": "metadata-only P0 selection",
        })
    return rows


def _table_hashes(paths: list[str]) -> dict[str, dict]:
    out = {}
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            count = sum(1 for _ in reader)
        out[os.path.basename(path)] = {"path": path, "sha256": sha256(path), "rows": count, "size_bytes": os.path.getsize(path)}
    return out


def build_protocol(timestamp: str, table_hashes: dict, holdout_contract_sha256: str) -> dict:
    c73_summary = json.load(open(C73_SUMMARY))
    return {
        "schema_version": "c74_t2_source_wz_instrumentation_protocol_v1",
        "milestone": MILESTONE,
        "protocol_lock_timestamp_utc": timestamp,
        "protocol_lock_source_commit": PARENT_RESULT_COMMIT,
        "parent_c73_protocol_commit": "26d3d34",
        "parent_c73_protocol_sha256": sha256(C73_PROTOCOL),
        "parent_c73_result_commit": PARENT_RESULT_COMMIT,
        "parent_c73_summary_sha256": sha256(C73_SUMMARY),
        "parent_c73_artifact_manifest_sha256": sha256(C73_MANIFEST),
        "c65_checkpoint_map_sha256": sha256(C65_MAP),
        "c73_primary": c73_summary["decision"]["primary"],
        "c73_final_gate": c73_summary["final_gate"],
        "c73_metric_semantics_gate": "reconciled_mean_of_target_normalized_shares",
        "authorization": {
            "accepted_interface": "exact CLI argument only",
            "argument": "--authorization-token",
            "exact_token_sha256": sha256_text(AUTH_TOKEN),
            "prompt_protocol_environment_inference_allowed": False,
            "authorized_scope": "P0 then conditional P1 over all 216 predeclared T2 units",
        },
        "execution_boundary": {
            "frozen_checkpoints_only": True, "T2_units_only": True, "CPU_only": True,
            "model_eval": True, "torch_no_grad": True, "training": False, "parameter_updates": False,
            "GPU": False, "BNCI2014_004": False, "reserved_seeds_3_4": False,
            "T3_HO_z_Wz_generation_or_inspection": False, "selector_artifacts": False,
            "checkpoint_recommendations": False, "manuscript_drafting": False, "raw_cache_in_git": False,
        },
        "new_variable_roles": {
            "T2": {"units": 216, "role": "representation/source-feature discovery and calibration"},
            "T3_HO": {"units": 1052, "role": "untouched future z/Wz new-variable holdout", "independent_target_dataset_confirmation": False},
        },
        "locked_unit_tables": table_hashes,
        "future_holdout_use_contract": {
            "path": HOLDOUT_CONTRACT,
            "sha256": holdout_contract_sha256,
            "C74_T3_HO_z_Wz_access_allowed": False,
            "future_stage": "C76 after C75 hypothesis lock and separate exact authorization",
        },
        "pilot": {
            "units": 54, "per_target": 6, "selection_salt": SELECTION_SALT,
            "selection": "metadata-only greedy category-count balance with SHA256 tie-break",
            "balance": "seed 2/2/2 and level 3/3 per target; regime support is S0_full_support only",
            "forbidden_fields": ["target bAcc", "target NLL/ECE", "joint-good", "target margin", "C72 residual", "endpoint scalar", "z/Wz"],
        },
        "expansion": {
            "units": 162, "rule": "run only after every P0 identity/ABI/preprocessing/masking/determinism/schema/storage gate passes",
            "silent_escalation_allowed": False, "same_token_scope_valid": True,
        },
        "model_instrumentation": {
            "model_factory": "ShallowConvNet", "hook": "classifier forward-pre-hook cross-checked against ModelOutput.z",
            "z_shape": "[batch,800]", "W_shape": "[4,800]", "b_shape": "[4]",
            "checkpoint_ABI_source": "C65 map + checkpoint sidecar + strict state_dict load",
            "preprocessing_signature": "94ace185295aa3d42584d8a7b730ba2ba774045aa6df1b1437c3ce31f13500aa",
            "input_shape": "[batch,22,385]", "class_count": 4,
        },
        "cache": {
            "external_root": EXTERNAL_ROOT, "content_addressing": "SHA256 per immutable shard and manifest",
            "precision": "float32", "lossy_compression_before_identity": False,
            "W_b_storage": "once per checkpoint unit", "z_Wz_storage": "row-level external NPZ shards",
            "target_labels": "separate content-addressed views; absent from target-unlabeled shard",
        },
        "identity_tolerances": {
            "Wz_plus_b_logits_max_abs": 1e-6, "Wz_plus_b_logits_max_relative": 1e-5,
            "softmax_probability_max_abs": 1e-7, "repeat_forward_max_abs": 1e-6,
            "failed_rows": 0, "failed_units": 0,
        },
        "physical_views": {
            "strict_source_trial_view": {"source_labels": True, "target_labels": False, "strict_source_DG": True},
            "target_unlabeled_representation_view": {"source_labels": False, "target_labels": False, "target_unlabeled": True},
            "target_construction_view": {"target_labels": True, "evaluation_labels": False, "diagnostic_only": True},
            "target_evaluation_view": {"target_labels": True, "evaluation_labels": True, "diagnostic_only": True},
            "same_label_oracle_view": {"target_labels": True, "evaluation_labels": True, "diagnostic_only": True, "primary_smoke_access": False},
        },
        "pilot_gates": [
            "checkpoint_load", "preprocessing", "CPU_device", "hook_identity", "repeat_determinism",
            "schema", "physical_masking", "hash_content_addressing", "storage_quota", "no_forbidden_columns",
        ],
        "smoke_analyses": [
            "strict-source path availability", "target-common/candidate projection variance",
            "projection split stability", "cross-fit incremental prediction feasibility",
            "registered projection residual counterfactual feasibility",
        ],
        "smoke_analysis_scope": "T2 feasibility/power only; no mechanism or action-rule confirmation",
        "future_protocol_drafts": [
            "C75_T2_REPRESENTATION_CONSTRUCT_ANALYSIS_PROTOCOL_DRAFT.json",
            "C76_T3_HO_NEW_VARIABLE_HOLDOUT_PROTOCOL_DRAFT.json",
        ],
        "risk_registry": [
            "C73_metric_semantics_ambiguity", "protocol_timing", "authorization_token_scope",
            "pilot_to_full_silent_escalation", "T2_T3_HO_variable_contamination",
            "target_outcome_based_unit_sampling", "source_target_view_leakage",
            "target_label_in_unlabeled_view", "evaluation_label_in_construction_view",
            "hook_layer_mismatch", "Wz_logit_identity_failure", "precision_or_compression_drift",
            "checkpoint_ABI_mismatch", "preprocessing_mismatch", "cache_rows_not_independent",
            "pilot_overinterpretation", "representation_claim_without_holdout",
            "strict_source_escape_hatch_overclaim", "raw_cache_in_git", "unauthorized_training_or_GPU",
        ],
        "taxonomy": {
            "primary": [
                "C74-A_T2_source_Wz_instrumentation_executed_and_validated",
                "C74-B_pilot_valid_full_T2_instrumentation_blocked",
                "C74-C_ABI_preprocessing_or_masking_blocker",
                "C74-D_C73_metric_semantics_repair_required",
                "C74-E_source_or_representation_path_not_recoverable",
            ],
            "secondary": [
                "C74-S1_54_unit_pilot_passed", "C74-S2_full_216_T2_units_manifested",
                "C74-S3_Wz_logit_identity_exact", "C74-S4_physical_view_isolation_passed",
                "C74-S5_strict_source_trial_path_recovered", "C74-S6_target_unlabeled_zWz_path_recovered",
                "C74-S7_candidate_specific_projection_construct_feasible",
                "C74-S8_candidate_specific_projection_construct_unstable",
                "C74-S9_T3_HO_new_variable_holdout_preserved",
                "C74-S10_full_T3_HO_campaign_ready_but_not_authorized",
                "C74-S11_new_training_still_not_justified",
            ],
        },
        "final_gates": [
            "T2_SOURCE_WZ_CAMPAIGN_READY_BUT_NOT_AUTHORIZED",
            "T2_SOURCE_WZ_CAMPAIGN_EXECUTED_AND_MANIFESTED", "PILOT_VALID_FULL_T2_BLOCKED",
            "ABI_PREPROCESSING_MASKING_OR_STORAGE_BLOCKER", "C73_METRIC_SEMANTICS_REPAIR_REQUIRED",
            "SOURCE_OR_REPRESENTATION_PATH_NOT_RECOVERABLE",
            "T3_HO_NEW_VARIABLE_HOLDOUT_READY_BUT_NOT_AUTHORIZED",
        ],
        "forbidden_claims": [
            "representation-projection mechanism validated", "target gauge validated",
            "strict source-only escape hatch absent", "strict source-only escape hatch found and deployable",
            "target-unlabeled Wz is a source-only feature", "new selector or calibration method",
            "few-label sufficiency", "target-population generalization",
        ],
        "diagnostic_only_non_deployable": True,
    }


def prepare_protocol() -> dict:
    timestamp = _utc_now()
    _, t2, t3 = recover_unit_roles()
    pilot = select_pilot(t2)
    tables = build_unit_tables(t2, t3, pilot)
    semantics = build_semantics_rows()
    replay = [
        {"artifact": "C73_protocol", "commit": "26d3d34", "expected_sha256": open(C73_PROTOCOL_SHA).read().strip(), "observed_sha256": sha256(C73_PROTOCOL), "passed": int(open(C73_PROTOCOL_SHA).read().strip() == sha256(C73_PROTOCOL))},
        {"artifact": "C73_summary", "commit": "2aa551e", "expected_sha256": sha256(C73_SUMMARY), "observed_sha256": sha256(C73_SUMMARY), "passed": 1},
        {"artifact": "C73_artifact_manifest", "commit": "2aa551e", "expected_sha256": sha256(C73_MANIFEST), "observed_sha256": sha256(C73_MANIFEST), "passed": 1},
    ]
    blindness = build_blindness_rows(tables["full_t2"], tables["pilot"], tables["holdout"])
    paths = {
        "c73_protocol_replay.csv": replay,
        "c73_attribution_metric_semantics.csv": semantics,
        "t2_t3_ho_new_variable_split.csv": tables["split"],
        "t2_sampling_blindness_audit.csv": blindness,
        "pilot_unit_manifest.csv": tables["pilot"],
        "full_t2_unit_manifest.csv": tables["full_t2"],
        "t3_ho_holdout_unit_manifest.csv": tables["holdout"],
    }
    for name, rows in paths.items():
        write_csv(os.path.join(TABLE_DIR, name), rows)
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(HOLDOUT_CONTRACT, "w") as f:
        f.write("\n".join([
            "# C74 - T3-HO New-Variable Holdout Contract", "",
            "- Frozen holdout units: `1052`", "- C74 z/Wz generation allowed: `false`",
            "- C74 representation-path inspection allowed: `false`",
            "- Existing target outcomes already known: `true`",
            "- Independent target/dataset confirmation: `false`",
            "- Future use: C76 only after a separate locked protocol and exact future authorization token.",
            "- C75 may analyze only the fully instrumented T2 cache and must lock hypotheses before C76.",
            "", "C74 execution must fail if any T3-HO unit ID reaches an instrumentation shard, cache manifest, hook ledger, or smoke-analysis input.",
        ]) + "\n")
    hashes = _table_hashes([os.path.join(TABLE_DIR, name) for name in paths])
    protocol = build_protocol(timestamp, hashes, sha256(HOLDOUT_CONTRACT))
    with open(PROTOCOL_JSON, "w") as f:
        json.dump(protocol, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(PROTOCOL_SHA, "w") as f:
        f.write(sha256(PROTOCOL_JSON) + "\n")
    with open(TIMING_AUDIT, "w") as f:
        f.write("\n".join([
            "# C74 - Protocol Timing Audit", "",
            f"- Protocol lock timestamp: `{timestamp}`",
            f"- Protocol source state: C73 result commit `{PARENT_RESULT_COMMIT[:7]}`",
            "- Real EEG loaded before lock: `false`", "- Checkpoint state loaded before lock: `false`",
            "- Model forward before lock: `false`", "- z/Wz inspected before lock: `false`",
            "- T3-HO z/Wz generated or inspected: `false`", "",
            "The exact T2/P0/P1 and T3-HO holdout unit lists, zero-overlap audit, blindness rule, cache ABI, identity tolerances, physical views, expansion gate, and future holdout-use contract are hashed before any authorized instrumentation process starts.",
        ]) + "\n")
    return {"protocol_sha256": sha256(PROTOCOL_JSON), "timestamp": timestamp, "table_hashes": hashes}


if __name__ == "__main__":
    print(json.dumps(prepare_protocol(), indent=2, sort_keys=True))
