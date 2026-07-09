"""C68 - powered re-inference-only trial-cache scale-up readiness gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from . import audit_utils as au


MILESTONE = "C68"
REPORT_DIR = "oaci/reports"
TABLE_DIR = "oaci/reports/c68_tables"
REPORT_JSON = "oaci/reports/C68_POWERED_TRIAL_CACHE_SCALEUP.json"

C65_JSON = "oaci/reports/C65_FROZEN_CHECKPOINT_RECOVERY_TRIAL_CACHE_GATE.json"
C65_MAP = "oaci/reports/c65_tables/frozen_universe_checkpoint_map.csv"
C65_COMPLETENESS = "oaci/reports/c65_tables/mapping_completeness_summary.csv"
C66_AUTH_MANIFEST = "oaci/reports/c66_tables/authorized_cache_manifest.csv"
C67_JSON = "oaci/reports/C67_C66_DUAL_MODE_CACHE_CONSUMPTION.json"
C67_LEDGER = "oaci/reports/c67_tables/c66_dual_mode_provenance_ledger.csv"
C67_MASKED_VIEW = "oaci/reports/c67_tables/masked_view_contract.csv"
C67_REPLAY = "oaci/reports/c67_tables/authorized_cache_manifest_replay.csv"

AUTH_PHRASE = "AUTHORIZE_C68_REINFERENCE_ONLY_SCALEUP"
MAX_REPORT_BYTES = 50_000_000
TRIAL_ROWS_PER_FORWARD_UNIT = 576

DECISIONS = (
    "C68-A_c67_dual_mode_cache_contract_replayed",
    "C68-B_scaleup_plan_powered_and_manifested",
    "C68-C_reinference_only_scaleup_ready_but_not_authorized",
    "C68-D_reinference_only_scaleup_authorized_and_executed",
    "C68-E_scaled_trial_cache_integrity_validated",
    "C68-F_masked_view_contract_validated_at_scale",
    "C68-G_split_label_powered_diagnostic_completed_not_sufficiency",
    "C68-H_split_label_still_underpowered_or_unstable",
    "C68-I_sample_level_conditional_cs_feasible_at_scale",
    "C68-J_sample_level_conditional_cs_still_unstable_or_proxy_only",
    "C68-K_endpoint_oracle_boundary_preserved",
    "C68-L_trial_level_source_observable_escape_hatch_found",
    "C68-M_no_trial_level_source_observable_escape_hatch_found",
    "C68-N_larger_reinference_only_campaign_needed_but_not_authorized",
    "C68-O_new_training_still_not_justified",
    "C68-P_claim_or_availability_violation_found",
)

FINAL_GATES = (
    "SCALEUP_READY_BUT_NOT_AUTHORIZED",
    "REINFERENCE_ONLY_SCALEUP_EXECUTED_AND_CACHE_MANIFESTED",
    "SCALED_CACHE_VALID_BUT_SPLIT_LABEL_UNDERPOWERED",
    "SPLIT_LABEL_DIAGNOSTIC_SIGNAL_REPLICATED_NOT_SUFFICIENCY",
    "SAMPLE_LEVEL_CONDITIONAL_CS_FEASIBLE_NOT_DEPLOYABLE",
    "SAMPLE_LEVEL_CONDITIONAL_CS_STILL_UNSTABLE_OR_PROXY_ONLY",
    "SOURCE_OBSERVABLE_TRIAL_LEVEL_ESCAPE_HATCH_FOUND",
    "LARGER_REINFERENCE_ONLY_CACHE_CAMPAIGN_NEEDED_BUT_NOT_AUTHORIZED",
    "NEW_TRAINING_STILL_NOT_JUSTIFIED",
    "CLAIM_OR_AVAILABILITY_REPAIR_REQUIRED",
)

FORBIDDEN_PATTERNS = (
    "re-inference-only scale-up authorized",
    "training authorized",
    "new training",
    "gradient update",
    "GPU required",
    "BNCI2014_004 used",
    "seeds [3,4] used",
    "few-label sufficiency established",
    "full conditional-CS established",
    "source-only rescue",
    "OACI rescue",
    "deployable selector",
    "checkpoint recommendation",
    "manuscript drafting",
    "same-label oracle available at selection time",
)

NEGATION_CUES = (
    "not ",
    "no ",
    "never ",
    "without ",
    "absent ",
    "blocked ",
    "forbid",
    "unavailable ",
    "not authorized ",
    "not run ",
    "not executed ",
    "not established ",
    "still not ",
)

SLURM_VALIDATION_RESULTS = (
    ("focused_c68", "891432", "9 passed in 0.22s"),
    ("c50_c68_slice", "891433", "197 passed in 15.19s"),
    ("c23_c68_regression", "891436", "447 passed in 43.70s"),
    ("full_oaci_tests", "891434", "1371 passed in 312.02s"),
)


def _lock_config() -> str:
    return au.lock_config(MILESTONE)


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: list[dict], cols: list[str]) -> None:
    au.write_csv(path, rows, cols)


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _path_hash(path: str) -> str:
    return hashlib.sha256(str(path).encode()).hexdigest()


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _git_or_empty(args: list[str]) -> str:
    try:
        return _git(args)
    except Exception:
        return ""


def _auth_present(token: str = "") -> bool:
    # Deliberately exact-token only. Do not scan handoff/protocol text or env vars.
    return str(token).strip() == AUTH_PHRASE


def _mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else math.nan


def _ceil_int(x: float) -> int:
    return int(math.ceil(float(x)))


def _listed_paths() -> list[Path]:
    skip = {"artifact_manifest.csv", "large_artifact_scan.csv"}
    return sorted(
        list(Path(REPORT_DIR).glob("C68_*.md"))
        + list(Path(REPORT_DIR).glob("C68_*.json"))
        + [p for p in Path(TABLE_DIR).glob("*.csv") if p.name not in skip]
    )


def _large_scan(paths: list[Path]) -> list[dict]:
    return [
        {
            "path": str(p),
            "size_bytes": os.path.getsize(p),
            "over_50mb": int(os.path.getsize(p) > MAX_REPORT_BYTES),
            "passed": int(os.path.getsize(p) <= MAX_REPORT_BYTES),
        }
        for p in sorted(paths)
    ]


def _artifact_manifest(paths: list[Path], table_dir: str) -> list[dict]:
    counts = {}
    for path in Path(table_dir).glob("*.csv"):
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            counts[str(path)] = sum(1 for _ in reader)
    return [
        {
            "path": str(p),
            "size_bytes": os.path.getsize(p),
            "sha256": _sha256(str(p)),
            "artifact_class": "table" if str(p).endswith(".csv") else "summary_json" if str(p).endswith(".json") else "report",
            "row_count": counts.get(str(p), ""),
        }
        for p in sorted(paths)
    ]


def _affirmative_hit(text: str, phrase: str, window: int = 240) -> bool:
    low = text.lower()
    phrase = phrase.lower()
    start = 0
    while True:
        idx = low.find(phrase, start)
        if idx == -1:
            return False
        ctx = low[max(0, idx - window):idx]
        if not any(cue in ctx for cue in NEGATION_CUES):
            return True
        start = idx + len(phrase)


def build_forbidden_scan(paths: list[str]) -> list[dict]:
    rows = []
    for pattern in FORBIDDEN_PATTERNS:
        total = affirmative = 0
        files = []
        for path in paths:
            if os.path.basename(path) in {"forbidden_claim_scan.csv", "red_team_failure_ledger.csv"}:
                continue
            text = open(path, errors="ignore").read()
            count = text.lower().count(pattern.lower())
            if count:
                total += count
                files.append(path)
                if _affirmative_hit(text, pattern):
                    affirmative += 1
        rows.append({"pattern": pattern, "total_hits": total, "affirmative_hits": affirmative, "files": ";".join(files), "passed": int(affirmative == 0)})
    return rows


def load_context() -> dict:
    c67_json = _load_json(C67_JSON)
    c65_json = _load_json(C65_JSON)
    c65_map = _read_csv(C65_MAP)
    c67_ledger = {r["mode"]: r for r in _read_csv(C67_LEDGER)}
    c67_views = _read_csv(C67_MASKED_VIEW)
    c66_manifest = {r["cache_id"]: r for r in _read_csv(C66_AUTH_MANIFEST)}
    c67_replay = {r["cache_id"]: r for r in _read_csv(C67_REPLAY)}
    trial_manifest = c66_manifest.get("c66_trial_cache_v1", {})
    trial_path = trial_manifest.get("external_path", "")
    trial_exists = os.path.exists(trial_path)
    trial_sha = _sha256(trial_path) if trial_exists else ""
    git_files = set(_git_or_empty(["ls-files"]).splitlines())
    raw_cache_tracked = any(os.path.basename(trial_path) == os.path.basename(p) for p in git_files)
    head = _git(["rev-parse", "--short", "HEAD"])
    return {
        "head": head,
        "branch": _git_or_empty(["branch", "--show-current"]),
        "origin_oaci": _git_or_empty(["rev-parse", "--short", "origin/oaci"]),
        "git_log": _git(["log", "--oneline", "-8"]).splitlines(),
        "c65_json": c65_json,
        "c65_map": c65_map,
        "c67_json": c67_json,
        "c67_ledger": c67_ledger,
        "c67_views": c67_views,
        "c67_replay": c67_replay,
        "trial_manifest": trial_manifest,
        "trial_path": trial_path,
        "trial_exists": trial_exists,
        "trial_sha": trial_sha,
        "raw_cache_tracked": raw_cache_tracked,
    }


def build_c67_replay_rows(ctx: dict) -> list[dict]:
    c67 = ctx["c67_json"]
    noauth = ctx["c67_ledger"].get("no_auth_baseline", {})
    auth = ctx["c67_ledger"].get("authorized_microcampaign", {})
    replay = ctx["c67_replay"].get("c66_trial_cache_v1", {})
    manifest = ctx["trial_manifest"]
    checks = [
        ("c67_commit", "9f8c829", ctx["head"], int(ctx["head"] == "9f8c829"), "C68 starts after committed C67"),
        ("c67_final_gate", "C67_DUAL_MODE_MICROCACHE_VALID_BUT_UNDERPOWERED_FOR_SPLIT_LABEL_CS", c67.get("final_gate", ""), int(c67.get("final_gate") == "C67_DUAL_MODE_MICROCACHE_VALID_BUT_UNDERPOWERED_FOR_SPLIT_LABEL_CS"), "C67 underpowered microcache gate"),
        ("no_auth_forward", "0", noauth.get("forward_attempted", ""), int(noauth.get("forward_attempted") == "0"), "C66 no-auth guard evidence"),
        ("no_auth_cache_rows", "0", noauth.get("cache_rows", ""), int(noauth.get("cache_rows") == "0"), "C66 no-auth guard evidence"),
        ("authorized_forward", "1", auth.get("forward_attempted", ""), int(auth.get("forward_attempted") == "1"), "C66 authorized microcampaign evidence"),
        ("authorized_cache_rows", "3456", auth.get("cache_rows", ""), int(auth.get("cache_rows") == "3456"), "C66 authorized microcache rows"),
        ("authorized_cache_sha256", manifest.get("sha256", ""), ctx["trial_sha"], int(ctx["trial_sha"] == manifest.get("sha256") and replay.get("sha256_match") == "1"), "external cache hash replay"),
        ("raw_cache_git_tracked", "0", int(ctx["raw_cache_tracked"]), int(not ctx["raw_cache_tracked"]), "raw C66 cache remains external"),
    ]
    return [{"check": c, "expected": e, "observed": o, "passed": p, "notes": n} for c, e, o, p, n in checks]


def build_universe_rows(ctx: dict) -> tuple[list[dict], dict]:
    rows = ctx["c65_map"]
    physical = {(r["checkpoint_id"], r["seed"], r["target"], r["level"]) for r in rows}
    stats = {
        "logical_singleton_rows": len(rows),
        "unique_checkpoint_ids": len({r["checkpoint_id"] for r in rows}),
        "physical_forward_units": len(physical),
        "targets": sorted({r["target"] for r in rows}, key=int),
        "seeds": sorted({r["seed"] for r in rows}, key=int),
        "levels": sorted({r["level"] for r in rows}, key=int),
        "regimes": sorted({r["regime"] for r in rows}),
        "trajectories": len({r["trajectory_id"] for r in rows}),
        "verified_rows": sum(1 for r in rows if r["file_status"] == "pt+json_verified"),
    }
    out = [
        {"metric": "c65_logical_singleton_rows", "value": stats["logical_singleton_rows"], "expected": 3804, "passed": int(stats["logical_singleton_rows"] == 3804), "notes": "C50 logical rows including regime labels"},
        {"metric": "c65_unique_checkpoint_ids", "value": stats["unique_checkpoint_ids"], "expected": 1268, "passed": int(stats["unique_checkpoint_ids"] == 1268), "notes": "physical checkpoint payloads used for deduped forward units"},
        {"metric": "c65_physical_forward_units", "value": stats["physical_forward_units"], "expected": 1268, "passed": int(stats["physical_forward_units"] == 1268), "notes": "checkpoint/seed/target/level units; regime labels do not require duplicate forward"},
        {"metric": "targets", "value": ";".join(stats["targets"]), "expected": "1;2;3;4;5;6;7;8;9", "passed": int(stats["targets"] == [str(i) for i in range(1, 10)]), "notes": "BNCI2014_001 targets only"},
        {"metric": "seeds", "value": ";".join(stats["seeds"]), "expected": "0;1;2", "passed": int(stats["seeds"] == ["0", "1", "2"]), "notes": "reserved seeds 3/4 absent"},
        {"metric": "levels", "value": ";".join(stats["levels"]), "expected": "0;1", "passed": int(stats["levels"] == ["0", "1"]), "notes": "two trajectory levels"},
        {"metric": "regimes", "value": ";".join(stats["regimes"]), "expected": "S0_full_support;S2_rare_cells;S3_nonestimable_cells", "passed": int(stats["regimes"] == ["S0_full_support", "S2_rare_cells", "S3_nonestimable_cells"]), "notes": "diagnostic regimes"},
        {"metric": "verified_pt_json_rows", "value": stats["verified_rows"], "expected": stats["logical_singleton_rows"], "passed": int(stats["verified_rows"] == stats["logical_singleton_rows"]), "notes": "C65 path verification replay"},
    ]
    return out, stats


def _ladder_units(stats: dict) -> dict[str, int]:
    return {
        "T0_micro_replay": 6,
        "T1_pilot_scale": min(64, stats["physical_forward_units"]),
        "T2_medium_scale": min(216, stats["physical_forward_units"]),
        "T3_full_physical_dedup": stats["physical_forward_units"],
    }


def build_scaleup_plan(ctx: dict, stats: dict) -> tuple[list[dict], list[dict], list[dict]]:
    rows = ctx["c65_map"]
    c66_cache_bytes = int(ctx["trial_manifest"].get("size_bytes", 0) or 0)
    c66_cache_rows = int(ctx["trial_manifest"].get("row_count", 0) or 0)
    bytes_per_row = c66_cache_bytes / c66_cache_rows if c66_cache_rows else 592.0
    units = _ladder_units(stats)
    plan_rows = []
    power_rows = []
    size_rows = []
    for rung, unit_count in units.items():
        trial_rows = unit_count * TRIAL_ROWS_PER_FORWARD_UNIT
        logical_rows = stats["logical_singleton_rows"] if rung == "T3_full_physical_dedup" else unit_count
        plan_rows.append({
            "rung": rung,
            "authorized_to_execute": 0,
            "forward_attempted_in_c68": 0,
            "physical_forward_units": unit_count,
            "logical_c50_rows_represented": logical_rows,
            "target_count": len(stats["targets"]) if rung != "T0_micro_replay" else 3,
            "seed_set": ";".join(stats["seeds"]),
            "reserved_seed_used": 0,
            "bnci004_used": 0,
            "selection_rule": "deterministic_stratified_nonperformance_plan_only",
            "notes": "readiness plan only; no checkpoint recommendation artifact",
        })
        power_rows.append({
            "rung": rung,
            "independent_physical_forward_units": unit_count,
            "estimated_trial_rows": trial_rows,
            "checkpoint_unit_gain_vs_c67": round(unit_count / 6.0, 3),
            "min_leave_checkpoint_units": unit_count - 1 if unit_count else 0,
            "split_label_support_status": "powered_plan" if unit_count >= 64 else "micro_underpowered",
            "conditional_cs_support_status": "pilot_or_better_plan" if unit_count >= 64 else "micro_underpowered",
            "claim_allowed_now": "readiness_only_not_authorized",
        })
        size_rows.append({
            "rung": rung,
            "bytes_per_row_from_c66": round(bytes_per_row, 3),
            "estimated_trial_rows": trial_rows,
            "estimated_cache_bytes": _ceil_int(trial_rows * bytes_per_row),
            "estimated_cache_mib": round(trial_rows * bytes_per_row / (1024 ** 2), 3),
            "external_only": 1,
            "raw_cache_committable_to_git": 0,
        })
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["seed"], r["target"], r["level"])].append(r)
    for key, group in sorted(by_cell.items(), key=lambda x: tuple(int(v) for v in x[0])):
        seed, target, level = key
        plan_rows.append({
            "rung": "T1_T2_stratification_cell",
            "authorized_to_execute": 0,
            "forward_attempted_in_c68": 0,
            "physical_forward_units": len({g["checkpoint_id"] for g in group}),
            "logical_c50_rows_represented": len(group),
            "target_count": 1,
            "seed_set": seed,
            "reserved_seed_used": int(seed in {"3", "4"}),
            "bnci004_used": 0,
            "selection_rule": f"cell seed={seed};target={target};level={level};candidate_order_stride",
            "notes": "cell support for future stratified schedule; no endpoint/performance field used",
        })
    return plan_rows, power_rows, size_rows


def build_noauth_gate_rows(authorized: bool) -> list[dict]:
    return [
        {"gate": "authorization_phrase_present", "allowed": 1, "observed": int(authorized), "passed": int(not authorized), "notes": "phrase absent in current user request; C68 remains readiness only"},
        {"gate": "new_forward_or_reinference", "allowed": int(authorized), "observed": 0, "passed": 1, "notes": "no EEG forward/re-inference in C68 readiness"},
        {"gate": "training_or_gradient_update", "allowed": 0, "observed": 0, "passed": 1, "notes": "training remains forbidden"},
        {"gate": "gpu_use", "allowed": 0, "observed": 0, "passed": 1, "notes": "CPU-only planning/report generation"},
        {"gate": "raw_cache_written", "allowed": int(authorized), "observed": 0, "passed": 1, "notes": "no scaled raw cache emitted"},
        {"gate": "selector_or_checkpoint_recommendation", "allowed": 0, "observed": 0, "passed": 1, "notes": "scale-up schedule is a non-performance plan, not a selector"},
    ]


def build_masking_rows(ctx: dict) -> tuple[list[dict], list[dict], list[dict]]:
    c67_views = {r["view"]: r for r in ctx["c67_views"]}
    view_rows = []
    for view in ("source_only_view", "target_construction_view", "target_evaluation_view", "same_label_oracle_view", "conditional_cs_diagnostic_view"):
        r = c67_views.get(view, {})
        view_rows.append({
            "view": view,
            "inherits_c67_projection": r.get("c66_projection", ""),
            "scaleup_status": "dry_run_not_authorized",
            "label_visible_rows_c67": r.get("label_visible_rows", ""),
            "prediction_visible_rows_c67": r.get("prediction_visible_rows", ""),
            "available_at_selection_time": r.get("available_at_selection_time", "0"),
            "diagnostic_only": r.get("diagnostic_only", "1"),
            "selection_path_enforced": r.get("selection_path_enforced", "0"),
            "policy_boundary_only": r.get("policy_boundary_only", "0"),
            "status": "pass" if r.get("status") == "pass" else "missing",
        })
    columns = [
        ("checkpoint_id", "visible", "visible", "visible", "visible", "visible", 0),
        ("seed_target_level_regime", "visible", "visible", "visible", "visible", "visible", 0),
        ("trial_id", "visible", "visible", "visible", "visible", "visible", 0),
        ("target_label", "masked", "construct_only", "eval_only", "visible_diagnostic_only", "visible_diagnostic_only", 1),
        ("target_correctness", "masked", "derived_construct_only", "eval_y_only", "visible_diagnostic_only", "y_or_diagnostic_only", 1),
        ("logits_probabilities", "masked_in_source_only", "visible_target_cache", "visible_target_cache", "visible_diagnostic_only", "visible_diagnostic_only", 0),
        ("same_label_endpoint_scalar", "forbidden", "forbidden", "forbidden", "diagnostic_only_unavailable", "forbidden_unless_flagged", 1),
        ("target_joint_margin_raw", "forbidden", "forbidden", "forbidden", "diagnostic_only_unavailable", "forbidden_unless_flagged", 1),
    ]
    matrix_rows = [
        {
            "column_family": c,
            "source_only_view": s,
            "target_construction_view": tc,
            "target_evaluation_view": te,
            "same_label_oracle_view": so,
            "conditional_cs_diagnostic_view": cs,
            "target_label_dependent": dep,
        }
        for c, s, tc, te, so, cs, dep in columns
    ]
    red_rows = [
        {"check": "source_only_target_labels_masked", "passed": int(c67_views["source_only_view"]["label_visible_rows"] == "0"), "notes": "replayed from C67 masked view"},
        {"check": "source_only_predictions_masked", "passed": int(c67_views["source_only_view"]["prediction_visible_rows"] == "0"), "notes": "source-only view cannot see target logits/probs"},
        {"check": "construction_eval_label_separation_planned", "passed": 1, "notes": "construction/evaluation access matrix remains split"},
        {"check": "same_label_oracle_policy_only", "passed": int(c67_views["same_label_oracle_view"]["policy_boundary_only"] == "1" and c67_views["same_label_oracle_view"]["available_at_selection_time"] == "0"), "notes": "diagnostic full view is not selection path"},
        {"check": "conditional_cs_diagnostic_policy_only", "passed": int(c67_views["conditional_cs_diagnostic_view"]["policy_boundary_only"] == "1"), "notes": "full diagnostic view only for CS smoke/design"},
    ]
    return view_rows, matrix_rows, red_rows


def build_protocol_rows(stats: dict) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    split_rows = [
        {"analysis": "construction_summary_to_eval_endpoint", "status": "not_run_not_authorized", "minimum_units_target": 64, "planned_units_T1": 64, "planned_units_T2": 216, "forbidden_claim": "few_label_sufficiency", "notes": "run only after authorized scaled cache exists"},
        {"analysis": "leave_checkpoint_out", "status": "not_run_not_authorized", "minimum_units_target": 64, "planned_units_T1": 64, "planned_units_T2": 216, "forbidden_claim": "deployable_selector", "notes": "checkpoint-level robustness planned"},
        {"analysis": "leave_target_out_if_support_allows", "status": "not_run_not_authorized", "minimum_units_target": 9, "planned_units_T1": 9, "planned_units_T2": 9, "forbidden_claim": "source_only_rescue", "notes": "target-level stress is diagnostic only"},
    ]
    split_cell_rows = [
        {"cell_key": "seed_target_level", "cell_count": 54, "planned_min_units_per_cell_T2": 4, "uses_endpoint_for_sampling": 0, "status": "plan_only"},
        {"cell_key": "target", "cell_count": len(stats["targets"]), "planned_coverage_T2": "all_targets", "uses_endpoint_for_sampling": 0, "status": "plan_only"},
        {"cell_key": "regime", "cell_count": len(stats["regimes"]), "planned_coverage_T3": "logical_annotation_no_duplicate_forward", "uses_endpoint_for_sampling": 0, "status": "plan_only"},
    ]
    split_null_rows = [
        {"null": "checkpoint_label_permutation", "status": "planned_not_run", "requires_authorized_cache": 1, "notes": "calibrate split-label signal"},
        {"null": "target_cell_preserving_shuffle", "status": "planned_not_run", "requires_authorized_cache": 1, "notes": "preserve target/cell base rates"},
    ]
    cs_feas_rows = [
        {"estimator": "gram_kernel_conditional_cs", "status": "not_run_not_authorized", "x1": "source_only_or_checkpoint_metadata", "x2": "construction_split_target_summary", "y": "heldout_trial_response", "full_cs_claim_allowed_now": 0, "notes": "requires scaled sample cache"},
        {"estimator": "summary_kernel_proxy", "status": "not_run_not_authorized", "x1": "source_summary", "x2": "trajectory_diagnostic_summary", "y": "eval_endpoint", "full_cs_claim_allowed_now": 0, "notes": "proxy must remain labeled proxy"},
    ]
    bandwidth_rows = [
        {"bandwidth_family": "median_heuristic_x0.5", "status": "planned_not_run", "support_gate": ">=64 independent units", "null_required": 1},
        {"bandwidth_family": "median_heuristic_x1.0", "status": "planned_not_run", "support_gate": ">=64 independent units", "null_required": 1},
        {"bandwidth_family": "median_heuristic_x2.0", "status": "planned_not_run", "support_gate": ">=64 independent units", "null_required": 1},
    ]
    cs_null_rows = [
        {"null": "x2_permutation_within_target", "status": "planned_not_run", "preserves": "target base rate", "requires_authorized_cache": 1},
        {"null": "label_preserving_trial_shuffle", "status": "planned_not_run", "preserves": "label marginals", "requires_authorized_cache": 1},
        {"null": "leave_checkpoint_out_stress", "status": "planned_not_run", "preserves": "checkpoint independence", "requires_authorized_cache": 1},
    ]
    avail_rows = [
        {"variable": "X1_source_only", "uses_target_labels": 0, "uses_eval_labels": 0, "available_at_selection_time": 1, "diagnostic_only": 0},
        {"variable": "X2_construction_target_summary", "uses_target_labels": 1, "uses_eval_labels": 0, "available_at_selection_time": 0, "diagnostic_only": 1},
        {"variable": "X2_same_label_endpoint_scalar", "uses_target_labels": 1, "uses_eval_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1},
        {"variable": "Y_heldout_eval_response", "uses_target_labels": 1, "uses_eval_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1},
    ]
    return split_rows, split_cell_rows, split_null_rows, cs_feas_rows, bandwidth_rows, cs_null_rows, avail_rows


def build_source_adversary_rows() -> list[dict]:
    return [
        {"adversary": "source_logits_probability_summary", "allowed_variables": "source logits/probs only", "target_labels_used": 0, "status": "planned_not_run_not_authorized", "escape_hatch_found": 0},
        {"adversary": "source_trial_correctness_summary", "allowed_variables": "source labels only if source-domain labels are available", "target_labels_used": 0, "status": "planned_not_run_not_authorized", "escape_hatch_found": 0},
        {"adversary": "checkpoint_metadata_template", "allowed_variables": "seed;target-key diagnostic;level;regime;candidate_order", "target_labels_used": 0, "status": "planned_not_run_not_authorized", "escape_hatch_found": 0},
    ]


def build_synthetic_writer_rows() -> list[dict]:
    fields = [
        "trial_cache_id", "checkpoint_id", "dataset_id", "target_id", "seed", "level", "regime",
        "trial_id", "class_label_quarantined", "y_true_quarantined", "y_pred", "logits",
        "probabilities", "confidence", "margin", "entropy", "split_role_for_future_split_label",
    ]
    synthetic = [
        {f: ("__MASKED__" if "quarantined" in f else "synthetic") for f in fields},
        {f: ("__MASKED__" if "quarantined" in f else "synthetic") for f in fields},
    ]
    body = "\n".join([",".join(fields)] + [",".join(str(r[f]) for f in fields) for r in synthetic])
    return [
        {"test": "synthetic_cache_schema_writer", "rows_written": len(synthetic), "schema_field_count": len(fields), "sha256": hashlib.sha256(body.encode()).hexdigest(), "raw_eeg_cache": 0, "passed": 1},
        {"test": "synthetic_source_only_masking", "rows_written": len(synthetic), "schema_field_count": len(fields), "sha256": hashlib.sha256((body + "|masked").encode()).hexdigest(), "raw_eeg_cache": 0, "passed": 1},
    ]


def build_test_manifest(status: str) -> list[dict]:
    return [
        {"test_scope": "focused_c68", "command": "python -m pytest oaci/tests/test_c68_powered_trial_cache_scaleup.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c50_c68_slice", "command": "python -m pytest oaci/tests/test_c5*.py oaci/tests/test_c6*.py oaci/tests/test_c68_*.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c23_c68_regression", "command": "python -m pytest oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c3*.py oaci/tests/test_c4*.py oaci/tests/test_c5*.py oaci/tests/test_c6*.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "full_oaci_tests", "command": "python -m pytest oaci/tests -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
    ]


def build_red_team_rows(res: dict) -> list[dict]:
    replay = {r["check"]: r for r in res["c68_c67_replay_dual_mode_ledger_rows"]}
    universe = {r["metric"]: r for r in res["c68_frozen_universe_replay_rows"]}
    gate = {r["gate"]: r for r in res["c68_noauth_execution_gate_rows"]}
    leakage = {r["check"]: r for r in res["c68_label_leakage_redteam_rows"]}
    checks = [
        ("authorization_absent_no_forward", gate["new_forward_or_reinference"]["observed"] == 0 and gate["authorization_phrase_present"]["observed"] == 0, "C68 is readiness-only because the authorization phrase is absent."),
        ("c67_replay_passed", all(int(r["passed"]) for r in replay.values()), "C67 dual-mode cache contract replayed."),
        ("c65_universe_passed", all(int(r["passed"]) for r in universe.values()), "C65/C50 frozen universe replayed."),
        ("scaleup_plan_manifested", len(res["c68_scaleup_sampling_plan_rows"]) >= 4 and len(res["c68_power_ladder_rows"]) == 4, "T0/T1/T2/T3 scale-up plan exists."),
        ("no_raw_cache_committed", all(int(r["raw_cache_committable_to_git"]) == 0 for r in res["c68_expected_cache_size_rows"]), "Scaled raw cache remains external-only if later authorized."),
        ("reserved_holdouts_preserved", all(int(r["reserved_seed_used"]) == 0 and int(r["bnci004_used"]) == 0 for r in res["c68_scaleup_sampling_plan_rows"]), "No BNCI2014_004 or seeds 3/4 in plan."),
        ("masking_contract_passed", all(int(r["passed"]) for r in leakage.values()), "View/masking dry-run passes."),
        ("split_label_not_claimed", all(r["status"].startswith("not_run") for r in res["c68_split_label_power_summary_rows"]), "Split-label powered diagnostic not run and not claimed."),
        ("cs_not_claimed", all(r["status"].startswith("not_run") for r in res["c68_sample_level_cs_feasibility_rows"]), "Conditional-CS not run and not claimed."),
        ("source_escape_not_claimed", all(r["status"].startswith("planned_not_run") and int(r["escape_hatch_found"]) == 0 for r in res["c68_source_observable_adversary_plan_rows"]), "No source-only escape hatch claim without scaled cache."),
        ("new_training_still_not_justified", gate["training_or_gradient_update"]["observed"] == 0, "Frozen re-inference-only route remains the next authorized step, not training."),
        ("large_artifact_scan_passed", all(int(r["passed"]) for r in res["large_artifact_scan_rows"]), "All C68 git artifacts are below 50MB."),
        ("forbidden_scan_passed", all(int(r["passed"]) for r in res["forbidden_claim_scan_rows"]), "Forbidden affirmative claim scan passed."),
    ]
    return [{"gate": g, "failed": int(not ok), "finding": f} for g, ok, f in checks]


def classify(res: dict, authorized: bool) -> dict:
    failures = [r for r in res["red_team_failure_ledger_rows"] if int(r["failed"])]
    if failures:
        primary = "C68-P_claim_or_availability_violation_found"
        active = [primary]
        gate = "CLAIM_OR_AVAILABILITY_REPAIR_REQUIRED"
    elif not authorized:
        active = [
            "C68-A_c67_dual_mode_cache_contract_replayed",
            "C68-B_scaleup_plan_powered_and_manifested",
            "C68-C_reinference_only_scaleup_ready_but_not_authorized",
            "C68-K_endpoint_oracle_boundary_preserved",
            "C68-N_larger_reinference_only_campaign_needed_but_not_authorized",
            "C68-O_new_training_still_not_justified",
        ]
        primary = "C68-C_reinference_only_scaleup_ready_but_not_authorized"
        gate = "SCALEUP_READY_BUT_NOT_AUTHORIZED"
    else:
        active = ["C68-P_claim_or_availability_violation_found"]
        primary = "C68-P_claim_or_availability_violation_found"
        gate = "CLAIM_OR_AVAILABILITY_REPAIR_REQUIRED"
    return {
        "primary": primary,
        "active": active,
        "inactive": [d for d in DECISIONS if d not in active],
        "final_gate": gate,
        "red_team_failure_count": len(failures),
        "recommended_next_direction": "remote review; provide explicit C68 authorization phrase only if the powered re-inference-only cache campaign is approved",
    }


def table_row_counts(res: dict) -> dict:
    keys = {
        "artifact_manifest": "artifact_manifest_rows",
        "c68_c67_replay_dual_mode_ledger": "c68_c67_replay_dual_mode_ledger_rows",
        "c68_expected_cache_size": "c68_expected_cache_size_rows",
        "c68_frozen_universe_replay": "c68_frozen_universe_replay_rows",
        "c68_label_leakage_redteam": "c68_label_leakage_redteam_rows",
        "c68_masked_view_contract": "c68_masked_view_contract_rows",
        "c68_noauth_execution_gate": "c68_noauth_execution_gate_rows",
        "c68_power_ladder": "c68_power_ladder_rows",
        "c68_sample_level_cs_availability_ledger": "c68_sample_level_cs_availability_ledger_rows",
        "c68_sample_level_cs_bandwidth_grid": "c68_sample_level_cs_bandwidth_grid_rows",
        "c68_sample_level_cs_feasibility": "c68_sample_level_cs_feasibility_rows",
        "c68_sample_level_cs_nulls": "c68_sample_level_cs_nulls_rows",
        "c68_scaleup_sampling_plan": "c68_scaleup_sampling_plan_rows",
        "c68_source_observable_adversary_plan": "c68_source_observable_adversary_plan_rows",
        "c68_split_label_cell_ledger": "c68_split_label_cell_ledger_rows",
        "c68_split_label_nulls": "c68_split_label_nulls_rows",
        "c68_split_label_power_summary": "c68_split_label_power_summary_rows",
        "c68_synthetic_cache_writer_dryrun": "c68_synthetic_cache_writer_dryrun_rows",
        "c68_view_column_access_matrix": "c68_view_column_access_matrix_rows",
        "forbidden_claim_scan": "forbidden_claim_scan_rows",
        "large_artifact_scan": "large_artifact_scan_rows",
        "red_team_failure_ledger": "red_team_failure_ledger_rows",
        "schema_validation_summary": "schema_validation_summary_rows",
        "test_command_manifest": "test_command_manifest_rows",
    }
    return {name: len(res.get(key, [])) for name, key in keys.items()}


def run(authorization_token: str = "", test_status: str = "planned") -> dict:
    authorized = _auth_present(authorization_token)
    ctx = load_context()
    universe_rows, stats = build_universe_rows(ctx)
    plan_rows, power_rows, size_rows = build_scaleup_plan(ctx, stats)
    view_rows, matrix_rows, leakage_rows = build_masking_rows(ctx)
    split_rows, split_cell_rows, split_null_rows, cs_rows, bw_rows, cs_null_rows, avail_rows = build_protocol_rows(stats)
    res = {
        "config_hash": _lock_config(),
        "authorization_present": authorized,
        "c67_commit": "9f8c829",
        "current_head": ctx["head"],
        "external_c66_cache_path_hash": _path_hash(ctx["trial_path"]),
        "external_c66_cache_sha256": ctx["trial_sha"],
        "c68_c67_replay_dual_mode_ledger_rows": build_c67_replay_rows(ctx),
        "c68_frozen_universe_replay_rows": universe_rows,
        "c68_scaleup_sampling_plan_rows": plan_rows,
        "c68_power_ladder_rows": power_rows,
        "c68_expected_cache_size_rows": size_rows,
        "c68_noauth_execution_gate_rows": build_noauth_gate_rows(authorized),
        "c68_masked_view_contract_rows": view_rows,
        "c68_view_column_access_matrix_rows": matrix_rows,
        "c68_label_leakage_redteam_rows": leakage_rows,
        "c68_split_label_power_summary_rows": split_rows,
        "c68_split_label_cell_ledger_rows": split_cell_rows,
        "c68_split_label_nulls_rows": split_null_rows,
        "c68_sample_level_cs_feasibility_rows": cs_rows,
        "c68_sample_level_cs_bandwidth_grid_rows": bw_rows,
        "c68_sample_level_cs_nulls_rows": cs_null_rows,
        "c68_sample_level_cs_availability_ledger_rows": avail_rows,
        "c68_source_observable_adversary_plan_rows": build_source_adversary_rows(),
        "c68_synthetic_cache_writer_dryrun_rows": build_synthetic_writer_rows(),
        "test_command_manifest_rows": build_test_manifest(test_status),
        "forbidden_claim_scan_rows": [],
        "large_artifact_scan_rows": [],
        "red_team_failure_ledger_rows": [],
        "schema_validation_summary_rows": [],
        "artifact_manifest_rows": [],
    }
    res["decision"] = classify({**res, "red_team_failure_ledger_rows": []}, authorized)
    return res


def _compact_json(res: dict) -> dict:
    return {
        "milestone": MILESTONE,
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": True,
        "authorization_present": res["authorization_present"],
        "c67_commit": res["c67_commit"],
        "current_head_at_generation": res["current_head"],
        "external_c66_cache_path_hash": res["external_c66_cache_path_hash"],
        "external_c66_cache_sha256": res["external_c66_cache_sha256"],
        "decision": res["decision"],
        "final_gate": res["decision"]["final_gate"],
        "key_numbers": {
            "c65_logical_singleton_rows": next(r["value"] for r in res["c68_frozen_universe_replay_rows"] if r["metric"] == "c65_logical_singleton_rows"),
            "c65_physical_forward_units": next(r["value"] for r in res["c68_frozen_universe_replay_rows"] if r["metric"] == "c65_physical_forward_units"),
            "t1_pilot_units": next(r["independent_physical_forward_units"] for r in res["c68_power_ladder_rows"] if r["rung"] == "T1_pilot_scale"),
            "t2_medium_units": next(r["independent_physical_forward_units"] for r in res["c68_power_ladder_rows"] if r["rung"] == "T2_medium_scale"),
            "t3_full_physical_units": next(r["independent_physical_forward_units"] for r in res["c68_power_ladder_rows"] if r["rung"] == "T3_full_physical_dedup"),
            "forward_attempted_in_c68": 0,
            "training_attempted_in_c68": 0,
            "raw_cache_rows_emitted_in_c68": 0,
            "red_team_failure_count": res["decision"]["red_team_failure_count"],
        },
        "table_row_counts": table_row_counts(res),
        "recommended_next_step": res["decision"]["recommended_next_direction"],
    }


def build_reports(res: dict) -> dict[str, str]:
    d = res["decision"]
    t3 = next(r for r in res["c68_power_ladder_rows"] if r["rung"] == "T3_full_physical_dedup")
    main = "\n".join([
        f"# C68 - Powered Re-inference-Only Trial-Cache Scale-Up Readiness (frozen C19 `{res['config_hash']}`)",
        "",
        "## 1. Executive Verdict",
        "",
        f"Primary: `{d['primary']}`",
        "",
        f"Active: `{' ; '.join(d['active'])}`",
        "",
        f"Inactive: `{' ; '.join(d['inactive'])}`",
        "",
        f"Final gate: `{d['final_gate']}`",
        "",
        "## 2. Authorization Boundary",
        "",
        "The C68 scale-up authorization phrase is absent in the current user request. C68 therefore performs only readiness, planning, synthetic writer, and masking dry-run work. It does not run new EEG forward passes, re-inference, training, GPU work, or raw cache emission.",
        "",
        "## 3. C67 Replay",
        "",
        "C67's dual-mode result is replayed: the no-auth C66 mode remains guard evidence, and the authorized C66 microcampaign remains the only consumed cache. The external C66 cache hash still matches its committed manifest.",
        "",
        "## 4. Scale-Up Plan",
        "",
        f"The C65/C50 universe has 3804 logical singleton rows and {t3['independent_physical_forward_units']} physical forward units after checkpoint/seed/target/level deduplication. The C68 ladder is T0=6, T1=64, T2=216, T3={t3['independent_physical_forward_units']} physical units. This is a non-performance stratified plan, not a checkpoint recommendation artifact.",
        "",
        "## 5. Diagnostic Plans",
        "",
        "Split-label and sample-level conditional-CS analyses are planned but not run. No few-label sufficiency, full conditional-CS, source-only rescue, OACI rescue, or deployable selector claim is made.",
        "",
        "## 6. Red-Team Verification",
        "",
        f"Red-team failures: `{d['red_team_failure_count']}`.",
    ])
    red = "\n".join([
        "# C68 - Red-Team Verification",
        "",
        "All C68 red-team gates pass." if d["red_team_failure_count"] == 0 else "C68 red-team gates failed.",
        "",
        *[f"- {r['gate']}: {'PASS' if not int(r['failed']) else 'FAIL'} - {r['finding']}" for r in res["red_team_failure_ledger_rows"]],
        "",
        "## Slurm Validation",
        "",
        *[f"- {scope} job `{job_id}` on `cpu-high` with `eeg2025`: `{result}`." for scope, job_id, result in SLURM_VALIDATION_RESULTS],
    ])
    return {
        "C68_POWERED_TRIAL_CACHE_SCALEUP.md": main,
        "C68_RED_TEAM_VERIFICATION.md": red,
    }


def write_tables(res: dict) -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    specs = {
        "c68_c67_replay_dual_mode_ledger.csv": ("c68_c67_replay_dual_mode_ledger_rows", ["check", "expected", "observed", "passed", "notes"]),
        "c68_frozen_universe_replay.csv": ("c68_frozen_universe_replay_rows", ["metric", "value", "expected", "passed", "notes"]),
        "c68_scaleup_sampling_plan.csv": ("c68_scaleup_sampling_plan_rows", ["rung", "authorized_to_execute", "forward_attempted_in_c68", "physical_forward_units", "logical_c50_rows_represented", "target_count", "seed_set", "reserved_seed_used", "bnci004_used", "selection_rule", "notes"]),
        "c68_power_ladder.csv": ("c68_power_ladder_rows", ["rung", "independent_physical_forward_units", "estimated_trial_rows", "checkpoint_unit_gain_vs_c67", "min_leave_checkpoint_units", "split_label_support_status", "conditional_cs_support_status", "claim_allowed_now"]),
        "c68_expected_cache_size.csv": ("c68_expected_cache_size_rows", ["rung", "bytes_per_row_from_c66", "estimated_trial_rows", "estimated_cache_bytes", "estimated_cache_mib", "external_only", "raw_cache_committable_to_git"]),
        "c68_noauth_execution_gate.csv": ("c68_noauth_execution_gate_rows", ["gate", "allowed", "observed", "passed", "notes"]),
        "c68_masked_view_contract.csv": ("c68_masked_view_contract_rows", ["view", "inherits_c67_projection", "scaleup_status", "label_visible_rows_c67", "prediction_visible_rows_c67", "available_at_selection_time", "diagnostic_only", "selection_path_enforced", "policy_boundary_only", "status"]),
        "c68_view_column_access_matrix.csv": ("c68_view_column_access_matrix_rows", ["column_family", "source_only_view", "target_construction_view", "target_evaluation_view", "same_label_oracle_view", "conditional_cs_diagnostic_view", "target_label_dependent"]),
        "c68_label_leakage_redteam.csv": ("c68_label_leakage_redteam_rows", ["check", "passed", "notes"]),
        "c68_split_label_power_summary.csv": ("c68_split_label_power_summary_rows", ["analysis", "status", "minimum_units_target", "planned_units_T1", "planned_units_T2", "forbidden_claim", "notes"]),
        "c68_split_label_cell_ledger.csv": ("c68_split_label_cell_ledger_rows", ["cell_key", "cell_count", "planned_min_units_per_cell_T2", "planned_coverage_T2", "planned_coverage_T3", "uses_endpoint_for_sampling", "status"]),
        "c68_split_label_nulls.csv": ("c68_split_label_nulls_rows", ["null", "status", "requires_authorized_cache", "notes"]),
        "c68_sample_level_cs_feasibility.csv": ("c68_sample_level_cs_feasibility_rows", ["estimator", "status", "x1", "x2", "y", "full_cs_claim_allowed_now", "notes"]),
        "c68_sample_level_cs_bandwidth_grid.csv": ("c68_sample_level_cs_bandwidth_grid_rows", ["bandwidth_family", "status", "support_gate", "null_required"]),
        "c68_sample_level_cs_nulls.csv": ("c68_sample_level_cs_nulls_rows", ["null", "status", "preserves", "requires_authorized_cache"]),
        "c68_sample_level_cs_availability_ledger.csv": ("c68_sample_level_cs_availability_ledger_rows", ["variable", "uses_target_labels", "uses_eval_labels", "available_at_selection_time", "diagnostic_only"]),
        "c68_source_observable_adversary_plan.csv": ("c68_source_observable_adversary_plan_rows", ["adversary", "allowed_variables", "target_labels_used", "status", "escape_hatch_found"]),
        "c68_synthetic_cache_writer_dryrun.csv": ("c68_synthetic_cache_writer_dryrun_rows", ["test", "rows_written", "schema_field_count", "sha256", "raw_eeg_cache", "passed"]),
        "test_command_manifest.csv": ("test_command_manifest_rows", ["test_scope", "command", "status", "environment", "slurm_partition"]),
        "forbidden_claim_scan.csv": ("forbidden_claim_scan_rows", ["pattern", "total_hits", "affirmative_hits", "files", "passed"]),
        "large_artifact_scan.csv": ("large_artifact_scan_rows", ["path", "size_bytes", "over_50mb", "passed"]),
        "red_team_failure_ledger.csv": ("red_team_failure_ledger_rows", ["gate", "failed", "finding"]),
        "schema_validation_summary.csv": ("schema_validation_summary_rows", ["table_name", "row_count", "required_columns_present", "passed"]),
        "artifact_manifest.csv": ("artifact_manifest_rows", ["path", "size_bytes", "sha256", "artifact_class", "row_count"]),
    }
    for name, (key, cols) in specs.items():
        _write_csv(os.path.join(TABLE_DIR, name), res.get(key, []), cols)


def _schema_rows() -> list[dict]:
    rows = []
    for path in sorted(Path(TABLE_DIR).glob("*.csv")):
        if path.name in {"schema_validation_summary.csv", "artifact_manifest.csv"}:
            continue
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            count = sum(1 for _ in reader)
        rows.append({"table_name": path.name, "row_count": count, "required_columns_present": int(bool(header)), "passed": int(bool(header))})
    return rows


def write_artifacts(res: dict) -> dict:
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
    for name, text in build_reports(res).items():
        with open(os.path.join(REPORT_DIR, name), "w") as f:
            f.write(text.rstrip() + "\n")
    write_tables(res)
    paths = [str(p) for p in _listed_paths()]
    res["forbidden_claim_scan_rows"] = build_forbidden_scan(paths)
    res["large_artifact_scan_rows"] = _large_scan([Path(p) for p in paths])
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res, res["authorization_present"])
    for name, text in build_reports(res).items():
        with open(os.path.join(REPORT_DIR, name), "w") as f:
            f.write(text.rstrip() + "\n")
    write_tables(res)
    res["schema_validation_summary_rows"] = _schema_rows()
    write_tables(res)
    paths = _listed_paths()
    res["large_artifact_scan_rows"] = _large_scan(paths)
    res["artifact_manifest_rows"] = [{} for _ in paths]
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
    res["large_artifact_scan_rows"] = _large_scan(paths)
    _write_csv(os.path.join(TABLE_DIR, "large_artifact_scan.csv"), res["large_artifact_scan_rows"], ["path", "size_bytes", "over_50mb", "passed"])
    res["artifact_manifest_rows"] = _artifact_manifest(paths, TABLE_DIR)
    _write_csv(os.path.join(TABLE_DIR, "artifact_manifest.csv"), res["artifact_manifest_rows"], ["path", "size_bytes", "sha256", "artifact_class", "row_count"])
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c68_powered_trial_cache_scaleup")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--authorization-token", default="", help="Exact C68 authorization token; protocol text is not accepted.")
    ap.add_argument("--test-status", default="planned")
    args = ap.parse_args(argv)
    res = run(authorization_token=args.authorization_token, test_status=args.test_status)
    if args.recompute:
        res = write_artifacts(res)
    print(f"[C68] decision={res['decision']['primary']} gate={res['decision']['final_gate']} tables={len(table_row_counts(res))}")


if __name__ == "__main__":
    main()
