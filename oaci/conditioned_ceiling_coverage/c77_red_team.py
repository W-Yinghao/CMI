"""Independent C77 protocol, provenance, simulation, and claim red-team."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from oaci.protocol.manifest_v2 import load_v2, optimization_manifest_hash

from . import c77_independent_multiregime_replication_protocol as analysis
from . import c77_protocol
from . import synthetic_multiregime_generator as synthetic


REPORT_PATH = c77_protocol.REPORT_DIR / "C77_RED_TEAM_VERIFICATION.md"
CHECKS_PATH = c77_protocol.TABLE_DIR / "red_team_checks.csv"
REPAIR_PATH = c77_protocol.TABLE_DIR / "red_team_repair_ledger.csv"
MAIN_REPORT = c77_protocol.REPORT_DIR / "C77_INDEPENDENT_MULTIREGIME_REPLICATION_PROTOCOL.md"
LOG_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c77-multiregime/logs")


def _rows(name: str) -> list[dict]:
    with open(c77_protocol.TABLE_DIR / name, newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _check(checks: list[dict], name: str, passed: bool, observed, expected, *, blocking: bool = True, note: str = "") -> None:
    checks.append({"check": name, "passed": int(bool(passed)), "blocking": int(blocking), "observed": observed, "expected": expected, "note": note})


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _git_blob_sha(commit: str, path: str | Path) -> str:
    return hashlib.sha256(subprocess.check_output(["git", "show", f"{commit}:{path}"])).hexdigest()


def _commit_time(commit: str) -> int:
    return int(_git("show", "-s", "--format=%ct", commit))


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def _current_and_historical_blobs() -> list[tuple[str, Path, str]]:
    return [
        (c77_protocol.REGIME_COMMITS["ERM"], c77_protocol.ERM_OBJECTIVE_PATH, "ERM objective"),
        (c77_protocol.REGIME_COMMITS["OACI"], c77_protocol.OACI_OBJECTIVE_PATH, "OACI objective"),
        (c77_protocol.REGIME_COMMITS["OACI"], c77_protocol.ENGINE_PATH, "training engine"),
        (c77_protocol.REGIME_COMMITS["SRC"], c77_protocol.SRC_OBJECTIVE_PATH, "SRC objective"),
        (c77_protocol.REGIME_COMMITS["SRC"], c77_protocol.SRC_ONEFOLD_PATH, "SRC runner"),
        (c77_protocol.REGIME_COMMITS["SRC"], c77_protocol.SRC_SELECTOR_PATH, "SRC source-only selector"),
        (c77_protocol.REGIME_COMMITS["manifest"], c77_protocol.MANIFEST_PATH, "confirmatory manifest"),
    ]


def _synthetic_reconstruction(checks: list[dict]) -> dict:
    phase = _rows("synthetic_transport_phase_diagram.csv")
    shard_paths = sorted((c77_protocol.TABLE_DIR / "synthetic_shards").glob("shard_*_of_08.csv"))
    shard_rows = []
    shard_counts = []
    for path in shard_paths:
        with open(path, newline="") as stream:
            rows = list(csv.DictReader(stream))
        shard_counts.append(len(rows)); shard_rows.extend(rows)
    _check(checks, "eight_shards_complete", len(shard_paths) == 8 and sum(shard_counts) == 486, f"files={len(shard_paths)};counts={shard_counts}", "8 files;486 cells")
    _check(checks, "synthetic_merge_exact", len(phase) == 486 and {row["cell_id"] for row in phase} == {row["cell_id"] for row in shard_rows}, len(phase), "486 unique shard cells")
    _check(checks, "synthetic_replicates_locked", all(int(row["replicates"]) == c77_protocol.SYNTHETIC_REPLICATES for row in phase), sorted({row["replicates"] for row in phase}), {"400"})

    null = [row for row in phase if float(row["association_strength"]) == 0]
    stable = [row for row in phase if float(row["association_strength"]) == 0.5 and float(row["transport_heterogeneity"]) == 0]
    heterogeneous = [row for row in phase if float(row["association_strength"]) == 0.5 and float(row["transport_heterogeneity"]) == 1.5]
    low_ties = [row for row in stable if int(row["effective_multiplicity"]) == 2]
    high_ties = [row for row in stable if int(row["effective_multiplicity"]) == 20]

    def rate(rows: list[dict], count: str) -> tuple[float, int, int]:
        successes = sum(int(row[count]) for row in rows)
        total = sum(int(row["replicates"]) for row in rows)
        return successes / total, successes, total

    fpr, fpr_success, fpr_total = rate(null, "association_detection_count")
    power, _, _ = rate(stable, "association_detection_count")
    stable_transport, _, _ = rate(stable, "transport_qualification_count")
    heterogeneous_transport, _, _ = rate(heterogeneous, "transport_qualification_count")
    low_action, _, _ = rate(low_ties, "actionability_qualification_count")
    high_action, _, _ = rate(high_ties, "actionability_qualification_count")
    ci = _wilson(fpr_success, fpr_total)
    reported = {row["gate"]: row for row in _rows("power_and_false_positive_plan.csv")}
    reconstructed = {
        "null_association_FPR": fpr,
        "stable_local_association_power": power,
        "heterogeneity_reduces_transport": stable_transport - heterogeneous_transport,
        "effective_multiplicity_reduces_actionability": low_action - high_action,
    }
    exact = all(abs(float(reported[name]["observed"]) - value) < 1e-15 for name, value in reconstructed.items())
    _check(checks, "synthetic_summary_exact_reconstruction", exact, reconstructed, "reported values bit-close")
    _check(checks, "null_FPR_and_Wilson_upper", fpr <= 0.075 and ci[1] <= 0.075, f"FPR={fpr:.6f};CI=[{ci[0]:.6f},{ci[1]:.6f}]", "point and upper 95% <=0.075")
    _check(checks, "stable_signal_power", power >= 0.80, power, ">=0.80")
    _check(checks, "transport_heterogeneity_direction", stable_transport > heterogeneous_transport, stable_transport - heterogeneous_transport, ">0")
    _check(checks, "multiplicity_direction_registered", low_action > high_action, low_action - high_action, ">0", note="direction passes; magnitude is small and cannot support a strong effect claim")
    _check(checks, "multiplicity_contrast_materiality_caveat", low_action - high_action >= 0.02, low_action - high_action, ">=0.02", blocking=False, note="not a registered C77 blocking gate; C78 must recalibrate rather than call this material")
    return {**reconstructed, "FPR_CI": ci, "stable_transport": stable_transport, "heterogeneous_transport": heterogeneous_transport, "low_action": low_action, "high_action": high_action}


def run_red_team() -> dict:
    # C77 is itself a protocol milestone: its required JSON is the prospectively
    # committed protocol and must exist before compute.  Only the narrative result
    # report is forbidden before red-team, and the protocol JSON is never overwritten.
    main_existed = MAIN_REPORT.exists()
    if main_existed:
        raise RuntimeError("C77 main report existed before independent red-team")
    checks: list[dict] = []
    protocol = json.loads(c77_protocol.PROTOCOL_PATH.read_text())
    c78 = json.loads(c77_protocol.C78_PROTOCOL_PATH.read_text())
    state = json.loads(analysis.STATE_PATH.read_text())
    protocol_commit = state["protocol_commit"]

    protocol_hash = c77_protocol.sha256(c77_protocol.PROTOCOL_PATH)
    c78_hash = c77_protocol.sha256(c77_protocol.C78_PROTOCOL_PATH)
    _check(checks, "C77_protocol_hash", protocol_hash == c77_protocol.PROTOCOL_SHA_PATH.read_text().strip(), protocol_hash, c77_protocol.PROTOCOL_SHA_PATH.read_text().strip())
    _check(checks, "C78_protocol_hash", c78_hash == c77_protocol.C78_PROTOCOL_SHA_PATH.read_text().strip(), c78_hash, c77_protocol.C78_PROTOCOL_SHA_PATH.read_text().strip())
    _check(checks, "protocol_commit_parent", _git("rev-parse", f"{protocol_commit}^") == c77_protocol.PARENT_COMMIT, _git("rev-parse", f"{protocol_commit}^"), c77_protocol.PARENT_COMMIT)
    shard_paths = sorted((c77_protocol.TABLE_DIR / "synthetic_shards").glob("*.csv"))
    _check(checks, "protocol_precedes_synthetic_compute", shard_paths and _commit_time(protocol_commit) <= min(int(path.stat().st_mtime) for path in shard_paths), _commit_time(protocol_commit), f"<= {min(int(path.stat().st_mtime) for path in shard_paths)}")
    _check(checks, "parent_C76_gate", protocol["parent_C76_result_commit"] == c77_protocol.PARENT_COMMIT, protocol["parent_C76_result_commit"], c77_protocol.PARENT_COMMIT)
    _check(checks, "main_report_absent_before_red_team", not main_existed, str(main_existed).lower(), "false")

    for item in protocol["locked_tables"].values():
        path = Path(item["path"])
        _check(checks, f"locked_registry_{path.name}", path.is_file() and c77_protocol.sha256(path) == item["sha256"] and path.stat().st_size == item["size_bytes"], c77_protocol.sha256(path) if path.is_file() else "missing", item["sha256"])

    metric_rows = _rows("c76_metric_identity_replay.csv")
    orbit = _rows("c76_orbit_identity_replay.csv")[0]
    _check(checks, "C76_metric_replay", len(metric_rows) == 10 and all(row["match"] == "1" for row in metric_rows), f"rows={len(metric_rows)};matches={sum(row['match']=='1' for row in metric_rows)}", "10/10")
    _check(checks, "C76_orbit_replay", orbit["orbit_variants"] == "29" and orbit["all_identity_pass"] == "1" and orbit["prediction_disagreements"] == "0", orbit, "29 variants; identity; zero disagreements")

    blob_rows = []
    blobs_exact = True
    for commit, path, label in _current_and_historical_blobs():
        historical = _git_blob_sha(commit, path)
        current = c77_protocol.sha256(path)
        exact = historical == current
        blobs_exact &= exact
        blob_rows.append({"component": label, "path": str(path), "historical_commit": commit, "historical_blob_sha256": historical, "current_blob_sha256": current, "byte_exact": int(exact)})
    _write_csv(c77_protocol.TABLE_DIR / "historical_code_blob_replay.csv", blob_rows)
    _check(checks, "historical_code_blobs_byte_exact", blobs_exact, sum(row["byte_exact"] for row in blob_rows), len(blob_rows))
    manifest = load_v2(str(c77_protocol.MANIFEST_PATH))
    _check(checks, "manifest_optimization_hash", optimization_manifest_hash(manifest) == "cfa2fdc7c7fafd62f4628aa75412fd7029eceaf1ca543ab51880ab4c6c681d30", optimization_manifest_hash(manifest), "cfa2fdc7c7fafd62f4628aa75412fd7029eceaf1ca543ab51880ab4c6c681d30")

    reconstruction = _rows("regime_reconstruction_status.csv")
    qualified = [row for row in reconstruction if row["qualifies_primary_R1"] == "1"]
    trajectory = [row for row in reconstruction if row["comparable_40_checkpoint_trajectory_per_level"] == "1"]
    _check(checks, "three_primary_two_trajectory_regimes", {row["regime_id"] for row in qualified} == {"ERM", "OACI", "SRC"} and {row["regime_id"] for row in trajectory} == {"OACI", "SRC"}, f"primary={[r['regime_id'] for r in qualified]};trajectory={[r['regime_id'] for r in trajectory]}", "ERM/OACI/SRC; OACI/SRC trajectories")
    inventory = {row["regime_id"]: row for row in _rows("historical_regime_inventory.csv")}
    _check(checks, "SRC_historical_context_not_whitewashed", "after C10" in inventory["SRC"]["historical_context"] and "C12 falsified" in inventory["SRC"]["historical_context"], inventory["SRC"]["historical_context"], "post-C10; C12 negative control")
    c11 = json.loads(c77_protocol.C11_RESULT_PATH.read_text())
    c12 = json.loads(c77_protocol.C12_RESULT_PATH.read_text())
    _check(checks, "historical_target_isolation_actual", c11["all_target_fit_ids_empty"] and c11["no_selector_read_target"], f"fit_empty={c11['all_target_fit_ids_empty']};selector_no_target={c11['no_selector_read_target']}", "true;true")
    _check(checks, "SRC_negative_result_actual", c12["verdict"]["verdict"] == "stop_SRC_pivot_measurement_only", c12["verdict"]["verdict"], "stop_SRC_pivot_measurement_only")

    matrix = c78["execution_matrix"]
    _check(checks, "C78_matrix_exact", len(matrix) == 54 and sum(int(row["retained_checkpoints"]) for row in matrix) == 1458 and {row["level"] for row in matrix} == {0, 1}, f"rows={len(matrix)};units={sum(int(r['retained_checkpoints']) for r in matrix)};levels={sorted({r['level'] for r in matrix})}", "54;1458;[0,1]")
    _check(checks, "ERM_asymmetry_explicit", c78["regimes"]["ERM_role"] == "shared_stage1_final_anchor_only" and c78["matrix_summary"]["training_phases"] == 54, c78["regimes"]["ERM_role"], "anchor only;54 phases")
    _check(checks, "pilot_outcome_blind_hash_rule", c78["pilot"]["target"] == 4 and c78["pilot"]["regime"] == "OACI" and c78["pilot"]["outcome_blind"], f"target={c78['pilot']['target']};regime={c78['pilot']['regime']}", "target4 OACI by SHA")
    _check(checks, "BNCI001_only", c78["execution_boundary"]["dataset_allowlist"] == ["BNCI2014_001"] and "BNCI2014_004" in c78["execution_boundary"]["dataset_denylist"], c78["execution_boundary"], "BNCI001 only; BNCI004 denied")
    _check(checks, "seed3_seed4_physical_role", {row["seed"] for row in matrix} == {3} and c78["execution_boundary"]["seed_denylist"] == [4], sorted({row["seed"] for row in matrix}), "[3];seed4 denied")
    _check(checks, "exact_CLI_token_no_prompt_auth", c78["authorization"]["accepted_channel"] == "exact_CLI_argument_only" and not c78["authorization"]["prompt_text_is_authorization"] and not c78["authorization"]["C77_authorized"], c78["authorization"], "exact CLI;C77 unauthorized")
    _check(checks, "C79_skeleton_not_final", json.loads(c77_protocol.C79_SKELETON_PATH.read_text())["status"] == "SKELETON_ONLY_NOT_FINAL_NOT_AUTHORIZED", json.loads(c77_protocol.C79_SKELETON_PATH.read_text())["status"], "SKELETON_ONLY_NOT_FINAL_NOT_AUTHORIZED")

    views = {row["view"]: row for row in _rows("physical_view_schema.csv")}
    view_pass = views["strict_source_trial_view"]["uses_target_rows"] == "0" and views["target_unlabeled_trial_view"]["uses_target_labels"] == "0" and views["target_construction_view"]["uses_evaluation_labels"] == "0" and all(row["physically_separate"] == "1" for row in views.values())
    _check(checks, "physical_view_isolation_contract", view_pass, view_pass, True)
    hypotheses = _rows("primary_hypothesis_registry.csv")
    action_gates = _rows("actionability_gate_registry.csv")
    _check(checks, "small_primary_family_and_conjunctive_action", [row["hypothesis"] for row in hypotheses] == [f"H{i}" for i in range(1, 8)] and len(action_gates) == 16 and all(row["all_required"] == "1" and row["association_p_alone_sufficient"] == "0" for row in action_gates), f"H={len(hypotheses)};gates={len(action_gates)}", "7;16;conjunctive")

    independent_dummy = analysis._dummy_abi()[0]
    reported_dummy = _rows("dummy_hook_ABI_validation.csv")[0]
    _check(checks, "independent_dummy_ABI", independent_dummy["passed"] == 1 and float(reported_dummy["Wz_plus_b_max_abs"]) == independent_dummy["Wz_plus_b_max_abs"] and int(reported_dummy["state_bytes"]) == independent_dummy["state_bytes"], independent_dummy, "exact Wz/logit;158424 bytes;no real data")
    synthetic_metrics = _synthetic_reconstruction(checks)

    storage = {row["campaign"]: row for row in _rows("compute_storage_plan.csv")}
    unit_check = int(storage["C78_seed3_full"]["retained_checkpoint_target_level_units"]) == 1458 and int(storage["R1_seed3_plus_seed4"]["retained_checkpoint_target_level_units"]) == 2916
    bytes_per_unit = int(_rows_from_path(c77_protocol.C74_STORAGE_PATH)[0]["external_size_bytes"]) / 216
    expected_gib = bytes_per_unit * 1458 / 2**30
    storage_exact = abs(float(storage["C78_seed3_full"]["trial_cache_GiB_projected"]) - expected_gib) < 1e-12
    _check(checks, "compute_storage_reconstruction", unit_check and storage_exact, f"units={storage['C78_seed3_full']['retained_checkpoint_target_level_units']};GiB={storage['C78_seed3_full']['trial_cache_GiB_projected']}", f"1458;{expected_gib}")
    _check(checks, "GPU_estimate_disclosed_as_unmeasured", all("unmeasured planning range" in row["GPU_estimate_basis"] for row in storage.values()), [row["GPU_estimate_basis"] for row in storage.values()], "unmeasured planning range")
    partitions = _rows("slurm_partition_snapshot.csv")
    _check(checks, "future_partitions_available_not_authorized", any(row["partition"] == "V100" and row["availability"] == "up" for row in partitions) and any(row["partition"] == "cpu-high" and row["availability"] == "up" for row in partitions) and all(row["C77_authorization"] == "0" for row in partitions), partitions, "V100/cpu-high up;authorization=0")

    attempts = _rows("execution_attempt_ledger.csv")
    boundary_pass = all(row["training"] == row["real_forward"] == row["GPU"] == row["seed3_access"] == row["seed4_access"] == row["BNCI2014_004_access"] == "0" for row in attempts)
    _check(checks, "zero_real_execution_boundary", boundary_pass and state["execution_boundary"] == {"BNCI2014_004_access": 0, "GPU": 0, "checkpoints_created": 0, "re_inference": 0, "real_forward": 0, "seed3_access": 0, "seed4_access": 0, "training": 0}, state["execution_boundary"], "all zero")
    risks = _rows("risk_register.csv")
    _check(checks, "no_open_blocking_risk", not [row for row in risks if row["blocking_open"] == "1"], sum(row["blocking_open"] == "1" for row in risks), 0)
    analysis_failures = _rows("analysis_failure_reason_ledger.csv")
    _check(checks, "analysis_failure_ledger_separate_and_clear", len(analysis_failures) == 5 and not [row for row in analysis_failures if row["blocking"] == "1"], f"rows={len(analysis_failures)};blocking={sum(row['blocking']=='1' for row in analysis_failures)}", "5;0;separate from locked protocol ledger")
    external = {row["requirement"]: row for row in _rows("external_dataset_readiness.csv")}
    _check(checks, "external_dataset_not_accessed", external["dataset_access"]["status"] == "not_accessed_in_C77", external["dataset_access"]["status"], "not_accessed_in_C77")

    log_ok, log_detail = True, []
    for index in range(8):
        stdout = LOG_ROOT / f"c77-synthetic_892728_{index}.out"
        stderr = LOG_ROOT / f"c77-synthetic_892728_{index}.err"
        text = stderr.read_text() if stderr.is_file() else ""
        ok = stdout.is_file() and stderr.is_file() and "finished" in text and "Traceback" not in text and "Error" not in text
        log_ok &= ok; log_detail.append(f"{index}:{int(ok)}")
    analyze_outputs = sorted(LOG_ROOT.glob("c77-analyze_*.out"), key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    analyze_out = analyze_outputs[-1] if analyze_outputs else Path("missing")
    analyze_err = analyze_out.with_suffix(".err")
    log_ok &= analyze_out.is_file() and analyze_err.is_file() and not analyze_err.read_text() and "READY_BUT_NOT_AUTHORIZED" in analyze_out.read_text()
    _check(checks, "slurm_jobs_complete_cleanly", log_ok, ";".join(log_detail), "8 synthetic complete;analysis stderr empty")

    tracked = _git("ls-files").splitlines()
    large = [(path, Path(path).stat().st_size) for path in tracked if Path(path).is_file() and Path(path).stat().st_size > 50 * 1024 * 1024]
    raw_extensions = [path for path in tracked if path.startswith("oaci/") and Path(path).suffix in {".pt", ".pth", ".npz", ".npy", ".parquet"}]
    _check(checks, "artifact_hygiene_no_large_or_raw_payload", not large and not raw_extensions, f"large={large};raw={raw_extensions}", "none")

    blocking_failures = [row for row in checks if row["blocking"] == 1 and row["passed"] == 0]
    repairs = [
        {"item": "R1_level_dimension", "status": "repaired_before_protocol_lock", "finding": "initial draft counted one level", "resolution": "C78 locks levels 0 and 1; 1458 units/seed"},
        {"item": "R2_ERM_trajectory_symmetry", "status": "repaired_before_protocol_lock", "finding": "ERM has no 40-point stage2 trajectory", "resolution": "ERM labeled one-point shared anchor; OACI/SRC are comparable trajectories"},
        {"item": "R3_SRC_history", "status": "repaired_before_protocol_lock", "finding": "blanket no-target-history wording was too broad", "resolution": "SRC disclosed as post-C10, pre-C14 negative control; C12 falsification retained"},
        {"item": "R4_synthetic_multiplicity_effect", "status": "claim_narrowed", "finding": f"registered directional contrast is only {synthetic_metrics['effective_multiplicity_reduces_actionability']:.6f}", "resolution": "passes locked direction only; no material-effect claim; C78 must recalibrate"},
        {"item": "R5_training_runtime", "status": "claim_narrowed", "finding": "no measured C78 GPU runtime exists", "resolution": "compute table reports a budget range, with mandatory P1 recalibration before P2"},
        {"item": "R6_locked_failure_ledger", "status": "repaired_after_blocking_red_team", "finding": "analysis overwrote a protocol-hash-locked failure ledger", "resolution": "restored locked bytes; post-compute outcomes moved to analysis_failure_reason_ledger.csv"},
        {"item": "R7_living_handoff_manifest", "status": "regression_contract_repaired", "finding": "C75 replay treated the later-updated handoff as immutable payload", "resolution": "preserve C75 historical row but replay current bytes only for immutable artifacts; C77 excludes handoff from its artifact manifest"},
    ]
    _write_csv(REPAIR_PATH, repairs)
    _write_csv(CHECKS_PATH, checks)
    status = "PASS" if not blocking_failures else "FAIL"
    REPORT_PATH.write_text(
        "# C77 Red-Team Verification\n\n"
        f"Final status: `{status}`\n\n"
        f"- Main C77 report existed before red-team: `{str(main_existed).lower()}`.\n"
        f"- Blocking checks passed: `{sum(row['blocking'] == 1 and row['passed'] == 1 for row in checks)}/{sum(row['blocking'] == 1 for row in checks)}`.\n"
        f"- Nonblocking caveats failed: `{sum(row['blocking'] == 0 and row['passed'] == 0 for row in checks)}`.\n"
        "- Real training / EEG forward / seed-3 / seed-4 / BNCI2014_004 access: `0 / 0 / 0 / 0 / 0`.\n"
        f"- Synthetic null FPR: `{synthetic_metrics['null_association_FPR']:.6f}`; stable-signal power: `{synthetic_metrics['stable_local_association_power']:.6f}`.\n"
        f"- Effective-multiplicity actionability contrast: `{synthetic_metrics['effective_multiplicity_reduces_actionability']:.6f}` (directional only; not material).\n\n"
        "## Adversarial repairs and limits\n\n"
        + "\n".join(f"- **{row['item']}**: {row['finding']} Resolution: {row['resolution']}" for row in repairs)
        + "\n\n## Claim boundary\n\n"
        "C77 can conclude that an exact, target-isolated, multi-regime seed-3 protocol is recoverable, powered in the registered synthetic benchmark, and compute/storage feasible. It cannot conclude that seed-3 training is authorized, that any EEG hypothesis replicated, that the representation mechanism is identified, or that a control rule is deployable.\n"
    )
    if blocking_failures:
        raise RuntimeError(f"C77 red-team blocking failures: {[row['check'] for row in blocking_failures]}")
    print(json.dumps({"status": status, "checks": len(checks), "blocking": sum(row["blocking"] == 1 for row in checks), "repairs": len(repairs)}, sort_keys=True))
    return {"status": status, "checks": checks, "repairs": repairs}


def _rows_from_path(path: Path) -> list[dict]:
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


if __name__ == "__main__":
    run_red_team()
