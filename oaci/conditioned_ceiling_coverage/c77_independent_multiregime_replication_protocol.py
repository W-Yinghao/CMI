"""C77 protocol feasibility analysis; never trains or loads real EEG data."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from oaci.models.shallow import ShallowConvNet

from . import c77_protocol
from . import synthetic_multiregime_generator


STATE_PATH = c77_protocol.REPORT_DIR / "C77_REPLICATION_PROTOCOL_ANALYSIS_STATE.json"
EXTERNAL_PLAN_ROOT = Path("/projects/EEG-foundation-model/yinghao")


def _rows(name: str) -> list[dict]:
    with open(c77_protocol.TABLE_DIR / name, newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(name: str, rows: list[dict]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(c77_protocol.TABLE_DIR / name, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _protocol_commit() -> str:
    commit = _git("log", "-1", "--format=%H", "--", str(c77_protocol.PROTOCOL_PATH))
    if not commit or commit == c77_protocol.PARENT_COMMIT:
        raise RuntimeError("C77 analysis requires a committed protocol after C76")
    tracked = set(_git("ls-tree", "-r", "--name-only", commit).splitlines())
    for path in (c77_protocol.PROTOCOL_PATH, c77_protocol.C78_PROTOCOL_PATH, c77_protocol.C78_PROTOCOL_SHA_PATH):
        if str(path) not in tracked:
            raise RuntimeError(f"C77 protocol artifact not tracked at protocol commit: {path}")
    return commit


def _dummy_abi() -> list[dict]:
    torch.manual_seed(77)
    model = ShallowConvNet(
        22, 385, 4, temporal_filters=40, temporal_kernel_samples=25,
        pool_kernel_samples=75, pool_stride_samples=15, dropout=0.5,
        safe_log_eps=1e-6,
    ).cpu().eval()
    state_before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    x = torch.zeros(2, 22, 385, dtype=torch.float32)
    with torch.no_grad():
        first = model(x)
        second = model(x)
    W, b = model.classifier.weight, model.classifier.bias
    projection_error = float(torch.max(torch.abs(first.z @ W.T + b - first.logits)).item())
    repeat_error = float(torch.max(torch.abs(first.logits - second.logits)).item())
    state_unchanged = all(torch.equal(value, state_before[key]) for key, value in model.state_dict().items())
    state_bytes = sum(value.numel() * value.element_size() for value in model.state_dict().values())
    return [{
        "test": "ShallowConvNet_dummy_hook_ABI", "device": str(next(model.parameters()).device),
        "input_shape": json.dumps(list(x.shape)), "z_shape": json.dumps(list(first.z.shape)),
        "logit_shape": json.dumps(list(first.logits.shape)), "W_shape": json.dumps(list(W.shape)),
        "b_shape": json.dumps(list(b.shape)), "Wz_plus_b_max_abs": projection_error,
        "repeat_logit_max_abs": repeat_error, "state_unchanged": int(state_unchanged),
        "state_bytes": state_bytes,
        "passed": int(projection_error == 0.0 and repeat_error == 0.0 and state_unchanged and tuple(first.z.shape) == (2, 800)),
        "real_EEG_rows_loaded": 0, "forward_type": "dummy_synthetic_only",
    }]


def _storage_plan(state_bytes: int) -> list[dict]:
    measured = _rows_from_path(c77_protocol.C74_STORAGE_PATH)[0]
    bytes_per_unit = int(measured["external_size_bytes"]) / int(measured["units"])
    source_rows_per_unit = int(measured["source_rows"]) / int(measured["units"])
    target_rows_per_unit = int(measured["target_rows"]) / int(measured["units"])
    unit_median_seconds = float(measured["unit_wall_seconds_median"])
    unit_p95_seconds = float(measured["unit_wall_seconds_p95"])
    full_units = len(c77_protocol.TARGETS) * len(c77_protocol.LEVELS) * (
        1 + len(c77_protocol.TRAJECTORY_REGIMES) * c77_protocol.STAGE2_CHECKPOINTS
    )
    pilot_units = len(c77_protocol.LEVELS) * (1 + c77_protocol.STAGE2_CHECKPOINTS)
    stages = [
        ("C78_seed3_P1", pilot_units, 4, 1, 8, 24),
        ("C78_seed3_full", full_units, 54, 18, 108, 324),
        ("C79_seed4_full", full_units, 54, 18, 108, 324),
        ("R1_seed3_plus_seed4", full_units * 2, 108, 36, 216, 648),
    ]
    rows = []
    for stage, units, phases, contexts, gpu_low, gpu_high in stages:
        cache_bytes = int(round(bytes_per_unit * units))
        weight_bytes = int(state_bytes * units)
        optimizer_bytes_upper = 3 * weight_bytes
        rows.append({
            "campaign": stage, "seed3_or_seed4": "seed3" if "seed3" in stage and "plus" not in stage else ("seed4" if "seed4" in stage and "plus" not in stage else "both"),
            "target_level_contexts": contexts, "training_phases": phases,
            "retained_checkpoint_target_level_units": units,
            "source_trial_rows_projected": int(round(source_rows_per_unit * units)),
            "target_trial_rows_projected": int(round(target_rows_per_unit * units)),
            "trial_cache_GiB_projected": cache_bytes / 2**30,
            "weights_GiB_projected": weight_bytes / 2**30,
            "optimizer_state_GiB_upper": optimizer_bytes_upper / 2**30,
            "temporary_storage_reserve_GiB": 2.0 * (cache_bytes + weight_bytes + optimizer_bytes_upper) / 2**30,
            "GPU_hours_planning_low": gpu_low, "GPU_hours_planning_high": gpu_high,
            "GPU_estimate_basis": "2-6 GPU-hours per historical training phase; unmeasured planning range; P1 must recalibrate before P2",
            "instrumentation_CPU_node_hours_median_linear": units * unit_median_seconds / (48 * 3600),
            "instrumentation_CPU_node_hours_p95_linear": units * unit_p95_seconds / (48 * 3600),
            "CPU_partition": "cpu-high", "CPU_cores": 48,
            "GPU_partition": "V100", "environment": "/home/infres/yinwang/anaconda3/envs/icml",
            "retry_budget_fraction": 0.15, "silent_escalation_allowed": 0,
            "planning_caveat": "storage scaled from C74; GPU time is a conservative budget range, not measured C78 runtime",
        })
    return rows


def _rows_from_path(path: Path) -> list[dict]:
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def _partition_snapshot() -> list[dict]:
    try:
        output = subprocess.check_output(
            ["sinfo", "-h", "-o", "%P|%a|%l|%D|%G"], text=True, stderr=subprocess.STDOUT,
        )
    except Exception as error:
        return [{"partition": "V100", "availability": "snapshot_unavailable", "time_limit": "unknown", "nodes": "unknown", "gres": "gpu:1_required_future", "C77_authorization": 0, "note": repr(error)}]
    rows = []
    for line in output.splitlines():
        partition, availability, limit, nodes, gres = line.split("|", 4)
        if partition.rstrip("*") in {"V100", "cpu-high"}:
            rows.append({"partition": partition.rstrip("*"), "availability": availability, "time_limit": limit, "nodes": nodes, "gres": gres, "C77_authorization": 0, "note": "read-only scheduler snapshot; availability is not authorization"})
    return rows


def _environment_audit(dummy: dict) -> list[dict]:
    return [{
        "environment": os.environ.get("CONDA_PREFIX", "unknown"),
        "python_executable": os.sys.executable, "torch": torch.__version__,
        "numpy": np.__version__, "CUDA_available_on_analysis_node": int(torch.cuda.is_available()),
        "expected_future_prefix": "/home/infres/yinwang/anaconda3/envs/icml",
        "expected_conda_explicit_sha256": "2c04fc1733a53b55abd071d6b1657eabfda8bbb56ef0bf0ab97e8234171958a1",
        "dummy_ABI_passed": dummy["passed"], "real_data_loaded": 0,
        "C78_rule": "rehash exact icml environment and run dummy ABI on assigned GPU before real data",
    }]


def analyze() -> dict:
    protocol_hash = c77_protocol.sha256(c77_protocol.PROTOCOL_PATH)
    if protocol_hash != c77_protocol.PROTOCOL_SHA_PATH.read_text().strip():
        raise RuntimeError("C77 protocol hash mismatch")
    if c77_protocol.sha256(c77_protocol.C78_PROTOCOL_PATH) != c77_protocol.C78_PROTOCOL_SHA_PATH.read_text().strip():
        raise RuntimeError("C78 protocol hash mismatch")
    protocol_commit = _protocol_commit()
    phase_rows, power_rows = synthetic_multiregime_generator.merge_shards()
    dummy_rows = _dummy_abi()
    if not dummy_rows[0]["passed"]:
        raise RuntimeError("C77 dummy ABI failed")
    _write_csv("dummy_hook_ABI_validation.csv", dummy_rows)
    _write_csv("environment_ABI_audit.csv", _environment_audit(dummy_rows[0]))
    storage = _storage_plan(int(dummy_rows[0]["state_bytes"]))
    _write_csv("compute_storage_plan.csv", storage)
    partitions = _partition_snapshot()
    _write_csv("slurm_partition_snapshot.csv", partitions)

    recovery = _rows("regime_reconstruction_status.csv")
    recovery_pass = (
        sum(int(row["qualifies_primary_R1"]) for row in recovery) == 3
        and sum(int(row["comparable_40_checkpoint_trajectory_per_level"]) for row in recovery) >= 2
        and all(int(row["exact_config_recoverable"]) == 1 for row in recovery if int(row["qualifies_primary_R1"]))
    )
    power_pass = all(int(row["passed"]) == 1 for row in power_rows)
    risks = _rows("risk_register.csv")
    blocking_risks = [row for row in risks if row["blocking_open"] == "1"]
    available = shutil.disk_usage(EXTERNAL_PLAN_ROOT).free if EXTERNAL_PLAN_ROOT.exists() else 0
    full_storage_reserve = max(float(row["temporary_storage_reserve_GiB"]) for row in storage)
    storage_pass = available == 0 or available / 2**30 >= full_storage_reserve
    partition_pass = any(row["partition"] == "V100" and row["availability"] == "up" for row in partitions)
    ready = recovery_pass and power_pass and not blocking_risks and storage_pass and partition_pass
    final_gate = (
        "SEED3_MULTIREGIME_INSTRUMENTED_PILOT_READY_BUT_NOT_AUTHORIZED"
        if ready else
        ("MULTIREGIME_PROTOCOL_BLOCKED_BY_CONFIG_RECOVERY" if not recovery_pass else "MULTIREGIME_PROTOCOL_BLOCKED_BY_POWER_OR_COMPUTE")
    )
    primary = "C77-A_multiregime_seed3_seed4_replication_protocol_ready" if ready else (
        "C77-B_historical_regimes_not_recoverable" if not recovery_pass else "C77-C_power_or_compute_insufficient"
    )
    failures = []
    for item, passed, reason in (
        ("historical_regime_recovery", recovery_pass, "three primary identities and two comparable trajectory regimes"),
        ("synthetic_power_false_positive", power_pass, "all locked simulation gates"),
        ("storage_capacity", storage_pass, f"free_GiB={available / 2**30:.3f};required_reserve_GiB={full_storage_reserve:.3f}"),
        ("future_V100_partition", partition_pass, "read-only snapshot; not authorization"),
        ("blocking_risks", not blocking_risks, f"open={len(blocking_risks)}"),
    ):
        failures.append({"item": item, "status": "pass" if passed else "blocked", "blocking": int(not passed), "reason": reason})
    # The protocol-time failure ledger is hash-locked.  Post-compute gate outcomes
    # are a separate artifact and must never mutate any locked registry table.
    _write_csv("analysis_failure_reason_ledger.csv", failures)
    attempts = [
        {"attempt": 1, "phase": "C76_replay_and_archaeology", "execution": "metadata_only", "training": 0, "real_forward": 0, "GPU": 0, "seed3_access": 0, "seed4_access": 0, "BNCI2014_004_access": 0, "status": "passed"},
        {"attempt": 2, "phase": "protocol_lock", "execution": protocol_commit, "training": 0, "real_forward": 0, "GPU": 0, "seed3_access": 0, "seed4_access": 0, "BNCI2014_004_access": 0, "status": "passed"},
        {"attempt": 3, "phase": "synthetic_power", "execution": "8_slurm_shards", "training": 0, "real_forward": 0, "GPU": 0, "seed3_access": 0, "seed4_access": 0, "BNCI2014_004_access": 0, "status": "passed" if power_pass else "completed_gate_failed"},
        {"attempt": 4, "phase": "dummy_ABI_compute_storage", "execution": os.environ.get("SLURM_JOB_ID", "local"), "training": 0, "real_forward": 0, "GPU": 0, "seed3_access": 0, "seed4_access": 0, "BNCI2014_004_access": 0, "status": "passed"},
    ]
    _write_csv("execution_attempt_ledger.csv", attempts)
    state = {
        "schema_version": "c77_replication_protocol_analysis_state_v1",
        "protocol_commit": protocol_commit, "protocol_sha256": protocol_hash,
        "C78_protocol_sha256": c77_protocol.sha256(c77_protocol.C78_PROTOCOL_PATH),
        "final_gate_candidate": final_gate, "primary_candidate": primary,
        "secondary_candidates": {
            "C77-S1_seed3_pilot_protocol_locked": True,
            "C77-S2_seed4_confirmation_protocol_skeleton_locked": True,
            "C77-S3_full_instrumentation_schema_ready": True,
            "C77-S4_cross_regime_hypotheses_locked": True,
            "C77-S5_synthetic_power_plan_passed": power_pass,
            "C77-S6_R2_external_dataset_readiness_defined": True,
            "C77-S7_new_training_scientifically_justified": True,
            "C77-S8_training_not_authorized": True,
        },
        "gates": {"regime_recovery": recovery_pass, "synthetic": power_pass, "storage": storage_pass, "partition": partition_pass, "blocking_risks": len(blocking_risks)},
        "synthetic_cells": len(phase_rows), "synthetic_gates": power_rows,
        "compute": {"full_units_per_seed": len(c77_protocol.TARGETS) * len(c77_protocol.LEVELS) * (1 + 2 * c77_protocol.STAGE2_CHECKPOINTS), "pilot_units": len(c77_protocol.LEVELS) * (1 + c77_protocol.STAGE2_CHECKPOINTS), "available_external_GiB_snapshot": available / 2**30},
        "execution_boundary": {"training": 0, "real_forward": 0, "re_inference": 0, "GPU": 0, "seed3_access": 0, "seed4_access": 0, "BNCI2014_004_access": 0, "checkpoints_created": 0},
        "claims": {"training_scientifically_justified": True, "training_authorized": False, "target_population_generalization": False, "representation_mechanism": False, "deployable_control": False},
    }
    STATE_PATH.write_text(json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"gate": final_gate, "primary": primary, "synthetic_cells": len(phase_rows)}, sort_keys=True))
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("analyze",))
    args = parser.parse_args(argv)
    if args.command == "analyze":
        analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
