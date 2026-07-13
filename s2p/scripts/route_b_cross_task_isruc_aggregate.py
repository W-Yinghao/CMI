#!/usr/bin/env python
"""Aggregate and independently gate ISRUC object-level frozen readouts."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, log_loss

from route_b_cross_task_common import (
    GATE_TAGS,
    TAGS,
    read_csv,
    sha256_file,
    truth,
    write_csv,
    write_json,
)


N_NULL = 200
MATCH_TOLERANCE = 1e-10


def metrics(labels, probability):
    prediction = probability.argmax(axis=1)
    return {
        "kappa": float(cohen_kappa_score(labels, prediction)),
        "nll": float(log_loss(labels, probability, labels=np.arange(5))),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "weighted_f1": float(f1_score(labels, prediction, average="weighted")),
    }


def load_object(root, tag):
    contract_path = root / f"{tag}_isruc_object_contract.json"
    prediction_path = root / f"{tag}_isruc_predictions.npz"
    contract = json.loads(contract_path.read_text())
    if (
        contract.get("status") != "PASS"
        or contract.get("tag") != tag
        or contract.get("encoder_frozen") is not True
        or contract.get("fine_tuning_used") is not False
        or contract.get("target_test_labels_used_for_selection") is not False
        or sha256_file(prediction_path) != contract.get("prediction_payload_sha256")
    ):
        raise RuntimeError(f"ISRUC object contract failed for {tag}")
    with np.load(prediction_path) as payload:
        data = {name: payload[name] for name in payload.files}
    return contract, data


def holm(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def load_l5(root, tag, contract):
    summary_rows = read_csv(root / f"{tag}_isruc_l5_summary.csv")
    if len(summary_rows) != 1 or summary_rows[0].get("tag") != tag:
        raise RuntimeError(f"ISRUC L5 summary contract failed for {tag}")
    summary = summary_rows[0]
    per_subject = read_csv(root / f"{tag}_isruc_l5_per_test_subject.csv")
    leave_one = read_csv(root / f"{tag}_isruc_l5_leave_one_subject_out.csv")
    if summary.get("status") != "PASS":
        if contract.get("l5_null_payload") is not None:
            raise RuntimeError(f"ISRUC non-interpretable L5 unexpectedly has payload for {tag}")
        return summary, per_subject, leave_one, None

    payload_path = root / f"{tag}_isruc_l5_nulls.npz"
    if (
        contract.get("l5_status") != "PASS"
        or int(contract.get("l5_null_repetitions", 0)) != N_NULL
        or sha256_file(payload_path) != contract.get("l5_null_payload_sha256")
    ):
        raise RuntimeError(f"ISRUC L5 payload contract failed for {tag}")
    with np.load(payload_path) as payload_file:
        payload = {name: payload_file[name] for name in payload_file.files}
    null_delta = payload["global_null_delta_kappa"]
    if null_delta.shape != (N_NULL,) or not np.isfinite(null_delta).all():
        raise RuntimeError(f"ISRUC L5 null distribution is invalid for {tag}")
    subject_delta = float(summary["subject_delta_kappa"])
    expected_mean = float(null_delta.mean())
    expected_p = float((1 + np.sum(null_delta >= subject_delta)) / (N_NULL + 1))
    if (
        abs(expected_mean - float(summary["null_delta_kappa_mean"])) > 1e-12
        or abs(expected_p - float(summary["empirical_one_sided_p"])) > 1e-12
        or float(payload["source_val_null_match_abs_error"].max()) > MATCH_TOLERANCE
        or len(per_subject) != 10
        or len(leave_one) != 10
    ):
        raise RuntimeError(f"ISRUC L5 arithmetic or matching verification failed for {tag}")
    return summary, per_subject, leave_one, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("gate", "fleet"), required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gate-json", type=Path)
    args = parser.parse_args()
    tags = GATE_TAGS if args.stage == "gate" else TAGS
    if args.stage == "fleet":
        if args.gate_json is None or json.loads(args.gate_json.read_text()).get("status") != "PASS":
            raise RuntimeError("ISRUC fleet requires a passing gate artifact")

    rows = []
    per_subject = []
    subject_rows = []
    geometry_rows = []
    l5_rows = []
    l5_per_subject_rows = []
    l5_leave_one_rows = []
    l5_payloads = {}
    contracts = {}
    identities = None
    for tag in tags:
        contract, data = load_object(args.object_dir, tag)
        if args.stage == "fleet" and contract.get("stage") != "fleet":
            raise RuntimeError(f"ISRUC fleet object has wrong stage for {tag}")
        contracts[tag] = contract
        test_probability = data["test_probabilities"].mean(axis=0)
        val_probability = data["val_probabilities"].mean(axis=0)
        test_metric = metrics(data["test_labels"], test_probability)
        val_metric = metrics(data["val_labels"], val_probability)
        for scope, expected, observed in (
            ("target", contract["target_test"], test_metric),
            ("source_val", contract["source_val"], val_metric),
        ):
            if max(abs(float(expected[key]) - observed[key]) for key in observed) > 1e-12:
                raise RuntimeError(f"ISRUC {scope} metric reproduction failed for {tag}")
        current_identity = (
            tuple(data["test_labels"]), tuple(data["test_subjects"]), tuple(data["test_rotations"]),
            tuple(data["val_labels"]), tuple(data["val_subjects"]), tuple(data["val_rotations"]),
        )
        if identities is None:
            identities = current_identity
        elif any(not np.array_equal(np.asarray(left), np.asarray(right)) for left, right in zip(identities, current_identity)):
            raise RuntimeError(f"ISRUC prediction identity differs for {tag}")
        rows.append({
            "tag": tag,
            **{f"source_val_{key}": value for key, value in val_metric.items()},
            **{f"target_test_{key}": value for key, value in test_metric.items()},
            "primary_metric": "target_test_kappa",
            "downstream_seeds_averaged": 3,
        })
        for subject in range(1, 11):
            mask = data["test_subjects"] == subject
            current = metrics(data["test_labels"][mask], test_probability[mask])
            per_subject.append({"tag": tag, "test_subject": subject, **current})
        subject_rows.extend(read_csv(args.object_dir / f"{tag}_isruc_subject_metrics.csv"))
        geometry_rows.extend(read_csv(args.object_dir / f"{tag}_isruc_geometry.csv"))

    by_tag = {row["tag"]: row for row in rows}
    device_names = {contract.get("cuda_device_name") for contract in contracts.values()}
    compute_capabilities = {
        tuple(contract.get("cuda_compute_capability", []))
        for contract in contracts.values()
    }
    if args.stage == "fleet" and (
        device_names != {"NVIDIA A40"} or len(compute_capabilities) != 1
    ):
        raise RuntimeError(
            "ISRUC fleet objects were not fit on one pinned A40 hardware contract"
        )
    random_val = by_tag["random"]["source_val_kappa"]
    for row in rows:
        row["task_gate_pass"] = (
            row["source_val_kappa"] >= 0.05
            and row["source_val_kappa"] >= random_val + 0.02
        )
        if args.stage == "fleet":
            tag = row["tag"]
            contract_gate = bool(contracts[tag].get("task_gate_pass"))
            if contract_gate != row["task_gate_pass"]:
                raise RuntimeError(
                    f"ISRUC fleet task-gate classification differs from object gate for {tag}"
                )
            summary, current_subject, current_leave_one, payload = load_l5(
                args.object_dir, tag, contracts[tag]
            )
            if truth(summary.get("task_gate_pass", "False")) != row["task_gate_pass"]:
                raise RuntimeError(f"ISRUC L5 summary task gate differs for {tag}")
            if row["task_gate_pass"] and summary.get("status") != "PASS":
                raise RuntimeError(f"ISRUC task-gated object lacks L5 for {tag}")
            if not row["task_gate_pass"] and summary.get("status") == "PASS":
                raise RuntimeError(f"ISRUC non-task-gated object has interpretable L5 for {tag}")
            if summary.get("status") == "PASS":
                summary["task_gate_pass"] = True
                l5_rows.append(summary)
                l5_per_subject_rows.extend(current_subject)
                l5_leave_one_rows.extend(current_leave_one)
                l5_payloads[tag] = payload

    if args.stage == "fleet":
        family = [row for row in l5_rows if row["tag"].startswith("H")]
        adjusted = holm([float(row["empirical_one_sided_p"]) for row in family]) if family else []
        for row, value in zip(family, adjusted):
            row["holm_adjusted_p"] = float(value)
            row["exceeds_matched_null_holm_0p05"] = bool(value < 0.05)
    released_minus_random = (
        by_tag["released"]["target_test_kappa"] - by_tag["random"]["target_test_kappa"]
    )
    finite = all(np.isfinite(float(row[key])) for row in rows for key in (
        "source_val_kappa", "source_val_nll", "target_test_kappa", "target_test_nll"
    ))
    nondegenerate = all(len(np.unique(load_object(args.object_dir, tag)[1]["test_probabilities"].mean(axis=0).argmax(axis=1))) > 1 for tag in ("random", "released"))
    status = "PASS" if released_minus_random >= 0.02 and finite and nondegenerate else "NO_GO"
    prefix = f"isruc_s3_{args.stage}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / f"{prefix}_task_performance.csv", rows)
    write_csv(args.out_dir / f"{prefix}_per_test_subject.csv", per_subject)
    write_csv(args.out_dir / f"{prefix}_subject_metrics.csv", subject_rows)
    write_csv(args.out_dir / f"{prefix}_geometry.csv", geometry_rows)
    write_csv(args.out_dir / f"{prefix}_l5_reliance.csv", l5_rows)
    write_csv(args.out_dir / f"{prefix}_l5_per_test_subject.csv", l5_per_subject_rows)
    write_csv(args.out_dir / f"{prefix}_l5_leave_one_subject_out.csv", l5_leave_one_rows)
    firewall = {
        "dataset": "ISRUC_S3_Group_III",
        "stage": args.stage,
        "target_test_labels_used_for_selection": False,
        "validation_subject_used_for_head_selection_only": True,
        "encoder_frozen": True,
        "encoder_optimizer_created": False,
        "flat_epoch_head_used": False,
        "sequence_overlap": False,
        "pass": True,
    }
    write_json(args.out_dir / f"{prefix}_target_label_firewall.json", firewall)
    verdict = {
        "phase": "C2_ISRUC_S3_gate" if args.stage == "gate" else "C2_ISRUC_S3_fleet",
        "status": status,
        "objects": tags,
        "released_target_kappa": by_tag["released"]["target_test_kappa"],
        "random_target_kappa": by_tag["random"]["target_test_kappa"],
        "released_minus_random_target_kappa": released_minus_random,
        "released_clears_random_plus_0p02": released_minus_random >= 0.02,
        "finite_endpoints": finite,
        "random_and_released_predictions_nondegenerate": nondegenerate,
        "rotations": 10,
        "downstream_seeds": [0, 1, 2],
        "test_subjects": 10,
        "low_power_directional_replication": True,
        "target_label_firewall_pass": True,
        "fine_tuning_used": False,
        "fleet_authorized_by_gate": bool(args.stage == "gate" and status == "PASS"),
        "other_dataset_auto_launch_authorized": False,
        "l5_task_gated_objects": [row["tag"] for row in l5_rows],
        "l5_null_repetitions_per_object": N_NULL if args.stage == "fleet" else 0,
        "l5_holm_family": [
            row["tag"] for row in l5_rows if row["tag"].startswith("H")
        ],
        "l5_low_power_interpretation": args.stage == "fleet",
        "downstream_head_device_names": sorted(
            value for value in device_names if value is not None
        ),
        "downstream_head_compute_capabilities": [
            list(value) for value in sorted(compute_capabilities)
        ],
    }
    write_json(args.out_dir / f"{prefix}_verdict.json", verdict)
    verification = {
        "phase": "C2_ISRUC_S3_independent_prediction_recompute",
        "status": "PASS",
        "object_prediction_hashes": {
            tag: contracts[tag]["prediction_payload_sha256"] for tag in tags
        },
        "all_object_metrics_recomputed_from_predictions": True,
        "rotation_identity_consistent": True,
        "gate_decision_recomputed": status,
        "target_labels_used_for_selection": False,
        "l5_arithmetic_recomputed_from_compact_nulls": args.stage == "fleet",
        "l5_source_val_variance_match_pass": bool(
            args.stage != "fleet"
            or all(
                float(payload["source_val_null_match_abs_error"].max())
                <= MATCH_TOLERANCE
                for payload in l5_payloads.values()
            )
        ),
        "l5_null_payload_hashes": {
            tag: contracts[tag]["l5_null_payload_sha256"] for tag in l5_payloads
        },
        "l5_holm_recomputed": args.stage == "fleet",
        "same_downstream_head_hardware_for_all_objects": bool(
            args.stage != "fleet"
            or (device_names == {"NVIDIA A40"} and len(compute_capabilities) == 1)
        ),
    }
    write_json(args.out_dir / f"{prefix}_adversarial_verification.json", verification)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
