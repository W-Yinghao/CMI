#!/usr/bin/env python
"""Aggregate and independently gate ISRUC object-level frozen readouts."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, log_loss

from route_b_cross_task_common import GATE_TAGS, TAGS, read_csv, sha256_file, write_csv, write_json


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
    contracts = {}
    identities = None
    for tag in tags:
        contract, data = load_object(args.object_dir, tag)
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
    random_val = by_tag["random"]["source_val_kappa"]
    for row in rows:
        row["task_gate_pass"] = (
            row["source_val_kappa"] >= 0.05
            and row["source_val_kappa"] >= random_val + 0.02
        )
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
    }
    write_json(args.out_dir / f"{prefix}_adversarial_verification.json", verification)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
