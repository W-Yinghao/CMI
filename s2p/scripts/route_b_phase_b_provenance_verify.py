#!/usr/bin/env python
"""Independent metadata red-team for the Phase-B provenance closure."""
import argparse
import csv
import hashlib
import json
import stat
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_TAGS = {
    "random",
    "released",
    "H200_s0",
    "H200_s1",
    "H500_s0",
    "H500_s1",
    "H1000_s0",
    "H1000_s1",
    "H2000_s0",
    "H2000_s1",
}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fobj:
        while True:
            chunk = fobj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as fobj:
        return list(csv.DictReader(fobj))


def read_json(path):
    return json.loads(Path(path).read_text())


def as_bool(value):
    return str(value).lower() == "true"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--closure-dir",
        default="results/s2p_route_b_phase_b_checkpoint_closure",
    )
    parser.add_argument(
        "--b0-go-nogo",
        default="results/s2p_route_b_representation_emergence_b0/phase_b0_go_nogo.json",
    )
    parser.add_argument(
        "--output",
        default="results/s2p_route_b_phase_b_checkpoint_closure/phase_b_provenance_redteam_verdict.json",
    )
    args = parser.parse_args()
    root = Path(args.closure_dir)
    manifest = read_csv(root / "phase_b_checkpoint_immutable_manifest.csv")
    copies = read_json(root / "phase_b_checkpoint_copy_verification.json")
    reload_rows = read_csv(root / "phase_b_checkpoint_reload_verification.csv")
    feature_rows = read_csv(root / "phase_b_feature_equivalence_rerun.csv")
    closure = read_json(root / "phase_b_provenance_closure.json")
    b0 = read_json(args.b0_go_nogo)

    failures = []
    manifest_tags = {row["tag"] for row in manifest}
    if len(manifest) != 10 or manifest_tags != EXPECTED_TAGS:
        failures.append("manifest_exact_ten_object_set")
    if closure.get("status") != "PASS":
        failures.append("closure_status")
    if {row["tag"] for row in reload_rows} != EXPECTED_TAGS:
        failures.append("reload_exact_ten_object_set")
    if {row["tag"] for row in feature_rows} != EXPECTED_TAGS:
        failures.append("feature_exact_ten_object_set")

    physical_hashes = {}
    mutable_path_consumed = False
    for row in manifest:
        tag = row["tag"]
        if tag == "random":
            if not row["immutable_path"].startswith("random_init_contract://sha256_"):
                failures.append("random_contract_path")
            if row["source_sha256"] != row["immutable_sha256"]:
                failures.append("random_contract_hash")
            continue
        path = Path(row["immutable_path"])
        if not path.is_file() or path.is_symlink():
            failures.append(f"direct_payload:{tag}")
            continue
        digest = sha256_file(path)
        physical_hashes[tag] = digest
        if digest != row["immutable_sha256"] or digest not in path.name:
            failures.append(f"content_address:{tag}")
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            failures.append(f"writable_payload:{tag}")
        if path.resolve() == Path(row["source_path"]).resolve():
            mutable_path_consumed = True
            failures.append(f"source_path_reused:{tag}")
        if int(row["immutable_size_bytes"]) != path.stat().st_size:
            failures.append(f"payload_size:{tag}")

    for row in reload_rows:
        if not as_bool(row["strict_reload_pass"]):
            failures.append(f"strict_reload:{row['tag']}")
        if not as_bool(row["parameter_exact_pass"]):
            failures.append(f"parameter_exact:{row['tag']}")
        if row["source_parameter_sha256"] != row["immutable_parameter_sha256"]:
            failures.append(f"parameter_hash:{row['tag']}")
        if int(row["missing_key_count"]) or int(row["unexpected_key_count"]):
            failures.append(f"state_keys:{row['tag']}")

    for row in feature_rows:
        if not as_bool(row["feature_equivalence_pass"]):
            failures.append(f"feature_equivalence:{row['tag']}")
        if float(row["source_repeat_max_abs_diff"]) != 0.0:
            failures.append(f"feature_repeat:{row['tag']}")
        if float(row["source_vs_immutable_max_abs_diff"]) != 0.0:
            failures.append(f"feature_copy:{row['tag']}")
        if row["source_feature_sha256"] != row["immutable_feature_sha256"]:
            failures.append(f"feature_hash:{row['tag']}")

    if copies.get("approved_copy_count") != 7 or copies.get("copied_or_reused_count") != 7:
        failures.append("approved_copy_count")
    for row in copies.get("copies", []):
        required = [
            "source_double_sha_stable",
            "destination_no_overwrite_contract",
            "byte_integrity_pass",
        ]
        if not all(row.get(key) is True for key in required):
            failures.append(f"copy_contract:{row.get('tag')}")
        if row.get("source_sha256_first") != row.get("source_sha256_second"):
            failures.append(f"source_double_hash:{row.get('tag')}")

    route_rows = [row for row in manifest if row["tag"].startswith("H")]
    if not all(row["selection_metric"] == "pretrain_val_loss_only" for row in route_rows):
        failures.append("checkpoint_selection")
    released = next(row for row in manifest if row["tag"] == "released")
    if released["training_provenance"] != "externally_released_locally_unverified":
        failures.append("released_provenance_boundary")
    required_b0_true = [
        "phase_b0_design_pass",
        "clip_grouping_pass",
        "feature_path_reproduction_pass",
        "all_checkpoint_sha256_pinned",
        "all_checkpoint_strict_reload_pass",
        "all_checkpoint_feature_equivalence_pass",
        "phase_b1_go_recommended",
    ]
    if not all(b0.get(key) is True for key in required_b0_true):
        failures.append("post_closure_b0_gate")
    if b0.get("phase_b1_compute_authorized") is not False:
        failures.append("b1_authorization_must_remain_false")
    if b0.get("immutable_blocking_tags") != []:
        failures.append("b0_blockers_not_empty")

    verdict = {
        "phase": "B_checkpoint_provenance_independent_redteam",
        "status": "PASS" if not failures else "NO_GO",
        "failures": failures,
        "checkpoint_count_expected": 10,
        "checkpoint_count_verified": len(manifest),
        "physical_payload_count_verified": len(physical_hashes),
        "deterministic_random_contract_count": 1,
        "all_physical_payloads_direct": not any(item.startswith("direct_payload") for item in failures),
        "all_physical_payloads_content_addressed": not any(item.startswith("content_address") for item in failures),
        "all_physical_payloads_read_only": not any(item.startswith("writable_payload") for item in failures),
        "all_strict_reload_pass": not any(item.startswith("strict_reload") for item in failures),
        "all_parameters_exact": not any(item.startswith("parameter_") for item in failures),
        "all_features_exact_max_abs_0": not any(
            item.startswith("feature_") for item in failures
        ),
        "mutable_checkpoint_path_used_by_b1": mutable_path_consumed,
        "checkpoint_selection_pretrain_val_only": "checkpoint_selection" not in failures,
        "target_labels_used": False,
        "released_training_provenance_verified": False,
        "released_reference_use": "path_validity_reference_only",
        "phase_b1_go_recommended": not failures,
        "phase_b1_compute_authorized": False,
        "requires_pm_review_before_b1": True,
        "scientific_metrics_computed": False,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
