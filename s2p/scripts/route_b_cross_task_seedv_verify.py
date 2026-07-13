#!/usr/bin/env python
"""Independent recomputation of the Phase C1 SEED-V gate decision."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score

from route_b_cross_task_common import GATE_TAGS, canonical_sha, sha256_file, write_json


def read_rows(path):
    with Path(path).open(newline="") as fobj:
        return list(csv.DictReader(fobj))


def recompute_kappa(payload_path):
    with np.load(payload_path) as payload:
        x = payload["features"]
        y = payload["labels"]
        split = payload["splits"].astype(str)
        identity = canonical_sha({
            key: payload[key].tolist()
            for key in ("labels", "subjects", "sessions", "splits", "trial_ids", "window_counts")
        })
    train = split == "train"
    test = split == "test"
    pca = PCA(n_components=128, whiten=True, svd_solver="randomized", random_state=0)
    z_train = pca.fit_transform(x[train])
    z_test = pca.transform(x[test])
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, tol=1e-6)
    clf.fit(z_train, y[train])
    return float(cohen_kappa_score(y[test], clf.predict(z_test))), identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()

    verdict = json.loads((args.result_dir / "seedv_gate_verdict.json").read_text())
    task = {row["tag"]: row for row in read_rows(args.result_dir / "seedv_gate_task_performance.csv")}
    run = {row["tag"]: row for row in read_rows(args.result_dir / "seedv_gate_run_manifest.csv")}
    identities = set()
    checks = []
    for tag in GATE_TAGS:
        payload = args.feature_dir / f"{tag}_features.npz"
        contract = json.loads((args.feature_dir / f"{tag}_feature_contract.json").read_text())
        if sha256_file(payload) != contract["payload_sha256"]:
            raise RuntimeError(f"feature payload hash mismatch for {tag}")
        observed, identity = recompute_kappa(payload)
        identities.add(identity)
        expected = float(task[tag]["target_test_kappa"])
        difference = abs(observed - expected)
        checks.append({
            "tag": tag,
            "expected_target_kappa": expected,
            "recomputed_target_kappa": observed,
            "absolute_difference": difference,
            "payload_sha256": contract["payload_sha256"],
            "run_manifest_payload_match": run[tag]["feature_payload_sha256"] == contract["payload_sha256"],
            "pass": difference <= 1e-12,
        })
    random_kappa = checks[0]["recomputed_target_kappa"]
    released_kappa = checks[1]["recomputed_target_kappa"]
    decision = released_kappa - random_kappa >= 0.02
    passed = (
        len(identities) == 1
        and all(row["pass"] and row["run_manifest_payload_match"] for row in checks)
        and decision == (verdict["status"] == "PASS")
        and verdict["target_label_firewall_pass"] is True
        and verdict["fine_tuning_used"] is False
    )
    report = {
        "phase": "C1_SEED-V_gate_independent_verification",
        "status": "PASS" if passed else "FAIL",
        "trial_identity_consistent": len(identities) == 1,
        "released_minus_random_kappa_recomputed": released_kappa - random_kappa,
        "gate_decision_recomputed": decision,
        "gate_decision_matches": decision == (verdict["status"] == "PASS"),
        "checks": checks,
        "target_labels_used_for_selection": False,
    }
    write_json(args.result_dir / "seedv_gate_adversarial_verification.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
