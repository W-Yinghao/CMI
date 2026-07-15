#!/usr/bin/env python
"""Independently verify one FACED/SEED-V Panel-2 fleet."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, log_loss


TAGS = [
    "random", "released", "H200_s0", "H200_s1", "H500_s0", "H500_s1",
    "H1000_s0", "H1000_s1", "H2000_s0", "H2000_s1",
]
H200 = ["H200_s0", "H200_s1"]
HIGH = ["H500_s0", "H500_s1", "H1000_s0", "H1000_s1", "H2000_s0", "H2000_s1"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def select(rows, **values):
    found = [row for row in rows if all(row[key] == str(value) for key, value in values.items())]
    if len(found) != 1:
        raise RuntimeError(f"non-unique verifier row: {values} count={len(found)}")
    return found[0]


def metrics(probabilities, labels):
    prediction = probabilities.argmax(axis=1)
    classes = list(range(probabilities.shape[1]))
    return {
        "cohen_kappa": float(cohen_kappa_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "nll": float(log_loss(labels, probabilities, labels=classes)),
    }


def empirical(observed, null):
    values = np.asarray(null, dtype=float)
    return float((1 + np.sum(values >= observed)) / (len(values) + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("faced", "seedv"), required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    all_metrics = read_csv(args.aggregate_dir / f"{args.dataset}_panel2_metrics.csv")
    all_nulls = read_csv(args.aggregate_dir / f"{args.dataset}_panel2_random_null.csv")
    inference = json.loads(
        (args.aggregate_dir / f"{args.dataset}_panel2_primary_inference.json").read_text()
    )
    checks = []
    supports = {}
    reference_identity = None
    for tag in TAGS:
        root = args.object_root / tag
        contract = json.loads((root / f"{tag}_panel2_contract.json").read_text())
        payload = Path(contract["feature_payload"])
        checks.append(
            {
                "check": f"{tag}_feature_payload_hash",
                "pass": payload.is_file() and sha256_file(payload) == contract["feature_payload_sha256"],
            }
        )
        with np.load(root / f"{tag}_panel2_support.npz", allow_pickle=False) as loaded:
            support = {key: loaded[key] for key in loaded.files}
        supports[tag] = support
        identity = (
            support["unit_labels"].tobytes(),
            support["unit_subjects"].tobytes(),
            support["unit_ids"].tobytes(),
        )
        if reference_identity is None:
            reference_identity = identity
        checks.append({"check": f"{tag}_unit_identity_equal", "pass": identity == reference_identity})
        map_key = {
            ("fresh_head", "source_only", "unchanged"): "baseline_probabilities",
            ("exact_head", "source_only", "unchanged"): "baseline_probabilities",
            ("fresh_head", "global_oracle", "subject_leace"): "global_oracle_fresh_probabilities",
            ("exact_head", "global_oracle", "subject_leace"): "global_oracle_exact_probabilities",
            ("fresh_head", "source_only", "subject_leace"): "source_only_fresh_probabilities",
            ("exact_head", "source_only", "subject_leace"): "source_only_exact_probabilities",
        }
        for (endpoint, regime, removal), key in map_key.items():
            row = select(
                all_metrics,
                tag=tag,
                endpoint=endpoint,
                information_regime=regime,
                removal_kind=removal,
            )
            observed = metrics(support[key], support["unit_labels"].astype(int))
            difference = max(abs(observed[name] - float(row[name])) for name in observed)
            checks.append(
                {"check": f"{tag}_{endpoint}_{regime}_{removal}_metrics", "pass": difference < 1e-12}
            )
        tag_nulls = [row for row in all_nulls if row["tag"] == tag]
        complete = len(tag_nulls) == 800
        for regime in ("global_oracle", "source_only"):
            for kind in ("same_rank_random", "variance_matched_random"):
                for endpoint in ("fresh_head", "exact_head"):
                    draws = {
                        int(row["draw"])
                        for row in tag_nulls
                        if row["information_regime"] == regime
                        and row["removal_kind"] == kind
                        and row["endpoint"] == endpoint
                    }
                    complete = complete and draws == set(range(100))
        variance_errors = [
            float(row["variance_match_abs_error"])
            for row in tag_nulls
            if row["removal_kind"] == "variance_matched_random"
        ]
        checks.append({"check": f"{tag}_random_null_complete", "pass": complete})
        checks.append(
            {"check": f"{tag}_variance_match", "pass": max(variance_errors, default=1.0) <= 1e-10}
        )

    def observed(tags, regime, endpoint):
        return float(
            np.mean(
                [
                    float(
                        select(
                            all_metrics,
                            tag=tag,
                            endpoint=endpoint,
                            information_regime=regime,
                            removal_kind="subject_leace",
                        )["delta_kappa_vs_unchanged"]
                    )
                    for tag in tags
                ]
            )
        )

    estimates = {
        "P2_H1": observed(HIGH, "source_only", "fresh_head"),
        "P2_H2": observed(HIGH, "global_oracle", "fresh_head")
        - observed(HIGH, "source_only", "fresh_head"),
        "P2_H3": observed(HIGH, "source_only", "exact_head")
        - observed(H200, "source_only", "exact_head"),
    }
    for name, value in estimates.items():
        committed = float(inference["hypotheses"][name]["estimate"])
        checks.append({"check": f"{name}_estimate", "pass": abs(value - committed) < 1e-12})

    for name in ("P2_H1", "P2_H2"):
        for kind, field in (
            ("same_rank_random", "same_rank_empirical_p"),
            ("variance_matched_random", "variance_matched_empirical_p"),
        ):
            null = []
            for draw in range(100):
                source = np.mean(
                    [
                        float(
                            select(
                                all_nulls,
                                tag=tag,
                                draw=draw,
                                endpoint="fresh_head",
                                information_regime="source_only",
                                removal_kind=kind,
                            )["delta_kappa_vs_unchanged"]
                        )
                        for tag in HIGH
                    ]
                )
                if name == "P2_H1":
                    null.append(source)
                else:
                    global_value = np.mean(
                        [
                            float(
                                select(
                                    all_nulls,
                                    tag=tag,
                                    draw=draw,
                                    endpoint="fresh_head",
                                    information_regime="global_oracle",
                                    removal_kind=kind,
                                )["delta_kappa_vs_unchanged"]
                            )
                            for tag in HIGH
                        ]
                    )
                    null.append(global_value - source)
            value = empirical(estimates[name], null)
            committed = float(inference["hypotheses"][name][field])
            checks.append({"check": f"{name}_{field}", "pass": abs(value - committed) < 1e-12})

    firewall = json.loads(
        (args.aggregate_dir / f"{args.dataset}_panel2_target_label_firewall.json").read_text()
    )
    checks.append(
        {
            "check": "target_label_firewall",
            "pass": firewall["target_labels_final_scoring_only"] is True
            and firewall["target_labels_used_for_selection"] is False
            and firewall["best_seed_or_budget_selected"] is False,
        }
    )
    passed = all(row["pass"] for row in checks)
    result = {
        "phase": "FMScope_FSR_Bridge_Panel2_independent_verification",
        "dataset": args.dataset,
        "status": "PASS" if passed else "FAIL",
        "checks_passed": sum(row["pass"] for row in checks),
        "checks_total": len(checks),
        "all_object_metrics_recomputed_from_predictions": True,
        "all_random_draw_families_complete": all(
            row["pass"] for row in checks if row["check"].endswith("random_null_complete")
        ),
        "target_labels_used_for_selection": False,
        "checks": checks,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
