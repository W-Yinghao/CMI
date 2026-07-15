#!/usr/bin/env python
"""Aggregate one FACED/SEED-V Panel-2 fleet and freeze primary inference."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score


TAGS = [
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
]
H200_TAGS = ["H200_s0", "H200_s1"]
HIGH_TAGS = ["H500_s0", "H500_s1", "H1000_s0", "H1000_s1", "H2000_s0", "H2000_s1"]
N_BOOTSTRAP = 5000


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def select(rows, **values):
    result = [row for row in rows if all(row[key] == str(value) for key, value in values.items())]
    if len(result) != 1:
        raise RuntimeError(f"row selection is not unique: {values} count={len(result)}")
    return result[0]


def empirical_greater(observed, null):
    values = np.asarray(null, dtype=float)
    return float((1 + np.sum(values >= observed)) / (len(values) + 1))


def holm(pvalues):
    names = list(pvalues)
    ordered = sorted(names, key=lambda name: pvalues[name])
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for index, name in enumerate(ordered):
        value = min(1.0, (count - index) * pvalues[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def kappa(probabilities, labels):
    return float(cohen_kappa_score(labels, probabilities.argmax(axis=1)))


def sampled_kappa(probabilities, labels, subjects, sampled):
    indices = np.concatenate([np.flatnonzero(subjects == subject) for subject in sampled])
    return kappa(probabilities[indices], labels[indices])


def load_support(root, tag):
    path = root / tag / f"{tag}_panel2_support.npz"
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def subject_delta(support, regime, endpoint, sampled=None):
    labels = support["unit_labels"].astype(int)
    subjects = support["unit_subjects"].astype(int)
    base = support["baseline_probabilities"]
    changed = support[f"{regime}_{endpoint}_probabilities"]
    if sampled is None:
        return kappa(changed, labels) - kappa(base, labels)
    return sampled_kappa(changed, labels, subjects, sampled) - sampled_kappa(
        base, labels, subjects, sampled
    )


def bootstrap_primary(supports, seed):
    reference = supports[TAGS[0]]
    subjects = np.unique(reference["unit_subjects"].astype(int))
    rng = np.random.default_rng(seed)
    h1 = np.empty(N_BOOTSTRAP)
    h2 = np.empty(N_BOOTSTRAP)
    h3 = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        sampled = rng.choice(subjects, len(subjects), replace=True)
        source_high = np.mean(
            [subject_delta(supports[tag], "source_only", "fresh", sampled) for tag in HIGH_TAGS]
        )
        global_high = np.mean(
            [subject_delta(supports[tag], "global_oracle", "fresh", sampled) for tag in HIGH_TAGS]
        )
        exact_high = np.mean(
            [subject_delta(supports[tag], "source_only", "exact", sampled) for tag in HIGH_TAGS]
        )
        exact_low = np.mean(
            [subject_delta(supports[tag], "source_only", "exact", sampled) for tag in H200_TAGS]
        )
        h1[index] = source_high
        h2[index] = global_high - source_high
        h3[index] = exact_high - exact_low
    return {"P2_H1": h1, "P2_H2": h2, "P2_H3": h3}


def interval(values):
    values = np.asarray(values, dtype=float)
    return {
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "bootstrap_p_one_sided": float((1 + np.sum(values <= 0)) / (len(values) + 1)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("faced", "seedv"), required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = []
    nulls = []
    transfers = []
    modifiers = []
    contracts = []
    supports = {}
    for tag in TAGS:
        root = args.object_root / tag
        metrics.extend(read_csv(root / f"{tag}_panel2_metrics.csv"))
        nulls.extend(read_csv(root / f"{tag}_panel2_random_null.csv"))
        transfers.extend(read_csv(root / f"{tag}_panel2_transferability.csv"))
        modifiers.extend(read_csv(root / f"{tag}_panel2_modifiers.csv"))
        contract = json.loads((root / f"{tag}_panel2_contract.json").read_text())
        if contract["status"] != "PASS_OBJECT" or contract["random_draws_per_null"] != 100:
            raise RuntimeError(f"object contract is not scientific PASS: {tag}")
        contracts.append(contract)
        supports[tag] = load_support(args.object_root, tag)

    expected_null_rows = len(TAGS) * 2 * 2 * 2 * 100
    if len(nulls) != expected_null_rows:
        raise RuntimeError(f"random null row count {len(nulls)} != {expected_null_rows}")

    def observed(tags, regime, endpoint):
        return float(
            np.mean(
                [
                    float(
                        select(
                            metrics,
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

    h1 = observed(HIGH_TAGS, "source_only", "fresh_head")
    h2 = observed(HIGH_TAGS, "global_oracle", "fresh_head") - h1
    h3 = observed(HIGH_TAGS, "source_only", "exact_head") - observed(
        H200_TAGS, "source_only", "exact_head"
    )

    null_distributions = {}
    for null_kind in ("same_rank_random", "variance_matched_random"):
        source = []
        global_minus_source = []
        for draw in range(100):
            source_value = np.mean(
                [
                    float(
                        select(
                            nulls,
                            tag=tag,
                            draw=draw,
                            endpoint="fresh_head",
                            information_regime="source_only",
                            removal_kind=null_kind,
                        )["delta_kappa_vs_unchanged"]
                    )
                    for tag in HIGH_TAGS
                ]
            )
            global_value = np.mean(
                [
                    float(
                        select(
                            nulls,
                            tag=tag,
                            draw=draw,
                            endpoint="fresh_head",
                            information_regime="global_oracle",
                            removal_kind=null_kind,
                        )["delta_kappa_vs_unchanged"]
                    )
                    for tag in HIGH_TAGS
                ]
            )
            source.append(source_value)
            global_minus_source.append(global_value - source_value)
        null_distributions[null_kind] = {
            "P2_H1": np.asarray(source),
            "P2_H2": np.asarray(global_minus_source),
        }

    bootstrap = bootstrap_primary(supports, 2026071501 if args.dataset == "faced" else 2026071502)
    raw_p = {
        "P2_H1": max(
            empirical_greater(h1, null_distributions["same_rank_random"]["P2_H1"]),
            empirical_greater(h1, null_distributions["variance_matched_random"]["P2_H1"]),
        ),
        "P2_H2": max(
            empirical_greater(h2, null_distributions["same_rank_random"]["P2_H2"]),
            empirical_greater(h2, null_distributions["variance_matched_random"]["P2_H2"]),
        ),
        "P2_H3": interval(bootstrap["P2_H3"])["bootstrap_p_one_sided"],
    }
    adjusted = holm(raw_p)
    values = {"P2_H1": h1, "P2_H2": h2, "P2_H3": h3}
    random_val = float(
        select(modifiers, tag="random", information_regime="source_only")[
            "baseline_source_val_kappa"
        ]
    )
    gate_rows = []
    gate_by_tag = {}
    for tag in TAGS:
        value = float(
            select(modifiers, tag=tag, information_regime="source_only")[
                "baseline_source_val_kappa"
            ]
        )
        passed = value >= 0.05 and value >= random_val + 0.02
        gate_by_tag[tag] = passed
        gate_rows.append(
            {
                "tag": tag,
                "source_val_kappa": value,
                "random_source_val_kappa": random_val,
                "task_gate_pass": passed,
            }
        )
    hypothesis_rows = []
    for name in ("P2_H1", "P2_H2", "P2_H3"):
        required_tags = HIGH_TAGS if name in ("P2_H1", "P2_H2") else H200_TAGS + HIGH_TAGS
        task_gate_pass = all(gate_by_tag[tag] for tag in required_tags)
        row = {
            "dataset": args.dataset,
            "hypothesis": name,
            "estimate": values[name],
            **interval(bootstrap[name]),
            "raw_primary_p": raw_p[name],
            "holm_adjusted_p": adjusted[name],
            "task_gate_all_required_cells_pass": task_gate_pass,
            "primary_pass": adjusted[name] <= 0.05 and values[name] > 0 and task_gate_pass,
        }
        if name in ("P2_H1", "P2_H2"):
            row.update(
                {
                    "same_rank_empirical_p": empirical_greater(
                        values[name], null_distributions["same_rank_random"][name]
                    ),
                    "variance_matched_empirical_p": empirical_greater(
                        values[name], null_distributions["variance_matched_random"][name]
                    ),
                }
            )
        hypothesis_rows.append(row)

    training_transfers = [row for row in transfers if row["tag"].startswith("H")]
    scatter = [float(row["target_subject_scatter_removed_fraction"]) for row in training_transfers]
    h4 = {
        "min_target_subject_scatter_removed_fraction": min(scatter),
        "mean_target_subject_scatter_removed_fraction": float(np.mean(scatter)),
        "all_training_checkpoints_above_half": all(value > 0.5 for value in scatter),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / f"{args.dataset}_panel2_metrics.csv", metrics)
    write_csv(args.out_dir / f"{args.dataset}_panel2_random_null.csv", nulls)
    write_csv(args.out_dir / f"{args.dataset}_panel2_transferability.csv", transfers)
    write_csv(args.out_dir / f"{args.dataset}_panel2_modifiers.csv", modifiers)
    write_csv(args.out_dir / f"{args.dataset}_panel2_task_gates.csv", gate_rows)
    write_csv(args.out_dir / f"{args.dataset}_panel2_primary_hypotheses.csv", hypothesis_rows)
    inference = {
        "phase": "FMScope_FSR_Bridge_Panel2",
        "dataset": args.dataset,
        "primary_utility_metric": "cohen_kappa",
        "random_draws_per_null": 100,
        "target_subject_bootstrap_replicates": N_BOOTSTRAP,
        "hypotheses": {row["hypothesis"]: row for row in hypothesis_rows},
        "P2_H4": h4,
        "target_labels_used_for_selection": False,
        "cross_dataset_pooled_p_value": False,
    }
    write_json(args.out_dir / f"{args.dataset}_panel2_primary_inference.json", inference)
    write_json(
        args.out_dir / f"{args.dataset}_panel2_target_label_firewall.json",
        {
            "dataset": args.dataset,
            "target_labels_final_scoring_only": True,
            "target_subject_ids_used_for_final_transferability_diagnostic": True,
            "target_labels_used_for_operator_fit": False,
            "target_labels_used_for_selection": False,
            "best_seed_or_budget_selected": False,
        },
    )
    verdict = {
        "dataset": args.dataset,
        "status": "PASS_AGGREGATE",
        "objects": len(contracts),
        "all_four_arm_objects_complete": True,
        "P2_H1_source_only_identity_specific_utility": hypothesis_rows[0]["primary_pass"],
        "P2_H2_global_source_gap": hypothesis_rows[1]["primary_pass"],
        "P2_H3_exact_head_cost_reduces": hypothesis_rows[2]["primary_pass"],
        "P2_H4_axis_transferability_high": h4["all_training_checkpoints_above_half"],
        "interaction_claim_allowed": False,
        "target_labels_used_for_selection": False,
    }
    write_json(args.out_dir / f"{args.dataset}_panel2_verdict.json", verdict)
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
