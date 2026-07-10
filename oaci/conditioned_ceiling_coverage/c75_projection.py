"""C75 projection variance and counterfactual sensitivity audits."""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from . import c74_analysis
from . import c74_cache
from . import c74_t2_source_wz_instrumentation as c74_runner
from . import c75_data
from . import c75_modeling
from . import c75_protocol


def _descriptor(manifest: dict, kind: str) -> dict:
    return next(item for item in manifest["shards"] if item["kind"] == kind)


def _load(descriptor: dict, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(descriptor["path"], allow_pickle=False) as shard:
        return {field: shard[field] for field in fields}


def _target_payloads() -> dict[int, list[dict]]:
    c75_data.load_protocol()
    c74_protocol = c75_protocol.C74_PROTOCOL.read_text()
    import json
    manifests = c74_analysis._primary_smoke_manifests(json.loads(c74_protocol))
    t3_ids = {
        row["checkpoint_id"] for row in c75_data.csv_dicts(c75_protocol.C74_T3_UNITS)
    }
    if len(manifests) != 216 or any(
        {item["kind"] for item in manifest["shards"]} != c75_data.ALLOWED_KINDS
        or manifest["checkpoint_id"] in t3_ids
        for manifest in manifests
    ):
        raise RuntimeError("C75 projection restricted-view/T3-HO isolation failed")
    grouped = defaultdict(list)
    for manifest in manifests:
        target_descriptor = _descriptor(manifest, "target_unlabeled_representation")
        wb_descriptor = _descriptor(manifest, "checkpoint_Wb")
        evaluation_descriptor = _descriptor(manifest, "target_evaluation_labels")
        c74_cache.verify_shard(target_descriptor, required_fields=c74_runner.TARGET_UNLABELED_FIELDS)
        c74_cache.verify_shard(wb_descriptor, required_fields=c74_runner.CHECKPOINT_FIELDS)
        c74_cache.verify_shard(evaluation_descriptor, required_fields=c74_runner.EVALUATION_FIELDS)
        target = _load(target_descriptor, ("target_trial_id", "logits", "probabilities", "Wz"))
        wb = _load(wb_descriptor, ("b",))
        evaluation = _load(evaluation_descriptor, ("target_trial_id", "target_class_label"))
        index = {str(trial_id): idx for idx, trial_id in enumerate(target["target_trial_id"])}
        eval_indices = np.asarray([index[str(trial_id)] for trial_id in evaluation["target_trial_id"]], dtype=int)
        grouped[int(manifest["target_id"])].append({
            "manifest": manifest, "trial_id": target["target_trial_id"],
            "logits": target["logits"].astype(float), "Wz": target["Wz"].astype(float),
            "bias": wb["b"].astype(float), "eval_indices": eval_indices,
            "eval_labels": evaluation["target_class_label"].astype(int),
        })
    for target, rows in grouped.items():
        rows.sort(key=lambda row: (row["manifest"]["seed"], row["manifest"]["level"], row["manifest"]["candidate_order"], row["manifest"]["unit_id"]))
        if len(rows) != 24 or any(not np.array_equal(rows[0]["trial_id"], row["trial_id"]) for row in rows):
            raise RuntimeError(f"C75 projection target alignment failed for {target}")
        if any(not np.array_equal(rows[0]["eval_indices"], row["eval_indices"]) or not np.array_equal(rows[0]["eval_labels"], row["eval_labels"]) for row in rows):
            raise RuntimeError(f"C75 evaluation view alignment failed for {target}")
    return dict(grouped)


def variance_audit(payloads: dict[int, list[dict]], bootstrap_repeats: int = c75_protocol.BOOTSTRAP_REPLICATES) -> dict[str, list[dict]]:
    rows = []
    for target, units in sorted(payloads.items()):
        stack = np.stack([unit["Wz"] for unit in units])
        for class_index in range(4):
            values = stack[:, :, class_index]
            grand = float(np.mean(values))
            candidate = np.mean(values, axis=1, keepdims=True) - grand
            trial = np.mean(values, axis=0, keepdims=True) - grand
            residual = values - grand - candidate - trial
            total_ss = float(np.sum((values - grand) ** 2))
            shares = {
                "target_common_trial": float(values.shape[0] * np.sum(trial ** 2) / total_ss),
                "checkpoint_candidate": float(values.shape[1] * np.sum(candidate ** 2) / total_ss),
                "candidate_x_trial_residual": float(np.sum(residual ** 2) / total_ss),
            }
            rows.append({
                "target_id": target, "class_index": class_index,
                "candidate_count": values.shape[0], "trial_count": values.shape[1],
                "estimand": "two_way_descriptive_ANOVA_Wz_candidate_x_trial_within_target_class",
                **{f"{key}_share": value for key, value in shares.items()},
                "accounting_sum": sum(shares.values()), "causal_interpretation": 0,
            })
    rng = np.random.default_rng(c75_protocol.RNG_SEED + 20)
    target_ids = sorted(payloads)
    replicates = defaultdict(list)
    by_target = {target: [row for row in rows if row["target_id"] == target] for target in target_ids}
    for _ in range(bootstrap_repeats):
        sampled = rng.choice(target_ids, size=len(target_ids), replace=True)
        selected = [row for target in sampled for row in by_target[int(target)]]
        for component in ("target_common_trial", "checkpoint_candidate", "candidate_x_trial_residual"):
            replicates[component].append(float(np.mean([row[f"{component}_share"] for row in selected])))
    bootstrap = []
    for component, values in replicates.items():
        bootstrap.append({
            "component": component, "point_mean": float(np.mean([row[f"{component}_share"] for row in rows])),
            "bootstrap_mean": float(np.mean(values)), "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)), "bootstrap_replicates": bootstrap_repeats,
            "bootstrap_unit": "target_then_all_four_classes",
        })
    estimand = [
        {"field": "observational_unit", "value": "Wz[candidate,trial,class] within each target x class"},
        {"field": "candidate_main", "value": "candidate mean minus target-class grand mean; SS multiplied by trial count"},
        {"field": "target_common_trial_main", "value": "across-candidate trial mean minus grand mean; SS multiplied by candidate count"},
        {"field": "interaction_residual", "value": "value-grand-candidate_main-trial_main"},
        {"field": "normalization", "value": "each component SS / total within-target-class SS"},
        {"field": "causal_interpretation", "value": "forbidden; descriptive crossed ANOVA only"},
    ]
    return {"by_target_class": rows, "bootstrap": bootstrap, "estimand": estimand}


def _endpoint_utility(logits: np.ndarray, eval_indices: np.ndarray, labels: np.ndarray) -> np.ndarray:
    metrics = [c75_data.endpoint_metrics(candidate[eval_indices], labels) for candidate in logits]
    oriented = np.column_stack((
        c75_data.midrank_percentile(np.asarray([row["bAcc"] for row in metrics])),
        c75_data.midrank_percentile(-np.asarray([row["NLL"] for row in metrics])),
        c75_data.midrank_percentile(-np.asarray([row["ECE"] for row in metrics])),
    ))
    return np.mean(oriented, axis=1)


def _flip_fraction(reference: np.ndarray, alternative: np.ndarray) -> float:
    flips = 0
    comparable = 0
    for left in range(len(reference)):
        for right in range(left + 1, len(reference)):
            first = np.sign(reference[left] - reference[right])
            second = np.sign(alternative[left] - alternative[right])
            if first and second:
                comparable += 1
                flips += int(first != second)
    return flips / comparable if comparable else math.nan


def counterfactual_audit(payloads: dict[int, list[dict]], null_repeats: int = c75_protocol.NULL_REPLICATES) -> dict[str, list[dict]]:
    rng = np.random.default_rng(c75_protocol.RNG_SEED + 30)
    observed_rows = []
    null_by_replicate = defaultdict(lambda: defaultdict(list))
    blocked_rows = []
    for target, units in sorted(payloads.items()):
        wz = np.stack([unit["Wz"] for unit in units])
        bias = np.stack([unit["bias"] for unit in units])
        common = np.mean(wz, axis=0, keepdims=True)
        residual = wz - common
        original_logits = wz + bias[:, None, :]
        eval_indices = units[0]["eval_indices"]
        labels = units[0]["eval_labels"]
        original_utility = _endpoint_utility(original_logits, eval_indices, labels)
        variants = {
            "residual_shrink_0.5": common + 0.5 * residual + bias[:, None, :],
            "target_common_replacement": common + bias[:, None, :],
        }
        for name, logits in variants.items():
            utility = _endpoint_utility(logits, eval_indices, labels)
            row = {
                "target_id": target, "intervention": name,
                "utility_spearman": (
                    c75_modeling.safe_spearman(original_utility, utility)
                    if np.std(utility) > 0 else math.nan
                ),
                "pairwise_rank_flip_fraction": _flip_fraction(original_utility, utility),
                "top1_agreement": int(int(np.argmax(original_utility)) == int(np.argmax(utility))),
                "best_utility_delta": float(np.max(utility) - np.max(original_utility)),
                "mechanism_origin_identified": 0,
            }
            observed_rows.append(row)
            blocked_rows.append(row)
        trajectory_groups = defaultdict(list)
        for index, unit in enumerate(units):
            trajectory_groups[unit["manifest"]["trajectory_id"]].append(index)
        for replicate in range(null_repeats):
            permutation = rng.permutation(len(units))
            trajectory_residual = residual.copy()
            for indices in trajectory_groups.values():
                shuffled = rng.permutation(indices)
                trajectory_residual[indices] = residual[shuffled]
            random_residual = rng.normal(size=residual.shape)
            random_residual -= np.mean(random_residual, axis=0, keepdims=True)
            random_residual *= np.linalg.norm(residual) / max(np.linalg.norm(random_residual), 1e-15)
            trial_permutation = rng.permutation(common.shape[1])
            random_common = common[:, trial_permutation, :]
            null_variants = {
                "candidate_permutation": common + residual[permutation] + bias[:, None, :],
                "trajectory_preserving_shuffle": common + trajectory_residual + bias[:, None, :],
                "magnitude_matched_random": common + random_residual + bias[:, None, :],
                "random_target_common_replacement": random_common + residual + bias[:, None, :],
            }
            for name, logits in null_variants.items():
                utility = _endpoint_utility(logits, eval_indices, labels)
                null_by_replicate[replicate][name].append(_flip_fraction(original_utility, utility))
    null_rows = []
    for replicate, variants in sorted(null_by_replicate.items()):
        for name, values in sorted(variants.items()):
            null_rows.append({
                "replicate": replicate, "null_family": name,
                "mean_pairwise_rank_flip_fraction": float(np.mean(values)),
                "target_count": len(values),
            })
    identity_rows = []
    max_null = [
        max(float(np.mean(values)) for values in variants.values())
        for _, variants in sorted(null_by_replicate.items())
    ]
    for intervention in ("residual_shrink_0.5", "target_common_replacement"):
        selected = [row for row in observed_rows if row["intervention"] == intervention]
        observed = float(np.mean([row["pairwise_rank_flip_fraction"] for row in selected]))
        all_null = [row["mean_pairwise_rank_flip_fraction"] for row in null_rows]
        identity_rows.append({
            "intervention": intervention, "observed_mean_flip": observed,
            "matched_null_mean": float(np.mean(all_null)), "matched_null_p95": float(np.quantile(max_null, 0.95)),
            "max_family_p": (1 + sum(value >= observed for value in max_null)) / (1 + len(max_null)),
            "function_changes_by_construction": 1, "factorization_origin_identified": 0,
            "conclusion": "counterfactual_sensitivity_not_mechanism_origin",
        })
    return {"identity": identity_rows, "nulls": null_rows, "blocked": blocked_rows}
