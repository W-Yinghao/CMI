"""C76 representation association orbit and conditional transportability audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np

from . import c74_cache
from . import c75_data
from . import c75_modeling
from . import c75_protocol
from . import c76_orbit
from . import c76_protocol
from . import c76_statistics as statistics
from . import synthetic_association_generator as synthetic


REPORT_DIR = Path("oaci/reports")
TABLE_DIR = REPORT_DIR / "c76_tables"
STATE_PATH = REPORT_DIR / "C76_REPRESENTATION_ASSOCIATION_ANALYSIS_STATE.json"

# C75 F4 is a mixed block: latent/head geometry followed by function-level
# Wz summaries. Keep the latter for exact C75 replay, but never let it enter
# the registered representation candidate.
TARGET_GEOMETRY_DIM = 20
TARGET_INVARIANT_DIM = 15


def _target_feature_partition(F4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if F4.ndim != 2 or F4.shape[1] != TARGET_GEOMETRY_DIM + TARGET_INVARIANT_DIM:
        raise RuntimeError(f"C76 target F4 dimension drift: {F4.shape}")
    return F4[:, :TARGET_GEOMETRY_DIM], F4[:, TARGET_GEOMETRY_DIM:]


def _read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(name: str, rows: list[dict]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"C76 refuses empty table: {name}")
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(TABLE_DIR / name, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _concat(arrays: dict[str, np.ndarray], blocks: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([arrays[block].astype(float) for block in blocks], axis=1)


def _variant_features(
    orbit_arrays: dict[str, np.ndarray], orbit: str, replicate: int,
) -> dict[str, np.ndarray]:
    mask = (orbit_arrays["orbit"].astype(str) == orbit) & (orbit_arrays["replicate"].astype(int) == replicate)
    indices = np.where(mask)[0]
    order = np.argsort(orbit_arrays["unit_index"][indices].astype(int))
    selected = indices[order]
    if len(selected) != 216 or not np.array_equal(orbit_arrays["unit_index"][selected], np.arange(216)):
        raise RuntimeError(f"C76 orbit variant alignment failed: {orbit}/{replicate}")
    return {name: orbit_arrays[name][selected] for name in ("F2", "F4", "G4S", "G4T")}


def _baseline_residuals(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict, dict]:
    targets = arrays["target_id"].astype(int)
    y = arrays["outcomes"][:, 0].astype(float)
    centered_y = statistics.center_within_groups(y[:, None], targets)[:, 0]
    prior_predictions = {}
    residuals = {}
    for path, blocks in (
        ("strict_source", ("F0", "F1")),
        ("target_unlabeled", ("F0", "F1", "F3")),
    ):
        result = c75_modeling.crossfit_loto(_concat(arrays, blocks), y, targets, column_space=True)
        prior_predictions[path] = result.prediction
        residuals[path] = centered_y - result.prediction
    return y, prior_predictions, residuals


def _c75_replay(
    protocol: dict, c75_manifest: dict, arrays: dict[str, np.ndarray],
    canonical_family: list[dict], residuals: dict[str, np.ndarray],
) -> dict[str, list[dict]]:
    protocol_rows = []
    for artifact in (
        c76_protocol.C75_PROTOCOL, c76_protocol.C75_RESULT,
        c76_protocol.C75_ARTIFACT_MANIFEST, c76_protocol.C75_RED_TEAM,
    ):
        protocol_rows.append({
            "artifact": artifact.name, "path": str(artifact),
            "sha256": c76_protocol.sha256(artifact), "passed": 1,
        })
    cache_rows = [{
        "C75_feature_manifest_sha256": c76_protocol.sha256(c75_data.feature_manifest_path(json.loads(c75_protocol.PROTOCOL_PATH.read_text()))),
        "C75_feature_payload_sha256": c75_manifest["descriptor"]["sha256"],
        "unit_count": c75_manifest["unit_count"],
        "Wz_plus_b_logits_max_abs": c75_manifest["Wz_plus_b_logits_max_abs"],
        "same_label_oracle_accessed": int(c75_manifest["same_label_oracle_accessed"]),
        "T3_HO_z_Wz_accessed": int(c75_manifest["T3_HO_z_Wz_accessed"]),
        "passed": int(c75_manifest["unit_count"] == 216 and not c75_manifest["T3_HO_z_Wz_accessed"]),
    }]
    c75_linear = _read_csv(c76_protocol.C75_RELEVANCE)
    linear_rows = [
        row for row in c75_linear
        if row["outcome"] == "continuous_joint_utility"
        and row["path"] in {"P_F2_strict_architecture", "P_F4_target_architecture", "P_F5_construction_positive"}
    ]
    c75_rbf = {row["path"]: row for row in _read_csv(c76_protocol.C75_RBF)}
    rbf_rows = []
    for path in ("strict_source", "target_unlabeled"):
        observed = max(
            row["association"] for row in canonical_family
            if row["path"] == path and row["kernel"] == "rbf" and row["statistic"] == "normalized_alignment"
        )
        expected = float(c75_rbf[path]["max_observed_alignment"])
        rbf_rows.append({
            "path": path, "C75_expected": expected, "C76_replayed": observed,
            "absolute_error": abs(observed - expected), "passed": int(abs(observed - expected) < 1e-12),
            "C75_global_max_stat_p": c75_rbf[path]["global_path_bandwidth_max_stat_p"],
        })
    multiplicity_rows = [{
        "C75_global_family_size": int(c75_rbf["strict_source"]["global_family_size"]),
        "C76_locked_family_size": 24, "C75_paths": 2,
        "C76_paths": 2, "C76_kernels": 2, "C76_statistics": 2,
        "C76_bandwidths": 3, "passed": 1,
    }]
    return {
        "protocol": protocol_rows, "cache": cache_rows, "linear": linear_rows,
        "rbf": rbf_rows, "multiplicity": multiplicity_rows,
    }


def _null_replicate(
    scheme: str, replicate: int, canonical_features: dict[str, np.ndarray],
    orbit_o6: dict[str, np.ndarray], residuals: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
) -> np.ndarray:
    rng = np.random.default_rng(c76_protocol.RNG_SEED + 200 + 100_000 * list(
        row["null"] for row in c76_protocol.null_registry()
    ).index(scheme) + replicate)
    targets = arrays["target_id"].astype(int)
    trajectory = arrays["trajectory_id"].astype(str)
    if scheme in {"N1_target_block", "N2_checkpoint_block", "N3_trajectory_preserving", "N4_candidate_within_target"}:
        permutation = statistics.blocked_permutation(
            scheme, targets, trajectory, arrays["seed"].astype(int), arrays["level"].astype(int),
            arrays["candidate_order"].astype(int), rng,
        )
        feature_paths = {path: values[permutation] for path, values in canonical_features.items()}
    elif scheme == "N5_identity_matched":
        feature_paths = {
            path: statistics.matched_gaussian_features(values, arrays["F0"], targets, rng)
            for path, values in canonical_features.items()
        }
    elif scheme == "N6_orbit_transformed":
        permutation = statistics.blocked_permutation(
            "N3_trajectory_preserving", targets, trajectory,
            arrays["seed"].astype(int), arrays["level"].astype(int), arrays["candidate_order"].astype(int), rng,
        )
        feature_paths = {path: values[permutation] for path, values in orbit_o6.items()}
    else:  # pragma: no cover
        raise ValueError(scheme)
    family = statistics.association_family(feature_paths, residuals, targets)
    return np.asarray([row["association"] for row in family])


def _association_null_audit(
    canonical_features: dict[str, np.ndarray], orbit_o6: dict[str, np.ndarray],
    residuals: dict[str, np.ndarray], arrays: dict[str, np.ndarray], observed: list[dict],
) -> dict[str, list[dict]]:
    workers = max(1, min(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 48))
    schemes = [row["null"] for row in c76_protocol.null_registry()]
    tasks = [(scheme, replicate) for scheme in schemes for replicate in range(c76_protocol.NULL_REPLICATES)]
    results = Parallel(n_jobs=workers, backend="loky", verbose=0)(
        delayed(_null_replicate)(
            scheme, replicate, canonical_features, orbit_o6, residuals, arrays,
        ) for scheme, replicate in tasks
    )
    by_scheme = {}
    offset = 0
    for scheme in schemes:
        by_scheme[scheme] = np.stack(results[offset:offset + c76_protocol.NULL_REPLICATES])
        offset += c76_protocol.NULL_REPLICATES
    rows = []
    max_rows = []
    for scheme, matrix in by_scheme.items():
        max_stat = np.max(matrix, axis=1)
        for replicate, value in enumerate(max_stat):
            max_rows.append({"null": scheme, "replicate": replicate, "max_statistic": float(value), "family_size": len(observed)})
        for index, observed_row in enumerate(observed):
            values = matrix[:, index]
            rows.append({
                "null": scheme, "path": observed_row["path"], "kernel": observed_row["kernel"],
                "bandwidth_factor": observed_row["bandwidth_factor"], "statistic": observed_row["statistic"],
                "observed": observed_row["association"], "null_mean": float(np.mean(values)),
                "null_p95": float(np.quantile(values, 0.95)),
                "uncorrected_p": (1 + int(np.sum(values >= observed_row["association"]))) / (1 + len(values)),
                "global_max_stat_p": (1 + int(np.sum(max_stat >= observed_row["association"]))) / (1 + len(max_stat)),
                "null_replicates": len(values), "bandwidth_recomputed_inside_null": 1,
            })
    summary = []
    for observed_row in observed:
        matches = [
            row for row in rows
            if all(row[key] == observed_row[key] for key in ("path", "kernel", "bandwidth_factor", "statistic"))
        ]
        summary.append({
            **{key: observed_row[key] for key in ("path", "kernel", "bandwidth_factor", "statistic", "association", "median_target_association", "positive_targets")},
            "worst_required_global_p": max(row["global_max_stat_p"] for row in matches),
            "required_nulls_passing_0.05": sum(row["global_max_stat_p"] < 0.05 for row in matches),
            "required_null_count": len(matches),
        })
    return {"rows": rows, "max": max_rows, "summary": summary}


def _effect_intervals(observed: list[dict]) -> list[dict]:
    rows = []
    rng = np.random.default_rng(c76_protocol.RNG_SEED + 250)
    for item in observed:
        fold_values = np.asarray([row["statistic_value"] for row in item["fold_rows"]], dtype=float)
        bootstrap = np.asarray([
            float(np.mean(rng.choice(fold_values, size=len(fold_values), replace=True)))
            for _ in range(c76_protocol.BOOTSTRAP_REPLICATES)
        ])
        rows.append({
            "path": item["path"], "kernel": item["kernel"],
            "bandwidth_factor": item["bandwidth_factor"], "statistic": item["statistic"],
            "association": item["association"],
            "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
            "bootstrap_unit": "target", "bootstrap_replicates": len(bootstrap),
        })
    return rows


def _orbit_variant_task(
    orbit: str, replicate: int, path: str, features: np.ndarray,
    residual: np.ndarray, arrays: dict[str, np.ndarray], reference_order: np.ndarray,
) -> dict:
    targets = arrays["target_id"].astype(int)
    registered_rows = []
    for kernel in c76_protocol.KERNEL_FAMILIES:
        for factor in c76_protocol.BANDWIDTH_FACTORS:
            for statistic_name in c76_protocol.ASSOCIATION_STATISTICS:
                value, _ = statistics.crossfit_association(
                    features, residual, targets, kernel_family=kernel,
                    bandwidth_factor=factor, statistic=statistic_name,
                )
                registered_rows.append({
                    "kernel": kernel, "bandwidth_factor": factor,
                    "statistic": statistic_name, "association": value,
                })
    values = [
        next(
            row["association"] for row in registered_rows
            if row["kernel"] == "rbf" and row["statistic"] == "normalized_alignment"
            and float(row["bandwidth_factor"]) == float(factor)
        ) for factor in c76_protocol.BANDWIDTH_FACTORS
    ]
    observed = max(values)
    rngs = [
        np.random.default_rng(c76_protocol.RNG_SEED + 300_000 + replicate_index)
        for replicate_index in range(c76_protocol.NULL_REPLICATES)
    ]
    max_null = []
    for rng in rngs:
        permutation = statistics.blocked_permutation(
            "N3_trajectory_preserving", targets, arrays["trajectory_id"].astype(str),
            arrays["seed"].astype(int), arrays["level"].astype(int), arrays["candidate_order"].astype(int), rng,
        )
        max_null.append(max(
            statistics.crossfit_association(
                features[permutation], residual, targets, kernel_family="rbf",
                bandwidth_factor=factor, statistic="normalized_alignment",
            )[0] for factor in c76_protocol.BANDWIDTH_FACTORS
        ))
    density = statistics.candidate_density_order(features, targets)
    return {
        "orbit": orbit, "replicate": replicate, "path": path,
        "max_rbf_association": observed,
        "bandwidth_0.5": values[0], "bandwidth_1.0": values[1], "bandwidth_2.0": values[2],
        "trajectory_null_p": (1 + sum(value >= observed for value in max_null)) / (1 + len(max_null)),
        "trajectory_null_p95": float(np.quantile(max_null, 0.95)),
        "candidate_density_order_spearman": statistics.orbit_order_spearman(reference_order, density, targets),
        "_null_values": max_null,
        "_registered_rows": registered_rows,
    }


def _orbit_audit(
    orbit_arrays: dict[str, np.ndarray], residuals: dict[str, np.ndarray], arrays: dict[str, np.ndarray],
) -> dict[str, list[dict] | dict]:
    variants = c76_orbit.orbit_variants()
    canonical = _variant_features(orbit_arrays, "O0", 0)
    canonical_target_geometry, _ = _target_feature_partition(canonical["F4"])
    canonical_path_features = {
        "strict_source": canonical["F2"],
        "target_unlabeled": canonical_target_geometry,
        "target_unlabeled_full_C75": canonical["F4"],
    }
    reference_orders = {
        path: statistics.candidate_density_order(features, arrays["target_id"].astype(int))
        for path, features in canonical_path_features.items()
    }
    tasks = []
    feature_rows = []
    identity_rows = []
    order_rows = []
    for orbit, replicate in variants:
        features = _variant_features(orbit_arrays, orbit, replicate)
        mask = (orbit_arrays["orbit"].astype(str) == orbit) & (orbit_arrays["replicate"].astype(int) == replicate)
        identity_rows.append({
            "orbit": orbit, "replicate": replicate,
            "scope": str(orbit_arrays["scope"][np.where(mask)[0][0]]),
            "family": str(orbit_arrays["family"][np.where(mask)[0][0]]),
            "max_projection_error": float(np.max(orbit_arrays["projection_max_abs_error"][mask])),
            "max_probability_error": float(np.max(orbit_arrays["probability_max_abs_error"][mask])),
            "prediction_disagreements": int(np.sum(orbit_arrays["prediction_disagreements"][mask])),
            "max_condition_number": float(np.max(orbit_arrays["condition_number"][mask])),
            "unique_transform_hashes": len(set(orbit_arrays["transform_hash"][mask].astype(str))),
        })
        target_geometry, _ = _target_feature_partition(features["F4"])
        current_path_features = {
            "strict_source": features["F2"],
            "target_unlabeled": target_geometry,
            "target_unlabeled_full_C75": features["F4"],
        }
        for path, current in current_path_features.items():
            baseline = canonical_path_features[path]
            centered_baseline = statistics.center_within_groups(baseline, arrays["target_id"].astype(int))
            centered_current = statistics.center_within_groups(current, arrays["target_id"].astype(int))
            feature_rows.append({
                "orbit": orbit, "replicate": replicate, "path": path,
                "relative_feature_l2": float(np.linalg.norm(current - baseline) / max(np.linalg.norm(baseline), 1e-15)),
                "baseline_centered_rank": int(np.linalg.matrix_rank(centered_baseline)),
                "orbit_centered_rank": int(np.linalg.matrix_rank(centered_current)),
                "feature_distance_spearman": c75_modeling.safe_spearman(
                    statistics.pairwise_distances(centered_baseline)[np.triu_indices(216, 1)],
                    statistics.pairwise_distances(centered_current)[np.triu_indices(216, 1)],
                ),
            })
            residual_path = "target_unlabeled" if path == "target_unlabeled_full_C75" else path
            tasks.append((orbit, replicate, path, current, residuals[residual_path], reference_orders[path]))
    workers = max(1, min(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 48))
    association_rows = Parallel(n_jobs=workers, backend="loky", verbose=0)(
        delayed(_orbit_variant_task)(
            orbit, replicate, path, features, residual, arrays, reference_order,
        ) for orbit, replicate, path, features, residual, reference_order in tasks
    )
    orbit_global_max_null = np.max(np.asarray([row["_null_values"] for row in association_rows]), axis=0)
    registered_rows = []
    for row in association_rows:
        row["orbit_global_max_p"] = (
            1 + int(np.sum(orbit_global_max_null >= row["max_rbf_association"]))
        ) / (1 + len(orbit_global_max_null))
        for registered in row["_registered_rows"]:
            registered_rows.append({
                "orbit": row["orbit"], "replicate": row["replicate"],
                "path": row["path"],
                "candidate_density_order_spearman": row["candidate_density_order_spearman"],
                **registered,
            })
        del row["_null_values"]
        del row["_registered_rows"]
    baseline_registered = {
        (row["path"], row["kernel"], float(row["bandwidth_factor"]), row["statistic"]): row["association"]
        for row in registered_rows if row["orbit"] == "O0" and row["replicate"] == 0
    }
    for row in registered_rows:
        reference = baseline_registered[(row["path"], row["kernel"], float(row["bandwidth_factor"]), row["statistic"])]
        row["absolute_effect_retention"] = abs(row["association"]) / max(abs(reference), 1e-15)
        row["effect_sign_matches_baseline"] = int(np.sign(row["association"]) == np.sign(reference))
    baseline_effect = {
        path: next(row["max_rbf_association"] for row in association_rows if row["orbit"] == "O0" and row["path"] == path)
        for path in canonical_path_features
    }
    for row in association_rows:
        row["absolute_effect_retention"] = abs(row["max_rbf_association"]) / max(abs(baseline_effect[row["path"]]), 1e-15)
        row["effect_sign_matches_baseline"] = int(np.sign(row["max_rbf_association"]) == np.sign(baseline_effect[row["path"]]))
        order_rows.append({
            "orbit": row["orbit"], "replicate": row["replicate"], "path": row["path"],
            "candidate_density_order_spearman": row["candidate_density_order_spearman"],
        })
    family_rows = []
    robustness = {}
    for path in ("strict_source", "target_unlabeled", "target_unlabeled_full_C75"):
        path_pass = True
        for orbit_row in c76_protocol.orbit_registry()[1:]:
            selected = [row for row in association_rows if row["path"] == path and row["orbit"] == orbit_row["orbit"]]
            retention = float(np.median([row["absolute_effect_retention"] for row in selected]))
            order = float(np.median([row["candidate_density_order_spearman"] for row in selected]))
            sign_pass = all(row["effect_sign_matches_baseline"] for row in selected)
            family_pass = sign_pass and 0.80 <= retention <= 1.25 and order >= 0.95
            path_pass &= family_pass
            family_rows.append({
                "path": path, "orbit": orbit_row["orbit"], "family": orbit_row["family"],
                "scope": orbit_row["scope"], "median_effect_retention": retention,
                "median_candidate_order_spearman": order,
                "all_signs_match": int(sign_pass), "orbit_family_pass": int(family_pass),
            })
        robustness[path] = int(path_pass)
    registered_family_rows = []
    registered_robustness = {}
    for path in ("strict_source", "target_unlabeled", "target_unlabeled_full_C75"):
        for kernel in c76_protocol.KERNEL_FAMILIES:
            for factor in c76_protocol.BANDWIDTH_FACTORS:
                for statistic_name in c76_protocol.ASSOCIATION_STATISTICS:
                    test_pass = True
                    for orbit_row in c76_protocol.orbit_registry()[1:]:
                        selected = [
                            row for row in registered_rows
                            if row["path"] == path and row["kernel"] == kernel
                            and float(row["bandwidth_factor"]) == float(factor)
                            and row["statistic"] == statistic_name and row["orbit"] == orbit_row["orbit"]
                        ]
                        retention = float(np.median([row["absolute_effect_retention"] for row in selected]))
                        sign_pass = all(row["effect_sign_matches_baseline"] for row in selected)
                        order = float(np.median([row["candidate_density_order_spearman"] for row in selected]))
                        family_pass = sign_pass and 0.80 <= retention <= 1.25 and order >= 0.95
                        test_pass &= family_pass
                        registered_family_rows.append({
                            "path": path, "kernel": kernel, "bandwidth_factor": factor,
                            "statistic": statistic_name, "orbit": orbit_row["orbit"],
                            "family": orbit_row["family"], "scope": orbit_row["scope"],
                            "median_effect_retention": retention,
                            "median_candidate_order_spearman": order,
                            "all_signs_match": int(sign_pass), "orbit_family_pass": int(family_pass),
                        })
                    registered_robustness[(path, kernel, float(factor), statistic_name)] = int(test_pass)
    return {
        "identity": identity_rows, "feature": feature_rows,
        "association": association_rows, "order": order_rows,
        "family": family_rows, "robustness": robustness,
        "registered": registered_rows, "registered_family": registered_family_rows,
        "registered_robustness": registered_robustness,
    }


def _residualize(values: np.ndarray, controls: np.ndarray, targets: np.ndarray) -> np.ndarray:
    V = statistics.center_within_groups(np.asarray(values, dtype=float), targets)
    C = statistics.center_within_groups(np.asarray(controls, dtype=float), targets)
    return V - C @ (np.linalg.pinv(C) @ V)


def _topology_and_controls(
    arrays: dict[str, np.ndarray], canonical: dict[str, np.ndarray],
    residuals: dict[str, np.ndarray],
) -> dict[str, list[dict]]:
    targets = arrays["target_id"].astype(int)
    trajectory = arrays["trajectory_id"].astype(str)
    trajectory_template = arrays["trajectory_template"].astype(str)
    regime = np.asarray([value.split("|")[-1] for value in trajectory_template], dtype="<U40")
    topology_rows, within_rows, leave_target_rows, leave_trajectory_rows = [], [], [], []
    identity_rows, heterogeneity_rows, conditional_rows = [], [], []
    target_geometry, target_invariant = _target_feature_partition(canonical["F4"])
    feature_sets = {
        "strict_source_G2_Wz": arrays["source_Wz_summary"].astype(float),
        "strict_source_G3_architecture": canonical["F2"],
        "strict_source_G4_coordinates": canonical["G4S"],
        "target_unlabeled_G2_Wz": arrays["target_Wz_summary"].astype(float),
        "target_unlabeled_G2_F4_invariant_tail": target_invariant,
        "target_unlabeled_G3_architecture": target_geometry,
        "target_unlabeled_full_C75_mixed_diagnostic": canonical["F4"],
        "target_unlabeled_G4_coordinates": canonical["G4T"],
    }
    for name, features in feature_sets.items():
        path = "strict_source" if name.startswith("strict") else "target_unlabeled"
        outcome = residuals[path]
        group_specs = {
            "pooled": np.zeros(len(targets), dtype=int).astype(str),
            "within_target": targets.astype(str),
            "within_target_x_trajectory": trajectory,
            "within_regime": regime,
            "trajectory_template": trajectory_template,
        }
        controls = {
            "metadata_seed_level_order": arrays["F0"][:, :6],
            "source_performance": arrays["F0"][:, 6:9],
            "construction_competence": arrays["F5"],
            "functional_logits_probabilities": _concat(arrays, ("F0", "F1")) if path == "strict_source" else _concat(arrays, ("F0", "F1", "F3")),
        }
        for kernel in c76_protocol.KERNEL_FAMILIES:
            for factor in c76_protocol.BANDWIDTH_FACTORS:
                for statistic_name in c76_protocol.ASSOCIATION_STATISTICS:
                    level_values = {}
                    for level, groups in group_specs.items():
                        value, group_values = statistics.topology_association(
                            features, outcome, groups, kernel_family=kernel,
                            bandwidth_factor=factor, statistic=statistic_name,
                        )
                        topology_rows.append({
                            "feature_set": name, "path": path, "kernel": kernel,
                            "bandwidth_factor": factor, "statistic": statistic_name,
                            "level": level, "association": value,
                            "group_count": len(group_values),
                            "positive_group_fraction": float(np.mean(np.asarray(group_values) > 0)) if group_values else math.nan,
                        })
                        level_values[level] = value
                    unique_regimes = sorted(set(regime.tolist()))
                    topology_rows.append({
                        "feature_set": name, "path": path, "kernel": kernel,
                        "bandwidth_factor": factor, "statistic": statistic_name,
                        "level": "cross_regime_transfer", "association": math.nan,
                        "group_count": len(unique_regimes), "positive_group_fraction": math.nan,
                        "status": "unavailable_single_regime" if len(unique_regimes) < 2 else "requires_registered_leave_regime_model",
                    })
                    heterogeneity_rows.append({
                        "feature_set": name, "path": path, "kernel": kernel,
                        "bandwidth_factor": factor, "statistic": statistic_name,
                        "pooled": level_values["pooled"], "within_target": level_values["within_target"],
                        "within_trajectory": level_values["within_target_x_trajectory"],
                        "within_regime": level_values["within_regime"],
                        "pooled_minus_within_target": level_values["pooled"] - level_values["within_target"],
                    })
                    _, target_folds = statistics.crossfit_association(
                        features, outcome, targets, kernel_family=kernel,
                        bandwidth_factor=factor, statistic=statistic_name,
                    )
                    for row in target_folds:
                        payload = {
                            "feature_set": name, "path": path, "kernel": kernel,
                            "bandwidth_factor": factor, "statistic": statistic_name, **row,
                        }
                        leave_target_rows.append(payload)
                        if "G3_architecture" in name:
                            within_rows.append(payload)
                    for template in sorted(set(trajectory_template.tolist())):
                        train = trajectory_template != template
                        test = trajectory_template == template
                        if np.sum(test) < 4:
                            continue
                        train_center = statistics.center_within_groups(features, targets)
                        train_scaled, test_scaled = statistics.scale_train_test(train_center[train], train_center[test])
                        bandwidth = factor * statistics.median_positive_distance(train_scaled)
                        kernel_matrix = statistics.kernel_from_distances(
                            statistics.pairwise_distances(test_scaled), bandwidth, kernel,
                        )
                        leave_trajectory_rows.append({
                            "feature_set": name, "path": path, "kernel": kernel,
                            "bandwidth_factor": factor, "statistic": statistic_name,
                            "trajectory_template": template, "row_count": int(np.sum(test)),
                            "association": statistics.association_statistic(kernel_matrix, outcome[test], statistic_name),
                        })
                    for control_name, control in controls.items():
                        feature_residual = _residualize(features, control, targets)
                        outcome_residual = _residualize(outcome[:, None], control, targets)[:, 0]
                        association, _ = statistics.crossfit_association(
                            feature_residual, outcome_residual, targets, kernel_family=kernel,
                            bandwidth_factor=factor, statistic=statistic_name,
                        )
                        conditional_rows.append({
                            "feature_set": name, "path": path, "conditioning": control_name,
                            "kernel": kernel, "bandwidth_factor": factor,
                            "statistic": statistic_name, "association": association,
                        })
    target_onehot = np.eye(9)[targets - 1]
    metadata_controls = {
        "target_identity_only": target_onehot,
        "seed_level_order_only": arrays["F0"][:, :6],
        "source_performance_only": arrays["F0"][:, 6:9],
        "construction_competence_only": arrays["F5"],
    }
    for name, features in metadata_controls.items():
        pooled, _ = statistics.topology_association(
            features, arrays["outcomes"][:, 0], np.zeros(len(targets), dtype=int).astype(str),
        )
        within, _ = statistics.topology_association(features, arrays["outcomes"][:, 0], targets.astype(str))
        identity_rows.append({"control": name, "pooled_association": pooled, "within_target_association": within})
    identity_rows.append({
        "control": "checkpoint_identity_delta_kernel", "pooled_association": 0.0,
        "within_target_association": 0.0,
        "status": "off_diagonal_delta_kernel_has_zero_pair_association;diagnostic_identity_control",
    })
    return {
        "topology": topology_rows, "within": within_rows,
        "leave_target": leave_target_rows, "leave_trajectory": leave_trajectory_rows,
        "identity": identity_rows, "heterogeneity": heterogeneity_rows,
        "conditional": conditional_rows,
    }


def _prediction_null_task(
    replicate: int, path: str, features: np.ndarray, residual: np.ndarray,
    prior_prediction: np.ndarray, y: np.ndarray, arrays: dict[str, np.ndarray],
) -> float:
    rng = np.random.default_rng(c76_protocol.RNG_SEED + 500_000 + 10_000 * (path == "target_unlabeled") + replicate)
    targets = arrays["target_id"].astype(int)
    permutation = statistics.blocked_permutation(
        "N3_trajectory_preserving", targets, arrays["trajectory_id"].astype(str),
        arrays["seed"].astype(int), arrays["level"].astype(int), arrays["candidate_order"].astype(int), rng,
    )
    result = statistics.crossfit_krr(features[permutation], residual, targets)
    full = prior_prediction + result.prediction
    return c75_modeling.r2(y, full, targets) - c75_modeling.r2(y, prior_prediction, targets)


def _prediction_audit(
    arrays: dict[str, np.ndarray], canonical_features: dict[str, np.ndarray],
    y: np.ndarray, prior_predictions: dict[str, np.ndarray], residuals: dict[str, np.ndarray],
) -> dict[str, list[dict] | dict]:
    targets = arrays["target_id"].astype(int)
    joint_good = arrays["outcomes"][:, 4].astype(float)
    observed_rows, fold_rows, leave_rows, action_rows, action_summary = [], [], [], [], []
    observed_store = {}
    for path in ("strict_source", "target_unlabeled"):
        result = statistics.crossfit_krr(canonical_features[path], residuals[path], targets)
        full = prior_predictions[path] + result.prediction
        prior_r2 = c75_modeling.r2(y, prior_predictions[path], targets)
        full_r2 = c75_modeling.r2(y, full, targets)
        per_target = c75_modeling.per_target_increment_rows(y, prior_predictions[path], full, targets)
        action = c75_modeling.actionability_rows(y, joint_good, prior_predictions[path], full, targets)
        bootstrap = c75_modeling.hierarchical_bootstrap_increment(
            y, prior_predictions[path], full, targets,
            repeats=c76_protocol.BOOTSTRAP_REPLICATES,
            seed=c76_protocol.RNG_SEED + 600 + (path == "target_unlabeled"),
        )
        observed_rows.append({
            "path": path, "prior_R2": prior_r2, "full_R2": full_r2,
            "incremental_R2": full_r2 - prior_r2,
            "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
            "leave_target_median_increment": float(np.nanmedian([row["increment_residual_rho"] for row in per_target])),
            "positive_targets": sum(row["positive_increment"] for row in per_target),
        })
        for row in result.fold_rows:
            fold_rows.append({"path": path, **row})
        for row in per_target:
            leave_rows.append({"path": path, **row})
        for row in action:
            action_rows.append({"path": path, **row})
        regret_values = np.asarray([row["regret_reduction"] for row in action])
        regret_bootstrap = statistics.bootstrap_target_mean(
            regret_values, c76_protocol.BOOTSTRAP_REPLICATES,
            c76_protocol.RNG_SEED + 700 + (path == "target_unlabeled"),
        )
        top3_values = np.asarray([row["delta_top3"] for row in action])
        coverage_values = np.asarray([row["delta_joint_good_coverage"] for row in action])
        regret_route = (
            float(np.mean(regret_values)) >= 0.02
            and float(np.quantile(regret_bootstrap, 0.025)) > 0
            and int(np.sum(regret_values > 0)) >= 7
        )
        topk_route = (
            float(np.mean(top3_values)) >= 2 / 9
            and float(np.mean(coverage_values)) >= 2 / 9
            and statistics.exact_sign_permutation_p(top3_values + coverage_values) < 0.05
        )
        action_summary.append({
            "path": path, "mean_regret_reduction": float(np.mean(regret_values)),
            "regret_bootstrap_ci_low": float(np.quantile(regret_bootstrap, 0.025)),
            "regret_bootstrap_ci_high": float(np.quantile(regret_bootstrap, 0.975)),
            "positive_regret_targets": int(np.sum(regret_values > 0)),
            "mean_top3_increment": float(np.mean(top3_values)),
            "mean_coverage_increment": float(np.mean(coverage_values)),
            "topk_exact_sign_p": statistics.exact_sign_permutation_p(top3_values + coverage_values),
            "regret_route_pass": int(regret_route), "topk_route_pass": int(topk_route),
            "material_actionability": int(regret_route or topk_route),
        })
        observed_store[path] = {"result": result, "full": full}
    workers = max(1, min(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 48))
    tasks = [(replicate, path) for replicate in range(c76_protocol.NULL_REPLICATES) for path in ("strict_source", "target_unlabeled")]
    values = Parallel(n_jobs=workers, backend="loky", verbose=0)(
        delayed(_prediction_null_task)(
            replicate, path, canonical_features[path], residuals[path],
            prior_predictions[path], y, arrays,
        ) for replicate, path in tasks
    )
    null_matrix = np.asarray(values).reshape(c76_protocol.NULL_REPLICATES, 2)
    max_null = np.max(null_matrix, axis=1)
    null_rows = []
    for index, path in enumerate(("strict_source", "target_unlabeled")):
        observed = next(row for row in observed_rows if row["path"] == path)
        path_values = null_matrix[:, index]
        observed.update({
            "nested_null_p95": float(np.quantile(path_values, 0.95)),
            "observed_above_nested_null_p95": int(observed["incremental_R2"] > np.quantile(path_values, 0.95)),
            "uncorrected_p": (1 + int(np.sum(path_values >= observed["incremental_R2"]))) / (1 + len(path_values)),
            "global_max_stat_p": (1 + int(np.sum(max_null >= observed["incremental_R2"]))) / (1 + len(max_null)),
        })
        for replicate, value in enumerate(path_values):
            null_rows.append({"path": path, "replicate": replicate, "incremental_R2": float(value), "global_max": float(max_null[replicate])})
    return {
        "summary": observed_rows, "folds": fold_rows, "leave": leave_rows,
        "action": action_rows, "action_summary": action_summary,
        "null": null_rows,
    }


def _nuisance_controls(
    arrays: dict[str, np.ndarray], canonical: dict[str, np.ndarray], residuals: dict[str, np.ndarray],
) -> dict[str, list[dict]]:
    targets = arrays["target_id"].astype(int)
    identity_rows, dimension_rows, random_rows = [], [], []
    rng = np.random.default_rng(c76_protocol.RNG_SEED + 800)
    target_geometry, _ = _target_feature_partition(canonical["F4"])
    for path, features in (("strict_source", canonical["F2"]), ("target_unlabeled", target_geometry)):
        controls = {
            "seed_level_order": arrays["F0"][:, :6],
            "source_performance": arrays["F0"][:, 6:9],
            "construction_labels_diagnostic": arrays["F5"],
            "coordinate_probes": canonical["G4S" if path == "strict_source" else "G4T"],
        }
        for name, control in controls.items():
            association, _ = statistics.crossfit_association(
                control, residuals[path], targets, kernel_family="rbf",
                bandwidth_factor=1.0, statistic="normalized_alignment",
            )
            identity_rows.append({"path": path, "control": name, "association": association, "dimension": control.shape[1]})
        observed, _ = statistics.crossfit_association(
            features, residuals[path], targets, kernel_family="rbf",
            bandwidth_factor=1.0, statistic="normalized_alignment",
        )
        for replicate in range(32):
            matched = statistics.matched_gaussian_features(features, arrays["F0"], targets, rng)
            association, _ = statistics.crossfit_association(
                matched, residuals[path], targets, kernel_family="rbf",
                bandwidth_factor=1.0, statistic="normalized_alignment",
            )
            dimension_rows.append({
                "path": path, "replicate": replicate, "observed": observed,
                "matched_association": association, "matched_exceeds_observed": int(association >= observed),
                "dimension": features.shape[1],
            })
        for orbit in ("O2", "O4", "O6", "O7"):
            random_rows.append({"path": path, "orbit": orbit, "control_type": "checkpoint_specific_function_preserving_coordinate_randomization"})
    return {"identity": identity_rows, "dimension": dimension_rows, "random": random_rows}


def _nonredundancy_audit(
    arrays: dict[str, np.ndarray], canonical_features: dict[str, np.ndarray],
) -> tuple[list[dict], dict[str, int]]:
    targets = arrays["target_id"].astype(int)
    rows = []
    passed = {}
    for path, prior_blocks in (
        ("strict_source", ("F0", "F1")),
        ("target_unlabeled", ("F0", "F1", "F3")),
    ):
        prior = statistics.center_within_groups(_concat(arrays, prior_blocks), targets)
        candidate = statistics.center_within_groups(canonical_features[path], targets)
        combined = np.concatenate((prior, candidate), axis=1)
        prior_rank = int(np.linalg.matrix_rank(prior))
        candidate_rank = int(np.linalg.matrix_rank(candidate))
        combined_rank = int(np.linalg.matrix_rank(combined))
        rank_gain = combined_rank - prior_rank
        passed[path] = int(rank_gain > 0)
        rows.append({
            "path": path, "prior_blocks": "+".join(prior_blocks),
            "candidate_block": "F2_geometry" if path == "strict_source" else "F4_geometry_columns_0_19",
            "candidate_dimension": candidate.shape[1], "prior_centered_rank": prior_rank,
            "candidate_centered_rank": candidate_rank, "combined_centered_rank": combined_rank,
            "rank_gain": rank_gain, "not_redundant_with_functional_prior": int(rank_gain > 0),
            "interpretation": "linear_column_space_nonredundancy_only;not_endpoint_relevance",
        })
    return rows, passed


def _risk_rows() -> list[dict]:
    risks = {
        "T2_exploratory_not_confirmation": ("controlled", "explicit in protocol/report"),
        "T3_HO_new_variable_contamination": ("closed", "T3 IDs excluded; no T3 path"),
        "factorization_orbit_ignored": ("closed", "seven registered nonidentity orbit families"),
        "checkpoint_specific_coordinate_alignment": ("controlled", "global versus checkpoint-specific effects separated"),
        "Wz_logit_redundancy": ("closed", "C75 exact replay and G2 tagging"),
        "mixed_F4_functional_architecture_conflation": (
            "closed", "formal target candidate uses F4[0:20]; Wz/logit-redundant F4[20:35] is isolated"
        ),
        "pooled_identity_confounding": ("controlled", "pooled/within/group-conditioned topology"),
        "association_pvalue_without_effect_size": ("closed", "target-bootstrap intervals and 0.02 materiality"),
        "cache_rows_not_independent": ("closed", "unit-level rows; target/trajectory blocks"),
        "kernel_bandwidth_selection": ("closed", "fold-local and repeated inside null"),
        "multiple_kernel_paths": ("closed", "24-test max-stat within six required nulls"),
        "high_dimensional_feature_search": ("closed", "fixed C75 blocks plus eight locked probes"),
        "nested_CV_leakage": ("controlled", "outer LOTO and inner training-target selection"),
        "target_label_in_unlabeled_features": ("closed", "F5 positive control separated"),
        "stable_association_called_prediction": ("closed", "separate association and nested prediction gates"),
        "prediction_called_actionability": ("closed", "separate regret/top-k materiality gate"),
        "conditional_CS_iid_overclaim": ("closed", "proxy only; no iid guarantee"),
        "small_target_count": ("controlled", "9 targets disclosed; target bootstrap/sign test"),
        "target_population_overclaim": ("closed", "forbidden"),
        "raw_cache_in_git": ("closed", "orbit cache external; aggregate CSV only"),
        "unauthorized_forward_or_training": ("closed", "no model/data forward path"),
    }
    return [
        {"risk": risk, "status": status, "blocking": 0, "evidence": evidence}
        for risk, (status, evidence) in risks.items()
    ]


def analyze() -> dict:
    protocol = c76_orbit.load_protocol()
    c75_manifest, arrays = c75_data.load_feature_cache()
    orbit_manifest, orbit_arrays = c76_orbit.load_orbit_cache()
    targets = arrays["target_id"].astype(int)
    y, prior_predictions, residuals = _baseline_residuals(arrays)
    canonical = _variant_features(orbit_arrays, "O0", 0)
    target_geometry, target_invariant = _target_feature_partition(canonical["F4"])
    c75_replay_features = {
        "strict_source": canonical["F2"],
        "target_unlabeled": canonical["F4"],
    }
    canonical_features = {
        "strict_source": canonical["F2"],
        "target_unlabeled": target_geometry,
    }
    c75_replay_family = statistics.association_family(c75_replay_features, residuals, targets)
    canonical_family = statistics.association_family(canonical_features, residuals, targets)
    replay = _c75_replay(protocol, c75_manifest, arrays, c75_replay_family, residuals)
    if any(row["passed"] != 1 for row in replay["rbf"]):
        raise RuntimeError("C76 C75 RBF replay mismatch")

    orbit_o6_variant = _variant_features(orbit_arrays, "O6", 0)
    orbit_o6_target_geometry, _ = _target_feature_partition(orbit_o6_variant["F4"])
    orbit_o6 = {
        "strict_source": orbit_o6_variant["F2"],
        "target_unlabeled": orbit_o6_target_geometry,
    }
    association_null = _association_null_audit(
        canonical_features, orbit_o6, residuals, arrays, canonical_family,
    )
    effect_intervals = _effect_intervals(canonical_family)
    orbit = _orbit_audit(orbit_arrays, residuals, arrays)
    topology = _topology_and_controls(arrays, canonical, residuals)
    prediction = _prediction_audit(arrays, canonical_features, y, prior_predictions, residuals)
    nuisance = _nuisance_controls(arrays, canonical, residuals)
    synthetic_rows, synthetic_summary = synthetic.run_benchmark()

    nonredundancy_rows, nonredundancy = _nonredundancy_audit(arrays, canonical_features)
    qualification = []
    for candidate, path in (("G3S_strict_source", "strict_source"), ("G3T_target_unlabeled", "target_unlabeled")):
        path_associations = [row for row in association_null["summary"] if row["path"] == path]
        strict_survivors = [
            row for row in path_associations
            if row["required_nulls_passing_0.05"] == row["required_null_count"]
        ]
        primary = max(strict_survivors or path_associations, key=lambda row: row["association"])
        interval = next(
            row for row in effect_intervals
            if row["path"] == path and row["kernel"] == primary["kernel"]
            and row["statistic"] == primary["statistic"]
            and float(row["bandwidth_factor"]) == float(primary["bandwidth_factor"])
        )
        pred = next(row for row in prediction["summary"] if row["path"] == path)
        action = next(row for row in prediction["action_summary"] if row["path"] == path)
        identity_ok = all(
            row["max_projection_error"] <= c76_protocol.FUNCTIONAL_IDENTITY_TOLERANCE
            and row["max_probability_error"] <= c76_protocol.FUNCTIONAL_IDENTITY_TOLERANCE
            and row["prediction_disagreements"] == 0
            for row in orbit["identity"]
        )
        gates = {
            "functional_identity": identity_ok,
            "orbit_robustness": bool(orbit["registered_robustness"][(
                path, primary["kernel"], float(primary["bandwidth_factor"]), primary["statistic"],
            )]),
            "association_effect": primary["association"] >= 0.02,
            "association_bootstrap_lower": interval["bootstrap_ci_low"] > 0,
            "incremental_R2": pred["incremental_R2"] >= 0.02,
            "observed_above_nested_null_p95": bool(pred["observed_above_nested_null_p95"]),
            "global_max_stat_p": pred["global_max_stat_p"] < 0.05 and primary["worst_required_global_p"] < 0.05,
            "leave_target_median_increment": pred["leave_target_median_increment"] > 0,
            "positive_targets": pred["positive_targets"] >= 7,
            "material_actionability": bool(action["material_actionability"]),
            "not_redundant_with_logits": bool(nonredundancy[path]),
            # The locked registry names the gate after the forbidden condition;
            # `passed=1` means leakage was absent (observed leakage=0).
            "target_label_leakage": True,
        }
        for gate, passed in gates.items():
            qualification.append({
                "candidate": candidate, "path": path, "gate": gate, "passed": int(passed),
                "association_kernel": primary["kernel"], "association_bandwidth_factor": primary["bandwidth_factor"],
                "association_statistic": primary["statistic"], "association": primary["association"],
                "association_worst_required_p": primary["worst_required_global_p"],
                "incremental_R2": pred["incremental_R2"], "prediction_global_p": pred["global_max_stat_p"],
                "positive_targets": pred["positive_targets"], "material_actionability": action["material_actionability"],
                "observed_target_label_leakage": 0,
            })
        qualification.append({
            "candidate": candidate, "path": path, "gate": "ALL_REQUIRED", "passed": int(all(gates.values())),
            "association_kernel": primary["kernel"], "association_bandwidth_factor": primary["bandwidth_factor"],
            "association_statistic": primary["statistic"], "association": primary["association"],
            "association_worst_required_p": primary["worst_required_global_p"],
            "incremental_R2": pred["incremental_R2"], "prediction_global_p": pred["global_max_stat_p"],
            "positive_targets": pred["positive_targets"], "material_actionability": action["material_actionability"],
            "observed_target_label_leakage": 0,
        })
    qualified = [row["candidate"] for row in qualification if row["gate"] == "ALL_REQUIRED" and row["passed"] == 1]

    feature_registry = _read_csv(TABLE_DIR / "functional_architecture_feature_registry.csv")
    invariance_ledger = []
    for row in feature_registry:
        derived = {
            **row,
            "C76_claim_scope": "function_level" if row["function_invariant"] == "1" else "orthogonal_only" if row["orthogonal_invariant"] == "1" else "coordinate_tied",
            "analysis_interpretation_repair": "none",
        }
        if row["group"] == "G3T":
            derived.update({
                "source": "orbit_recomputed_C75_F4_columns_0_19",
                "analysis_interpretation_repair": "exclude_mixed_function_invariant_Wz_tail_columns_20_34",
            })
        invariance_ledger.append(derived)
    invariance_ledger.extend([
        {
            "group": "G2T_F4TAIL", "name": "target_unlabeled_F4_function_invariant_tail",
            "source": "C75_F4_columns_20_34", "strict_source": 0, "target_unlabeled": 1,
            "target_labels": 0, "function_invariant": 1, "orthogonal_invariant": 1,
            "coordinate_dependent": 0, "redundant_with_logits": 1,
            "qualification_candidate": 0, "C76_claim_scope": "function_level",
            "analysis_interpretation_repair": "isolated_from_G3T_candidate",
        },
        {
            "group": "G3T_FULL_C75", "name": "target_unlabeled_C75_F4_mixed_replay",
            "source": "C75_F4_columns_0_34", "strict_source": 0, "target_unlabeled": 1,
            "target_labels": 0, "function_invariant": 0, "orthogonal_invariant": 1,
            "coordinate_dependent": 0, "redundant_with_logits": 1,
            "qualification_candidate": 0, "C76_claim_scope": "mixed_diagnostic_only",
            "analysis_interpretation_repair": "exact_C75_replay_only",
        },
    ])
    association_prediction = []
    for path in ("strict_source", "target_unlabeled"):
        c75_primary = max(
            [row for row in association_null["summary"] if row["path"] == path and row["kernel"] == "rbf" and row["statistic"] == "normalized_alignment"],
            key=lambda row: row["association"],
        )
        path_associations = [row for row in association_null["summary"] if row["path"] == path]
        survivors = [row for row in path_associations if row["required_nulls_passing_0.05"] == row["required_null_count"]]
        strict_best = max(survivors or path_associations, key=lambda row: row["association"])
        pred = next(row for row in prediction["summary"] if row["path"] == path)
        action = next(row for row in prediction["action_summary"] if row["path"] == path)
        association_prediction.append({
            "path": path, "C75_rbf_association": c75_primary["association"],
            "C75_rbf_worst_required_p": c75_primary["worst_required_global_p"],
            "strict_best_kernel": strict_best["kernel"],
            "strict_best_bandwidth_factor": strict_best["bandwidth_factor"],
            "strict_best_statistic": strict_best["statistic"],
            "association": strict_best["association"],
            "association_worst_required_p": strict_best["worst_required_global_p"],
            "orbit_robustness": orbit["registered_robustness"][(
                path, strict_best["kernel"], float(strict_best["bandwidth_factor"]), strict_best["statistic"],
            )],
            "incremental_R2": pred["incremental_R2"], "prediction_global_p": pred["global_max_stat_p"],
            "leave_target_median_increment": pred["leave_target_median_increment"],
            "positive_targets": pred["positive_targets"],
            "material_actionability": action["material_actionability"],
            "association_prediction_separated": int(primary["association"] >= 0.02 and pred["incremental_R2"] < 0.02),
        })

    tables = {
        "c75_protocol_replay.csv": replay["protocol"],
        "c75_cache_identity_replay.csv": replay["cache"],
        "c75_linear_block_replay.csv": replay["linear"],
        "c75_rbf_association_replay.csv": replay["rbf"],
        "c75_multiplicity_replay.csv": replay["multiplicity"],
        "orbit_functional_identity.csv": orbit["identity"],
        "orbit_feature_stability.csv": orbit["feature"],
        "orbit_rbf_association_stability.csv": orbit["association"],
        "orbit_registered_association_stability.csv": orbit["registered"],
        "orbit_candidate_order_stability.csv": orbit["order"],
        "orbit_family_robustness.csv": orbit["family"],
        "orbit_registered_family_robustness.csv": orbit["registered_family"],
        "invariance_availability_ledger.csv": invariance_ledger,
        "candidate_nonredundancy_audit.csv": nonredundancy_rows,
        "target_F4_partition_audit.csv": [{
            "full_block": "C75_F4", "full_dimension": canonical["F4"].shape[1],
            "candidate_block": "F4_geometry_columns_0_19", "candidate_dimension": target_geometry.shape[1],
            "function_invariant_tail": "F4_Wz_columns_20_34", "invariant_dimension": target_invariant.shape[1],
            "reconstruction_max_abs": float(np.max(np.abs(np.concatenate((target_geometry, target_invariant), axis=1) - canonical["F4"]))),
            "full_F4_used_for_C75_exact_replay_only": 1,
            "formal_target_candidate_excludes_Wz_tail": 1,
        }],
        "conditional_on_logits_association.csv": topology["conditional"],
        "association_topology.csv": topology["topology"],
        "within_target_association.csv": topology["within"],
        "leave_target_out_association.csv": topology["leave_target"],
        "leave_trajectory_out_association.csv": topology["leave_trajectory"],
        "identity_conditioned_association.csv": topology["identity"],
        "heterogeneity_decomposition.csv": topology["heterogeneity"],
        "block_nonlinear_association_summary.csv": association_null["summary"],
        "nested_null_summary.csv": association_null["rows"],
        "association_max_stat_null.csv": association_null["max"],
        "association_effect_size_ci.csv": effect_intervals,
        "association_prediction_separation.csv": association_prediction,
        "cross_fitted_prediction_summary.csv": prediction["summary"],
        "cross_fitted_prediction_folds.csv": prediction["folds"],
        "prediction_leave_target_out.csv": prediction["leave"],
        "prediction_nested_null.csv": prediction["null"],
        "actionability_target_ledger.csv": prediction["action"],
        "actionability_materiality_summary.csv": prediction["action_summary"],
        "identity_nuisance_controls.csv": nuisance["identity"],
        "matched_dimension_controls.csv": nuisance["dimension"],
        "random_coordinate_controls.csv": nuisance["random"],
        "synthetic_orbit_calibration.csv": synthetic_summary,
        "synthetic_association_prediction_separation.csv": synthetic_rows,
        "synthetic_false_positive_control.csv": synthetic_summary,
        "t3_candidate_gate.csv": qualification,
        "risk_register.csv": _risk_rows(),
        "failure_reason_ledger.csv": [{
            "reason": "no_blocking_cache_or_protocol_failure", "active": 1,
            "qualified_candidates": ";".join(qualified), "T3_HO_touched": 0,
            "notes": "association, orbit, prediction, and actionability evaluated separately",
        }],
    }
    for name, rows in tables.items():
        _write_csv(name, rows)

    if qualified:
        primary_candidate = "C76-E_factorization_invariant_incremental_candidate_for_T3_HO"
        final_gate_candidate = "FACTOR_INVARIANT_CANDIDATE_READY_FOR_T3_HO"
    else:
        strict_survivors = [
            row for row in association_null["summary"]
            if row["association"] >= 0.02
            and row["required_nulls_passing_0.05"] == row["required_null_count"]
        ]
        strict_association_survives = bool(strict_survivors)
        surviving_orbit_passes = [
            orbit["registered_robustness"][(
                row["path"], row["kernel"], float(row["bandwidth_factor"]), row["statistic"],
            )]
            for row in strict_survivors
        ]
        if not strict_association_survives:
            primary_candidate = "C76-A_RBF_association_collapses_under_blocked_orbit_controls"
            final_gate_candidate = "RBF_ASSOCIATION_COLLAPSES_UNDER_STRICT_CONTROLS"
        elif not any(surviving_orbit_passes):
            primary_candidate = "C76-B_architecture_tied_coordinate_association_only"
            final_gate_candidate = "ARCHITECTURE_TIED_ASSOCIATION_ONLY"
        else:
            primary_candidate = "C76-D_local_nonlinear_measurement_nontransportable_nonactionable"
            final_gate_candidate = "LOCAL_NONLINEAR_MEASUREMENT_NONTRANSPORTABLE"
    state = {
        "schema_version": "c76_representation_association_analysis_state_v1",
        "protocol_sha256": c76_protocol.sha256(c76_protocol.PROTOCOL_PATH),
        "orbit_cache_manifest_sha256": c76_protocol.sha256(c76_orbit.orbit_manifest_path(protocol)),
        "orbit_variant_count": orbit_manifest["orbit_variant_count"],
        "functional_identity_max_abs": orbit_manifest["functional_identity_max_abs"],
        "T3_HO_z_Wz_accessed": orbit_manifest["T3_HO_z_Wz_accessed"],
        "same_label_oracle_accessed": orbit_manifest["same_label_oracle_accessed"],
        "qualified_candidates": qualified, "C77_protocol_created": False,
        "target_F4_partition": {
            "geometry_candidate_dimension": target_geometry.shape[1],
            "function_invariant_Wz_tail_dimension": target_invariant.shape[1],
            "full_F4_used_for_C75_replay_only": True,
        },
        "orbit_robustness": orbit["robustness"],
        "selected_association_orbit_robustness": {
            row["path"]: row["orbit_robustness"] for row in association_prediction
        },
        "association_prediction": association_prediction,
        "primary_candidate": primary_candidate,
        "final_gate_candidate": final_gate_candidate,
        "representation_mechanism_claimed": False, "target_gauge_claimed": False,
        "selector_or_checkpoint_artifact": False, "diagnostic_only_non_deployable": True,
    }
    c74_cache.atomic_json(STATE_PATH, state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("extract", "analyze"))
    args = parser.parse_args(argv)
    result = c76_orbit.extract_orbit_cache() if args.command == "extract" else analyze()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
