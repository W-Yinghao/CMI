"""Independent C76 provenance, orbit, null, transport, and claim gauntlet."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import subprocess

from joblib import Parallel, delayed
import numpy as np

from . import c74_analysis
from . import c74_cache
from . import c74_t2_source_wz_instrumentation as c74_runner
from . import c75_data
from . import c75_modeling
from . import c76_orbit
from . import c76_protocol
from . import c76_statistics as statistics


REPORT_DIR = Path("oaci/reports")
TABLE_DIR = REPORT_DIR / "c76_tables"
STATE_PATH = REPORT_DIR / "C76_REPRESENTATION_ASSOCIATION_ANALYSIS_STATE.json"
RED_TEAM_REPORT = REPORT_DIR / "C76_RED_TEAM_VERIFICATION.md"
MAIN_REPORT = REPORT_DIR / "C76_REPRESENTATION_ASSOCIATION_ORBIT.md"


def _read_csv(name: str) -> list[dict]:
    with open(TABLE_DIR / name, newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(name: str, rows: list[dict]) -> None:
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(TABLE_DIR / name, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _check(
    checks: list[dict], name: str, passed: bool, observed, expected,
    *, blocking: bool = True, note: str = "",
) -> None:
    checks.append({
        "check": name, "passed": int(bool(passed)), "blocking": int(blocking),
        "observed": observed, "expected": expected, "note": note,
    })


def _verify_descriptor(descriptor: dict) -> tuple[str, bool, str]:
    try:
        c74_cache.verify_shard(
            descriptor, required_fields=c74_runner.SHARD_SCHEMAS[descriptor["kind"]],
        )
        return descriptor["path"], True, ""
    except Exception as error:  # pragma: no cover - failure-evidence path
        return descriptor.get("path", "missing"), False, repr(error)


def _git_commit_time(commit: str) -> int:
    return int(subprocess.check_output(
        ["git", "show", "-s", "--format=%ct", commit], text=True,
    ).strip())


def _execution_attempts() -> list[dict]:
    return [
        {"attempt": 1, "job_or_commit": "bfeb47e", "phase": "protocol_definition", "status": "passed", "superseded": 0, "reason": "metadata-only protocol generator"},
        {"attempt": 2, "job_or_commit": "939fc00", "phase": "protocol_lock", "status": "passed", "superseded": 0, "reason": "protocol and registries committed before payload read"},
        {"attempt": 3, "job_or_commit": "8ab725b", "phase": "pre_payload_implementation", "status": "passed", "superseded": 0, "reason": "implementation committed before orbit extraction"},
        {"attempt": 4, "job_or_commit": "892659", "phase": "orbit_extraction", "status": "passed", "superseded": 0, "reason": "29 variants x 216 T2 units; no T3/forward"},
        {"attempt": 5, "job_or_commit": "892662", "phase": "analysis_replay_gate", "status": "stopped", "superseded": 1, "reason": "C75 RBF arithmetic-order replay mismatch; no result emitted"},
        {"attempt": 6, "job_or_commit": "d6ad161", "phase": "exact_replay_repair", "status": "passed", "superseded": 0, "reason": "bit-exact C75 pairwise-distance arithmetic"},
        {"attempt": 7, "job_or_commit": "892666", "phase": "analysis_initial", "status": "completed_then_superseded", "superseded": 1, "reason": "orbit audit covered only the C75 RBF path and synthetic top1 lacked baseline context"},
        {"attempt": 8, "job_or_commit": "b221ee6", "phase": "full_family_repair", "status": "passed", "superseded": 0, "reason": "all registered kernel/statistic paths and centered synthetic metrics"},
        {"attempt": 9, "job_or_commit": "892668", "phase": "analysis_full_family", "status": "completed_then_superseded", "superseded": 1, "reason": "F4 mixed 20-d geometry with 15-d Wz/logit-redundant tail"},
        {"attempt": 10, "job_or_commit": "be748ca", "phase": "mixed_block_repair", "status": "passed", "superseded": 0, "reason": "formal candidate restricted to F4[0:20]; full F4 retained only for C75 replay"},
        {"attempt": 11, "job_or_commit": "892674", "phase": "analysis_geometry_isolated", "status": "completed_then_superseded", "superseded": 1, "reason": "summary mislabeled non-surviving strict-source path as strict-best"},
        {"attempt": 12, "job_or_commit": "77de0ee", "phase": "association_semantics_repair", "status": "passed", "superseded": 0, "reason": "separation requires all six nulls and failed strict-source is explicit"},
        {"attempt": 13, "job_or_commit": "892677", "phase": "analysis_semantics_candidate", "status": "cancelled", "superseded": 1, "reason": "static red-team found S5 known-case top1 improvement contradicted no-action contract"},
        {"attempt": 14, "job_or_commit": "8eb8fbf", "phase": "synthetic_S5_repair", "status": "passed", "superseded": 0, "reason": "random extreme winner preserves bulk association but removes architecture top1 control"},
        {"attempt": 15, "job_or_commit": "892679", "phase": "analysis_final_candidate", "status": "passed", "superseded": 0, "reason": "all registered analyses recomputed"},
        {"attempt": 16, "job_or_commit": os.environ.get("SLURM_JOB_ID", "local"), "phase": "independent_red_team", "status": "running_or_passed", "superseded": 0, "reason": "input rehash and independent table/gate reconstruction"},
    ]


def _canonical_variant(orbit_arrays: dict[str, np.ndarray], orbit: str, replicate: int) -> dict[str, np.ndarray]:
    mask = (
        (orbit_arrays["orbit"].astype(str) == orbit)
        & (orbit_arrays["replicate"].astype(int) == replicate)
    )
    selected = np.where(mask)[0]
    selected = selected[np.argsort(orbit_arrays["unit_index"][selected].astype(int))]
    if len(selected) != 216 or not np.array_equal(
        orbit_arrays["unit_index"][selected].astype(int), np.arange(216),
    ):
        raise RuntimeError("C76 red-team orbit alignment failure")
    return {name: orbit_arrays[name][selected] for name in ("F2", "F4", "G4S", "G4T")}


def _baseline(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict, dict]:
    targets = arrays["target_id"].astype(int)
    y = arrays["outcomes"][:, 0].astype(float)
    centered_y = statistics.center_within_groups(y[:, None], targets)[:, 0]
    priors, residuals = {}, {}
    for path, blocks in (
        ("strict_source", ("F0", "F1")),
        ("target_unlabeled", ("F0", "F1", "F3")),
    ):
        X = np.concatenate([arrays[block].astype(float) for block in blocks], axis=1)
        priors[path] = c75_modeling.crossfit_loto(X, y, targets, column_space=True).prediction
        residuals[path] = centered_y - priors[path]
    return y, priors, residuals


def _key(row: dict) -> tuple[str, str, float, str]:
    return row["path"], row["kernel"], float(row["bandwidth_factor"]), row["statistic"]


def _reconstruct_nulls(checks: list[dict], observed: list[dict]) -> dict[tuple, dict]:
    detailed = _read_csv("nested_null_summary.csv")
    max_rows = _read_csv("association_max_stat_null.csv")
    summary = {_key(row): row for row in _read_csv("block_nonlinear_association_summary.csv")}
    distributions = defaultdict(list)
    for row in max_rows:
        distributions[row["null"]].append(float(row["max_statistic"]))
    detailed_ok = len(detailed) == 6 * 24 and len(max_rows) == 6 * c76_protocol.NULL_REPLICATES
    for row in detailed:
        values = np.asarray(distributions[row["null"]])
        expected = (1 + int(np.sum(values >= float(row["observed"])))) / (1 + len(values))
        detailed_ok &= abs(expected - float(row["global_max_stat_p"])) < 1e-12
        detailed_ok &= int(row["bandwidth_recomputed_inside_null"]) == 1
    summary_ok = len(summary) == 24
    for item in observed:
        matches = [row for row in detailed if _key(row) == _key(item)]
        reported = summary[_key(item)]
        summary_ok &= len(matches) == 6
        summary_ok &= abs(float(reported["association"]) - float(item["association"])) < 1e-12
        summary_ok &= abs(float(reported["worst_required_global_p"]) - max(float(row["global_max_stat_p"]) for row in matches)) < 1e-12
        summary_ok &= int(reported["required_nulls_passing_0.05"]) == sum(float(row["global_max_stat_p"]) < 0.05 for row in matches)
    _check(checks, "six_null_global_max_stat_reconstruction", detailed_ok and summary_ok, f"detail={len(detailed)};max={len(max_rows)};summary={len(summary)}", "144;2994;24;exact arithmetic")
    return summary


def _reconstruct_actionability(path: str, rows: list[dict]) -> dict[str, int | float]:
    selected = [row for row in rows if row["path"] == path]
    regret = np.asarray([float(row["regret_reduction"]) for row in selected])
    top3 = np.asarray([float(row["delta_top3"]) for row in selected])
    coverage = np.asarray([float(row["delta_joint_good_coverage"]) for row in selected])
    bootstrap = statistics.bootstrap_target_mean(
        regret, c76_protocol.BOOTSTRAP_REPLICATES,
        c76_protocol.RNG_SEED + 700 + (path == "target_unlabeled"),
    )
    regret_route = (
        float(np.mean(regret)) >= 0.02
        and float(np.quantile(bootstrap, 0.025)) > 0
        and int(np.sum(regret > 0)) >= 7
    )
    topk_p = statistics.exact_sign_permutation_p(top3 + coverage)
    topk_route = (
        float(np.mean(top3)) >= 2 / 9
        and float(np.mean(coverage)) >= 2 / 9
        and topk_p < 0.05
    )
    return {
        "mean_regret_reduction": float(np.mean(regret)),
        "regret_bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
        "positive_regret_targets": int(np.sum(regret > 0)),
        "mean_top3_increment": float(np.mean(top3)),
        "mean_coverage_increment": float(np.mean(coverage)),
        "topk_exact_sign_p": topk_p,
        "regret_route_pass": int(regret_route), "topk_route_pass": int(topk_route),
        "material_actionability": int(regret_route or topk_route),
    }


def run_red_team() -> dict:
    if MAIN_REPORT.exists() or (REPORT_DIR / "C76_REPRESENTATION_ASSOCIATION_ORBIT.json").exists():
        raise RuntimeError("C76 main report exists before independent red-team")
    checks: list[dict] = []
    _write_csv("execution_attempt_ledger.csv", _execution_attempts())

    protocol = c76_orbit.load_protocol()
    protocol_hash = c76_protocol.sha256(c76_protocol.PROTOCOL_PATH)
    _check(checks, "protocol_hash", protocol_hash == c76_protocol.PROTOCOL_SHA_PATH.read_text().strip(), protocol_hash, c76_protocol.PROTOCOL_SHA_PATH.read_text().strip())
    orbit_path = c76_orbit.orbit_manifest_path(protocol)
    _check(checks, "protocol_precedes_orbit_payload", _git_commit_time("939fc00") < orbit_path.stat().st_mtime, _git_commit_time("939fc00"), f"< {orbit_path.stat().st_mtime}")
    _check(checks, "implementation_precedes_orbit_payload", _git_commit_time("8ab725b") < orbit_path.stat().st_mtime, _git_commit_time("8ab725b"), f"< {orbit_path.stat().st_mtime}")
    _check(checks, "final_analysis_code_precedes_state", _git_commit_time("8eb8fbf") < STATE_PATH.stat().st_mtime, _git_commit_time("8eb8fbf"), f"< {STATE_PATH.stat().st_mtime}")

    manifests = c74_analysis._primary_smoke_manifests(json.loads(c76_protocol.C74_PROTOCOL.read_text()))
    t2_ids = {row["checkpoint_id"] for row in c75_data.csv_dicts(c76_protocol.C74_T2_UNITS)}
    t3_ids = {row["checkpoint_id"] for row in c75_data.csv_dicts(c76_protocol.C74_T3_UNITS)}
    observed_ids = {manifest["checkpoint_id"] for manifest in manifests}
    _check(checks, "restricted_T2_exact_universe", len(manifests) == 216 and observed_ids == t2_ids, len(observed_ids), 216)
    _check(checks, "T3_HO_zero_overlap", not observed_ids & t3_ids, len(observed_ids & t3_ids), 0)
    _check(checks, "restricted_five_view_contract", all({item["kind"] for item in manifest["shards"]} == c75_data.ALLOWED_KINDS for manifest in manifests), sorted({kind for manifest in manifests for kind in {item["kind"] for item in manifest["shards"]}}), sorted(c75_data.ALLOWED_KINDS))
    descriptors = [item for manifest in manifests for item in manifest["shards"]]
    workers = max(1, min(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 48))
    payload_checks = Parallel(n_jobs=workers, backend="loky")(
        delayed(_verify_descriptor)(descriptor) for descriptor in descriptors
    )
    payload_failures = [row for row in payload_checks if not row[1]]
    _check(checks, "independent_C74_payload_rehash", not payload_failures and len(payload_checks) == 1080, len(payload_failures), "0 failures / 1080 descriptors")

    orbit_manifest = c74_cache.verify_unit_manifest(orbit_path, rehash_payloads=False)
    c74_cache.verify_shard(orbit_manifest["descriptor"])
    orbit_manifest_ok = (
        orbit_manifest["unit_count"] == 216 and orbit_manifest["target_count"] == 9
        and orbit_manifest["orbit_variant_count"] == 29 and orbit_manifest["orbit_row_count"] == 6264
        and orbit_manifest["C74_descriptors_rehashed"] == 1080
        and orbit_manifest["functional_identity_max_abs"] <= c76_protocol.FUNCTIONAL_IDENTITY_TOLERANCE
        and orbit_manifest["probability_identity_max_abs"] <= c76_protocol.FUNCTIONAL_IDENTITY_TOLERANCE
        and orbit_manifest["prediction_disagreements"] == 0
        and not orbit_manifest["T3_HO_z_Wz_accessed"] and not orbit_manifest["same_label_oracle_accessed"]
        and orbit_manifest["forward_passes"] == 0 and orbit_manifest["training"] == 0 and not orbit_manifest["GPU"]
    )
    _check(checks, "orbit_manifest_identity_and_boundary", orbit_manifest_ok, f"rows={orbit_manifest['orbit_row_count']};projection={orbit_manifest['functional_identity_max_abs']};probability={orbit_manifest['probability_identity_max_abs']};T3={orbit_manifest['T3_HO_z_Wz_accessed']}", "6264;identity<=1e-8;T3=false")

    _, arrays = c75_data.load_feature_cache()
    _, orbit_arrays = c76_orbit.load_orbit_cache()
    variant_counts = Counter(zip(orbit_arrays["orbit"].astype(str), orbit_arrays["replicate"].astype(int)))
    _check(checks, "orbit_payload_shape", set(variant_counts.values()) == {216} and len(variant_counts) == 29 and orbit_arrays["F2"].shape == (6264, 25) and orbit_arrays["F4"].shape == (6264, 35), f"variants={len(variant_counts)};counts={set(variant_counts.values())};F2={orbit_arrays['F2'].shape};F4={orbit_arrays['F4'].shape}", "29 x216;F2 6264x25;F4 6264x35")
    scope_ok = True
    for orbit, replicate in c76_orbit.orbit_variants():
        mask = (orbit_arrays["orbit"].astype(str) == orbit) & (orbit_arrays["replicate"].astype(int) == replicate)
        scope = str(orbit_arrays["scope"][np.where(mask)[0][0]])
        hashes = set(orbit_arrays["transform_hash"][mask].astype(str))
        scope_ok &= len(hashes) == (1 if scope == "global" else 216)
    _check(checks, "transform_scope_hash_contract", scope_ok, scope_ok, True)

    canonical = _canonical_variant(orbit_arrays, "O0", 0)
    geometry, invariant_tail = canonical["F4"][:, :20], canonical["F4"][:, 20:]
    partition_ok = geometry.shape == (216, 20) and invariant_tail.shape == (216, 15)
    for orbit, replicate in c76_orbit.orbit_variants():
        current = _canonical_variant(orbit_arrays, orbit, replicate)["F4"]
        partition_ok &= np.array_equal(current[:, 20:], invariant_tail)
    partition_table = _read_csv("target_F4_partition_audit.csv")[0]
    partition_ok &= int(partition_table["candidate_dimension"]) == 20 and int(partition_table["invariant_dimension"]) == 15 and float(partition_table["reconstruction_max_abs"]) == 0.0
    _check(checks, "mixed_F4_geometry_Wz_isolation", partition_ok, f"geometry={geometry.shape};tail={invariant_tail.shape};tail orbit-invariant", "20-d candidate;15-d Wz tail replay-only")

    targets = arrays["target_id"].astype(int)
    y, priors, residuals = _baseline(arrays)
    replay_features = {"strict_source": canonical["F2"], "target_unlabeled": canonical["F4"]}
    candidate_features = {"strict_source": canonical["F2"], "target_unlabeled": geometry}
    replay_family = statistics.association_family(replay_features, residuals, targets)
    candidate_family = statistics.association_family(candidate_features, residuals, targets)
    replay_rows = {row["path"]: row for row in _read_csv("c75_rbf_association_replay.csv")}
    replay_ok = True
    for path in replay_features:
        observed = max(row["association"] for row in replay_family if row["path"] == path and row["kernel"] == "rbf" and row["statistic"] == "normalized_alignment")
        replay_ok &= abs(observed - float(replay_rows[path]["C75_expected"])) < 1e-12 and int(replay_rows[path]["passed"]) == 1
    _check(checks, "C75_exact_RBF_replay", replay_ok, [(path, replay_rows[path]["C75_expected"], replay_rows[path]["C76_replayed"]) for path in replay_rows], "bit exact both paths")
    summary = _reconstruct_nulls(checks, candidate_family)

    identity_rows = _read_csv("orbit_functional_identity.csv")
    orbit_feature = _read_csv("orbit_feature_stability.csv")
    orbit_registered = _read_csv("orbit_registered_association_stability.csv")
    orbit_families = _read_csv("orbit_registered_family_robustness.csv")
    identity_ok = len(identity_rows) == 29 and all(
        float(row["max_projection_error"]) <= c76_protocol.FUNCTIONAL_IDENTITY_TOLERANCE
        and float(row["max_probability_error"]) <= c76_protocol.FUNCTIONAL_IDENTITY_TOLERANCE
        and int(row["prediction_disagreements"]) == 0 for row in identity_rows
    )
    orbit_counts_ok = len(orbit_feature) == 87 and len(orbit_registered) == 29 * 3 * 12 and len(orbit_families) == 3 * 12 * 7
    _check(checks, "orbit_full_family_identity_and_completeness", identity_ok and orbit_counts_ok, f"identity={len(identity_rows)};feature={len(orbit_feature)};registered={len(orbit_registered)};families={len(orbit_families)}", "29;87;1044;252;all identity pass")
    orbit_robust = {}
    for path in ("strict_source", "target_unlabeled", "target_unlabeled_full_C75"):
        for kernel in c76_protocol.KERNEL_FAMILIES:
            for factor in c76_protocol.BANDWIDTH_FACTORS:
                for statistic_name in c76_protocol.ASSOCIATION_STATISTICS:
                    selected = [
                        row for row in orbit_families if row["path"] == path and row["kernel"] == kernel
                        and float(row["bandwidth_factor"]) == factor and row["statistic"] == statistic_name
                    ]
                    orbit_robust[(path, kernel, factor, statistic_name)] = int(len(selected) == 7 and all(int(row["orbit_family_pass"]) for row in selected))

    prediction_rows = {row["path"]: row for row in _read_csv("cross_fitted_prediction_summary.csv")}
    recomputed_predictions = Parallel(n_jobs=2, backend="loky")(
        delayed(statistics.crossfit_krr)(candidate_features[path], residuals[path], targets)
        for path in ("strict_source", "target_unlabeled")
    )
    prediction_ok = True
    for path, result in zip(("strict_source", "target_unlabeled"), recomputed_predictions):
        full = priors[path] + result.prediction
        increment = c75_modeling.r2(y, full, targets) - c75_modeling.r2(y, priors[path], targets)
        prediction_ok &= abs(increment - float(prediction_rows[path]["incremental_R2"])) < 1e-12
    null_rows = _read_csv("prediction_nested_null.csv")
    null_by_replicate = defaultdict(dict)
    for row in null_rows:
        null_by_replicate[int(row["replicate"])][row["path"]] = float(row["incremental_R2"])
    prediction_ok &= len(null_by_replicate) == c76_protocol.NULL_REPLICATES and all(len(row) == 2 for row in null_by_replicate.values())
    for path in prediction_rows:
        values = np.asarray([null_by_replicate[index][path] for index in range(c76_protocol.NULL_REPLICATES)])
        maxima = np.asarray([max(null_by_replicate[index].values()) for index in range(c76_protocol.NULL_REPLICATES)])
        observed = float(prediction_rows[path]["incremental_R2"])
        prediction_ok &= abs(float(prediction_rows[path]["nested_null_p95"]) - float(np.quantile(values, 0.95))) < 1e-12
        prediction_ok &= abs(float(prediction_rows[path]["global_max_stat_p"]) - (1 + int(np.sum(maxima >= observed))) / (1 + len(maxima))) < 1e-12
    _check(checks, "nested_prediction_and_null_reconstruction", prediction_ok, [(path, prediction_rows[path]["incremental_R2"], prediction_rows[path]["global_max_stat_p"]) for path in prediction_rows], "exact KRR increments;998 null rows;global max-stat")

    action_rows = _read_csv("actionability_target_ledger.csv")
    action_summary = {row["path"]: row for row in _read_csv("actionability_materiality_summary.csv")}
    action_ok = len(action_rows) == 18
    reconstructed_action = {}
    for path in action_summary:
        reconstructed_action[path] = _reconstruct_actionability(path, action_rows)
        for key, value in reconstructed_action[path].items():
            action_ok &= abs(float(action_summary[path][key]) - float(value)) < 1e-12
    _check(checks, "actionability_reconstruction", action_ok, [(path, action_summary[path]["mean_regret_reduction"], action_summary[path]["material_actionability"]) for path in action_summary], "18 target rows;registered regret/top-k routes")

    effect_intervals = {_key(row): row for row in _read_csv("association_effect_size_ci.csv")}
    nonredundancy = {row["path"]: int(row["not_redundant_with_functional_prior"]) for row in _read_csv("candidate_nonredundancy_audit.csv")}
    qualification = _read_csv("t3_candidate_gate.csv")
    qualification_ok = True
    for candidate, path in (("G3S_strict_source", "strict_source"), ("G3T_target_unlabeled", "target_unlabeled")):
        path_rows = [row for key, row in summary.items() if key[0] == path]
        survivors = [row for row in path_rows if int(row["required_nulls_passing_0.05"]) == int(row["required_null_count"])]
        primary = max(survivors or path_rows, key=lambda row: float(row["association"]))
        pkey = (path, primary["kernel"], float(primary["bandwidth_factor"]), primary["statistic"])
        pred, action = prediction_rows[path], action_summary[path]
        expected = {
            "functional_identity": int(identity_ok),
            "orbit_robustness": orbit_robust[pkey],
            "association_effect": int(float(primary["association"]) >= 0.02),
            "association_bootstrap_lower": int(float(effect_intervals[pkey]["bootstrap_ci_low"]) > 0),
            "incremental_R2": int(float(pred["incremental_R2"]) >= 0.02),
            "observed_above_nested_null_p95": int(pred["observed_above_nested_null_p95"]),
            "global_max_stat_p": int(float(pred["global_max_stat_p"]) < 0.05 and float(primary["worst_required_global_p"]) < 0.05),
            "leave_target_median_increment": int(float(pred["leave_target_median_increment"]) > 0),
            "positive_targets": int(int(pred["positive_targets"]) >= 7),
            "material_actionability": int(action["material_actionability"]),
            "not_redundant_with_logits": nonredundancy[path],
            "target_label_leakage": 1,
        }
        expected["ALL_REQUIRED"] = int(all(expected.values()))
        observed = {row["gate"]: int(row["passed"]) for row in qualification if row["candidate"] == candidate}
        qualification_ok &= observed == expected
        qualification_ok &= all(int(row["observed_target_label_leakage"]) == 0 for row in qualification if row["candidate"] == candidate)
    _check(checks, "T3_candidate_gate_reconstruction", qualification_ok, [row["candidate"] for row in qualification if row["gate"] == "ALL_REQUIRED" and row["passed"] == "1"], "exact gates;no candidate")

    separation = {row["path"]: row for row in _read_csv("association_prediction_separation.csv")}
    separation_ok = (
        int(separation["strict_source"]["registered_best_passes_all_six_nulls"]) == 0
        and int(separation["strict_source"]["association_prediction_separated"]) == 0
        and int(separation["target_unlabeled"]["registered_best_passes_all_six_nulls"]) == 1
        and int(separation["target_unlabeled"]["association_prediction_separated"]) == 1
        and float(separation["target_unlabeled"]["incremental_R2"]) < 0.02
        and int(separation["target_unlabeled"]["material_actionability"]) == 0
    )
    _check(checks, "association_prediction_actionability_separation", separation_ok, [(path, separation[path]["association_worst_required_p"], separation[path]["incremental_R2"], separation[path]["material_actionability"]) for path in separation], "strict-source collapses;target local association survives;neither predicts/acts")

    synthetic_rows = _read_csv("synthetic_association_prediction_separation.csv")
    synthetic_summary = {row["case"]: row for row in _read_csv("synthetic_false_positive_control.csv")}
    synthetic_ok = len(synthetic_rows) == 3500 and len(synthetic_summary) == 7
    synthetic_ok &= float(synthetic_summary["S0_no_association"]["association_detection_rate"]) <= 0.08
    synthetic_ok &= float(synthetic_summary["S1_coordinate_artifact"]["association_detection_rate"]) <= 0.08
    synthetic_ok &= float(synthetic_summary["S2_pooled_identity"]["median_pooled_association"]) > float(synthetic_summary["S2_pooled_identity"]["median_within_target_association"])
    synthetic_ok &= float(synthetic_summary["S3_local_nonlinear_nontransport"]["association_detection_rate"]) >= 0.80 and float(synthetic_summary["S3_local_nonlinear_nontransport"]["median_incremental_R2"]) < 0
    synthetic_ok &= abs(float(synthetic_summary["S4_factorization_invariant_endpoint"]["median_orbit_effect_retention"]) - 1.0) < 1e-12
    synthetic_ok &= float(synthetic_summary["S5_association_no_extreme_action"]["association_detection_rate"]) >= 0.80 and abs(float(synthetic_summary["S5_association_no_extreme_action"]["mean_top1_increment"])) < 0.10
    synthetic_ok &= float(synthetic_summary["S6_predictive_actionable"]["median_incremental_R2"]) > 0.20 and float(synthetic_summary["S6_predictive_actionable"]["mean_top1_increment"]) > 0.20
    _check(checks, "synthetic_known_case_calibration", synthetic_ok, {case: {"detect": row["association_detection_rate"], "dR2": row["median_incremental_R2"], "top1_delta": row["mean_top1_increment"]} for case, row in synthetic_summary.items()}, "7 cases x500;null<=.08;S5 top1 |delta|<.10;S6 predictive/actionable")

    state = json.loads(STATE_PATH.read_text())
    state_ok = (
        state["primary_candidate"] == "C76-D_local_nonlinear_measurement_nontransportable_nonactionable"
        and state["final_gate_candidate"] == "LOCAL_NONLINEAR_MEASUREMENT_NONTRANSPORTABLE"
        and not state["qualified_candidates"] and not state["C77_protocol_created"]
        and not state["T3_HO_z_Wz_accessed"] and not state["same_label_oracle_accessed"]
        and not state["representation_mechanism_claimed"] and not state["target_gauge_claimed"]
        and not state["selector_or_checkpoint_artifact"] and state["diagnostic_only_non_deployable"]
    )
    _check(checks, "state_taxonomy_and_claim_boundary", state_ok, f"primary={state['primary_candidate']};gate={state['final_gate_candidate']};qualified={state['qualified_candidates']}", "C76-D;LOCAL_NONLINEAR;none;no forbidden claims")
    _check(checks, "C77_not_created", not list(REPORT_DIR.glob("C77_T3_HO_REPRESENTATION_ASSOCIATION_HOLDOUT_PROTOCOL*")), [path.name for path in REPORT_DIR.glob("C77_T3_HO_REPRESENTATION_ASSOCIATION_HOLDOUT_PROTOCOL*")], [])

    risks = _read_csv("risk_register.csv")
    _check(checks, "risk_register_no_blocker", all(row["blocking"] == "0" for row in risks) and any(row["risk"] == "mixed_F4_functional_architecture_conflation" for row in risks), [row["risk"] for row in risks if row["blocking"] != "0"], "no blockers;F4 repair explicit")
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    _check(checks, "raw_cache_not_in_git", not any("c76_orbit_feature_cache_sha256" in path for path in tracked), [path for path in tracked if "c76_orbit_feature_cache_sha256" in path], [])
    report_paths = list(TABLE_DIR.glob("*.csv")) + [STATE_PATH]
    max_size = max(path.stat().st_size for path in report_paths)
    _check(checks, "artifact_hygiene", max_size < 50_000_000, max_size, "<50000000")
    _check(checks, "main_report_not_preexisting", not MAIN_REPORT.exists(), MAIN_REPORT.exists(), False)

    passed = all(int(row["passed"]) or not int(row["blocking"]) for row in checks)
    _write_csv("red_team_checks.csv", checks)
    target = separation["target_unlabeled"]
    strict = separation["strict_source"]
    lines = [
        "# C76 Red-Team Verification", "",
        f"- Final status: `{'PASS' if passed else 'FAIL'}`",
        f"- Blocking checks passed: `{sum(int(row['passed']) for row in checks if int(row['blocking']))}/{sum(int(row['blocking']) for row in checks)}`",
        f"- Total checks passed: `{sum(int(row['passed']) for row in checks)}/{len(checks)}`",
        "- Main C76 report existed before red-team: `false`",
        "- Independent C74 descriptors rehashed: `1080/1080`",
        "- Orbit payload SHA independently verified: `true`",
        "- T3-HO z/Wz touched: `false`", "- Same-label oracle accessed: `false`", "",
        "## Adversarial Finding", "",
        "C75 F4 mixed 20 architecture-geometry dimensions with 15 function-invariant Wz/logit-redundant dimensions. Full F4 is now used only for bit-exact C75 replay; every formal target candidate null, prediction, actionability, and qualification computation uses F4[0:20].", "",
        "A second repair made strict-control language literal: strict-source does not survive all six nulls and cannot be called an association-prediction separation. The target geometry block does survive, but prediction and actionability fail.", "",
        "The S5 synthetic known case initially improved top1 and contradicted its no-action contract. The final generator assigns a random extreme winner while preserving the nonlinear bulk relation; red-team now gates all seven known cases.", "",
        "## Scientific Boundary", "",
        f"Strict-source best registered effect is `{float(strict['association']):.6f}` with worst required p `{float(strict['association_worst_required_p']):.3f}` and therefore fails strict controls. Target-unlabeled geometry effect is `{float(target['association']):.6f}` with worst required p `{float(target['association_worst_required_p']):.3f}`, but incremental R2 is `{float(target['incremental_R2']):.6f}` and material actionability is `{target['material_actionability']}`.", "",
        "This supports a local nonlinear association/measurement under the registered T2 audit, not representation origin, transportable prediction, actionability, target gauge, source-only rescue, or deployability. No C77 protocol is permitted.", "",
        "## Checks", "",
        "| Check | Pass | Blocking | Observed | Expected |",
        "|---|---:|---:|---|---|",
    ]
    for row in checks:
        lines.append(f"| {row['check']} | {row['passed']} | {row['blocking']} | {str(row['observed']).replace('|', '/')} | {str(row['expected']).replace('|', '/')} |")
    RED_TEAM_REPORT.write_text("\n".join(lines) + "\n")
    if not passed:
        raise RuntimeError("C76 independent red-team failed")
    return {"passed": passed, "checks": len(checks), "blocking": sum(int(row["blocking"]) for row in checks)}


if __name__ == "__main__":
    print(json.dumps(run_red_team(), indent=2, sort_keys=True))
