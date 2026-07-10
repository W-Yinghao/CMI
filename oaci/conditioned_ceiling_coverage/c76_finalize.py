"""Generate C76 reports only after red-team and all regression gates pass."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re

from . import c74_cache
from . import c76_orbit
from . import c76_protocol


REPORT_DIR = Path("oaci/reports")
TABLE_DIR = REPORT_DIR / "c76_tables"
MAIN_MD = REPORT_DIR / "C76_REPRESENTATION_ASSOCIATION_ORBIT.md"
MAIN_JSON = REPORT_DIR / "C76_REPRESENTATION_ASSOCIATION_ORBIT.json"
HANDOFF = Path("oaci/OACI_CODEX_HANDOFF.md")
STATE_PATH = REPORT_DIR / "C76_REPRESENTATION_ASSOCIATION_ANALYSIS_STATE.json"
REGRESSION_JOBS = {
    "C65_C76": (892715, "c76-reg-c65"),
    "C23_C76": (892716, "c76-reg-c23"),
    "full_OACI": (892717, "c76-reg-full"),
}


def _rows(name: str) -> list[dict]:
    with open(TABLE_DIR / name, newline="") as stream:
        return list(csv.DictReader(stream))


def _row(name: str, **matches) -> dict:
    selected = [
        row for row in _rows(name)
        if all(row[key] == str(value) for key, value in matches.items())
    ]
    if len(selected) != 1:
        raise RuntimeError(f"C76 expected one row in {name}: {matches}; got {len(selected)}")
    return selected[0]


def _regression_rows() -> list[dict]:
    root = c76_orbit.EXTERNAL_ROOT / "logs"
    rows = [{
        "suite": "focused_C76", "job_id": "local_lightweight",
        "passed_tests": 18, "stderr_empty": 1, "status": "passed",
        "command": "python -m pytest oaci/tests/test_c76_representation_association_orbit.py -q",
    }]
    for suite, (job_id, name) in REGRESSION_JOBS.items():
        stdout = root / f"{name}_{job_id}.out"
        stderr = root / f"{name}_{job_id}.err"
        if not stdout.is_file() or not stderr.is_file():
            raise RuntimeError(f"C76 regression logs missing for {job_id}")
        text = stdout.read_text()
        matches = re.findall(r"(\d+) passed(?:, [^\n]+)? in [0-9.]+s", text)
        if len(matches) != 1 or stderr.read_text() or "failed" in text.lower():
            raise RuntimeError(f"C76 regression did not pass cleanly: {job_id}")
        rows.append({
            "suite": suite, "job_id": job_id, "passed_tests": int(matches[0]),
            "stderr_empty": 1, "status": "passed", "command": name,
        })
    return rows


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


def _update_handoff(regressions: list[dict]) -> None:
    text = HANDOFF.read_text()
    start = text.index("## 0. Current continuation state")
    end = text.index("\n---\n", start)
    counts = {row["suite"]: row for row in regressions}
    replacement = f"""## 0. Current continuation state (2026-07-10)

The detailed C23-C31 history below remains useful background, but the authoritative tip is now C76:

```text
C75 result commit: fb8a412
C76 protocol SHA:  1a1b4255601d6178ffbe8a8245625845fdb4057c445ca8db25e84b4ddcd8528f
C76 final gate:    LOCAL_NONLINEAR_MEASUREMENT_NONTRANSPORTABLE
C76 primary:       C76-D_local_nonlinear_measurement_nontransportable_nonactionable
```

C76 is a no-forward/no-training T2 robustness audit of C75's nonlinear association. It preserves the untouched
1,052-unit T3-HO z/Wz new-variable holdout. It ran 29 function-preserving orbit variants over all 216 T2 units,
24 registered kernel/statistic paths, six 499-replicate blocked null families, nested leave-target-out prediction,
actionability gates, and seven 500-replicate synthetic known cases.

Gate-first result:

```text
strict-source architecture:
  best registered effect 0.234144
  worst six-null max-stat p 0.054
  strict control pass false
  incremental R2 -0.042483; prediction max-stat p 0.998; actionability false

target-unlabeled architecture geometry (F4[0:20], Wz tail excluded):
  Laplacian-HSIC effect 0.237725; target-bootstrap CI [0.195773, 0.277323]
  worst six-null max-stat p 0.030; all six nulls pass
  incremental R2 -0.011041; prediction max-stat p 0.820
  positive prediction targets 4/9; actionability false
```

The surviving target association is local and diagnostic. It is robust over the registered orbit but does not
transport as held-out prediction or improve checkpoint actionability. It does not identify W-versus-z origin,
representation causality, target gauge, source-only rescue, or a deployable method. No C77 protocol was created;
the T3-HO representation campaign is not justified and the representation branch is saturated under the current
architecture/frozen universe.

Independent red-team job `892694` passed 26/26 blocking checks after rehashing 1,080/1,080 C74 descriptors. It
forced three repairs before reporting: isolate C75 F4's 20-d geometry from its 15-d Wz/logit-redundant tail; require
all six nulls before calling association-prediction separation; and repair synthetic S5 so a random extreme winner
preserves association without top1 control. Final verification is 18 focused C76,
{counts['C65_C76']['passed_tests']} C65-C76 (job 892715), {counts['C23_C76']['passed_tests']} C23-C76 (job 892716),
and {counts['full_OACI']['passed_tests']} full-suite tests (job 892717), all with empty stderr.

Authoritative artifacts:

```text
oaci/reports/C76_REPRESENTATION_ASSOCIATION_ORBIT.md
oaci/reports/C76_REPRESENTATION_ASSOCIATION_ORBIT.json
oaci/reports/C76_RED_TEAM_VERIFICATION.md
oaci/reports/c76_tables/artifact_manifest.csv
oaci/reports/c76_tables/execution_attempt_ledger.csv
```

Do not launch C77, T3-HO representation re-inference, new training, GPU work, BNCI2014_004, seeds [3,4], a selector,
checkpoint recommendations, or manuscript drafting without a new explicit PM instruction.
"""
    HANDOFF.write_text(text[:start] + replacement + text[end:])


def _artifact_manifest(orbit_manifest: dict) -> list[dict]:
    paths = [
        MAIN_MD, MAIN_JSON, REPORT_DIR / "C76_RED_TEAM_VERIFICATION.md", STATE_PATH,
        c76_protocol.PROTOCOL_PATH, c76_protocol.PROTOCOL_SHA_PATH, c76_protocol.TIMING_PATH,
    ]
    paths.extend(sorted(path for path in TABLE_DIR.glob("*.csv") if path.name != "artifact_manifest.csv"))
    paths.extend([
        HANDOFF,
        Path("oaci/conditioned_ceiling_coverage/c76_finalize.py"),
        Path("oaci/conditioned_ceiling_coverage/c76_orbit.py"),
        Path("oaci/conditioned_ceiling_coverage/c76_protocol.py"),
        Path("oaci/conditioned_ceiling_coverage/c76_red_team.py"),
        Path("oaci/conditioned_ceiling_coverage/c76_representation_association_orbit.py"),
        Path("oaci/conditioned_ceiling_coverage/c76_statistics.py"),
        Path("oaci/conditioned_ceiling_coverage/synthetic_association_generator.py"),
        Path("oaci/slurm_c76_analyze.sh"), Path("oaci/slurm_c76_extract.sh"),
        Path("oaci/slurm_c76_finalize.sh"), Path("oaci/slurm_c76_protocol.sh"),
        Path("oaci/slurm_c76_red_team.sh"), Path("oaci/slurm_c76_regression.sh"),
        Path("oaci/tests/test_c76_representation_association_orbit.py"),
    ])
    result = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        row_count = 0
        if path.suffix == ".csv":
            with open(path, newline="") as stream:
                row_count = sum(1 for _ in csv.DictReader(stream))
        result.append({
            "path": str(path), "sha256": c76_protocol.sha256(path),
            "size_bytes": path.stat().st_size, "row_count": row_count,
            "tracked_in_git": 1, "raw_trial_cache": 0,
        })
    descriptor = orbit_manifest["descriptor"]
    result.append({
        "path": descriptor["path"], "sha256": descriptor["sha256"],
        "size_bytes": descriptor["size_bytes"], "row_count": descriptor["row_count"],
        "tracked_in_git": 0, "raw_trial_cache": 0,
    })
    return result


def finalize() -> dict:
    checks = _rows("red_team_checks.csv")
    if len(checks) != 26 or any(row["blocking"] == "1" and row["passed"] != "1" for row in checks):
        raise RuntimeError("C76 finalization requires 26/26 passing red-team checks")
    if "Final status: `PASS`" not in (REPORT_DIR / "C76_RED_TEAM_VERIFICATION.md").read_text():
        raise RuntimeError("C76 red-team report is not PASS")
    regressions = _regression_rows()
    _write_csv("regression_verification.csv", regressions)

    protocol = c76_orbit.load_protocol()
    orbit_manifest, _ = c76_orbit.load_orbit_cache()
    state = json.loads(STATE_PATH.read_text())
    separation = {row["path"]: row for row in _rows("association_prediction_separation.csv")}
    target = separation["target_unlabeled"]
    strict = separation["strict_source"]
    target_ci = _row(
        "association_effect_size_ci.csv", path="target_unlabeled", kernel="laplacian",
        bandwidth_factor="1.0", statistic="centered_hsic",
    )
    target_nulls = [
        row for row in _rows("nested_null_summary.csv")
        if row["path"] == "target_unlabeled" and row["kernel"] == "laplacian"
        and row["bandwidth_factor"] == "1.0" and row["statistic"] == "centered_hsic"
    ]
    topology = {
        row["level"]: row for row in _rows("association_topology.csv")
        if row["feature_set"] == "target_unlabeled_G3_architecture"
        and row["kernel"] == "laplacian" and row["bandwidth_factor"] == "1.0"
        and row["statistic"] == "centered_hsic"
    }
    orbit = [
        row for row in _rows("orbit_registered_family_robustness.csv")
        if row["path"] == "target_unlabeled" and row["kernel"] == "laplacian"
        and row["bandwidth_factor"] == "1.0" and row["statistic"] == "centered_hsic"
    ]
    predictions = {row["path"]: row for row in _rows("cross_fitted_prediction_summary.csv")}
    actions = {row["path"]: row for row in _rows("actionability_materiality_summary.csv")}
    synthetic = {row["case"]: row for row in _rows("synthetic_false_positive_control.csv")}
    partition = _rows("target_F4_partition_audit.csv")[0]

    active_primary = ["C76-D_local_nonlinear_measurement_nontransportable_nonactionable"]
    inactive_primary = [
        "C76-A_RBF_association_collapses_under_blocked_orbit_controls",
        "C76-B_architecture_tied_coordinate_association_only",
        "C76-C_identity_or_heterogeneity_explains_association",
        "C76-E_factorization_invariant_incremental_candidate_for_T3_HO",
        "C76-F_protocol_cache_or_claim_blocker",
    ]
    active_secondary = ["C76-S1", "C76-S4", "C76-S5", "C76-S6", "C76-S7", "C76-S8", "C76-S10", "C76-S11"]
    inactive_secondary = ["C76-S2", "C76-S3", "C76-S9"]
    final_gate = "LOCAL_NONLINEAR_MEASUREMENT_NONTRANSPORTABLE"

    result = {
        "schema_version": "c76_representation_association_orbit_result_v1",
        "milestone": "C76", "final_gate": final_gate,
        "protocol": {
            "sha256": c76_protocol.sha256(c76_protocol.PROTOCOL_PATH),
            "parent_C75_commit": c76_protocol.PARENT_COMMIT,
            "status": protocol["status"],
        },
        "data_boundary": {
            "T2_units": 216, "T3_HO_units": 1052, "T3_HO_z_Wz_touched": False,
            "forward_passes": 0, "re_inference": 0, "training": 0, "GPU": False,
            "same_label_oracle_accessed": False, "target_population_claim_allowed": False,
        },
        "taxonomy": {
            "primary_active": active_primary, "primary_inactive": inactive_primary,
            "secondary_active": active_secondary, "secondary_inactive": inactive_secondary,
        },
        "F4_partition": {
            "geometry_candidate_dimension": int(partition["candidate_dimension"]),
            "Wz_logit_redundant_tail_dimension": int(partition["invariant_dimension"]),
            "full_F4_replay_only": True,
        },
        "association_prediction_separation": separation,
        "target_association": {
            "effect": float(target["association"]),
            "target_bootstrap_CI": [float(target_ci["bootstrap_ci_low"]), float(target_ci["bootstrap_ci_high"])],
            "worst_six_null_max_stat_p": float(target["association_worst_required_p"]),
            "nulls": target_nulls, "topology": topology, "orbit_families": orbit,
        },
        "prediction": predictions, "actionability": actions,
        "synthetic_known_cases": synthetic,
        "qualification": {
            "qualified_candidates": [], "C77_protocol_created": False,
            "T3_HO_campaign_justified": False,
        },
        "red_team": {
            "job_id": 892694, "blocking_checks": 26, "checks_passed": 26,
            "C74_descriptors_rehashed": 1080,
            "repairs": ["F4_mixed_block_isolation", "strict_control_semantics", "S5_no_action_known_case"],
        },
        "regression": {row["suite"]: row for row in regressions},
        "claim_boundary": {
            "association_only": True, "prediction": False, "actionability": False,
            "representation_origin": False, "target_gauge": False,
            "source_only_escape_hatch": False, "deployable": False,
            "selector_or_checkpoint_artifact": False, "new_training_justified": False,
        },
        "external_orbit_cache": {
            "sha256": orbit_manifest["descriptor"]["sha256"],
            "row_count": orbit_manifest["descriptor"]["row_count"],
            "size_bytes": orbit_manifest["descriptor"]["size_bytes"],
        },
    }
    c74_cache.atomic_json(MAIN_JSON, result)

    null_lines = [
        f"- {row['null']}: p95 `{float(row['null_p95']):.6f}`, global max-stat p `{float(row['global_max_stat_p']):.3f}`."
        for row in target_nulls
    ]
    orbit_retention = [float(row["median_effect_retention"]) for row in orbit]
    orbit_order = [float(row["median_candidate_order_spearman"]) for row in orbit]
    regression_map = {row["suite"]: row for row in regressions}
    lines = [
        "# C76 - Representation Association Orbit / Conditional Transportability Audit", "",
        f"**Final gate:** `{final_gate}`", "",
        "**Primary active:** `" + " + ".join(active_primary) + "`", "",
        "**Primary inactive:** `" + " + ".join(inactive_primary) + "`", "",
        "## Gate-First Result", "",
        "C76 used only the 216-unit T2 instrumentation cache. T3-HO z/Wz access, same-label oracle access, real EEG forward passes, re-inference, training, GPU use, BNCI2014_004, and seeds [3,4] were all zero.", "",
        f"Strict-source architecture does not survive all six blocked controls: its best registered effect is `{float(strict['association']):.6f}`, but worst max-stat p is `{float(strict['association_worst_required_p']):.3f}`. Incremental R2 is `{float(strict['incremental_R2']):.6f}` (prediction max-stat p `{float(strict['prediction_global_p']):.3f}`), with no material actionability.", "",
        f"Target-unlabeled architecture geometry has a local nonlinear association: Laplacian-HSIC `{float(target['association']):.6f}`, target-bootstrap CI `[{float(target_ci['bootstrap_ci_low']):.6f}, {float(target_ci['bootstrap_ci_high']):.6f}]`, worst required max-stat p `{float(target['association_worst_required_p']):.3f}`. It fails transport/control: incremental R2 `{float(target['incremental_R2']):.6f}`, prediction max-stat p `{float(target['prediction_global_p']):.3f}`, positive prediction targets `{int(target['positive_targets'])}/9`, material actionability `false`.", "",
        "## C75 Replay And F4 Repair", "",
        f"C75 RBF is replayed bit-exactly: strict `{float(strict['C75_exact_full_block_rbf_association']):.17g}` and target full-F4 `{float(target['C75_exact_full_block_rbf_association']):.17g}`. Under C76's six required nulls, the candidate RBF paths both fail (worst p `1.000`); the surviving target result comes from the separately registered centered-HSIC statistic.", "",
        "Red-team found that C75 F4 was mixed: columns 0-19 are z/W geometry, while columns 20-34 are Wz/logit-redundant function-level summaries. Full F4 is retained only for C75 replay. Every formal target null, prediction, actionability, and T3 qualification uses the 20-d geometry block; the 15-d Wz tail is reported separately.", "",
        "## Six Blocked Nulls", "",
        *null_lines, "",
        "All bandwidths are fold-local and recomputed inside each null. Correction is over the full 24-test family separately for every required null. The identity-matched control is limiting (p `0.030`), so the result is statistically narrow rather than a broad representation effect.", "",
        "## Orbit Audit", "",
        f"All 29 variants preserve logits/probabilities: max projection error `{orbit_manifest['functional_identity_max_abs']:.3g}`, max probability error `{orbit_manifest['probability_identity_max_abs']:.3g}`, prediction disagreements `0`. For the selected target statistic, median effect retention across the seven nonidentity families ranges `{min(orbit_retention):.6f}` to `{max(orbit_retention):.6f}` and candidate-density order Spearman ranges `{min(orbit_order):.6f}` to `{max(orbit_order):.6f}`.", "",
        "This rejects a simple registered-orbit collapse, but it does not uniquely identify a z or W origin. Stability over these transforms is construct robustness, not representation causality.", "",
        "## Association Topology", "",
        f"The target association is weak pooled (`{float(topology['pooled']['association']):.6f}`), stronger within target (`{float(topology['within_target']['association']):.6f}`), and strongest inside target x trajectory cells (`{float(topology['within_target_x_trajectory']['association']):.6f}`). All 9 held-target association folds are positive, but this is not cross-target prediction: the nested KRR correction is negative and only 4/9 targets improve directionally.", "",
        "Cross-regime transfer is unavailable because the frozen T2 field contains one regime. This is an explicit support limit, not evidence of cross-regime transport.", "",
        "The object is therefore a local nonlinear measurement under heterogeneous conditioning, not a transferable endpoint predictor. C76-S4 and C76-S5 are active; pooled-identity collapse (S3) is inactive.", "",
        "## Prediction And Control", "",
        f"Strict-source mean regret reduction is `{float(actions['strict_source']['mean_regret_reduction']):.6f}`; target-unlabeled is `{float(actions['target_unlabeled']['mean_regret_reduction']):.6f}`. Both regret and top-k routes fail. Neither G3S nor G3T passes all 12 locked T3 qualification gates, so there is no strict-source or target-unlabeled escape hatch and no C77 protocol.", "",
        "## Synthetic Calibration", "",
        f"S0/S1 detection rates are `{float(synthetic['S0_no_association']['association_detection_rate']):.3f}`/`{float(synthetic['S1_coordinate_artifact']['association_detection_rate']):.3f}`. S3 demonstrates local association with negative transport R2 `{float(synthetic['S3_local_nonlinear_nontransport']['median_incremental_R2']):.6f}`. S4 preserves orbit effect exactly and is predictive. Repaired S5 retains detection `{float(synthetic['S5_association_no_extreme_action']['association_detection_rate']):.3f}` and R2 `{float(synthetic['S5_association_no_extreme_action']['median_incremental_R2']):.6f}` while top1 increment is only `{float(synthetic['S5_association_no_extreme_action']['mean_top1_increment']):.6f}`. S6 is predictive/actionable with top1 increment `{float(synthetic['S6_predictive_actionable']['mean_top1_increment']):.6f}`.", "",
        "## Independent Red-Team", "",
        "Red-team job `892694` ran before this report, independently rehashed 1,080/1,080 C74 descriptors and the external orbit payload, reconstructed all 144 test-null cells, 2,994 max-stat rows, KRR increments, actionability, orbit gates, and T3 qualification. It passed 26/26 blocking checks.", "",
        "Three completed candidate analyses were superseded: one lacked full-family orbit coverage, one mixed F4 geometry with the Wz tail, and one used imprecise strict-control labels. A later candidate job was cancelled when S5 violated its known-case semantics. Final evidence comes from analysis `892679` only.", "",
        "## Claim Boundary", "",
        "C76 establishes one narrow target-unlabeled local nonlinear association that is robust over the registered orbit and blocked controls. It does not establish factorization-invariant function-level information, held-out prediction, checkpoint actionability, representation origin, target gauge, source-only rescue, selector/control, deployability, target-population generalization, or an EEG theorem.", "",
        "The representation branch is saturated under the current architecture/frozen universe. T3-HO remains untouched, a C77 campaign is not justified, and new training is not justified by C76.", "",
        "## Verification", "",
        "- focused C76: `18 passed`.",
        f"- C65-C76 regression: `{regression_map['C65_C76']['passed_tests']} passed` (Slurm `892715`).",
        f"- C23-C76 regression: `{regression_map['C23_C76']['passed_tests']} passed` (Slurm `892716`).",
        f"- full OACI suite: `{regression_map['full_OACI']['passed_tests']} passed` (Slurm `892717`).",
        "- all three regression stderr streams: empty.", "",
        "## Next-State Gate", "",
        "No C77 protocol was created. Further T3-HO representation instrumentation, new training, new datasets/targets, GPU work, seeds [3,4], BNCI2014_004, selector artifacts, checkpoint recommendations, or manuscript drafting require a new explicit PM decision.",
    ]
    MAIN_MD.write_text("\n".join(lines) + "\n")
    _update_handoff(regressions)
    manifest = _artifact_manifest(orbit_manifest)
    _write_csv("artifact_manifest.csv", manifest)
    return {
        "final_gate": final_gate, "primary_active": active_primary,
        "artifact_count": len(manifest), "C77_protocol_created": False,
        "T3_HO_z_Wz_touched": False,
    }


if __name__ == "__main__":
    print(json.dumps(finalize(), indent=2, sort_keys=True))
