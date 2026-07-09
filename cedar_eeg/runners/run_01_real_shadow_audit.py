"""CEDAR_01 real frozen-latent shadow audit over CEDAR_01F features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from cedar_eeg.config import CEDAR_01_DROP_KS, DEFAULT_P0_THRESHOLDS, DEFAULT_PROBE, P0Thresholds, parse_drop_ks
from cedar_eeg.data.feature_handoff import validate_handoff_manifest
from cedar_eeg.data.feature_schema import sha256_file, stable_json_hash
from cedar_eeg.data.load_frozen_features import load_frozen_feature_npz
from cedar_eeg.eval.noninferiority import crossfit_task_metrics, fit_source_eval_target_metrics
from cedar_eeg.probes.crossfit_grouped import crossfit_conditional_domain_probe, make_folds
from cedar_eeg.red_team import RedTeamFailure, validate_p0_result
from cedar_eeg.surgery.latent_mask import (
    apply_diagonal_mask,
    candidate_drop_sets_by_k,
    effective_rank,
    latent_dimension_scores,
    mask_from_drop_dims,
    rank_latent_dimensions,
)
from cedar_eeg.surgery.selection import (
    SurgeryCandidate,
    decide_p0,
    score_candidate,
    select_best_accept,
    sort_candidate_records,
    source_side_rank_components,
    target_eval_warnings,
)


def _positive_adv(report: dict[str, object]) -> float:
    return max(0.0, float(report["advantage_mean"]))


def _encode(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    raw = np.asarray(values).astype(str)
    mapping = {v: i for i, v in enumerate(sorted(np.unique(raw)))}
    return np.array([mapping[v] for v in raw], dtype=np.int64), mapping


def _grouped_split_report(groups: np.ndarray, *, n_splits: int, seed: int) -> dict[str, Any]:
    folds = make_folds(len(groups), groups=groups, n_splits=n_splits, seed=seed)
    overlaps = []
    for fold_id, (tr, ev) in enumerate(folds):
        overlap = sorted(set(groups[tr].astype(str)) & set(groups[ev].astype(str)))
        overlaps.append({"fold_id": int(fold_id), "overlap_count": int(len(overlap))})
    return {
        "required": True,
        "groups_present": True,
        "n_groups": int(len(np.unique(groups.astype(str)))),
        "n_splits": int(n_splits),
        "n_folds": int(len(folds)),
        "split_policy": "group_disjoint_crossfit",
        "train_eval_group_overlap": overlaps,
        "overlap_free": bool(all(x["overlap_count"] == 0 for x in overlaps)),
    }


def _random_control_stats(
    z: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    groups: np.ndarray,
    *,
    n_classes: int,
    n_domains: int,
    base_leakage: float,
    k_drop: int,
    exclude_dims: tuple[int, ...],
    repeats: int,
    seed: int,
    probe: str,
    n_splits: int,
) -> dict[str, Any]:
    if base_leakage <= 1e-8:
        return {
            "matched_k": int(k_drop),
            "drop_frac": 0.0,
            "drop_abs": 0.0,
            "after_mean": 0.0,
            "after_values": [],
            "repeats": int(repeats),
            "exclude_selected_dims": tuple(int(x) for x in exclude_dims),
        }
    rng = np.random.default_rng(seed)
    excluded = set(int(x) for x in exclude_dims)
    pool = np.asarray([i for i in range(z.shape[1]) if i not in excluded], dtype=np.int64)
    if len(pool) < k_drop:
        return {
            "matched_k": int(k_drop),
            "drop_frac": 0.0,
            "drop_abs": 0.0,
            "after_mean": base_leakage,
            "after_values": [],
            "repeats": int(repeats),
            "exclude_selected_dims": tuple(int(x) for x in exclude_dims),
            "skipped": "insufficient_unselected_dimensions",
        }
    leakages = []
    for rep in range(max(1, repeats)):
        drop = rng.choice(pool, size=k_drop, replace=False)
        keep = mask_from_drop_dims(z.shape[1], drop)
        z_rand = apply_diagonal_mask(z, keep)
        leakages.append(
            _positive_adv(
                crossfit_conditional_domain_probe(
                    z_rand,
                    y,
                    d,
                    n_classes=n_classes,
                    n_domains=n_domains,
                    groups=groups,
                    n_splits=n_splits,
                    probe=probe,
                    seed=seed + 17 + rep,
                )
            )
        )
    after_mean = float(np.mean(leakages))
    drop_abs = float(base_leakage - after_mean)
    return {
        "matched_k": int(k_drop),
        "drop_frac": float(drop_abs / max(base_leakage, 1e-8)),
        "drop_abs": drop_abs,
        "after_mean": after_mean,
        "after_values": [float(x) for x in leakages],
        "repeats": int(repeats),
        "exclude_selected_dims": tuple(int(x) for x in exclude_dims),
    }


def _target_rows(bundle) -> dict[str, np.ndarray]:
    role = np.asarray(bundle.role).astype(str)
    keep = np.char.startswith(role, "target_")
    if not np.any(keep):
        raise ValueError("target diagnostics require target_* rows")
    return {
        "z": bundle.z[keep],
        "y": bundle.y[keep],
        "domain": bundle.domain[keep],
        "groups": bundle.groups[keep],
        "role": role[keep],
    }


def _run_artifact(path: str | Path, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    bundle = load_frozen_feature_npz(path)
    source = bundle.source_selection_view()
    target = _target_rows(bundle)
    z = np.asarray(source["z"], dtype=np.float64)
    y = np.asarray(source["y"]).astype(np.int64, copy=False)
    d, domain_mapping = _encode(np.asarray(source["domain"]))
    groups = np.asarray(source["groups"]).astype(str)
    n_classes = int(max(y.max(initial=0), np.asarray(target["y"]).astype(np.int64).max(initial=0))) + 1
    n_domains = int(len(domain_mapping))
    drop_ks = parse_drop_ks(args.drop_ks)
    thresholds = P0Thresholds(
        min_leakage_drop_frac=args.min_leakage_drop_frac,
        max_source_bacc_drop=args.max_source_bacc_drop,
        max_target_bacc_drop=args.max_target_bacc_drop,
        max_r3_delta=args.max_r3_delta,
        max_random_control_drop_frac=args.max_random_control_drop_frac,
        min_stability=args.min_stability,
    )

    grouped_split = _grouped_split_report(groups, n_splits=args.n_splits, seed=args.seed)
    base_leak_report = crossfit_conditional_domain_probe(
        z,
        y,
        d,
        n_classes=n_classes,
        n_domains=n_domains,
        groups=groups,
        n_splits=args.n_splits,
        probe=args.probe,
        seed=args.seed,
    )
    perm_report = crossfit_conditional_domain_probe(
        z,
        y,
        d,
        n_classes=n_classes,
        n_domains=n_domains,
        groups=groups,
        n_splits=args.n_splits,
        probe=args.probe,
        seed=args.seed,
        permutation=True,
    )
    base_leakage = _positive_adv(base_leak_report)
    base_source_metrics = crossfit_task_metrics(
        z,
        y,
        groups=groups,
        n_classes=n_classes,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    base_erank = effective_rank(z)
    ranked = rank_latent_dimensions(z, y, d)
    scores = latent_dimension_scores(z, y, d)
    drop_sets, skipped_drop_ks = candidate_drop_sets_by_k(ranked, drop_ks)

    candidates: list[dict[str, Any]] = []
    target_candidate_inputs: list[tuple[str, tuple[int, ...]]] = []
    for cand_id, (name, drop_dims) in enumerate(drop_sets):
        keep = mask_from_drop_dims(z.shape[1], drop_dims)
        z_masked = apply_diagonal_mask(z, keep)
        leak_report = crossfit_conditional_domain_probe(
            z_masked,
            y,
            d,
            n_classes=n_classes,
            n_domains=n_domains,
            groups=groups,
            n_splits=args.n_splits,
            probe=args.probe,
            seed=args.seed + 100 + cand_id,
        )
        source_metrics = crossfit_task_metrics(
            z_masked,
            y,
            groups=groups,
            n_classes=n_classes,
            n_splits=args.n_splits,
            seed=args.seed + 100 + cand_id,
        )
        perm_candidate_report = crossfit_conditional_domain_probe(
            z_masked,
            y,
            d,
            n_classes=n_classes,
            n_domains=n_domains,
            groups=groups,
            n_splits=args.n_splits,
            probe=args.probe,
            seed=args.seed + 500 + cand_id,
            permutation=True,
        )
        random_control = _random_control_stats(
            z,
            y,
            d,
            groups,
            n_classes=n_classes,
            n_domains=n_domains,
            base_leakage=base_leakage,
            k_drop=len(drop_dims),
            exclude_dims=tuple(int(x) for x in drop_dims),
            repeats=args.random_control_repeats,
            seed=args.seed + 1000 + cand_id,
            probe=args.probe,
            n_splits=args.n_splits,
        )
        after_erank = effective_rank(z_masked)
        cand = SurgeryCandidate(
            name=name,
            dropped_units=tuple(int(x) for x in drop_dims),
            leakage_before=base_leakage,
            leakage_after=_positive_adv(leak_report),
            source_bacc_before=float(base_source_metrics["bacc"]),
            source_bacc_after=float(source_metrics["bacc"]),
            target_bacc_before=None,
            target_bacc_after=None,
            r3_before=0.0,
            r3_after=0.0,
            stability=max(0.0, 1.0 - float(leak_report["advantage_std"])),
            random_control_drop_frac=float(random_control["drop_frac"]),
        )
        decision, reasons = decide_p0(cand, thresholds)
        cand_payload = cand.to_dict()
        source_bacc_drop = float(cand_payload["source_bacc_drop"])
        source_ce_delta = float(source_metrics["ce"]) - float(base_source_metrics["ce"])
        source_nll_delta = float(source_metrics["nll"]) - float(base_source_metrics["nll"])
        record = {
            "candidate": cand_payload,
            "decision": decision.value,
            "reasons": reasons,
            "target_eval_warnings": target_eval_warnings(cand, thresholds),
            "utility": score_candidate(cand),
            "leakage_report": leak_report,
            "permutation_null": perm_candidate_report,
            "random_control": random_control,
            "source_utility_delta": {
                "source_bacc_drop": source_bacc_drop,
                "source_ce_delta": source_ce_delta,
                "source_nll_delta": source_nll_delta,
                "r3_delta": cand_payload["r3_delta"],
                "r3_status": "not_available_for_feature_only_shadow",
                "source_bacc_before": base_source_metrics["bacc"],
                "source_bacc_after": source_metrics["bacc"],
                "source_ce_before": base_source_metrics["ce"],
                "source_ce_after": source_metrics["ce"],
                "source_nll_before": base_source_metrics["nll"],
                "source_nll_after": source_metrics["nll"],
            },
            "collapse_guard": {
                "effective_rank_before": base_erank,
                "effective_rank_after": after_erank,
                "effective_rank_ratio": float(after_erank / max(base_erank, 1e-8)),
                "passed": bool(after_erank >= 1.0 and after_erank / max(base_erank, 1e-8) >= 0.5),
            },
            "grouped_split": grouped_split,
        }
        record["rank_key"] = source_side_rank_components(record)
        candidates.append(record)
        target_candidate_inputs.append((name, tuple(int(x) for x in drop_dims)))

    candidates = sort_candidate_records(candidates)
    selected = select_best_accept(candidates)
    result = {
        "project": "CEDAR-EEG",
        "phase": "P0_frozen_latent",
        "audit_stage": "CEDAR_01_real_frozen_latent_shadow",
        "feature_npz": str(path),
        "feature_file_sha256": bundle.metadata["file_sha256"],
        "dataset": bundle.metadata.get("dataset"),
        "backbone": bundle.metadata.get("backbone"),
        "fold_id": bundle.metadata.get("fold_id"),
        "seed": bundle.metadata.get("seed"),
        "probe": args.probe,
        "n_splits": args.n_splits,
        "groups_present": True,
        "grouped_split": grouped_split,
        "candidate_universe": {
            "drop_ks_requested": tuple(int(k) for k in drop_ks),
            "skipped_drop_ks": skipped_drop_ks,
            "selection_policy": "fixed_drop_ks",
            "bottom_leakage_negative_control": False,
        },
        "target_label_role": "diagnostic_only",
        "mask_selection_regime": "source_only_shadow",
        "deployable": False,
        "mask_materialized": False,
        "thresholds": thresholds.__dict__,
        "baseline": {
            "leakage": base_leak_report,
            "permutation_null": perm_report,
            "source_metrics": base_source_metrics,
            "target_bacc_eval_only": None,
        },
        "candidates": candidates,
        "selected": selected,
        "shadow_selected_candidate": selected,
        "latent_dimension_scores": scores,
        "claim_boundary": (
            "Mask selection is source-side. Target metrics, when present, are evaluation-only. "
            "Leakage reduction is not a target-generalization guarantee."
        ),
    }

    target_base = fit_source_eval_target_metrics(
        z,
        y,
        np.asarray(target["z"], dtype=np.float64),
        np.asarray(target["y"]).astype(np.int64, copy=False),
        n_classes=n_classes,
        seed=args.seed,
    )
    target_records = []
    for cand_id, (name, drop_dims) in enumerate(target_candidate_inputs):
        keep = mask_from_drop_dims(z.shape[1], drop_dims)
        target_after = fit_source_eval_target_metrics(
            apply_diagonal_mask(z, keep),
            y,
            apply_diagonal_mask(np.asarray(target["z"], dtype=np.float64), keep),
            np.asarray(target["y"]).astype(np.int64, copy=False),
            n_classes=n_classes,
            seed=args.seed + 2000 + cand_id,
        )
        target_records.append(
            {
                "candidate_name": name,
                "dropped_units": tuple(int(x) for x in drop_dims),
                "target_metrics_before": target_base,
                "target_metrics_after": target_after,
                "target_bacc_drop": float(target_base["bacc"]) - float(target_after["bacc"]),
                "target_ce_delta": float(target_after["ce"]) - float(target_base["ce"]),
                "target_nll_delta": float(target_after["nll"]) - float(target_base["nll"]),
                "target_label_role": "diagnostic_only",
            }
        )
    target_diag = {
        "feature_npz": str(path),
        "feature_file_sha256": bundle.metadata["file_sha256"],
        "dataset": bundle.metadata.get("dataset"),
        "backbone": bundle.metadata.get("backbone"),
        "fold_id": bundle.metadata.get("fold_id"),
        "computed_after_source_selection": True,
        "target_label_role": "diagnostic_only",
        "selection_uses_target_metrics": False,
        "baseline": target_base,
        "candidates": target_records,
        "shadow_selected_candidate_name": selected["candidate"]["name"] if selected else None,
    }
    table_rows = [_candidate_table_row(result, rec) for rec in candidates]
    return result, target_diag, table_rows


def _candidate_table_row(result: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    cand = rec["candidate"]
    source = rec["source_utility_delta"]
    random_control = rec["random_control"]
    collapse = rec["collapse_guard"]
    return {
        "dataset": result["dataset"],
        "backbone": result["backbone"],
        "fold_id": result["fold_id"],
        "feature_npz": result["feature_npz"],
        "candidate_name": cand["name"],
        "dropped_units": ";".join(str(x) for x in cand["dropped_units"]),
        "decision": rec["decision"],
        "utility": rec["utility"],
        "leakage_before": cand["leakage_before"],
        "leakage_after": cand["leakage_after"],
        "leakage_drop": cand["leakage_drop"],
        "leakage_drop_frac": cand["leakage_drop_frac"],
        "random_drop_abs": random_control["drop_abs"],
        "random_drop_frac": random_control["drop_frac"],
        "source_bacc_before": cand["source_bacc_before"],
        "source_bacc_after": cand["source_bacc_after"],
        "source_bacc_drop": cand["source_bacc_drop"],
        "source_ce_delta": source["source_ce_delta"],
        "source_nll_delta": source["source_nll_delta"],
        "effective_rank_ratio": collapse["effective_rank_ratio"],
        "collapse_guard_passed": collapse["passed"],
        "permutation_null_advantage": rec["permutation_null"]["advantage_mean"],
        "grouped_overlap_free": rec["grouped_split"]["overlap_free"],
        "target_label_role": result["target_label_role"],
        "deployable": result["deployable"],
        "mask_materialized": result["mask_materialized"],
    }


def _write_json(payload: Any, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_candidate_table(rows: list[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _red_team(results: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for result in results:
        try:
            rt = validate_p0_result(result)
            records.append(
                {
                    "feature_npz": result["feature_npz"],
                    "dataset": result["dataset"],
                    "backbone": result["backbone"],
                    "fold_id": result["fold_id"],
                    **rt.to_dict(),
                }
            )
        except RedTeamFailure as exc:
            records.append(
                {
                    "feature_npz": result["feature_npz"],
                    "dataset": result["dataset"],
                    "backbone": result["backbone"],
                    "fold_id": result["fold_id"],
                    "passed": False,
                    "checks": [],
                    "warnings": [],
                    "failure": str(exc),
                }
            )
    warnings = [x for rec in records for x in rec.get("warnings", [])]
    failures = [rec for rec in records if not rec.get("passed")]
    return {
        "project": "CEDAR-EEG",
        "phase": "CEDAR_01_real_frozen_latent_shadow",
        "passed": bool(not failures),
        "red_team_zero_warnings": bool(not failures and not warnings),
        "n_records": len(records),
        "n_failures": len(failures),
        "n_warnings": len(warnings),
        "records": records,
    }


def _selected_pm_flags(result: dict[str, Any]) -> dict[str, Any]:
    selected = result.get("shadow_selected_candidate")
    if not selected:
        return {
            "has_selected_accept": False,
            "source_noninferiority": False,
            "ce_nll_not_materially_worse": False,
            "leakage_beats_random": False,
            "collapse_guard_passed": False,
            "grouped_probe_valid": result["grouped_split"]["overlap_free"],
            "permutation_null_low": abs(float(result["baseline"]["permutation_null"]["advantage_mean"])) <= 0.05,
        }
    cand = selected["candidate"]
    source = selected["source_utility_delta"]
    random_control = selected["random_control"]
    leakage_drop = float(cand["leakage_drop"])
    random_drop = max(0.0, float(random_control["drop_abs"]))
    return {
        "has_selected_accept": True,
        "source_noninferiority": float(cand["source_bacc_drop"]) <= 0.01,
        "ce_nll_not_materially_worse": (
            float(source["source_ce_delta"]) <= 0.02 and float(source["source_nll_delta"]) <= 0.02
        ),
        "leakage_beats_random": bool(leakage_drop >= 3.0 * random_drop or leakage_drop - random_drop >= 0.15),
        "collapse_guard_passed": bool(selected["collapse_guard"]["passed"]),
        "grouped_probe_valid": result["grouped_split"]["overlap_free"],
        "permutation_null_low": abs(float(result["baseline"]["permutation_null"]["advantage_mean"])) <= 0.05,
        "selected_candidate_name": cand["name"],
        "selected_dropped_units": cand["dropped_units"],
        "leakage_drop": leakage_drop,
        "random_drop_abs": random_drop,
        "source_bacc_drop": cand["source_bacc_drop"],
        "source_ce_delta": source["source_ce_delta"],
        "source_nll_delta": source["source_nll_delta"],
    }


def _source_decision_summary(results: list[dict[str, Any]], red_team: dict[str, Any]) -> dict[str, Any]:
    artifact_flags = []
    by_backbone: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        flags = {
            "feature_npz": result["feature_npz"],
            "dataset": result["dataset"],
            "backbone": result["backbone"],
            "fold_id": result["fold_id"],
            **_selected_pm_flags(result),
        }
        artifact_flags.append(flags)
        by_backbone.setdefault(str(result["backbone"]), []).append(flags)

    backbone_summary = {}
    for backbone, rows in sorted(by_backbone.items()):
        core_keys = (
            "has_selected_accept",
            "source_noninferiority",
            "ce_nll_not_materially_worse",
            "leakage_beats_random",
            "collapse_guard_passed",
            "grouped_probe_valid",
            "permutation_null_low",
        )
        backbone_summary[backbone] = {
            "n_folds": len(rows),
            "all_core_pass": bool(rows and all(all(row[k] for k in core_keys) for row in rows)),
            "accepted_folds": int(sum(bool(row["has_selected_accept"]) for row in rows)),
            "source_noninferior_folds": int(sum(bool(row["source_noninferiority"]) for row in rows)),
            "leakage_beats_random_folds": int(sum(bool(row["leakage_beats_random"]) for row in rows)),
            "collapse_guard_passed_folds": int(sum(bool(row["collapse_guard_passed"]) for row in rows)),
            "grouped_probe_valid_folds": int(sum(bool(row["grouped_probe_valid"]) for row in rows)),
            "permutation_null_low_folds": int(sum(bool(row["permutation_null_low"]) for row in rows)),
        }

    passing_backbones = [k for k, v in backbone_summary.items() if v["all_core_pass"]]
    if red_team.get("red_team_zero_warnings") and len(passing_backbones) == 2:
        gate_outcome = "PASS_WITH_R3_CAVEAT"
    elif red_team.get("red_team_zero_warnings") and len(passing_backbones) == 1:
        gate_outcome = "CONDITIONAL_PASS_SINGLE_BACKBONE_WITH_R3_CAVEAT"
    else:
        gate_outcome = "FAIL_OR_DIAGNOSTIC_ONLY"

    return {
        "project": "CEDAR-EEG",
        "phase": "CEDAR_01_real_frozen_latent_shadow",
        "source_only_shadow_selector": True,
        "target_label_role": "diagnostic_only",
        "deployable": False,
        "mask_materialized": False,
        "gate_outcome": gate_outcome,
        "red_team_zero_warnings": red_team.get("red_team_zero_warnings"),
        "r3_status": "not_available_for_feature_only_shadow",
        "r3_caveat": "P1 or method claims require an explicit R3 bridge before continuation.",
        "backbone_summary": backbone_summary,
        "artifact_flags": artifact_flags,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    handoff = validate_handoff_manifest(args.handoff_manifest)
    handoff_hash = sha256_file(args.handoff_manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = handoff["per_artifact_hashes"]

    results = []
    target_diagnostics = []
    table_rows = []
    for rec in artifacts:
        result, target_diag, rows = _run_artifact(rec["path"], args)
        result["feature_supply_handoff_hash"] = handoff_hash
        results.append(result)
        target_diagnostics.append(target_diag)
        table_rows.extend(rows)

    candidate_table = out_dir / "candidate_table.csv"
    _write_candidate_table(table_rows, candidate_table)
    candidate_hash = sha256_file(candidate_table)
    (out_dir / "candidate_table.hash").write_text(f"{candidate_hash}  candidate_table.csv\n")

    red_team = _red_team(results)
    source_summary = _source_decision_summary(results, red_team)
    target_payload = {
        "project": "CEDAR-EEG",
        "phase": "CEDAR_01_real_frozen_latent_shadow",
        "target_label_role": "diagnostic_only",
        "computed_after_source_selection": True,
        "selection_uses_target_metrics": False,
        "deployable": False,
        "records": target_diagnostics,
    }
    _write_json(red_team, out_dir / "red_team.json")
    _write_json(source_summary, out_dir / "source_decision_summary.json")
    _write_json(target_payload, out_dir / "target_diagnostics_DIAGNOSTIC_ONLY.json")
    (out_dir / "feature_supply_handoff_hash.txt").write_text(f"{handoff_hash}\n")

    run_manifest = {
        "project": "CEDAR-EEG",
        "phase": "CEDAR_01_real_frozen_latent_shadow",
        "source_only_shadow_selector": True,
        "target_label_role": "diagnostic_only",
        "deployable": False,
        "mask_materialized": False,
        "feature_supply_handoff_manifest": str(args.handoff_manifest),
        "feature_supply_handoff_hash": handoff_hash,
        "feature_supply_source_commit": handoff.get("source_commit"),
        "cluster_provenance": handoff.get("cluster_provenance"),
        "n_artifacts": len(artifacts),
        "candidate_table": str(candidate_table),
        "candidate_table_hash": candidate_hash,
        "red_team": str(out_dir / "red_team.json"),
        "source_decision_summary": str(out_dir / "source_decision_summary.json"),
        "target_diagnostics": str(out_dir / "target_diagnostics_DIAGNOSTIC_ONLY.json"),
        "artifact_results": results,
        "run_manifest_hash": "",
    }
    run_manifest["run_manifest_hash"] = stable_json_hash(run_manifest)
    _write_json(run_manifest, out_dir / "run_manifest.json")
    return run_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--handoff-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--probe", default=DEFAULT_PROBE.probe, choices=("linear", "mlp"))
    ap.add_argument("--n-splits", type=int, default=DEFAULT_PROBE.n_splits)
    ap.add_argument("--seed", type=int, default=DEFAULT_PROBE.seed)
    ap.add_argument("--drop-ks", default=",".join(str(k) for k in CEDAR_01_DROP_KS))
    ap.add_argument("--min-leakage-drop-frac", type=float, default=DEFAULT_P0_THRESHOLDS.min_leakage_drop_frac)
    ap.add_argument("--max-source-bacc-drop", type=float, default=DEFAULT_P0_THRESHOLDS.max_source_bacc_drop)
    ap.add_argument("--max-target-bacc-drop", type=float, default=DEFAULT_P0_THRESHOLDS.max_target_bacc_drop)
    ap.add_argument("--max-r3-delta", type=float, default=DEFAULT_P0_THRESHOLDS.max_r3_delta)
    ap.add_argument(
        "--max-random-control-drop-frac",
        type=float,
        default=DEFAULT_P0_THRESHOLDS.max_random_control_drop_frac,
    )
    ap.add_argument("--random-control-repeats", type=int, default=5)
    ap.add_argument("--min-stability", type=float, default=DEFAULT_P0_THRESHOLDS.min_stability)
    args = ap.parse_args()
    manifest = run(args)
    print(
        json.dumps(
            {
                "run_manifest": str(Path(args.out_dir) / "run_manifest.json"),
                "run_manifest_hash": manifest["run_manifest_hash"],
                "candidate_table_hash": manifest["candidate_table_hash"],
                "feature_supply_handoff_hash": manifest["feature_supply_handoff_hash"],
                "n_artifacts": manifest["n_artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
