#!/usr/bin/env python
"""Exact Phase-B1 forensic replay and variance-excluded core persistence.

The scientific implementation is loaded byte-for-byte from the frozen replay
commit. This wrapper only persists the already-computed frame state after the
pre-registered variance stability exception.
"""
import argparse
import csv
import hashlib
import json
import os
import stat
import subprocess
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FROZEN_COMMIT = "960104e77dd897c7e695924da2e3bff45968cb10"
FROZEN_SCRIPT = "s2p/scripts/route_b_phase_b1_decomposition.py"
ORIGINAL_FAILED_TAGS = ["H1000_s0", "released"]
COMPONENTS = ("subject_frac", "class_frac", "interaction_frac", "residual_frac")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows):
    path = Path(path)
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fobj:
        while True:
            chunk = fobj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_module(repo_root):
    source = subprocess.check_output(
        ["git", "show", f"{FROZEN_COMMIT}:{FROZEN_SCRIPT}"],
        cwd=repo_root,
    )
    module = types.ModuleType("s2p_phase_b1_frozen_replay")
    module.__file__ = f"git:{FROZEN_COMMIT}:{FROZEN_SCRIPT}"
    module.__dict__["__name__"] = module.__name__
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def find_run_locals(exc):
    trace = exc.__traceback__
    while trace is not None:
        local_keys = trace.tb_frame.f_locals.keys()
        if (
            "variance_rows" in local_keys
            and "unstable_tags" in local_keys
        ):
            return dict(trace.tb_frame.f_locals)
        trace = trace.tb_next
    raise RuntimeError("frozen B1 exception did not retain the run frame")


def traceback_frame_self_test():
    def frozen_run():
        variance_rows = ["sentinel"]
        unstable_tags = ["sentinel"]
        raise RuntimeError("sentinel")

    def run():
        try:
            frozen_run()
        except RuntimeError as exc:
            captured = find_run_locals(exc)
            if captured.get("variance_rows") != ["sentinel"]:
                raise RuntimeError("traceback frame selector chose the wrapper frame")

    run()


def ensure_hashes_still_match(hash_rows):
    for row in hash_rows:
        if row["tag"] == "random":
            if row["sha256_at_global_end"] != "logical_contract":
                raise RuntimeError("random logical hash contract changed")
            continue
        path = Path(row["immutable_path"])
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"immutable payload missing after replay: {row['tag']}")
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise RuntimeError(f"immutable payload became writable: {row['tag']}")
        if sha256_file(path) != row["expected_sha256"]:
            raise RuntimeError(f"immutable payload hash changed: {row['tag']}")


def persist_variance_forensic(out, values, frozen_source_sha):
    tags = values["TAGS"]
    variance_rows = values["variance_rows"]
    variance_samples = values["variance_samples"]
    feature_lookup = {row["tag"]: row for row in values["feature_rows"]}
    rows = []
    gate_components = {}
    replay_failed = []
    for tag in tags:
        folds = sorted(
            [row for row in variance_rows if row["tag"] == tag and row["scope"] == "fold"],
            key=lambda row: int(row["fold"]),
        )
        if len(folds) != 3:
            raise RuntimeError(f"forensic replay lacks three variance folds: {tag}")
        tag_details = {}
        tag_failed = False
        residual_min = min(float(row["residual_frac"]) for row in folds)
        for component in COMPONENTS:
            fold_values = np.asarray([float(row[component]) for row in folds])
            samples = np.asarray(variance_samples[tag][component], dtype=np.float64)
            max_deviation = float(np.max(np.abs(fold_values - fold_values.mean())))
            ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
            ci_width = float(ci_high - ci_low)
            fold_gate = max_deviation <= 0.10
            bootstrap_gate = ci_width <= 0.20
            residual_gate = residual_min >= -0.01 if component == "residual_frac" else True
            passed = fold_gate and bootstrap_gate and residual_gate
            tag_failed = tag_failed or not passed
            detail = {
                "fold_values": fold_values.tolist(),
                "fold_mean": float(fold_values.mean()),
                "max_fold_deviation": max_deviation,
                "max_fold_deviation_threshold": 0.10,
                "fold_stability_pass": fold_gate,
                "bootstrap_ci95_low": float(ci_low),
                "bootstrap_ci95_high": float(ci_high),
                "bootstrap_ci95_width": ci_width,
                "bootstrap_ci95_width_threshold": 0.20,
                "bootstrap_stability_pass": bootstrap_gate,
                "minimum_residual_fraction": residual_min,
                "minimum_residual_fraction_threshold": -0.01,
                "residual_stability_pass": residual_gate,
                "component_gate_pass": passed,
            }
            tag_details[component] = detail
            rows.append({
                "tag": tag,
                "component": component,
                "fold0": fold_values[0],
                "fold1": fold_values[1],
                "fold2": fold_values[2],
                **{key: value for key, value in detail.items() if key != "fold_values"},
                "rng_seed": values["BOOTSTRAP_SEED"],
                "bootstrap_reps": values["N_BOOTSTRAP"],
                "input_feature_sha256": feature_lookup[tag]["full_feature_sha256"],
                "frozen_source_commit": FROZEN_COMMIT,
            })
        if tag_failed:
            replay_failed.append(tag)
        gate_components[tag] = {
            "components": tag_details,
            "checkpoint_gate_pass": not tag_failed,
            "input_feature_sha256": feature_lookup[tag]["full_feature_sha256"],
        }
    replay_failed = sorted(replay_failed)
    original = sorted(ORIGINAL_FAILED_TAGS)
    match = replay_failed == original
    write_csv(out / "phase_b1_variance_forensic_replay.csv", rows)
    write_json(out / "phase_b1_variance_gate_components.json", {
        "phase": "B1_variance_forensic_replay",
        "frozen_source_commit": FROZEN_COMMIT,
        "frozen_source_sha256": frozen_source_sha,
        "bootstrap_seed": values["BOOTSTRAP_SEED"],
        "bootstrap_reps": values["N_BOOTSTRAP"],
        "thresholds_changed": False,
        "components": gate_components,
        "failed_checkpoint_tags": replay_failed,
    })
    write_json(out / "phase_b1_variance_reproducibility.json", {
        "phase": "B1_variance_forensic_reproducibility",
        "original_attempt_job_id": "893011",
        "original_failed_checkpoint_tags": original,
        "replay_failed_checkpoint_tags": replay_failed,
        "failed_tag_set_exact_match": match,
        "component_numeric_comparison_to_original_available": False,
        "status": "REPRODUCED_AT_TAG_GATE_LEVEL" if match else "NONDETERMINISTIC_UNRELIABLE",
        "variance_scientific_pass_restored": False,
    })
    write_json(out / "phase_b1_variance_family_disposition.json", {
        "phase": "B1_variance_family_disposition",
        "status": "FAILED_STABILITY_NOT_INTERPRETABLE",
        "all_checkpoints_excluded_from_variance_claims": True,
        "selective_survivor_reporting_allowed": False,
        "variance_used_in_primary_inference": False,
        "variance_used_in_mechanism_verdict": False,
        "interaction_claim_allowed": False,
        "forensic_use_only": True,
        "b1_core_may_continue_if_provenance_clean": True,
    })
    return replay_failed, match


def persist_core(out, values, frozen_source_sha, replay_match):
    tags = values["TAGS"]
    primary_rows = values["primary_rows"]
    sensitivity_rows = [
        row for row in values["sensitivity_rows"]
        if not row["analysis"].startswith("variance_")
    ]
    subject_uninformative = bool(values["primary_subject_uninformative"])
    subject_early = (
        None if subject_uninformative
        else bool(primary_rows[0]["reject_holm_0p05"] and values["p1_point"] > 0)
    )
    continuing_subject = (
        None if subject_uninformative
        else bool(
            sensitivity_rows[0]["ci95_low"] > 0
            and sensitivity_rows[0]["point"] > 0
        )
    )
    task_later = bool(primary_rows[1]["reject_holm_0p05"] and values["p2_point"] > 0)
    overlap_significant = bool(primary_rows[2]["reject_holm_0p05"])
    overlap_point = float(values["p3_point"])
    overlap_trend = (
        "increase" if overlap_significant and overlap_point > 0
        else "decrease" if overlap_significant and overlap_point < 0
        else "no_detectable_change"
    )
    subspaces_stable = bool(values["all_subspaces_stable"])
    if subject_early is not True or not task_later or not subspaces_stable:
        mechanism = "D"
    elif overlap_trend == "increase":
        mechanism = "C"
    elif continuing_subject:
        mechanism = "B"
    else:
        mechanism = "A"
    mechanism_labels = {
        "A": "SUBJECT_EARLY_TASK_LATER_LOW_OVERLAP",
        "B": "SUBJECT_CONTINUES_TASK_LATER_LOW_OVERLAP",
        "C": "TASK_LATER_WITH_INCREASING_SUBJECT_TASK_OVERLAP",
        "D": "CORE_GEOMETRY_UNRESOLVED",
    }

    run_manifest = {
        "phase": "B1_core_subject_task_geometry",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "launch_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "frozen_scientific_source_commit": FROZEN_COMMIT,
        "frozen_scientific_source_sha256": frozen_source_sha,
        "checkpoint_objects": ";".join(tags),
        "pca_dim": values["PCA_DIM"],
        "probe_C": values["PROBE_C"],
        "subspace_rank": values["SUBSPACE_RANK"],
        "bootstrap_reps": values["N_BOOTSTRAP"],
        "bootstrap_seed": values["BOOTSTRAP_SEED"],
        "variance_partition_status": "FAILED_STABILITY_NOT_INTERPRETABLE",
        "variance_used_in_core": False,
        "target_labels_used_for_selection": False,
        "phase_b2_authorized": False,
        "new_training": False,
        "fine_tuning": False,
        "h4000": False,
        "codebrain": False,
    }
    write_csv(out / "phase_b1_core_run_manifest.csv", [run_manifest])
    write_csv(out / "phase_b1_core_checkpoint_hash_recheck.csv", values["hash_rows"])
    write_csv(out / "phase_b1_core_feature_manifest.csv", values["feature_rows"])
    write_csv(out / "phase_b1_core_subject_metrics.csv", values["subject_metric_rows"])
    write_csv(out / "phase_b1_core_subject_cluster_scores.csv", values["subject_cluster_rows"])
    write_csv(out / "phase_b1_core_subject_pair_metrics.csv", values["subject_pair_rows"])
    write_csv(out / "phase_b1_core_task_metrics.csv", values["task_metric_rows"])
    write_csv(out / "phase_b1_core_task_sample_scores.csv", values["task_sample_rows"])
    write_csv(out / "phase_b1_core_geometry.csv", values["geometry_rows"])
    write_csv(out / "phase_b1_core_sensitivity.csv", sensitivity_rows)
    write_csv(out / "phase_b1_core_primary_bootstrap_samples.csv", [
        {
            "replicate": index,
            "p1_subject_delta": float(values["p1_samples"][index]),
            "p2_task_nll_delta": float(values["p2_samples"][index]),
            "p3_overlap_delta": float(values["p3_samples"][index]),
        }
        for index in range(values["N_BOOTSTRAP"])
    ])
    write_json(out / "phase_b1_core_primary_inference.json", {
        "phase": "B1_core_primary_inference",
        "primary_subject_metric": values["primary_subject_metric"],
        "primary_subject_metric_uninformative": subject_uninformative,
        "subject_metric_saturation_reasons": values["saturation_reasons"],
        "fallback_definition_source": (
            f"pre_run_code_commit_{FROZEN_COMMIT}" if subject_uninformative else "not_needed"
        ),
        "primary_task_metric": "target_task_nll",
        "primary_overlap_metric": "rank8_normalized_projection_overlap",
        "bootstrap_reps": values["N_BOOTSTRAP"],
        "bootstrap_seed": values["BOOTSTRAP_SEED"],
        "holm_family_size": 3,
        "contrasts": primary_rows,
        "variance_in_primary_family": False,
        "target_labels_used_for_selection": False,
    })
    firewall = {
        "target_labels_used_for_pca": False,
        "target_labels_used_for_probe_fit": False,
        "target_labels_used_for_probe_selection": False,
        "target_labels_used_for_rank_selection": False,
        "target_labels_used_for_metric_selection": False,
        "target_labels_used_for_checkpoint_selection": False,
        "target_labels_used_for_final_task_scoring": True,
        "target_labels_used_for_target_subject_cluster_bootstrap_scoring": True,
        "target_labels_used_for_selection": False,
    }
    write_json(out / "phase_b1_core_target_label_firewall.json", firewall)
    verdict = {
        "phase": "B1_core_subject_task_geometry",
        "status": "PASS",
        "independent_verification_status": "PENDING",
        "variance_partition_status": "FAILED_STABILITY_NOT_INTERPRETABLE",
        "variance_forensic_failure_set_reproduced": replay_match,
        "variance_used_in_mechanism_verdict": False,
        "checkpoint_objects": 10,
        "all_hashes_reverified": True,
        "all_features_match_closure": True,
        "clip_group_crossfit_pass": True,
        "primary_subject_metric": values["primary_subject_metric"],
        "primary_subject_metric_uninformative": subject_uninformative,
        "subject_early_contrast": primary_rows[0],
        "subject_structure_early": subject_early,
        "subject_continues_after_h200": continuing_subject,
        "primary_task_metric": "target_task_nll",
        "task_later_contrast": primary_rows[1],
        "task_structure_later": task_later,
        "primary_overlap_metric": "rank8_normalized_projection_overlap",
        "overlap_higher_minus_h200": primary_rows[2],
        "subject_task_overlap_trend": overlap_trend,
        "all_subject_task_subspaces_stable": subspaces_stable,
        "mechanism_verdict": mechanism,
        "mechanism_label": mechanism_labels[mechanism],
        "interaction_claim_allowed": False,
        "geometry_claim_scope": "measured_rank8_linear_effect_subspaces_only",
        "target_labels_used_for_selection": False,
        "recommend_phase_b2_layerwise": mechanism in ("A", "B"),
        "phase_b2_authorized": False,
    }
    write_json(out / "phase_b1_core_verdict.json", verdict)

    support = {
        "tags": np.asarray(tags),
        "source_counts": values["source_counts"].astype(np.int16),
        "target_counts": values["target_counts"].astype(np.int16),
        "subject_nll_samples": np.stack([values["subject_nll_samples"][tag] for tag in tags]),
        "subject_margin_samples": np.stack([values["subject_margin_samples"][tag] for tag in tags]),
        "subject_auc_samples": np.stack([values["subject_auc_samples"][tag] for tag in tags]),
        "target_nll_samples": np.stack([values["target_nll_samples"][tag] for tag in tags]),
        "target_kappa_samples": np.stack([values["target_kappa_samples"][tag] for tag in tags]),
        "target_bacc_samples": np.stack([values["target_bacc_samples"][tag] for tag in tags]),
        "geometry_overlap_samples": np.stack([values["geometry_samples"][tag] for tag in tags]),
    }
    for tag in tags:
        for fold in range(3):
            support[f"geometry_cell__{tag}__fold{fold}"] = values["geometry_cells"][tag][fold]
            support[f"geometry_hold_cell__{tag}__fold{fold}"] = values["geometry_hold_cells"][tag][fold]
    np.savez_compressed(out / "phase_b1_core_support.npz", **support)
    return verdict


def run(args):
    if not args.pm_authorized:
        raise RuntimeError("B1-Core replay requires explicit PM authorization")
    repo_root = Path(args.repo_root).resolve()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frozen, frozen_source_sha = load_frozen_module(repo_root)
    frozen.run_self_tests()
    frozen_args = argparse.Namespace(
        checkpoint_manifest=args.checkpoint_manifest,
        closure_json=args.closure_json,
        provenance_redteam=args.provenance_redteam,
        b0_go_nogo=args.b0_go_nogo,
        copy_verification_json=args.copy_verification_json,
        protocol_doc=args.protocol_doc,
        redteam_doc=args.redteam_doc,
        faced_lmdb=args.faced_lmdb,
        out_dir=str(out / "forensic_scratch"),
        device=args.device,
        batch_size=args.batch_size,
        pm_authorized=True,
        self_test=False,
        checksum_canary=False,
    )
    try:
        frozen.run(frozen_args)
    except RuntimeError as exc:
        if not str(exc).startswith("variance partition failed the frozen stability gate:"):
            raise
        frame = find_run_locals(exc)
    else:
        raise RuntimeError("forensic replay unexpectedly passed the frozen variance gate")

    required_locals = [
        "hash_rows", "feature_rows", "subject_metric_rows", "subject_cluster_rows",
        "subject_pair_rows", "task_metric_rows", "task_sample_rows", "geometry_rows",
        "variance_rows", "variance_samples", "source_counts", "target_counts",
        "subject_nll_samples", "subject_margin_samples", "subject_auc_samples",
        "target_nll_samples", "target_kappa_samples", "target_bacc_samples",
        "geometry_samples", "geometry_cells", "geometry_hold_cells", "primary_rows",
        "sensitivity_rows", "p1_samples", "p2_samples", "p3_samples",
    ]
    missing = [key for key in required_locals if key not in frame]
    if missing:
        raise RuntimeError(f"forensic traceback lacks required frozen state: {missing}")
    for key in (
        "TAGS", "BOOTSTRAP_SEED", "N_BOOTSTRAP", "PCA_DIM", "PROBE_C", "SUBSPACE_RANK"
    ):
        frame[key] = getattr(frozen, key)
    ensure_hashes_still_match(frame["hash_rows"])
    replay_failed, replay_match = persist_variance_forensic(out, frame, frozen_source_sha)
    if replay_failed != sorted(frame["unstable_tags"]):
        raise RuntimeError("forensic component reconstruction disagrees with frozen gate")
    verdict = persist_core(out, frame, frozen_source_sha, replay_match)
    print(json.dumps({
        "variance_failed_tags": replay_failed,
        "variance_replay_match": replay_match,
        "core_status": verdict["status"],
        "core_mechanism_verdict": verdict["mechanism_verdict"],
    }, indent=2, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", default="results/s2p_route_b_representation_emergence_b1_core")
    parser.add_argument("--checkpoint-manifest", default="results/s2p_route_b_phase_b_checkpoint_closure/phase_b_checkpoint_immutable_manifest.csv")
    parser.add_argument("--closure-json", default="results/s2p_route_b_phase_b_checkpoint_closure/phase_b_provenance_closure.json")
    parser.add_argument("--provenance-redteam", default="results/s2p_route_b_phase_b_checkpoint_closure/phase_b_provenance_redteam_verdict.json")
    parser.add_argument("--b0-go-nogo", default="results/s2p_route_b_representation_emergence_b0/phase_b0_go_nogo.json")
    parser.add_argument("--copy-verification-json", default="results/s2p_route_b_phase_b_checkpoint_closure/phase_b_checkpoint_copy_verification.json")
    parser.add_argument("--protocol-doc", default="docs/S2P_19_REPRESENTATION_EMERGENCE_PROTOCOL.md")
    parser.add_argument("--redteam-doc", default="docs/S2P_20_REPRESENTATION_EMERGENCE_REDTEAM.md")
    parser.add_argument("--faced-lmdb", default="/projects/EEG-foundation-model/FACED_data/processed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--pm-authorized", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        module, source_sha = load_frozen_module(Path(args.repo_root).resolve())
        module.run_self_tests()
        traceback_frame_self_test()
        print(f"Frozen source replay self-test: PASS commit={FROZEN_COMMIT} sha256={source_sha}")
        return
    run(args)


if __name__ == "__main__":
    main()
