"""Build the CPU-only P12 preregistration packet from frozen P9 artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from h2cmi import run_fp_gem as runner
from h2cmi import analyze_fp_gem as analyzer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "h2cmi/results/fp_gem_main"
SOURCE_INDEX = OUT / "p9_source_checkpoint_index.json"
UNITS = OUT / "fp_gem_units.csv"
METHOD_FREEZE = OUT / "FP_GEM_METHOD_FREEZE.md"
INTEGRATION_AUDIT = OUT / "fp_gem_integration_audit.md"
COMMAND_LOG = OUT / "COMMAND_LOG.md"
AMENDMENT_MD = OUT / "FP_GEM_SOURCE_PROVENANCE_AMENDMENT.md"
AMENDMENT_JSON = OUT / "FP_GEM_SOURCE_PROVENANCE_AMENDMENT.json"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_units(config, source_hashes, manifests):
    rows = []
    group_counts = Counter()
    for dataset in runner.SELECTED_DATASETS:
        targets = sorted(key[1] for key in manifests if key[0] == dataset and key[2] == 0)
        for target in targets:
            for seed in runner.SELECTED_SEEDS:
                key = (dataset, target, seed)
                hardware = "A100" if dataset == "Lee2019_MI" and seed == 2 and target >= 27 else "V100"
                rows.append({
                    "unit_index": len(rows),
                    "hardware_group": hardware,
                    "hardware_group_index": group_counts[hardware],
                    "dataset": dataset,
                    "target_subject": target,
                    "source_seed": seed,
                    "p9_reference_source_model_sha256": source_hashes[key],
                    "split_hash": manifests[key]["split_hash"],
                    "n_adapt": manifests[key]["n_adapt"],
                    "n_eval": manifests[key]["n_eval"],
                })
                group_counts[hardware] += 1
    with UNITS.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows, dict(group_counts)


def write_method_freeze(config, runner_sha, analyzer_sha, source_index_sha, units_sha, dryrun):
    density = config["source_density"]
    gem = config["geometry_em"]
    lines = [
        "# Fixed-Prior Geometry EM Method Freeze",
        "",
        "Status: **FROZEN BEFORE TARGET PERFORMANCE OBSERVATION; SOURCE-PROVENANCE AMENDMENT RECORDED BEFORE A REPLACEMENT SMOKE**.",
        "",
        "The paper method name is **Fixed-Prior Geometry EM (FP-GEM)**. The only ablation is **Joint-GEM**. No other method name or method is part of P12.",
        "",
        "## Frozen Question",
        "",
        "On the exact official P9 TSMNet backbone, classifier, repaired split, source seeds, and source-training configuration, does the existing H2CMI fixed-prior iterative diagonal geometry operator improve balanced accuracy relative to the same-pipeline source-only, RCT, official SPDIM, and Joint-GEM comparators?",
        "",
        "## TSMNet Integration",
        "",
        "- Base before GEM: the RCT-refitted TSMNet used before P9 SPDIM geodesic/bias adaptation.",
        "- Exact feature hook: `TSMNet.classifier.register_forward_pre_hook`.",
        "- Captured value: the sole tensor input to the final classifier, after `logeig` and the model's device/dtype conversion.",
        f"- Frozen feature dimension: `{config['feature_space']['dimension']}`.",
        "- Semantic gate: the normal TSMNet forward logits must equal `classifier(captured_feature)` to maximum absolute error `<=1e-7`.",
        "- The transformed feature is passed through the unchanged `TSMNet.classifier`; no replacement decoder is allowed.",
        "- TSMNet parameters and classifier are hash-checked before and after RCT/GEM; RCT may update only its intended domain-statistic buffers before the model is frozen for GEM.",
        "",
        "## Source Density",
        "",
        f"- Family: {density['family']}.",
        f"- Implementation: `{density['implementation']}`.",
        f"- Components/class: `{density['n_components']}`; covariance rank: `{density['cov_rank']}`; df: `{density['degrees_of_freedom']}`; variance floor: `{density['eigenvalue_floor']}`.",
        "- Fit data: exact P9 source-training-split pre-classifier features and source labels only. P9 source validation rows and all target labels are excluded.",
        f"- Optimizer: `{density['optimizer']}`, lr `{density['learning_rate']}`, weight decay `{density['weight_decay']}`, batch size `{density['batch_size']}`, cosine schedule, gradient clip `{density['gradient_clip']}`.",
        f"- Stopping rule: {density['stopping_rule']}.",
        "",
        "## Geometry EM",
        "",
        "```text",
        "T(z) = diag(exp(a)) z + b",
        "r_iy proportional to pi_fit[y] * p_y(T(z_i))",
        "```",
        "",
        f"- Initialization: {gem['initialization']}.",
        f"- Source prior: {gem['source_prior_definition']}.",
        f"- Iterations: `{gem['outer_iterations']}` responsibility/geometry rounds with `{gem['transform_steps_per_iteration']}` transform-gradient steps each.",
        f"- Transform optimizer: `{gem['optimizer']}`, lr `{gem['learning_rate']}`.",
        f"- Regularization: logdet `{gem['logdet_weight']}`, scale trust `{gem['trust_region_a']}`, shift trust `{gem['trust_region_b']}`.",
        f"- Stopping rule: {gem['stopping_rule']}.",
        "- FP-GEM pins `pi_fit` to the source empirical prior and has no target-prior M-step.",
        "- Joint-GEM is identical except for the responsibility prior M-step with Dirichlet/source anchoring.",
        "- Neither fit prior is injected into classifier logits. The only classifier input change is `T(z)`.",
        "",
        "## Frozen Scope And Statistics",
        "",
        "- Datasets: `BNCI2014_001`, `Lee2019_MI` only.",
        "- Source seeds: `0,1,2`; all 63 target subjects; 189 target-seed units.",
        "- New methods: Joint-GEM and FP-GEM only, 378 rows.",
        "- Same-checkpoint controls: 756 rows for source-only TSMNet, RCT, SPDIM geodesic, and SPDIM bias, rerun without tuning from each unit's reproduced source checkpoint.",
        "- Direct P9 row reuse: 0. P9 checkpoint weights were not persisted, so its 756 selected rows are provenance references rather than exact-checkpoint inputs.",
        "- Final same-backbone table: 1,134 rows, six methods.",
        "- Average seeds within dataset x target subject x method first.",
        "- Cluster bootstrap: 10,000 replicates, seed 20260710, dataset-stratified dataset x target-subject clusters, paired methods preserved.",
        "- Report bAcc and accuracy per dataset, subject-weighted, and dataset-macro; primary inference is the five frozen FP-GEM paired contrasts.",
        "",
        "## Frozen Smoke Gate",
        "",
        "One V100 unit only: `BNCI2014_001`, target `1`, source seed `0`. It reports no accuracy or bAcc. It must reproduce the exact P9 source-training configuration, persist and hash the resulting checkpoint, prove all six methods share that checkpoint, validate shape/hook/numerics/leakage, and leave all frozen scientific settings unchanged.",
        "",
        "## Source Checkpoint Policy",
        "",
        "P9 did not persist checkpoint files. Byte equality to a hash without recoverable weights is not a valid retraining gate: the first clean V100 retrain matched P9's logged trajectory through epoch 10 and differed by only 2e-4 in the printed final loss, while its full-state SHA differed. P12 therefore follows the user-approved fallback: reproduce the exact P9 training configuration on the recorded GPU family, persist the actual state, and run all six methods from that one state. The P9 state hash remains recorded as a reference and is never presented as the loaded checkpoint.",
        "",
        "## Precommitted Interpretation Grid",
        "",
        "| observed paired bAcc result | permitted interpretation | prohibited interpretation |",
        "|---|---|---|",
        "| FP-GEM minus all four non-GEM comparators has CI lower bounds above zero | evidence that FP-GEM improves this frozen same-backbone two-dataset pipeline | broad benchmark superiority or a third-dataset claim |",
        "| FP-GEM minus Joint-GEM has CI lower bound above zero, but one or more non-GEM contrasts span/include zero | evidence against the prior M-step in this pipeline, without evidence of overall baseline superiority | claim that FP-GEM beats official SPDIM generally |",
        "| FP-GEM contrasts span zero | no detected improvement under P12; report estimates/CIs | equivalence or noninferiority |",
        "| FP-GEM contrast CI upper bound is below zero | FP-GEM is worse for that frozen estimand | post-result tuning, renaming, or dataset removal |",
        "| heterogeneous dataset signs | report both dataset cells and both aggregate estimands | concealment by a single pooled headline |",
        "",
        "Smoke performance, target labels, and target performance may not alter this grid, configuration, method list, or dataset pair.",
        "",
        "## Frozen Provenance",
        "",
        f"- runner SHA-256: `{runner_sha}`",
        f"- analyzer SHA-256: `{analyzer_sha}`",
        f"- config SHA-256: `{runner.FROZEN_CONFIG_SHA256}`",
        f"- source checkpoint hash index SHA-256: `{source_index_sha}`",
        f"- execution unit manifest SHA-256: `{units_sha}`",
        f"- repaired manifest semantic hash: `{runner.MANIFEST_HASH}`",
        f"- P9 result SHA-256: `{runner.P9_RESULTS_SHA256}`",
        f"- P9 runner SHA-256: `{runner.P9_RUNNER_SHA256}`",
        f"- P9 config SHA-256: `{runner.P9_CONFIG_SHA256}`",
        f"- external SPDIM commit: `{runner.OFFICIAL_SHA}`",
        f"- CPU dry-run: `{'PASS' if dryrun['dryrun_pass'] else 'BLOCKED'}`",
    ]
    METHOD_FREEZE.write_text("\n".join(lines) + "\n")


def write_integration_audit(config, dryrun, source_index_sha, units_sha):
    lines = [
        "# FP-GEM Integration Audit",
        "",
        "- phase: `P12A_SOURCE_PROVENANCE_AMENDMENT_PRECOMPUTE`",
        "- CPU dry-run status: `PASS`",
        "- GPU smoke status: `PENDING`",
        "- approve single smoke: `true`",
        "- approve P12B fleet before smoke: `false`",
        "",
        "## Static And Artifact Gates",
        "",
        f"- selected datasets: `{dryrun['datasets']}`",
        f"- selected source seeds: `{dryrun['source_seeds']}`",
        f"- target-seed units: `{dryrun['unit_count']}`",
        f"- expected new-method rows: `{dryrun['new_method_rows_expected']}`",
        f"- expected same-checkpoint control rows: `{dryrun['within_unit_control_rows_expected']}`",
        f"- expected reused P9 rows: `{dryrun['reused_p9_rows_expected']}`",
        f"- P9 reference rows: `{dryrun['p9_reference_rows_expected']}`",
        f"- expected final rows: `{dryrun['final_rows_expected']}`",
        f"- exact unit keys match repaired manifest: `{dryrun['unit_keys_match_manifest']}`",
        f"- exact unit keys match P9 source hashes: `{dryrun['unit_keys_match_p9']}`",
        f"- every unit reference hash matches its P9 row: `{dryrun['unit_reference_hashes_match_p9']}`",
        f"- every unit split hash/count matches the repaired manifest: `{dryrun['unit_split_fields_match_manifest']}`",
        f"- every unit hardware group matches the frozen P9-family mapping: `{dryrun['unit_hardware_groups_match_freeze']}`",
        f"- all adaptation/evaluation IDs disjoint: `{dryrun['all_adapt_eval_disjoint']}`",
        f"- all adaptation splits have both classes in the frozen manifest: `{dryrun['all_adapt_both_classes']}`",
        f"- all evaluation splits have both classes: `{dryrun['all_eval_both_classes']}`",
        f"- target-label leakage detected: `{dryrun['target_label_leakage_detected']}`",
        f"- target-performance selection detected: `{dryrun['target_performance_selection_detected']}`",
        f"- actual CPU feature-hook probe: `{dryrun['feature_hook_cpu_probe']}`",
        f"- V100 reproduction units: `{dryrun['hardware_group_counts']['V100']}`",
        f"- A100 reproduction units: `{dryrun['hardware_group_counts']['A100']}`",
        f"- source checkpoint index SHA-256: `{source_index_sha}`",
        f"- execution unit manifest SHA-256: `{units_sha}`",
        "",
        "## Feature Hook Gate",
        "",
        "Official TSMNet computes `logeig -> dtype/device conversion -> classifier`. The runner registers a forward pre-hook on `TSMNet.classifier`, so it captures the exact classifier input rather than reconstructing or replacing the decoder. The smoke must prove direct classifier replay within 1e-7 and a 210-dimensional feature.",
        "",
        "## Leakage Boundary",
        "",
        "Source labels are used only for exact P9 source training and the post-hoc source class-conditional density. Adaptation receives target X/features and a dummy-zero label tensor only for the official RCT API. Evaluation labels are first read after both GEM fits complete. The smoke does not read evaluation labels or compute performance.",
        "",
        "## Checkpoint Availability",
        "",
        "No P9 TSMNet checkpoint files were found or recorded. The committed P9 `source_model_sha256` column supplies a provenance reference per dataset x target x seed, but not recoverable weights. Every P12 unit must persist one exact-config retrain and run all six methods from that actual hashed checkpoint. Direct P9 row reuse is prohibited when the actual state hash differs.",
    ]
    INTEGRATION_AUDIT.write_text("\n".join(lines) + "\n")


def write_source_provenance_amendment() -> None:
    payload = {
        "status": "frozen_before_replacement_smoke",
        "target_performance_observed": False,
        "p9_checkpoint_files_found": False,
        "p9_checkpoint_weights_recoverable": False,
        "scientific_method_changed": False,
        "dataset_seed_split_hyperparameters_changed": False,
        "source_reproduction_policy": "exact_p9_configuration_retrain",
        "direct_p9_result_rows_reused": 0,
        "within_unit_controls_rerun": list(runner.P9_METHODS),
        "new_methods": list(runner.NEW_METHODS),
        "failed_pre_amendment_launches": [
            {
                "job_id": "893415",
                "classification": "zero_result_infrastructure_failure",
                "reason": "compute node could not access the /tmp launch worktree",
                "stdout_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "stderr_sha256": "8913d578a2b0a2a1c753c7a21370cc4501ff5b46c80dea63719e288ace9c56a1",
                "target_performance_observed": False,
                "accepted_rows": 0,
            },
            {
                "job_id": "893416",
                "classification": "zero_result_overstrict_provenance_gate_failure",
                "reason": "exact-config V100 retrain produced a different byte-level state hash from unrecoverable P9 weights",
                "p9_reference_source_model_sha256": "f21981a86a61ca0c5129c642a5ecaee301fff0a98466a3fa09d7f89c719b3c43",
                "retrained_source_model_sha256": "623fd92a6204d2a106e98d4189b5ca86a335e82ea47a2d91c6aef0cb7b6c1d17",
                "p9_epoch_0_10_trace_matches_printed_precision": True,
                "p9_epoch_19_printed_loss_absolute_difference": 0.0002,
                "stdout_sha256": "5e37e1285a1dc3bf5d5e889ef244de5ca28c0aac15774aee0f1c0714a36d361f",
                "stderr_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "gem_executed": False,
                "evaluation_labels_accessed": False,
                "target_performance_observed": False,
                "accepted_rows": 0,
            },
        ],
        "rationale": (
            "P9 explicitly allowed exact source-training configuration reproduction when checkpoints were unavailable. "
            "Because no P9 checkpoint file exists, direct P9 result reuse would not be an exact-checkpoint head-to-head. "
            "The amended design reruns the four frozen controls and both GEM methods from one persisted source state per unit."
        ),
    }
    write_json(AMENDMENT_JSON, payload)
    AMENDMENT_MD.write_text(
        "# FP-GEM Source-Provenance Amendment\n\n"
        "Status: **FROZEN BEFORE REPLACEMENT SMOKE; NO TARGET PERFORMANCE OBSERVED**.\n\n"
        "## Trigger\n\n"
        "P9 persisted source-state hashes but no TSMNet checkpoint files. Job `893416` reproduced the exact P9 "
        "BNCI2014-001 target-1 seed-0 training configuration on a Tesla V100. Its epoch 0 and 10 trace matched "
        "the committed P9 stdout to printed precision; epoch 19 differed by only `0.0002` in printed loss, while "
        "the full byte-level state SHA differed. The runner stopped before RCT, GEM, evaluation-label access, or any "
        "target metric. Job `893415` was an earlier zero-result workdir visibility failure.\n\n"
        "## Frozen Correction\n\n"
        "For each dataset x target x seed unit, P12 reproduces the exact P9 source-training configuration on the "
        "recorded GPU family, persists the actual checkpoint, and runs source-only TSMNet, RCT, SPDIM geodesic, "
        "SPDIM bias, Joint-GEM, and FP-GEM from that same state. The four official controls are unchanged methods "
        "with unchanged P9 adaptation settings; rerunning them is required because the committed P9 checkpoint "
        "weights cannot be loaded. Direct P9 row reuse is therefore zero.\n\n"
        "No dataset, split, source seed, source-training setting, adaptation hyperparameter, FP-GEM definition, "
        "Joint-GEM definition, bootstrap rule, or interpretation boundary changed.\n"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = runner.load_config()
    runner.validate_frozen_inputs(config)
    source_hashes = runner.p9_source_hashes(config)
    manifests = runner.manifest_map(config)
    source_entries = [
        {
            "dataset": key[0],
            "target_subject": key[1],
            "source_seed": key[2],
            "source_model_sha256": source_hashes[key],
        }
        for key in sorted(source_hashes)
    ]
    write_json(SOURCE_INDEX, {
        "status": "frozen_p9_reference_source_states",
        "p9_results_path": config["p9_pipeline"]["results_path"],
        "p9_results_sha256": runner.P9_RESULTS_SHA256,
        "entry_count": len(source_entries),
        "entries": source_entries,
    })
    units, hardware_counts = build_units(config, source_hashes, manifests)
    dryrun = runner.dry_run(config)
    if not dryrun["dryrun_pass"] or hardware_counts != {"V100": 161, "A100": 28}:
        raise RuntimeError(f"P12A dry-run gate failed: {dryrun}")
    source_index_sha = sha256_file(SOURCE_INDEX)
    units_sha = sha256_file(UNITS)
    write_method_freeze(
        config,
        sha256_file(runner.__file__),
        sha256_file(analyzer.__file__),
        source_index_sha,
        units_sha,
        dryrun,
    )
    write_integration_audit(config, dryrun, source_index_sha, units_sha)
    write_source_provenance_amendment()
    COMMAND_LOG.write_text(
        "# FP-GEM Command Log\n\n"
        "- P12A CPU-only freeze generation: `python -m h2cmi.prepare_fp_gem_freeze`. The method, scope, "
        "statistical estimands, and interpretation grid were frozen before target performance observation.\n"
        "- Pre-amendment smoke job `893415`: zero-result infrastructure failure because `/tmp` was not visible "
        "on the compute node.\n"
        "- Pre-amendment smoke job `893416`: clean V100 exact-config source retrain; stopped on the overstrict "
        "unrecoverable P9 byte-hash gate before RCT/GEM/evaluation labels/metrics; accepted rows `0`.\n"
        "- P12A source-provenance amendment: `python -m h2cmi.prepare_fp_gem_freeze`. Direct P9 row reuse is "
        "replaced by frozen same-checkpoint reruns of the four official controls; no scientific setting changed "
        "and no target performance had been observed.\n"
    )
    print(json.dumps({
        "status": "pass",
        "units": len(units),
        "hardware_counts": hardware_counts,
        "runner_sha256": sha256_file(runner.__file__),
        "analyzer_sha256": sha256_file(analyzer.__file__),
        "config_sha256": sha256_file(runner.CONFIG_PATH),
        "source_index_sha256": source_index_sha,
        "units_sha256": units_sha,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
