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
                    "expected_source_model_sha256": source_hashes[key],
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
        "Status: **FROZEN BEFORE GPU SMOKE OR TARGET PERFORMANCE OBSERVATION**.",
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
        "- New computation: Joint-GEM and FP-GEM only, 378 rows.",
        "- Reuse: 756 exact-key P9 rows for source-only TSMNet, RCT, SPDIM geodesic, and SPDIM bias.",
        "- Final same-backbone table: 1,134 rows, six methods.",
        "- Average seeds within dataset x target subject x method first.",
        "- Cluster bootstrap: 10,000 replicates, seed 20260710, dataset-stratified dataset x target-subject clusters, paired methods preserved.",
        "- Report bAcc and accuracy per dataset, subject-weighted, and dataset-macro; primary inference is the five frozen FP-GEM paired contrasts.",
        "",
        "## Frozen Smoke Gate",
        "",
        "One V100 unit only: `BNCI2014_001`, target `1`, source seed `0`. It reports no accuracy or bAcc. It must reproduce P9 source state `f21981a86a61ca0c5129c642a5ecaee301fff0a98466a3fa09d7f89c719b3c43`, validate shape/hook/numerics/leakage, and leave all frozen settings unchanged.",
        "",
        "## Source Checkpoint Policy",
        "",
        "P9 did not persist checkpoint files. P12 therefore reproduces the exact P9 source-training configuration on the original P9 GPU family, requires exact `hash_state` equality to the committed P9 row before adaptation, and then persists that verified state for retries. A mismatch blocks the unit; it never falls back to a scientifically different source model.",
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
        "- phase: `P12A_PRECOMPUTE`",
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
        f"- expected new rows: `{dryrun['new_rows_expected']}`",
        f"- expected reused P9 rows: `{dryrun['reused_p9_rows_expected']}`",
        f"- expected final rows: `{dryrun['final_rows_expected']}`",
        f"- exact unit keys match repaired manifest: `{dryrun['unit_keys_match_manifest']}`",
        f"- exact unit keys match P9 source hashes: `{dryrun['unit_keys_match_p9']}`",
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
        "No P9 TSMNet checkpoint files were found or recorded. The committed P9 `source_model_sha256` column supplies one consistent expected state hash per dataset x target x seed. The runner must reproduce that state exactly before it may execute RCT or GEM.",
    ]
    INTEGRATION_AUDIT.write_text("\n".join(lines) + "\n")


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
        "status": "frozen_expected_source_states",
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
    COMMAND_LOG.write_text(
        "# FP-GEM Command Log\n\n"
        "- P12A CPU-only freeze generation: `python -m h2cmi.prepare_fp_gem_freeze`. "
        "No GPU was used. The method/config, 189-unit manifest, exact P9 source-state index, "
        "smoke gate, statistical estimands, and interpretation grid were frozen before target "
        "performance observation.\n"
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
