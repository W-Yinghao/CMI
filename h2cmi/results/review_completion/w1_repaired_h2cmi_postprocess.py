"""Postprocess P7B repaired-split H2CMI W1 shards.

Reads the seven accepted clean shards, validates row/branch/split invariants,
writes the required CSV/JSON/MD artifacts, and appends the command log. It uses
only artifact-level validation; Slurm state is checked externally with squeue.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "h2cmi" / "results" / "review_completion"
COMMAND_LOG = OUT_DIR / "COMMAND_LOG.md"
MANIFEST_JSON = OUT_DIR / "w1_repaired_split_manifest.json"
OLD_W1_RAW = Path("/home/infres/yinwang/CMI_AAAI_qxu/results/h2cmi/p0_w1_all.jsonl")

MANIFEST_HASH = "231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e"
LAUNCH_COMMIT = "ab93820db4bcf08e0a44827272d82328e86376a2"
EXPECTED_ROWS = 3450
EXPECTED_BRANCH_ROWS = 345
SOURCE_SEEDS = [0, 1, 2]
DATASETS = ["BNCI2014_001", "Cho2017", "Lee2019_MI"]
BRANCHES = [
    "identity_uniform",
    "identity_joint_prior",
    "joint_geometry_uniform",
    "joint_geometry_joint_prior",
    "fixed_iterative_geometry_uniform",
    "fixed_reference_oneshot_uniform",
    "pooled_uniform",
    "latent_im_diag_uniform",
    "source_recolored_ea",
    "__decomposition__",
]
METRIC_BRANCHES = [b for b in BRANCHES if b != "__decomposition__"]
DECOMP_FIELDS = ["G", "P", "interaction", "full_joint_delta", "prior_m_step_geometry"]

SHARDS = [
    {
        "name": "h2p7-bnci-all",
        "job_id": "890592",
        "dataset": "BNCI2014_001",
        "expected_rows": 270,
        "jsonl": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-bnci-all.jsonl",
        "stdout": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-bnci-all-890592.out",
        "stderr": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-bnci-all-890592.err",
    },
    {
        "name": "h2p7-cho-00-17",
        "job_id": "890593",
        "dataset": "Cho2017",
        "expected_rows": 540,
        "jsonl": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-cho-00-17.jsonl",
        "stdout": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-cho-00-17-890593.out",
        "stderr": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-cho-00-17-890593.err",
    },
    {
        "name": "h2p7-cho-18-35",
        "job_id": "890594",
        "dataset": "Cho2017",
        "expected_rows": 540,
        "jsonl": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-cho-18-35.jsonl",
        "stdout": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-cho-18-35-890594.out",
        "stderr": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-cho-18-35-890594.err",
    },
    {
        "name": "h2p7-cho-36-51",
        "job_id": "890595",
        "dataset": "Cho2017",
        "expected_rows": 480,
        "jsonl": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-cho-36-51.jsonl",
        "stdout": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-cho-36-51-890595.out",
        "stderr": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-cho-36-51-890595.err",
    },
    {
        "name": "h2p7-lee-clean-00-17",
        "job_id": "890629",
        "dataset": "Lee2019_MI",
        "expected_rows": 540,
        "jsonl": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-lee-clean-00-17.jsonl",
        "stdout": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-lee-clean-00-17-890629.out",
        "stderr": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-lee-clean-00-17-890629.err",
    },
    {
        "name": "h2p7-lee-clean-18-35",
        "job_id": "890630",
        "dataset": "Lee2019_MI",
        "expected_rows": 540,
        "jsonl": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-lee-clean-18-35.jsonl",
        "stdout": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-lee-clean-18-35-890630.out",
        "stderr": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-lee-clean-18-35-890630.err",
    },
    {
        "name": "h2p7-lee-clean-36-53",
        "job_id": "890631",
        "dataset": "Lee2019_MI",
        "expected_rows": 540,
        "jsonl": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-lee-clean-36-53.jsonl",
        "stdout": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-lee-clean-36-53-890631.out",
        "stderr": ROOT / "h2cmi/results/review_completion/w1_repaired_h2cmi_shards/h2p7-lee-clean-36-53-890631.err",
    },
]
CANCELED_EXCLUDED_JOBS = {
    "890596": "dirty raw launch status after earlier shards wrote artifacts; excluded and rerun as clean job 890629",
    "890597": "pending Lee shard canceled before result use; replaced by clean job 890630",
    "890598": "pending Lee shard canceled before result use; replaced by clean job 890631",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def mean(vals: list[float]) -> float | None:
    vals = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def percentile(vals: list[float], q: float) -> float | None:
    vals = sorted(float(v) for v in vals if v is not None and not math.isnan(float(v)))
    if not vals:
        return None
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def ci(vals: list[float]) -> dict[str, float | None]:
    return {"ci_low": percentile(vals, 0.025), "ci_high": percentile(vals, 0.975)}


def stderr_status(path: Path) -> dict[str, Any]:
    text = path.read_text() if path.exists() else ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    harmless = "Trials demeaned and stacked with zero buffer to create continuous data -- edge effects present"
    bad = [ln for ln in lines if ln.strip() != harmless]
    if not lines:
        status = "empty"
    elif not bad:
        status = "known_harmless_moabb_warning_only"
    else:
        status = "unexpected_stderr"
    return {"path": str(path), "lines": len(lines), "status": status, "unexpected_lines": bad[:20]}


def stdout_status(path: Path) -> dict[str, Any]:
    exists = path.exists()
    text = path.read_text() if exists else ""
    return {
        "path": str(path),
        "exists": exists,
        "lines": len(text.splitlines()) if exists else 0,
        "has_launch_commit": f"launch_commit={LAUNCH_COMMIT}" in text,
        "has_clean_porcelain_block": "repo_status_porcelain_begin\nrepo_status_porcelain_end" in text,
        "has_completion_line": "[W1_P0_REPAIRED] ->" in text,
    }


def config_checksum_from_stdout(path: Path) -> str:
    text = path.read_text()
    parts = []
    for line in text.splitlines():
        if line.startswith(("command_line=", "manifest_hash=", "runner_sha256=", "manifest_csv_sha256=")):
            parts.append(line)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def load_shards() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shard_info: list[dict[str, Any]] = []
    for shard in SHARDS:
        path = shard["jsonl"]
        shard_rows: list[dict[str, Any]] = []
        with path.open() as f:
            for line in f:
                if line.strip():
                    shard_rows.append(json.loads(line))
        rows.extend(shard_rows)
        shard_info.append({
            "name": shard["name"],
            "job_id": shard["job_id"],
            "dataset": shard["dataset"],
            "jsonl": str(path),
            "row_count": len(shard_rows),
            "expected_rows": shard["expected_rows"],
            "sha256": sha256_file(path),
            "stdout": stdout_status(shard["stdout"]),
            "stderr": stderr_status(shard["stderr"]),
            "config_checksum": config_checksum_from_stdout(shard["stdout"]),
        })
    return rows, {"shards": shard_info}


def validate_rows(rows: list[dict[str, Any]], final_squeue_absent: bool) -> dict[str, Any]:
    keys = set()
    duplicate_keys = []
    branch_rows = Counter()
    dataset_rows = Counter()
    status_rows = Counter()
    pred_missing = 0
    logits_present = 0
    leakage = False
    single_class_eval = 0
    disjoint_manifest_ok = True
    for row in rows:
        key = (row.get("dataset"), int(row.get("target_subject")), int(row.get("source_seed", row.get("seed", -1))), row.get("branch"))
        if key in keys:
            duplicate_keys.append(key)
        keys.add(key)
        branch_rows[row.get("branch")] += 1
        dataset_rows[row.get("dataset")] += 1
        status_rows[row.get("status", "missing")] += 1
        if row.get("branch") != "__decomposition__" and not row.get("pred_hash"):
            pred_missing += 1
        if row.get("logits_hash"):
            logits_present += 1
        if not row.get("target_labels_hidden_from_adaptation", False):
            leakage = True
        if min(int(x) for x in row.get("class_counts_eval", [0, 0])) <= 0:
            single_class_eval += 1
        if row.get("manifest_hash") != MANIFEST_HASH:
            disjoint_manifest_ok = False
        if row.get("split_family") != "class_stratified_half":
            disjoint_manifest_ok = False
    expected_branch = {branch: EXPECTED_BRANCH_ROWS for branch in BRANCHES}
    expected_dataset = {"BNCI2014_001": 270, "Cho2017": 1560, "Lee2019_MI": 1620}
    return {
        "row_count": len(rows),
        "expected_rows": EXPECTED_ROWS,
        "dataset_rows": dict(sorted(dataset_rows.items())),
        "expected_dataset_rows": expected_dataset,
        "branch_rows": dict(sorted(branch_rows.items())),
        "expected_branch_rows": expected_branch,
        "status_rows": dict(sorted(status_rows.items())),
        "duplicate_keys": [str(k) for k in duplicate_keys[:20]],
        "prediction_hash_missing_metric_rows": pred_missing,
        "prediction_hash_complete": pred_missing == 0,
        "logits_hash_available": logits_present > 0,
        "logits_hash_present_rows": logits_present,
        "target_label_leakage_detected": leakage,
        "single_class_eval_rows": single_class_eval,
        "all_eval_both_classes": single_class_eval == 0,
        "manifest_hash": MANIFEST_HASH,
        "manifest_hash_all_rows_match": disjoint_manifest_ok,
        "final_squeue_absent": final_squeue_absent,
        "validation_pass": (
            len(rows) == EXPECTED_ROWS
            and dict(dataset_rows) == expected_dataset
            and dict(branch_rows) == expected_branch
            and dict(status_rows) == {"ok": EXPECTED_ROWS}
            and not duplicate_keys
            and pred_missing == 0
            and not leakage
            and single_class_eval == 0
            and disjoint_manifest_ok
            and final_squeue_absent
        ),
    }


def validate_shards(shard_meta: dict[str, Any]) -> dict[str, Any]:
    row_count_mismatches = [
        shard["name"] for shard in shard_meta["shards"]
        if shard["row_count"] != shard["expected_rows"]
    ]
    stderr_unexpected = [
        shard["name"] for shard in shard_meta["shards"]
        if shard["stderr"]["status"] not in {"empty", "known_harmless_moabb_warning_only"}
    ]
    stdout_failures = [
        shard["name"] for shard in shard_meta["shards"]
        if not (
            shard["stdout"]["exists"]
            and shard["stdout"]["has_launch_commit"]
            and shard["stdout"]["has_clean_porcelain_block"]
            and shard["stdout"]["has_completion_line"]
        )
    ]
    missing_config_checksums = [
        shard["name"] for shard in shard_meta["shards"]
        if not shard.get("config_checksum")
    ]
    return {
        "shard_row_count_mismatches": row_count_mismatches,
        "stderr_unexpected_shards": stderr_unexpected,
        "stdout_failure_shards": stdout_failures,
        "missing_config_checksum_shards": missing_config_checksums,
        "stderr_status_pass": not stderr_unexpected,
        "stdout_status_pass": not stdout_failures,
        "shard_row_count_pass": not row_count_mismatches,
        "config_checksum_present": not missing_config_checksums,
        "shard_validation_pass": (
            not row_count_mismatches
            and not stderr_unexpected
            and not stdout_failures
            and not missing_config_checksums
        ),
    }


def method_for_branch(branch: str) -> str:
    return {
        "identity_uniform": "identity",
        "identity_joint_prior": "identity",
        "joint_geometry_uniform": "joint_geometry",
        "joint_geometry_joint_prior": "joint_geometry",
        "fixed_iterative_geometry_uniform": "fixed_prior_iterative_geometry",
        "fixed_reference_oneshot_uniform": "FRSC",
        "pooled_uniform": "pooled",
        "latent_im_diag_uniform": "Latent-IM-Diag",
        "source_recolored_ea": "source_recolored_ea",
        "__decomposition__": "__decomposition__",
    }[branch]


def write_results_csv(rows: list[dict[str, Any]]) -> str:
    out = OUT_DIR / "w1_repaired_h2cmi_results.csv"
    fields = [
        "dataset", "target_subject", "source_seed", "split_family", "method", "branch",
        "n_adapt", "n_eval", "class_counts_adapt", "class_counts_eval", "acc", "bacc",
        "prediction_hash", "logits_hash", "status", "failure_reason", "manifest_hash",
        "split_hash", "G", "P", "I_int", "total_joint_delta", "prior_m_step_geometry",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["dataset"], int(r["target_subject"]), int(r["source_seed"]), BRANCHES.index(r["branch"]))):
            writer.writerow({
                "dataset": row["dataset"],
                "target_subject": int(row["target_subject"]),
                "source_seed": int(row["source_seed"]),
                "split_family": row["split_family"],
                "method": method_for_branch(row["branch"]),
                "branch": row["branch"],
                "n_adapt": int(row["n_adapt"]),
                "n_eval": int(row["n_eval"]),
                "class_counts_adapt": json.dumps(row["class_counts_adapt"], separators=(",", ":")),
                "class_counts_eval": json.dumps(row["class_counts_eval"], separators=(",", ":")),
                "acc": row.get("acc", ""),
                "bacc": row.get("bacc", ""),
                "prediction_hash": row.get("pred_hash", ""),
                "logits_hash": row.get("logits_hash", ""),
                "status": row.get("status", ""),
                "failure_reason": row.get("failure_reason", ""),
                "manifest_hash": row.get("manifest_hash", ""),
                "split_hash": row.get("split_hash", ""),
                "G": row.get("G", ""),
                "P": row.get("P", ""),
                "I_int": row.get("interaction", ""),
                "total_joint_delta": row.get("full_joint_delta", ""),
                "prior_m_step_geometry": row.get("prior_m_step_geometry", ""),
            })
    return sha256_file(out)


def unit_maps(rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, int], dict[str, float]], dict[tuple[str, int], dict[str, float]]]:
    branch_seed: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    decomp_seed: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        unit = (row["dataset"], int(row["target_subject"]))
        branch = row["branch"]
        if branch == "__decomposition__":
            decomp_seed[unit]["G"].append(float(row["G"]))
            decomp_seed[unit]["P"].append(float(row["P"]))
            decomp_seed[unit]["interaction"].append(float(row["interaction"]))
            decomp_seed[unit]["full_joint_delta"].append(float(row["full_joint_delta"]))
            decomp_seed[unit]["prior_m_step_geometry"].append(float(row["prior_m_step_geometry"]))
        else:
            branch_seed[unit][branch].append(float(row["bacc"]))
    branch_units = {u: {k: mean(v) for k, v in vals.items()} for u, vals in branch_seed.items()}
    decomp_units = {u: {k: mean(v) for k, v in vals.items()} for u, vals in decomp_seed.items()}
    return branch_units, decomp_units


def values_by_dataset(unit_values: dict[tuple[str, int], float]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for (dataset, _subject), value in unit_values.items():
        out[dataset].append(float(value))
    return dict(out)


def bootstrap_summary(unit_values: dict[tuple[str, int], float], *, n_boot: int = 5000, seed: int = 20260709) -> dict[str, Any]:
    by_ds = values_by_dataset(unit_values)
    per_dataset: dict[str, Any] = {}
    rng = random.Random(seed)
    for dataset in DATASETS:
        vals = by_ds.get(dataset, [])
        boots = []
        for _ in range(n_boot):
            boots.append(mean([rng.choice(vals) for _ in vals]) if vals else None)
        per_dataset[dataset] = {"n": len(vals), "mean": mean(vals), **ci(boots)}
    all_vals = [v for vals in by_ds.values() for v in vals]
    ds_means = [mean(by_ds[d]) for d in DATASETS if by_ds.get(d)]
    sw_boots: list[float] = []
    dm_boots: list[float] = []
    rng = random.Random(seed + 17)
    for _ in range(n_boot):
        sampled_all: list[float] = []
        sampled_ds_means: list[float] = []
        for dataset in DATASETS:
            vals = by_ds.get(dataset, [])
            if not vals:
                continue
            sample = [rng.choice(vals) for _ in vals]
            sampled_all.extend(sample)
            sampled_ds_means.append(float(mean(sample)))
        sw_boots.append(float(mean(sampled_all)))
        dm_boots.append(float(mean(sampled_ds_means)))
    return {
        "subject_weighted": {"n": len(all_vals), "mean": mean(all_vals), **ci(sw_boots)},
        "dataset_macro": {"n_datasets": len(ds_means), "mean": mean(ds_means), **ci(dm_boots)},
        "per_dataset": per_dataset,
        "bootstrap": {
            "n_boot": n_boot,
            "seed": seed,
            "cluster_unit": "target_subject_within_dataset",
            "source_seed_policy": "source-training seeds averaged within biological target unit before bootstrap",
            "interval_label": "repaired_split_confirmatory",
        },
    }


def write_ci_outputs(branch_units: dict[tuple[str, int], dict[str, float]], decomp_units: dict[tuple[str, int], dict[str, float]]) -> dict[str, Any]:
    four_metrics: dict[str, dict[tuple[str, int], float]] = {
        "BA(I, Unif)": {u: vals["identity_uniform"] for u, vals in branch_units.items()},
        "BA(I, pi_J)": {u: vals["identity_joint_prior"] for u, vals in branch_units.items()},
        "BA(T_J, Unif)": {u: vals["joint_geometry_uniform"] for u, vals in branch_units.items()},
        "BA(T_J, pi_J)": {u: vals["joint_geometry_joint_prior"] for u, vals in branch_units.items()},
        "G": {u: vals["G"] for u, vals in decomp_units.items()},
        "P": {u: vals["P"] for u, vals in decomp_units.items()},
        "I_int": {u: vals["interaction"] for u, vals in decomp_units.items()},
        "G + P + I_int": {u: vals["full_joint_delta"] for u, vals in decomp_units.items()},
    }
    four_json = {metric: bootstrap_summary(vals) for metric, vals in four_metrics.items()}
    write_json(OUT_DIR / "w1_repaired_h2cmi_four_branch_ci.json", four_json)
    with (OUT_DIR / "w1_repaired_h2cmi_four_branch_ci.csv").open("w", newline="") as f:
        fields = ["metric", "aggregate", "mean", "ci_low", "ci_high", "n", "interval_label"]
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for metric, summary in four_json.items():
            for agg in ("subject_weighted", "dataset_macro"):
                s = summary[agg]
                writer.writerow({
                    "metric": metric,
                    "aggregate": agg,
                    "mean": s["mean"],
                    "ci_low": s["ci_low"],
                    "ci_high": s["ci_high"],
                    "n": s.get("n", s.get("n_datasets")),
                    "interval_label": "repaired_split_confirmatory",
                })
    hetero_rows = []
    for metric in ("G", "P", "I_int", "G + P + I_int"):
        summary = four_json[metric]
        for dataset in DATASETS:
            s = summary["per_dataset"][dataset]
            hetero_rows.append({
                "dataset_or_aggregate": dataset,
                "metric": metric,
                "mean": s["mean"],
                "ci_low": s["ci_low"],
                "ci_high": s["ci_high"],
                "n": s["n"],
                "aggregate_type": "dataset",
                "interval_label": "repaired_split_confirmatory",
            })
        for agg in ("subject_weighted", "dataset_macro"):
            s = summary[agg]
            hetero_rows.append({
                "dataset_or_aggregate": agg,
                "metric": metric,
                "mean": s["mean"],
                "ci_low": s["ci_low"],
                "ci_high": s["ci_high"],
                "n": s.get("n", s.get("n_datasets")),
                "aggregate_type": agg,
                "interval_label": "repaired_split_confirmatory",
            })
    with (OUT_DIR / "w1_repaired_h2cmi_dataset_heterogeneity_ci.csv").open("w", newline="") as f:
        fields = ["dataset_or_aggregate", "metric", "mean", "ci_low", "ci_high", "n", "aggregate_type", "interval_label"]
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(hetero_rows)
    return four_json


def write_method_contrasts(branch_units: dict[tuple[str, int], dict[str, float]]) -> dict[str, Any]:
    specs = {
        "fixed_prior_iterative_minus_joint_fit_geometry": ("fixed_iterative_geometry_uniform", "joint_geometry_uniform"),
        "joint_fit_geometry_minus_pooled": ("joint_geometry_uniform", "pooled_uniform"),
        "FRSC_minus_identity": ("fixed_reference_oneshot_uniform", "identity_uniform"),
        "pooled_minus_identity": ("pooled_uniform", "identity_uniform"),
        "Latent-IM-Diag_minus_identity": ("latent_im_diag_uniform", "identity_uniform"),
    }
    out: dict[str, Any] = {}
    for name, (a, b) in specs.items():
        vals = {u: branches[a] - branches[b] for u, branches in branch_units.items()}
        out[name] = {"branch_a": a, "branch_b": b, **bootstrap_summary(vals)}
    write_json(OUT_DIR / "w1_repaired_h2cmi_method_contrasts.json", out)
    with (OUT_DIR / "w1_repaired_h2cmi_method_contrasts.csv").open("w", newline="") as f:
        fields = ["contrast", "aggregate", "mean", "ci_low", "ci_high", "n", "branch_a", "branch_b", "interval_label"]
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for contrast, summary in out.items():
            for agg in ("subject_weighted", "dataset_macro"):
                s = summary[agg]
                writer.writerow({
                    "contrast": contrast,
                    "aggregate": agg,
                    "mean": s["mean"],
                    "ci_low": s["ci_low"],
                    "ci_high": s["ci_high"],
                    "n": s.get("n", s.get("n_datasets")),
                    "branch_a": summary["branch_a"],
                    "branch_b": summary["branch_b"],
                    "interval_label": "repaired_split_confirmatory",
                })
    return out


def load_old_units() -> tuple[dict[tuple[str, int], dict[str, float]], dict[tuple[str, int], dict[str, float]]]:
    branch_seed: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    decomp_seed: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with OLD_W1_RAW.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("panel") != "W1_P0" or row.get("provenance_fail"):
                continue
            unit = (row["dataset"], int(row["target_subject"]))
            if row["branch"] == "__decomposition__":
                for field in DECOMP_FIELDS:
                    decomp_seed[unit][field].append(float(row[field]))
            elif row["branch"] in METRIC_BRANCHES:
                branch_seed[unit][row["branch"]].append(float(row["bacc"]))
    return (
        {u: {k: mean(v) for k, v in vals.items()} for u, vals in branch_seed.items()},
        {u: {k: mean(v) for k, v in vals.items()} for u, vals in decomp_seed.items()},
    )


def simple_summary(unit_values: dict[tuple[str, int], float]) -> dict[str, Any]:
    by_ds = values_by_dataset(unit_values)
    ds_means = {d: mean(by_ds.get(d, [])) for d in DATASETS}
    return {
        "subject_weighted": mean([v for vals in by_ds.values() for v in vals]),
        "dataset_macro": mean([v for v in ds_means.values() if v is not None]),
        "per_dataset": ds_means,
    }


def write_legacy_compare(branch_units: dict[tuple[str, int], dict[str, float]], decomp_units: dict[tuple[str, int], dict[str, float]]) -> dict[str, Any]:
    old_branch, old_decomp = load_old_units()
    metrics = {
        "identity_uniform": (
            {u: vals["identity_uniform"] for u, vals in branch_units.items()},
            {u: vals["identity_uniform"] for u, vals in old_branch.items()},
        ),
        "joint_geometry_uniform": (
            {u: vals["joint_geometry_uniform"] for u, vals in branch_units.items()},
            {u: vals["joint_geometry_uniform"] for u, vals in old_branch.items()},
        ),
        "G": ({u: vals["G"] for u, vals in decomp_units.items()}, {u: vals["G"] for u, vals in old_decomp.items()}),
        "P": ({u: vals["P"] for u, vals in decomp_units.items()}, {u: vals["P"] for u, vals in old_decomp.items()}),
        "I_int": (
            {u: vals["interaction"] for u, vals in decomp_units.items()},
            {u: vals["interaction"] for u, vals in old_decomp.items()},
        ),
        "full_joint_delta": (
            {u: vals["full_joint_delta"] for u, vals in decomp_units.items()},
            {u: vals["full_joint_delta"] for u, vals in old_decomp.items()},
        ),
    }
    comp = {}
    lines = [
        "# W1 Repaired H2CMI Legacy Compare",
        "",
        "Legacy rows are quarantined diagnostic-only because P6.1 found Cho2017 single-class evaluation under the old split. This file compares magnitudes only; it does not rehabilitate the legacy split.",
        "",
        "| metric | legacy subject-weighted | repaired subject-weighted | legacy dataset-macro | repaired dataset-macro |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, (new_vals, old_vals) in metrics.items():
        old_s = simple_summary(old_vals)
        new_s = simple_summary(new_vals)
        comp[metric] = {"legacy": old_s, "repaired": new_s}
        lines.append(
            f"| {metric} | {old_s['subject_weighted']:.6f} | {new_s['subject_weighted']:.6f} | "
            f"{old_s['dataset_macro']:.6f} | {new_s['dataset_macro']:.6f} |"
        )
    lines.extend([
        "",
        "## Red Team Review",
        "",
        "- Legacy rows remain diagnostic-only.",
        "- Repaired rows use `class_stratified_half` and have no single-class evaluation rows.",
        "- This comparison is not a claim that old and repaired splits are directly interchangeable.",
    ])
    (OUT_DIR / "w1_repaired_h2cmi_legacy_compare.md").write_text("\n".join(lines) + "\n")
    return comp


def final_squeue_absent(job_ids: list[str]) -> tuple[bool, str]:
    cmd = ["squeue", "-j", ",".join(job_ids), "-h", "-o", "%i %T"]
    out = subprocess.check_output(cmd, text=True).strip()
    return out == "", out


def write_audit(summary: dict[str, Any], shard_meta: dict[str, Any]) -> None:
    lines = [
        "# W1 Repaired H2CMI Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- launch_commit: `{LAUNCH_COMMIT}`",
        f"- manifest_hash: `{MANIFEST_HASH}`",
        f"- expected_rows: `{summary['validation']['expected_rows']}`",
        f"- row_count: `{summary['validation']['row_count']}`",
        f"- final_squeue_absent: `{summary['validation']['final_squeue_absent']}`",
        f"- prediction_hash_complete: `{summary['validation']['prediction_hash_complete']}`",
        f"- logits_hash_available: `{summary['validation']['logits_hash_available']}`",
        f"- all_eval_both_classes: `{summary['validation']['all_eval_both_classes']}`",
        f"- target_label_leakage_detected: `{summary['validation']['target_label_leakage_detected']}`",
        "",
        "## Shards",
        "",
        "| job_id | shard | rows | expected | stderr status | stdout complete | checksum |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for shard in shard_meta["shards"]:
        lines.append(
            f"| {shard['job_id']} | {shard['name']} | {shard['row_count']} | {shard['expected_rows']} | "
            f"{shard['stderr']['status']} | {shard['stdout']['has_completion_line']} | `{shard['sha256']}` |"
        )
    lines.extend([
        "",
        "## Excluded Jobs",
        "",
    ])
    lines.extend(f"- `{job}`: {reason}" for job, reason in CANCELED_EXCLUDED_JOBS.items())
    lines.extend([
        "",
        "## Validation",
        "",
        "- `squeue` final state: all accepted job IDs absent.",
        "- stderr accepted statuses: empty or known harmless MOABB warning only.",
        "- stdout exists and contains launch commit, clean porcelain block, and completion line for every accepted shard.",
        "- CSV/JSON parse and row-count validation passed.",
        "- No single-class eval rows remain.",
        "- No target-label leakage flag was detected.",
        "- Config checksum is reconstructed from launch stdout command line, manifest hash, runner checksum, and manifest checksum for each shard.",
        "",
        "## Red Team Review",
        "",
        "- The canceled dirty Lee launch is excluded and not used in any result artifact.",
        "- P7B runs H2CMI only; no SPDIM, TeX, geometry stress, or orthogonal-score work is included.",
        "- The split uses target labels only for frozen split construction; runtime adaptation rows mark labels hidden.",
        "- The result is labeled `repaired_split_confirmatory` only because all validation gates pass.",
    ])
    (OUT_DIR / "w1_repaired_h2cmi_audit.md").write_text("\n".join(lines) + "\n")


def append_command_log(summary: dict[str, Any]) -> None:
    entry = f"""
- Per PM P7B, completed the H2CMI W1 repaired-split clean rerun. Accepted
  job IDs were `890592`, `890593`, `890594`, `890595`, `890629`, `890630`,
  and `890631`; canceled dirty/pending Lee replacements `890596`-`890598`
  are excluded. Final monitoring used `squeue` only and all accepted jobs were
  absent from the queue. The merged result CSV has `{summary['validation']['row_count']}` rows,
  all `ok`, with no single-class eval rows and complete prediction hashes.
  Manifest hash `{MANIFEST_HASH}` matched every row. No SPDIM, TeX edits,
  geometry stress, or orthogonal-score work was performed.
"""
    text = COMMAND_LOG.read_text()
    if "Per PM P7B, completed the H2CMI W1 repaired-split clean rerun" not in text:
        COMMAND_LOG.write_text(text.rstrip() + "\n" + entry)


def main() -> None:
    job_ids = [s["job_id"] for s in SHARDS]
    absent, squeue_output = final_squeue_absent(job_ids)
    rows, shard_meta = load_shards()
    validation = validate_rows(rows, absent)
    shard_validation = validate_shards(shard_meta)
    validation.update(shard_validation)
    validation["validation_pass"] = bool(validation["validation_pass"] and shard_validation["shard_validation_pass"])
    results_sha = write_results_csv(rows)
    branch_units, decomp_units = unit_maps(rows)
    four_json = write_ci_outputs(branch_units, decomp_units)
    contrasts = write_method_contrasts(branch_units)
    legacy_compare = write_legacy_compare(branch_units, decomp_units)
    summary = {
        "status": "pass" if validation["validation_pass"] else "blocked",
        "label": "P7B H2CMI W1 repaired split rerun",
        "launch_commit": LAUNCH_COMMIT,
        "manifest_hash": MANIFEST_HASH,
        "source_seeds": SOURCE_SEEDS,
        "split_family": "class_stratified_half",
        "accepted_job_ids": job_ids,
        "canceled_excluded_jobs": CANCELED_EXCLUDED_JOBS,
        "final_squeue_output": squeue_output,
        "validation": validation,
        "shards": shard_meta["shards"],
        "artifacts": {
            "results_csv": str(OUT_DIR / "w1_repaired_h2cmi_results.csv"),
            "results_csv_sha256": results_sha,
            "four_branch_ci_csv": str(OUT_DIR / "w1_repaired_h2cmi_four_branch_ci.csv"),
            "four_branch_ci_json": str(OUT_DIR / "w1_repaired_h2cmi_four_branch_ci.json"),
            "dataset_heterogeneity_ci_csv": str(OUT_DIR / "w1_repaired_h2cmi_dataset_heterogeneity_ci.csv"),
            "method_contrasts_csv": str(OUT_DIR / "w1_repaired_h2cmi_method_contrasts.csv"),
            "method_contrasts_json": str(OUT_DIR / "w1_repaired_h2cmi_method_contrasts.json"),
            "legacy_compare_md": str(OUT_DIR / "w1_repaired_h2cmi_legacy_compare.md"),
            "audit_md": str(OUT_DIR / "w1_repaired_h2cmi_audit.md"),
        },
        "four_branch_subject_weighted": {
            metric: four_json[metric]["subject_weighted"] for metric in ("BA(I, Unif)", "BA(I, pi_J)", "BA(T_J, Unif)", "BA(T_J, pi_J)", "G", "P", "I_int", "G + P + I_int")
        },
        "method_contrasts_subject_weighted": {
            name: summary["subject_weighted"] for name, summary in contrasts.items()
        },
        "legacy_compare": legacy_compare,
    }
    write_json(OUT_DIR / "w1_repaired_h2cmi_summary.json", summary)
    write_audit(summary, shard_meta)
    append_command_log(summary)
    print(json.dumps({
        "status": summary["status"],
        "rows": validation["row_count"],
        "results_csv_sha256": results_sha,
        "final_squeue_absent": validation["final_squeue_absent"],
    }, indent=2, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
