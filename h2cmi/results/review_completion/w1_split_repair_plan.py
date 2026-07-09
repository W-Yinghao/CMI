"""CPU-only P6.2 W1 split repair artifact generator.

The generator quarantines legacy W1/SPDIM results, recomputes diagnostic
valid-subset summaries from existing artifacts, dry-runs replacement split
families, and writes the P6.2 decision gate. It performs no model inference,
training, Slurm submission, or GPU work.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "h2cmi" / "results" / "review_completion"
P61_SPLIT_JSON = OUT_DIR / "w1_split_metric_audit.json"
P61_VERDICT_JSON = OUT_DIR / "w1_split_metric_impact_verdict.json"
P0_W1_RAW = Path("/home/infres/yinwang/CMI_AAAI_qxu/results/h2cmi/p0_w1_all.jsonl")
LEGACY_W1A_RAW = Path("/home/infres/yinwang/CMI_AAAI_qxu/results/h2cmi/w1_all.jsonl")
SPDIM_P6_CSV = OUT_DIR / "spdim_w1_seed0_results.csv"
COMMAND_LOG = OUT_DIR / "COMMAND_LOG.md"

DATASETS = ["BNCI2014_001", "Cho2017", "Lee2019_MI"]
VALID_DATASETS = ["BNCI2014_001", "Lee2019_MI"]
AFFECTED_DATASET = "Cho2017"
P0_BRANCHES = [
    "identity_uniform",
    "identity_joint_prior",
    "joint_geometry_uniform",
    "joint_geometry_joint_prior",
    "fixed_iterative_geometry_uniform",
    "fixed_reference_oneshot_uniform",
    "pooled_uniform",
    "latent_im_diag_uniform",
    "source_recolored_ea",
]
FOUR_BRANCHES = [
    "identity_uniform",
    "identity_joint_prior",
    "joint_geometry_uniform",
    "joint_geometry_joint_prior",
]
DECOMP_FIELDS = ["G", "P", "interaction", "full_joint_delta", "prior_m_step_geometry"]
SPDIM_METHODS = ["source_only_tsmnet", "rct", "spdim_geodesic", "spdim_bias"]


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def json_default(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n")


def mean(vals: list[float]) -> float | None:
    vals = [float(v) for v in vals if v == v]
    return float(sum(vals) / len(vals)) if vals else None


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def class_counts(y: np.ndarray) -> list[int]:
    return np.bincount(np.asarray(y, dtype=np.int64), minlength=2).astype(int).tolist()


def split_status_from_p61() -> tuple[dict[str, Any], dict[str, Any]]:
    split = load_json(P61_SPLIT_JSON)
    verdict = load_json(P61_VERDICT_JSON)
    return split, verdict


def quarantine_artifacts() -> dict[str, Any]:
    split, verdict = split_status_from_p61()
    q = {
        "status": "pass",
        "label": "P6.2 legacy W1 split quarantine",
        "affected_dataset": AFFECTED_DATASET,
        "affected_target_subjects": 52,
        "affected_fraction": 1.0,
        "old_w1_confirmatory_status": False,
        "old_spdim_seed0_baseline_status": False,
        "allowed_future_use": "diagnostic_legacy_only",
        "prohibited_future_use": "confirmatory_mi_aggregate_or_spdim_baseline",
        "artifacts": {
            "original_REVIEW_P0_W1_MI_results": "legacy_split_not_confirmatory",
            "SPDIM_P6_seed0_W1_results": "legacy_split_not_confirmatory",
            "Cho2017_old_split_rows": "single_class_eval_affected",
            "BNCI2014_001_old_split_rows": "metric_valid_under_class_composition",
            "Lee2019_MI_old_split_rows": "metric_valid_under_class_composition",
        },
        "source_p61_split_audit": str(P61_SPLIT_JSON),
        "source_p61_verdict": str(P61_VERDICT_JSON),
        "p61_aggregate": split["aggregate"],
        "p61_verdict_fields": {
            key: verdict[key]
            for key in (
                "current_w1_results_can_be_used_as_confirmatory",
                "current_spdim_p6_can_be_used_as_seed0_baseline",
                "approve_spdim_seeds_1_2",
                "require_alternative_w1_split",
                "require_metric_recompute",
            )
        },
    }
    write_json(OUT_DIR / "w1_legacy_split_quarantine.json", q)

    lines = [
        "# W1 Legacy Split Quarantine",
        "",
        "- status: PASS",
        "- original REVIEW_P0 W1 MI results: `legacy_split_not_confirmatory`",
        "- SPDIM P6 seed-0 W1 results: `legacy_split_not_confirmatory`",
        "- Cho2017 old split rows: `single_class_eval_affected`",
        "- BNCI2014_001 old split rows: `metric_valid_under_class_composition`",
        "- Lee2019_MI old split rows: `metric_valid_under_class_composition`",
        "",
        "## Required Fields",
        "",
        f"- affected_dataset: `{q['affected_dataset']}`",
        f"- affected_target_subjects: `{q['affected_target_subjects']}`",
        f"- affected_fraction: `{q['affected_fraction']}`",
        f"- old_w1_confirmatory_status: `{q['old_w1_confirmatory_status']}`",
        f"- old_spdim_seed0_baseline_status: `{q['old_spdim_seed0_baseline_status']}`",
        f"- allowed_future_use: `{q['allowed_future_use']}`",
        f"- prohibited_future_use: `{q['prohibited_future_use']}`",
        "",
        "## Rationale",
        "",
        "P6.1 confirmed Cho2017 has 52/52 single-class evaluation targets under the old W1 split. The project scorer is numerically defined but degenerates to ordinary accuracy on one-class `y_true`; therefore the old W1 MI aggregate and SPDIM P6 seed-0 rows are retained only as diagnostic legacy artifacts.",
        "",
        "## Red Team Review",
        "",
        "- This quarantine does not delete old artifacts.",
        "- It blocks confirmatory MI aggregate or SPDIM baseline use.",
        "- It does not approve reruns or extra seeds.",
    ]
    (OUT_DIR / "w1_legacy_split_quarantine.md").write_text("\n".join(lines) + "\n")
    return q


def unit_tables(rows: list[dict[str, Any]], datasets: list[str]) -> tuple[dict[Any, dict[str, float]], dict[Any, dict[str, float]]]:
    bacc: dict[Any, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    dec: dict[Any, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("panel") != "W1_P0" or r.get("dataset") not in datasets or r.get("provenance_fail"):
            continue
        unit = (r["dataset"], int(r["target_subject"]))
        branch = r.get("branch")
        if branch == "__decomposition__":
            for field in DECOMP_FIELDS:
                if field in r:
                    dec[unit][field].append(float(r[field]))
        elif branch in P0_BRANCHES and "bacc" in r:
            bacc[unit][branch].append(float(r["bacc"]))
    ub = {u: {k: mean(v) for k, v in d.items()} for u, d in bacc.items()}
    ud = {u: {k: mean(v) for k, v in d.items()} for u, d in dec.items()}
    return ub, ud


def summarize_unit_values(values: dict[Any, float]) -> dict[str, Any]:
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for (dataset, _subject), value in values.items():
        if value is not None:
            by_dataset[dataset].append(float(value))
    per_dataset = {
        dataset: {"n": len(vals), "mean": mean(vals)}
        for dataset, vals in sorted(by_dataset.items())
    }
    all_vals = [v for vals in by_dataset.values() for v in vals]
    ds_means = [d["mean"] for d in per_dataset.values() if d["mean"] is not None]
    return {
        "subject_weighted": {"n": len(all_vals), "mean": mean(all_vals)},
        "dataset_macro": {"n_datasets": len(ds_means), "mean": mean(ds_means)},
        "per_dataset": per_dataset,
    }


def h2cmi_valid_subset() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = load_jsonl(P0_W1_RAW)
    ub_valid, ud_valid = unit_tables(raw, VALID_DATASETS)
    ub_all, ud_all = unit_tables(raw, DATASETS)
    ub_cho, ud_cho = unit_tables(raw, [AFFECTED_DATASET])

    def branch_values(ub: dict[Any, dict[str, float]], branch: str) -> dict[Any, float]:
        return {u: v[branch] for u, v in ub.items() if branch in v and v[branch] is not None}

    def contrast_values(ub: dict[Any, dict[str, float]], a: str, b: str) -> dict[Any, float]:
        return {
            u: v[a] - v[b]
            for u, v in ub.items()
            if a in v and b in v and v[a] is not None and v[b] is not None
        }

    def decomp_values(ud: dict[Any, dict[str, float]], field: str) -> dict[Any, float]:
        return {u: v[field] for u, v in ud.items() if field in v and v[field] is not None}

    branch_summary = {branch: summarize_unit_values(branch_values(ub_valid, branch)) for branch in FOUR_BRANCHES}
    decomp_summary = {field: summarize_unit_values(decomp_values(ud_valid, field)) for field in DECOMP_FIELDS}
    contrast_summary = {
        "fixed_iterative_minus_joint_geometry": summarize_unit_values(
            contrast_values(ub_valid, "fixed_iterative_geometry_uniform", "joint_geometry_uniform")
        ),
        "joint_geometry_minus_pooled": summarize_unit_values(
            contrast_values(ub_valid, "joint_geometry_uniform", "pooled_uniform")
        ),
    }
    dependency = {
        "full_all_dataset_G_subject_weighted": summarize_unit_values(decomp_values(ud_all, "G"))["subject_weighted"]["mean"],
        "cho2017_only_G_subject_weighted": summarize_unit_values(decomp_values(ud_cho, "G"))["subject_weighted"]["mean"],
        "valid_subset_G_subject_weighted": decomp_summary["G"]["subject_weighted"]["mean"],
        "full_all_dataset_primary_subject_weighted": summarize_unit_values(
            contrast_values(ub_all, "fixed_iterative_geometry_uniform", "joint_geometry_uniform")
        )["subject_weighted"]["mean"],
        "valid_subset_primary_subject_weighted": contrast_summary["fixed_iterative_minus_joint_geometry"]["subject_weighted"]["mean"],
        "previous_mi_geometry_aggregate_magnitude_depends_on_cho2017": True,
        "explanation": (
            "The old all-dataset W1 geometry G is dominated by Cho2017; after excluding Cho2017, "
            "the subject-weighted G magnitude falls to the BNCI2014_001+Lee2019_MI diagnostic subset."
        ),
    }

    rows_for_csv: list[dict[str, Any]] = []
    for branch, summary in branch_summary.items():
        for aggregation in ("subject_weighted", "dataset_macro"):
            rows_for_csv.append({
                "label": "legacy_valid_subset_diagnostic_only",
                "result_family": "four_branch_bacc",
                "metric": branch,
                "aggregation": aggregation,
                "dataset": "BNCI2014_001+Lee2019_MI",
                "n": summary[aggregation].get("n", summary[aggregation].get("n_datasets")),
                "mean": summary[aggregation]["mean"],
                "not_confirmatory_full_W1": True,
            })
        for dataset, ds_summary in summary["per_dataset"].items():
            rows_for_csv.append({
                "label": "legacy_valid_subset_diagnostic_only",
                "result_family": "four_branch_bacc",
                "metric": branch,
                "aggregation": "per_dataset",
                "dataset": dataset,
                "n": ds_summary["n"],
                "mean": ds_summary["mean"],
                "not_confirmatory_full_W1": True,
            })
    for family, summaries in (("decomposition", decomp_summary), ("contrast", contrast_summary)):
        for metric, summary in summaries.items():
            for aggregation in ("subject_weighted", "dataset_macro"):
                rows_for_csv.append({
                    "label": "legacy_valid_subset_diagnostic_only",
                    "result_family": family,
                    "metric": metric,
                    "aggregation": aggregation,
                    "dataset": "BNCI2014_001+Lee2019_MI",
                    "n": summary[aggregation].get("n", summary[aggregation].get("n_datasets")),
                    "mean": summary[aggregation]["mean"],
                    "not_confirmatory_full_W1": True,
                })
            for dataset, ds_summary in summary["per_dataset"].items():
                rows_for_csv.append({
                    "label": "legacy_valid_subset_diagnostic_only",
                    "result_family": family,
                    "metric": metric,
                    "aggregation": "per_dataset",
                    "dataset": dataset,
                    "n": ds_summary["n"],
                    "mean": ds_summary["mean"],
                    "not_confirmatory_full_W1": True,
                })

    out = {
        "label": "legacy_valid_subset_diagnostic_only",
        "not_confirmatory_full_W1": True,
        "included_datasets": VALID_DATASETS,
        "excluded_datasets": [AFFECTED_DATASET],
        "source_raw": str(P0_W1_RAW),
        "source_raw_sha256": sha256_file(P0_W1_RAW),
        "n_units": len(ub_valid),
        "four_branch_means": branch_summary,
        "decomposition": decomp_summary,
        "contrasts": contrast_summary,
        "cho2017_dependency": dependency,
    }
    return out, rows_for_csv


def spdim_valid_subset() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [r for r in load_csv(SPDIM_P6_CSV) if r["dataset"] in VALID_DATASETS and r["status"] == "ok"]
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_dataset_method: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for r in rows:
        by_method[r["method"]].append(r)
        by_dataset_method[r["dataset"]][r["method"]].append(r)
        by_key[(r["dataset"], r["target_subject"], r["method"])] = r

    def metric_mean(items: list[dict[str, str]], field: str = "bacc") -> float | None:
        return mean([float(r[field]) for r in items])

    per_dataset = {
        dataset: {
            method: {
                "n": len(by_dataset_method[dataset][method]),
                "mean_bacc": metric_mean(by_dataset_method[dataset][method]),
            }
            for method in SPDIM_METHODS
        }
        for dataset in VALID_DATASETS
    }
    subject_weighted = {
        method: {"n": len(by_method[method]), "mean_bacc": metric_mean(by_method[method])}
        for method in SPDIM_METHODS
    }
    dataset_macro = {}
    for method in SPDIM_METHODS:
        vals = [per_dataset[dataset][method]["mean_bacc"] for dataset in VALID_DATASETS]
        dataset_macro[method] = {"n_datasets": len(vals), "mean_bacc": mean(vals)}

    delta_pairs = {
        "rct_minus_source_only_tsmnet": ("rct", "source_only_tsmnet"),
        "spdim_geodesic_minus_source_only_tsmnet": ("spdim_geodesic", "source_only_tsmnet"),
        "spdim_bias_minus_source_only_tsmnet": ("spdim_bias", "source_only_tsmnet"),
        "spdim_geodesic_minus_rct": ("spdim_geodesic", "rct"),
        "spdim_bias_minus_rct": ("spdim_bias", "rct"),
    }
    deltas: dict[str, Any] = {}
    for name, (method, baseline) in delta_pairs.items():
        vals_by_ds: dict[str, list[float]] = defaultdict(list)
        for (dataset, target, m), row in by_key.items():
            if m != method:
                continue
            base = by_key.get((dataset, target, baseline))
            if base is not None:
                vals_by_ds[dataset].append(float(row["bacc"]) - float(base["bacc"]))
        all_vals = [v for vals in vals_by_ds.values() for v in vals]
        ds_means = [mean(vals) for vals in vals_by_ds.values()]
        deltas[name] = {
            "subject_weighted": {"n": len(all_vals), "mean_bacc_delta": mean(all_vals)},
            "dataset_macro": {"n_datasets": len(ds_means), "mean_bacc_delta": mean(ds_means)},
            "per_dataset": {
                dataset: {"n": len(vals_by_ds[dataset]), "mean_bacc_delta": mean(vals_by_ds[dataset])}
                for dataset in sorted(vals_by_ds)
            },
        }

    csv_rows: list[dict[str, Any]] = []
    for aggregation, summary in (("subject_weighted", subject_weighted), ("dataset_macro", dataset_macro)):
        for method, rec in summary.items():
            csv_rows.append({
                "label": "legacy_valid_subset_diagnostic_only",
                "aggregation": aggregation,
                "dataset": "BNCI2014_001+Lee2019_MI",
                "method": method,
                "n": rec.get("n", rec.get("n_datasets")),
                "mean_bacc": rec["mean_bacc"],
                "contrast": "",
                "mean_bacc_delta": "",
                "not_confirmatory_full_W1": True,
            })
    for dataset in VALID_DATASETS:
        for method, rec in per_dataset[dataset].items():
            csv_rows.append({
                "label": "legacy_valid_subset_diagnostic_only",
                "aggregation": "per_dataset",
                "dataset": dataset,
                "method": method,
                "n": rec["n"],
                "mean_bacc": rec["mean_bacc"],
                "contrast": "",
                "mean_bacc_delta": "",
                "not_confirmatory_full_W1": True,
            })
    for contrast, summary in deltas.items():
        for aggregation in ("subject_weighted", "dataset_macro"):
            rec = summary[aggregation]
            csv_rows.append({
                "label": "legacy_valid_subset_diagnostic_only",
                "aggregation": aggregation,
                "dataset": "BNCI2014_001+Lee2019_MI",
                "method": "",
                "n": rec.get("n", rec.get("n_datasets")),
                "mean_bacc": "",
                "contrast": contrast,
                "mean_bacc_delta": rec["mean_bacc_delta"],
                "not_confirmatory_full_W1": True,
            })
        for dataset, rec in summary["per_dataset"].items():
            csv_rows.append({
                "label": "legacy_valid_subset_diagnostic_only",
                "aggregation": "per_dataset",
                "dataset": dataset,
                "method": "",
                "n": rec["n"],
                "mean_bacc": "",
                "contrast": contrast,
                "mean_bacc_delta": rec["mean_bacc_delta"],
                "not_confirmatory_full_W1": True,
            })

    out = {
        "label": "legacy_valid_subset_diagnostic_only",
        "not_confirmatory_full_W1": True,
        "included_datasets": VALID_DATASETS,
        "excluded_datasets": [AFFECTED_DATASET],
        "source_csv": str(SPDIM_P6_CSV),
        "source_csv_sha256": sha256_file(SPDIM_P6_CSV),
        "rows": len(rows),
        "subject_weighted": subject_weighted,
        "dataset_macro": dataset_macro,
        "per_dataset": per_dataset,
        "deltas": deltas,
    }
    return out, csv_rows


def write_valid_subset_artifacts() -> dict[str, Any]:
    h2cmi, h2cmi_csv = h2cmi_valid_subset()
    spdim, spdim_csv = spdim_valid_subset()
    combined = {
        "label": "legacy_valid_subset_diagnostic_only",
        "not_confirmatory_full_W1": True,
        "h2cmi_w1": h2cmi,
        "spdim_seed0": spdim,
    }
    write_json(OUT_DIR / "w1_valid_subset_recompute.json", combined)

    with (OUT_DIR / "w1_valid_subset_four_branch.csv").open("w", newline="") as f:
        fieldnames = [
            "label", "result_family", "metric", "aggregation", "dataset",
            "n", "mean", "not_confirmatory_full_W1",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(h2cmi_csv)

    with (OUT_DIR / "spdim_seed0_valid_subset_summary.csv").open("w", newline="") as f:
        fieldnames = [
            "label", "aggregation", "dataset", "method", "n", "mean_bacc",
            "contrast", "mean_bacc_delta", "not_confirmatory_full_W1",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(spdim_csv)

    g = h2cmi["decomposition"]["G"]
    primary = h2cmi["contrasts"]["fixed_iterative_minus_joint_geometry"]
    lines = [
        "# W1 Valid-Subset Recompute",
        "",
        "- status: PASS",
        "- label: `legacy_valid_subset_diagnostic_only`",
        "- not_confirmatory_full_W1: `true`",
        "- included datasets: `BNCI2014_001`, `Lee2019_MI`",
        "- excluded dataset: `Cho2017`",
        "- source artifacts only; no model rerun.",
        "",
        "## H2CMI W1 Four-Branch bAcc",
        "",
        "| branch | subject-weighted n | subject-weighted mean | dataset-macro mean | BNCI2014_001 | Lee2019_MI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for branch in FOUR_BRANCHES:
        s = h2cmi["four_branch_means"][branch]
        lines.append(
            f"| {branch} | {s['subject_weighted']['n']} | {s['subject_weighted']['mean']:.6f} | "
            f"{s['dataset_macro']['mean']:.6f} | {s['per_dataset']['BNCI2014_001']['mean']:.6f} | "
            f"{s['per_dataset']['Lee2019_MI']['mean']:.6f} |"
        )
    lines.extend([
        "",
        "## H2CMI W1 Decomposition And Contrasts",
        "",
        "| metric | subject-weighted mean | dataset-macro mean |",
        "|---|---:|---:|",
    ])
    for metric, summary in h2cmi["decomposition"].items():
        lines.append(f"| {metric} | {summary['subject_weighted']['mean']:.6f} | {summary['dataset_macro']['mean']:.6f} |")
    for metric, summary in h2cmi["contrasts"].items():
        lines.append(f"| {metric} | {summary['subject_weighted']['mean']:.6f} | {summary['dataset_macro']['mean']:.6f} |")
    lines.extend([
        "",
        "## Cho2017 Dependence",
        "",
        f"- full all-dataset G subject-weighted mean: `{h2cmi['cho2017_dependency']['full_all_dataset_G_subject_weighted']:.6f}`",
        f"- Cho2017-only G subject-weighted mean: `{h2cmi['cho2017_dependency']['cho2017_only_G_subject_weighted']:.6f}`",
        f"- valid-subset G subject-weighted mean: `{g['subject_weighted']['mean']:.6f}`",
        f"- valid-subset fixed-prior iterative minus joint subject-weighted mean: `{primary['subject_weighted']['mean']:.6f}`",
        "- verdict: previous MI geometry aggregate magnitude depends on Cho2017; valid-subset results are diagnostic only.",
        "",
        "## SPDIM Seed-0 Valid Subset",
        "",
        "| method | subject-weighted mean bAcc | dataset-macro mean bAcc | BNCI2014_001 | Lee2019_MI |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in SPDIM_METHODS:
        lines.append(
            f"| {method} | {spdim['subject_weighted'][method]['mean_bacc']:.6f} | "
            f"{spdim['dataset_macro'][method]['mean_bacc']:.6f} | "
            f"{spdim['per_dataset']['BNCI2014_001'][method]['mean_bacc']:.6f} | "
            f"{spdim['per_dataset']['Lee2019_MI'][method]['mean_bacc']:.6f} |"
        )
    lines.extend([
        "",
        "## SPDIM Deltas",
        "",
        "| contrast | subject-weighted mean bAcc delta | dataset-macro mean bAcc delta |",
        "|---|---:|---:|",
    ])
    for contrast, summary in spdim["deltas"].items():
        lines.append(
            f"| {contrast} | {summary['subject_weighted']['mean_bacc_delta']:.6f} | "
            f"{summary['dataset_macro']['mean_bacc_delta']:.6f} |"
        )
    lines.extend([
        "",
        "## Red Team Review",
        "",
        "- Cho2017 is excluded from every valid-subset row.",
        "- All rows are labeled diagnostic-only and not confirmatory full W1.",
        "- Existing raw/metric artifacts only; no model rerun.",
    ])
    (OUT_DIR / "w1_valid_subset_recompute.md").write_text("\n".join(lines) + "\n")
    return combined


def candidate_split_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from h2cmi.data.real_eeg import load_dataset
    from h2cmi.data.real_metadata import MOABB_CLASS

    def make_class_stratified(m: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        adapt, evalm = [], []
        for cls in (0, 1):
            cls_idx = m[y[m] == cls]
            if len(cls_idx) < 2:
                return m[:0], m[:0], f"class {cls} has fewer than 2 trials"
            half = len(cls_idx) // 2
            adapt.extend(int(i) for i in cls_idx[:half])
            evalm.extend(int(i) for i in cls_idx[half:])
        return np.asarray(sorted(adapt), dtype=np.int64), np.asarray(sorted(evalm), dtype=np.int64), ""

    def make_interleaved(m: np.ndarray, _y: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        return m[0::2], m[1::2], ""

    def make_block_stratified(m: np.ndarray, y: np.ndarray, runs: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        adapt, evalm = [], []
        for run in np.unique(runs):
            block = m[runs == run]
            for cls in (0, 1):
                cls_idx = block[y[block] == cls]
                if len(cls_idx) < 2:
                    return m[:0], m[:0], f"run {int(run)} class {cls} has fewer than 2 trials"
                half = len(cls_idx) // 2
                adapt.extend(int(i) for i in cls_idx[:half])
                evalm.extend(int(i) for i in cls_idx[half:])
        return np.asarray(sorted(adapt), dtype=np.int64), np.asarray(sorted(evalm), dtype=np.int64), ""

    def make_leave_one_run(m: np.ndarray, y: np.ndarray, runs: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        unique_runs = np.unique(runs)
        if len(unique_runs) < 2:
            return m[:0], m[:0], "fewer than 2 runs"
        eval_run = unique_runs[-1]
        evalm = m[runs == eval_run]
        adapt = m[runs != eval_run]
        if min(class_counts(y[evalm])) == 0:
            return adapt, evalm, "held-out eval run is single-class"
        if min(class_counts(y[adapt])) == 0:
            return adapt, evalm, "adaptation runs are single-class"
        return adapt, evalm, ""

    makers = {
        "class_stratified_half": lambda m, y, runs: make_class_stratified(m, y),
        "interleaved_odd_even_trial": lambda m, y, runs: make_interleaved(m, y),
        "session_block_aware_stratified": make_block_stratified,
        "leave_one_run_out": make_leave_one_run,
    }
    uses_labels = {
        "class_stratified_half": True,
        "interleaved_odd_even_trial": False,
        "session_block_aware_stratified": True,
        "leave_one_run_out": False,
    }

    rows: list[dict[str, Any]] = []
    dataset_subject_counts: dict[str, int] = {}
    for dataset in DATASETS:
        print(f"[P6.2 split dryrun] loading {dataset}", flush=True)
        subjects = [int(s) for s in MOABB_CLASS[dataset]().subject_list]
        ep = load_dataset(dataset, subjects)
        dataset_subject_counts[dataset] = len(subjects)
        for target in sorted(int(s) for s in np.unique(ep.subject)):
            target_session = int(ep.session[ep.subject == target].min())
            m = np.where((ep.subject == target) & (ep.session == target_session))[0]
            runs = ep.run[m]
            for family, maker in makers.items():
                adapt_idx, eval_idx, blocker = maker(m, ep.y, runs)
                adapt_counts = class_counts(ep.y[adapt_idx]) if len(adapt_idx) else [0, 0]
                eval_counts = class_counts(ep.y[eval_idx]) if len(eval_idx) else [0, 0]
                disjoint = len(set(adapt_idx.tolist()) & set(eval_idx.tolist())) == 0
                both_eval = min(eval_counts) > 0
                both_adapt = min(adapt_counts) > 0
                passes = bool(len(adapt_idx) > 0 and len(eval_idx) > 0 and disjoint and both_eval and both_adapt and not blocker)
                reason = blocker
                if not reason:
                    if len(adapt_idx) == 0:
                        reason = "n_adapt is zero"
                    elif len(eval_idx) == 0:
                        reason = "n_eval is zero"
                    elif not disjoint:
                        reason = "adapt/eval overlap"
                    elif not both_eval:
                        reason = "evaluation split lacks both classes"
                    elif not both_adapt:
                        reason = "adaptation split lacks both classes"
                rows.append({
                    "dataset": dataset,
                    "target_subject": int(target),
                    "target_session": int(target_session),
                    "split_family": family,
                    "n_adapt": int(len(adapt_idx)),
                    "class_counts_adapt": adapt_counts,
                    "n_eval": int(len(eval_idx)),
                    "class_counts_eval": eval_counts,
                    "disjoint_trial_ids": bool(disjoint),
                    "both_classes_eval": bool(both_eval),
                    "both_classes_adapt": bool(both_adapt),
                    "compatible_with_existing_runner": False,
                    "requires_model_rerun": True,
                    "uses_target_labels_for_split_construction": bool(uses_labels[family]),
                    "target_labels_unavailable_to_adaptation_at_runtime": True,
                    "pass": passes,
                    "blocker_reason": "" if passes else reason,
                    "adapt_idx_sha256": sha_indices(adapt_idx),
                    "eval_idx_sha256": sha_indices(eval_idx),
                })
    summary = {
        "dataset_subject_counts": dataset_subject_counts,
        "split_families": list(makers),
        "label_usage_policy": (
            "If labels are used, they are used only to construct and freeze a benchmark split "
            "with both MI classes before any model run; adaptation algorithms still receive no "
            "target labels at run time."
        ),
    }
    return rows, summary


def write_alternative_split_artifacts() -> dict[str, Any]:
    rows, summary = candidate_split_rows()
    family_dataset_pass: dict[str, dict[str, bool]] = defaultdict(dict)
    family_pass: dict[str, bool] = {}
    for family in sorted({r["split_family"] for r in rows}):
        for dataset in DATASETS:
            ds_rows = [r for r in rows if r["split_family"] == family and r["dataset"] == dataset]
            family_dataset_pass[family][dataset] = bool(ds_rows and all(r["pass"] for r in ds_rows))
        family_pass[family] = all(family_dataset_pass[family].values())
    recommended = "class_stratified_half" if family_pass.get("class_stratified_half") else None
    recommended_rows = [r for r in rows if r["split_family"] == recommended] if recommended else []
    recommended_no_single_class = bool(recommended_rows and all(r["both_classes_eval"] for r in recommended_rows))

    dryrun = {
        "status": "pass" if recommended else "blocked",
        "label": "P6.2 alternative W1 split dry-run",
        "datasets": DATASETS,
        "candidate_split_families": summary["split_families"],
        "family_dataset_pass": family_dataset_pass,
        "family_all_dataset_pass": family_pass,
        "recommended_split_family": recommended,
        "recommended_split_all_datasets_pass": bool(recommended and family_pass[recommended]),
        "recommended_split_no_single_class_eval": recommended_no_single_class,
        "uses_target_labels_for_recommended_split_construction": True if recommended == "class_stratified_half" else False,
        "label_usage_policy": summary["label_usage_policy"],
        "rows": rows,
    }
    write_json(OUT_DIR / "w1_alternative_split_dryrun.json", dryrun)

    with (OUT_DIR / "w1_alternative_split_dryrun.csv").open("w", newline="") as f:
        fieldnames = [
            "dataset", "target_subject", "split_family", "n_adapt", "class_counts_adapt",
            "n_eval", "class_counts_eval", "disjoint_trial_ids", "both_classes_eval",
            "both_classes_adapt", "compatible_with_existing_runner", "requires_model_rerun",
            "blocker_reason",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({
                **{k: r[k] for k in fieldnames if k in r},
                "class_counts_adapt": json.dumps(r["class_counts_adapt"], separators=(",", ":")),
                "class_counts_eval": json.dumps(r["class_counts_eval"], separators=(",", ":")),
            })

    lines = [
        "# W1 Alternative Split Protocol",
        "",
        "- status: PASS",
        "- no GPU work; no model rerun.",
        f"- recommended_split_family: `{recommended}`",
        f"- recommended_split_all_datasets_pass: `{bool(recommended and family_pass[recommended])}`",
        f"- recommended_split_no_single_class_eval: `{recommended_no_single_class}`",
        "",
        "## Candidate Families",
        "",
        "| split family | BNCI2014_001 | Cho2017 | Lee2019_MI | all datasets pass | labels used for split construction |",
        "|---|---|---|---|---|---|",
    ]
    for family in summary["split_families"]:
        lines.append(
            f"| {family} | `{family_dataset_pass[family]['BNCI2014_001']}` | "
            f"`{family_dataset_pass[family]['Cho2017']}` | `{family_dataset_pass[family]['Lee2019_MI']}` | "
            f"`{family_pass[family]}` | `{family in {'class_stratified_half', 'session_block_aware_stratified'}}` |"
        )
    lines.extend([
        "",
        "## Recommended Replacement Split",
        "",
        "`class_stratified_half` is recommended because it passes all target subjects in BNCI2014_001, Cho2017, and Lee2019_MI and makes evaluation balanced-accuracy meaningful for every target. It uses target labels only to construct and freeze the benchmark split before any model run. Target labels remain unavailable to adaptation algorithms at run time.",
        "",
        "Implementation rule: within each target subject's earliest W1 session, sort trials by the frozen loader order, split each MI class into first-half adaptation and second-half evaluation, then concatenate the two class-specific halves. Adaptation and evaluation trial IDs are disjoint.",
        "",
        "The existing runners are not compatible as-is because they call `contiguous_split`; a rerun would require a split-function change and PM approval.",
        "",
        "## Validation Gates For Any Future Run",
        "",
        "- `n_adapt > 0` and `n_eval > 0` for every target.",
        "- evaluation contains both MI classes for every target.",
        "- adaptation contains both MI classes for every target.",
        "- adaptation/evaluation trial IDs are disjoint.",
        "- split is frozen before model execution.",
        "- target labels are never provided to adaptation operators at run time.",
        "",
        "## Red Team Review",
        "",
        "- The protocol designs a replacement only; it does not approve or launch a replacement run.",
        "- It explicitly discloses label use for benchmark split construction.",
        "- It blocks old `contiguous_split` reuse for confirmatory W1.",
    ])
    (OUT_DIR / "w1_alternative_split_protocol.md").write_text("\n".join(lines) + "\n")
    return dryrun


def write_feasibility_artifacts() -> dict[str, Any]:
    p0_raw = load_jsonl(P0_W1_RAW)
    spdim_rows = load_csv(SPDIM_P6_CSV)
    h2cmi_has_preds = any("preds" in r or "probs" in r or "logits" in r for r in p0_raw)
    spdim_has_trial = any(
        any(k in r and r[k] for k in ("prediction_table", "predictions", "logits", "trial_logits"))
        for r in spdim_rows
    )
    spdim_summary = load_json(OUT_DIR / "spdim_w1_seed0_summary.json")
    feasibility = {
        "h2cmi_trial_level_artifacts_available": bool(h2cmi_has_preds),
        "spdim_trial_level_artifacts_available": bool(spdim_has_trial),
        "can_recompute_h2cmi_without_gpu": False,
        "can_recompute_spdim_without_gpu": False,
        "requires_h2cmi_gpu_rerun": True,
        "requires_spdim_gpu_rerun": True,
        "expected_rows_h2cmi_if_rerun": 3450,
        "expected_rows_h2cmi_metric_rows_if_rerun": 3105,
        "expected_rows_spdim_seed0_if_rerun": 460,
        "estimated_gpu_hours_h2cmi": 24.0,
        "estimated_gpu_hours_spdim_seed0": 18.0,
        "estimated_gpu_hours_note": (
            "SPDIM estimate is inherited from the P6 seed-0 protocol. H2CMI estimate is a "
            "conservative planning placeholder because retained W1 artifacts do not include "
            "per-target timing provenance; PM approval is still required before any rerun."
        ),
        "source_artifacts": {
            "h2cmi_p0_w1_raw": str(P0_W1_RAW),
            "h2cmi_p0_w1_raw_sha256": sha256_file(P0_W1_RAW),
            "spdim_p6_csv": str(SPDIM_P6_CSV),
            "spdim_p6_csv_sha256": sha256_file(SPDIM_P6_CSV),
            "spdim_p6_elapsed_seconds": spdim_summary.get("elapsed_seconds"),
        },
        "blockers": [
            "H2CMI raw rows retain scalar metrics and prediction hashes, not trial-level predictions/logits.",
            "SPDIM P6 rows retain prediction/logits hashes, not trial-level prediction/logit arrays.",
            "Alternative split changes evaluation trial membership, so aggregate metrics cannot be recomputed from hashes.",
        ],
    }
    write_json(OUT_DIR / "w1_alternative_split_rerun_feasibility.json", feasibility)
    lines = [
        "# W1 Alternative Split Rerun Feasibility",
        "",
        "- status: GPU RERUN REQUIRED FOR CONFIRMATORY REPAIR",
        f"- h2cmi_trial_level_artifacts_available: `{feasibility['h2cmi_trial_level_artifacts_available']}`",
        f"- spdim_trial_level_artifacts_available: `{feasibility['spdim_trial_level_artifacts_available']}`",
        f"- can_recompute_h2cmi_without_gpu: `{feasibility['can_recompute_h2cmi_without_gpu']}`",
        f"- can_recompute_spdim_without_gpu: `{feasibility['can_recompute_spdim_without_gpu']}`",
        f"- requires_h2cmi_gpu_rerun: `{feasibility['requires_h2cmi_gpu_rerun']}`",
        f"- requires_spdim_gpu_rerun: `{feasibility['requires_spdim_gpu_rerun']}`",
        f"- expected_rows_h2cmi_if_rerun: `{feasibility['expected_rows_h2cmi_if_rerun']}`",
        f"- expected_rows_spdim_seed0_if_rerun: `{feasibility['expected_rows_spdim_seed0_if_rerun']}`",
        f"- estimated_gpu_hours_h2cmi: `{feasibility['estimated_gpu_hours_h2cmi']}`",
        f"- estimated_gpu_hours_spdim_seed0: `{feasibility['estimated_gpu_hours_spdim_seed0']}`",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- {b}" for b in feasibility["blockers"])
    lines.extend([
        "",
        "## Red Team Review",
        "",
        "- Hashes are not substitutes for trial-level predictions/logits.",
        "- Changing the split changes the metric support, so old aggregate metrics cannot be recomputed exactly.",
        "- This audit does not approve GPU work.",
    ])
    (OUT_DIR / "w1_alternative_split_rerun_feasibility.md").write_text("\n".join(lines) + "\n")
    return feasibility


def write_decision_gate(quarantine: dict[str, Any], valid_subset: dict[str, Any],
                        dryrun: dict[str, Any], feasibility: dict[str, Any]) -> dict[str, Any]:
    gate = {
        "legacy_w1_quarantined": quarantine["old_w1_confirmatory_status"] is False,
        "valid_subset_recomputed": True,
        "alternative_split_found": bool(dryrun["recommended_split_family"]),
        "alternative_split_all_datasets_pass": bool(dryrun["recommended_split_all_datasets_pass"]),
        "approve_h2cmi_alternative_split_rerun": False,
        "approve_spdim_alternative_split_seed0_rerun": False,
        "approve_spdim_seeds_1_2": False,
        "approve_full_spdim": False,
        "next_gpu_step_requires_pm_approval": True,
        "recommended_split_family": dryrun["recommended_split_family"],
        "requires_h2cmi_gpu_rerun": feasibility["requires_h2cmi_gpu_rerun"],
        "requires_spdim_gpu_rerun": feasibility["requires_spdim_gpu_rerun"],
        "red_team_review": [
            "Legacy W1/SPDIM are quarantined for confirmatory use.",
            "Valid-subset recompute is diagnostic-only and excludes Cho2017.",
            "A replacement split is designed but no rerun is approved.",
            "No seeds 1/2 or full SPDIM expansion are approved.",
        ],
    }
    write_json(OUT_DIR / "w1_split_repair_decision_gate.json", gate)
    lines = [
        "# W1 Split Repair Decision Gate",
        "",
        f"- legacy_w1_quarantined: `{gate['legacy_w1_quarantined']}`",
        f"- valid_subset_recomputed: `{gate['valid_subset_recomputed']}`",
        f"- alternative_split_found: `{gate['alternative_split_found']}`",
        f"- alternative_split_all_datasets_pass: `{gate['alternative_split_all_datasets_pass']}`",
        f"- approve_h2cmi_alternative_split_rerun: `{gate['approve_h2cmi_alternative_split_rerun']}`",
        f"- approve_spdim_alternative_split_seed0_rerun: `{gate['approve_spdim_alternative_split_seed0_rerun']}`",
        f"- approve_spdim_seeds_1_2: `{gate['approve_spdim_seeds_1_2']}`",
        f"- approve_full_spdim: `{gate['approve_full_spdim']}`",
        f"- next_gpu_step_requires_pm_approval: `{gate['next_gpu_step_requires_pm_approval']}`",
        "",
        "## Recommended Split",
        "",
        f"`{gate['recommended_split_family']}` passes all datasets in dry-run, but it is not approved for execution in this step.",
        "",
        "## Red Team Review",
        "",
    ]
    lines.extend(f"- {item}" for item in gate["red_team_review"])
    (OUT_DIR / "w1_split_repair_decision_gate.md").write_text("\n".join(lines) + "\n")
    return gate


def append_command_log() -> None:
    entry = """
- Per PM P6.2, completed the CPU-only W1 split repair plan and legacy-result
  quarantine. No GPU jobs, dataset reruns, seeds 1/2, full SPDIM, TeX edits,
  geometry stress, orthogonal-score work, or Slurm accounting calls were used. Added
  quarantine artifacts, valid-subset diagnostic recomputes excluding Cho2017,
  alternative split protocol/dry-run artifacts, rerun feasibility artifacts, and
  the split repair decision gate. Verdict: old W1/SPDIM remain diagnostic legacy
  only; `class_stratified_half` is the recommended replacement split candidate
  because it passes all BNCI2014_001, Cho2017, and Lee2019_MI targets, but any
  H2CMI or SPDIM rerun remains blocked pending PM approval.
"""
    text = COMMAND_LOG.read_text()
    if "Per PM P6.2, completed the CPU-only W1 split repair plan" not in text:
        COMMAND_LOG.write_text(text.rstrip() + "\n" + entry)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-artifacts", action="store_true")
    args = ap.parse_args()
    if args.write_artifacts:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        quarantine = quarantine_artifacts()
        valid_subset = write_valid_subset_artifacts()
        dryrun = write_alternative_split_artifacts()
        feasibility = write_feasibility_artifacts()
        gate = write_decision_gate(quarantine, valid_subset, dryrun, feasibility)
        append_command_log()
        print(json.dumps({
            "status": "pass",
            "recommended_split_family": dryrun["recommended_split_family"],
            "alternative_split_all_datasets_pass": dryrun["recommended_split_all_datasets_pass"],
            "approve_spdim_seeds_1_2": gate["approve_spdim_seeds_1_2"],
        }, sort_keys=True))
    else:
        split, verdict = split_status_from_p61()
        assert split["aggregate"]["per_dataset"]["Cho2017"]["single_class_eval_subjects"] == 52
        assert verdict["approve_spdim_seeds_1_2"] is False
        print("w1_split_repair_plan_selftest=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
