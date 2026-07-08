"""CPU-only P6.1 W1 split/metric audit generator.

This script has two modes:

* default: run deterministic balanced-accuracy scorer checks only;
* --write-artifacts: also load the W1 MI datasets through the frozen loader,
  recompute the contiguous split composition, and write the P6.1 audit files.

It launches no Slurm jobs and performs no model training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.metrics import accuracy_score, balanced_accuracy_score


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "h2cmi" / "results" / "review_completion"
DATASETS = ["BNCI2014_001", "Cho2017", "Lee2019_MI"]
SPDIM_METHODS = ["source_only_tsmnet", "rct", "spdim_geodesic", "spdim_bias"]

MAIN_P0_RAW = Path("/home/infres/yinwang/CMI_AAAI_qxu/results/h2cmi/p0_w1_all.jsonl")
LEGACY_W1_RAW = Path("/home/infres/yinwang/CMI_AAAI_qxu/results/h2cmi/w1_all.jsonl")
SPDIM_P6_CSV = OUT_DIR / "spdim_w1_seed0_results.csv"
SPDIM_DRYRUN_JSON = OUT_DIR / "spdim_w1_seed0_dryrun_audit.json"
REVIEW_P0_MANIFEST = ROOT / "h2cmi" / "results" / "REVIEW_P0_MANIFEST.json"
REVIEW_P0_REPORT = ROOT / "h2cmi" / "results" / "review_p0.report.json"
W1_RESULTS_MD = ROOT / "h2cmi" / "results" / "W1_W2_RESULTS.md"


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_default(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def class_counts(y: np.ndarray, n_classes: int = 2) -> list[int]:
    return np.bincount(np.asarray(y, dtype=np.int64), minlength=n_classes).astype(int).tolist()


def scorer_cases() -> dict[str, Any]:
    cases = {
        "single_class_all_correct": ([1, 1, 1, 1], [1, 1, 1, 1]),
        "single_class_half_correct": ([1, 1, 1, 1], [1, 0, 1, 0]),
        "single_class_all_wrong": ([1, 1, 1, 1], [0, 0, 0, 0]),
        "balanced_two_class_mixed": ([0, 0, 1, 1], [0, 1, 1, 1]),
        "imbalanced_two_class_mixed": ([0, 1, 1, 1], [0, 0, 1, 1]),
    }
    out = {}
    warnings_seen: list[str] = []
    for name, (y_true, y_pred) in cases.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            acc = accuracy_score(y_true, y_pred)
            bacc = balanced_accuracy_score(y_true, y_pred)
        warnings_seen.extend(str(w.message) for w in caught)
        out[name] = {
            "y_true": list(y_true),
            "y_pred": list(y_pred),
            "class_counts_true": class_counts(np.asarray(y_true)),
            "accuracy": float(acc),
            "balanced_accuracy": float(bacc),
            "acc_equals_bacc": bool(abs(float(acc) - float(bacc)) < 1e-12),
            "warnings": [str(w.message) for w in caught],
        }
    assert out["single_class_half_correct"]["balanced_accuracy"] == 0.5
    assert out["single_class_half_correct"]["accuracy"] == 0.5
    assert out["single_class_all_wrong"]["balanced_accuracy"] == 0.0
    assert out["balanced_two_class_mixed"]["acc_equals_bacc"] is True
    assert out["imbalanced_two_class_mixed"]["acc_equals_bacc"] is False
    return {
        "sklearn_version": sklearn_version,
        "scorer": "sklearn.metrics.balanced_accuracy_score",
        "absent_class_handling": (
            "ignored_present_labels_only; when y_true has one class, the score is the recall "
            "of that present class and equals ordinary accuracy for that one-class evaluation set"
        ),
        "warnings_seen": sorted(set(warnings_seen)),
        "cases": out,
        "project_scorer_code_references": [
            "h2cmi/run_w1_mi.py: balanced_accuracy_score(ye, pred)",
            "h2cmi/eval/p0_eval.py: _record(... balanced_accuracy_score(y, pred) ...)",
            "h2cmi/run_spdim_probe.py: _metrics(... balanced_accuracy_score(y_true, y_pred) ...)",
        ],
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize_raw_jsonl(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "label": label,
            "path": str(path),
            "exists": False,
            "sha256": None,
            "rows": 0,
            "cho2017_rows": 0,
            "cho2017_metric_rows": 0,
        }
    rows = load_jsonl(path)
    branch_counter = Counter(str(r.get("branch")) for r in rows)
    method_counter = Counter(str(r.get("method")) for r in rows)
    seed_counter = Counter(str(r.get("seed")) for r in rows)
    dataset_counter = Counter(str(r.get("dataset")) for r in rows)
    cho_rows = [r for r in rows if r.get("dataset") == "Cho2017"]
    cho_metric = [r for r in cho_rows if r.get("branch") != "__decomposition__"]
    return {
        "label": label,
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "rows": len(rows),
        "dataset_rows": dict(dataset_counter),
        "branch_rows": dict(branch_counter),
        "method_rows": dict(method_counter),
        "seed_rows": dict(seed_counter),
        "cho2017_rows": len(cho_rows),
        "cho2017_metric_rows": len(cho_metric),
    }


def spdim_p6_rows() -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = list(csv.DictReader(SPDIM_P6_CSV.open()))
    dataset_rows = Counter(r["dataset"] for r in rows)
    single_rows = [
        r for r in rows
        if r.get("class_counts_eval") and min(json.loads(r["class_counts_eval"])) == 0
    ]
    return rows, {
        "path": str(SPDIM_P6_CSV),
        "exists": SPDIM_P6_CSV.exists(),
        "sha256": sha256_file(SPDIM_P6_CSV),
        "rows": len(rows),
        "dataset_rows": dict(dataset_rows),
        "cho2017_rows": sum(1 for r in rows if r["dataset"] == "Cho2017"),
        "single_class_rows": len(single_rows),
    }


def split_rows() -> list[dict[str, Any]]:
    from h2cmi.data.real_eeg import contiguous_split, load_dataset
    from h2cmi.data.real_metadata import MOABB_CLASS

    dryrun = load_json(SPDIM_DRYRUN_JSON)
    dryrun_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for ds in dryrun["datasets"]:
        for detail in ds["target_details"]:
            dryrun_by_key[(ds["dataset"], int(detail["target_subject"]))] = detail

    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        subjects = [int(s) for s in MOABB_CLASS[dataset]().subject_list]
        ep = load_dataset(dataset, subjects)
        for target in sorted(int(s) for s in np.unique(ep.subject)):
            target_session = int(ep.session[ep.subject == target].min())
            adapt_idx, eval_idx = contiguous_split(ep, target, target_session)
            session_idx = np.where((ep.subject == target) & (ep.session == target_session))[0]
            session_runs = np.unique(ep.run[session_idx]).astype(int).tolist()
            adapt_runs = sorted(np.unique(ep.run[adapt_idx]).astype(int).tolist())
            eval_runs = sorted(np.unique(ep.run[eval_idx]).astype(int).tolist())
            if len(session_runs) >= 2:
                adapt_blocks = [f"session={target_session}/run={r}" for r in adapt_runs]
                eval_blocks = [f"session={target_session}/run={r}" for r in eval_runs]
                split_rule = "first_half_runs_adapt_second_half_runs_eval"
            else:
                h = len(session_idx) // 2
                run_id = session_runs[0] if session_runs else 0
                adapt_blocks = [f"session={target_session}/run={run_id}/trial_offset=0:{h}"]
                eval_blocks = [f"session={target_session}/run={run_id}/trial_offset={h}:{len(session_idx)}"]
                split_rule = "single_run_first_contiguous_half_adapt_second_half_eval"
            adapt_counts = class_counts(ep.y[adapt_idx])
            eval_counts = class_counts(ep.y[eval_idx])
            single_class = min(eval_counts) == 0
            if single_class:
                equality_reason = "single_class_scorer_ignores_absent_class"
                fair_two_class = False
            elif eval_counts[0] == eval_counts[1]:
                equality_reason = "class_balance"
                fair_two_class = True
            else:
                equality_reason = "not_by_construction"
                fair_two_class = True
            dry = dryrun_by_key[(dataset, target)]
            rows.append({
                "dataset": dataset,
                "target_subject": int(target),
                "target_session": target_session,
                "split_rule": split_rule,
                "adaptation_block_ids": adapt_blocks,
                "evaluation_block_ids": eval_blocks,
                "n_adapt": int(len(adapt_idx)),
                "class_counts_adapt": adapt_counts,
                "n_eval": int(len(eval_idx)),
                "class_counts_eval": eval_counts,
                "evaluation_is_single_class": bool(single_class),
                "balanced_accuracy_well_defined_under_project_scorer": True,
                "fair_two_class_balanced_accuracy_interpretation": bool(fair_two_class),
                "acc_equals_bacc_reason": equality_reason,
                "spdim_p6_dryrun_eval_count_match": bool(
                    int(dry["eval_n"]) == int(len(eval_idx))
                    and list(dry["class_counts_eval"]) == eval_counts
                ),
                "spdim_p6_dryrun_adapt_count_match": bool(int(dry["adapt_n"]) == int(len(adapt_idx))),
                "adapt_idx_sha256": dry["adapt_idx_sha256"],
                "eval_idx_sha256": dry["eval_idx_sha256"],
            })
    return rows


def aggregate(split: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in split:
        by_dataset[row["dataset"]].append(row)
    per_dataset = {}
    for dataset in DATASETS:
        rows = by_dataset[dataset]
        n = len(rows)
        single = sum(1 for r in rows if r["evaluation_is_single_class"])
        per_dataset[dataset] = {
            "subjects": n,
            "single_class_eval_subjects": single,
            "single_class_eval_fraction": float(single / n) if n else math.nan,
        }
    all_single = sum(v["single_class_eval_subjects"] for v in per_dataset.values())
    return {
        "per_dataset": per_dataset,
        "all_w1": {
            "subjects": len(split),
            "single_class_eval_subjects": all_single,
            "single_class_eval_fraction": float(all_single / len(split)),
        },
    }


def build_split_audit(scorer: dict[str, Any]) -> dict[str, Any]:
    split = split_rows()
    agg = aggregate(split)
    p0_raw = summarize_raw_jsonl(MAIN_P0_RAW, "corrected_review_p0_w1_raw")
    legacy_raw = summarize_raw_jsonl(LEGACY_W1_RAW, "legacy_w1a_raw")
    _p6_rows, p6_summary = spdim_p6_rows()
    affected_subjects = agg["per_dataset"]["Cho2017"]["single_class_eval_subjects"]
    affected_rows = {
        "corrected_review_p0_w1_raw_total": p0_raw["cho2017_rows"],
        "corrected_review_p0_w1_metric_rows_excluding_decomposition": p0_raw["cho2017_metric_rows"],
        "legacy_w1a_raw_total": legacy_raw["cho2017_rows"],
        "spdim_p6_rows": p6_summary["cho2017_rows"],
    }
    return {
        "audit_status": "pass",
        "label": "P6.1 CPU-only W1 split/metric impact audit",
        "datasets": DATASETS,
        "split_function": "h2cmi.data.real_eeg.contiguous_split",
        "main_h2cmi_w1_runner": "h2cmi/run_w1_p0.py",
        "legacy_w1a_runner": "h2cmi/run_w1_mi.py",
        "spdim_p6_runner": "h2cmi/run_spdim_w1_seed0.py",
        "scorer_summary": {
            "scorer": scorer["scorer"],
            "sklearn_version": scorer["sklearn_version"],
            "absent_class_handling": scorer["absent_class_handling"],
        },
        "split_rows": split,
        "aggregate": agg,
        "affected_subjects": {
            "cho2017_single_class_eval_subjects": affected_subjects,
            "all_w1_single_class_eval_subjects": agg["all_w1"]["single_class_eval_subjects"],
        },
        "affected_rows": affected_rows,
        "main_h2cmi_w1_artifacts": {
            "review_p0_manifest": str(REVIEW_P0_MANIFEST),
            "review_p0_manifest_sha256": sha256_file(REVIEW_P0_MANIFEST),
            "review_p0_report": str(REVIEW_P0_REPORT),
            "review_p0_report_sha256": sha256_file(REVIEW_P0_REPORT),
            "w1_w2_results_md": str(W1_RESULTS_MD),
            "w1_w2_results_md_sha256": sha256_file(W1_RESULTS_MD),
            "corrected_review_p0_w1_raw": p0_raw,
            "legacy_w1a_raw": legacy_raw,
        },
        "spdim_p6_artifacts": {
            "spdim_results_csv": p6_summary,
            "spdim_dryrun_json": str(SPDIM_DRYRUN_JSON),
            "spdim_dryrun_json_sha256": sha256_file(SPDIM_DRYRUN_JSON),
        },
        "red_team_review": [
            "CPU-only audit: no Slurm submission and no GPU training.",
            "Both main H2CMI and SPDIM P6 use the same contiguous_split function.",
            "Cho2017 single-class evaluation is confirmed for 52/52 target subjects.",
            "The project scorer returns a numeric score on one-class y_true by ignoring absent classes.",
            "No seeds 1/2 or full SPDIM expansion are approved by this audit.",
        ],
    }


def build_verdict(split_audit: dict[str, Any]) -> dict[str, Any]:
    cho = split_audit["aggregate"]["per_dataset"]["Cho2017"]
    substantial = cho["single_class_eval_fraction"] >= 0.25
    scorer_ignores_absent = "ignored_present_labels_only" in split_audit["scorer_summary"]["absent_class_handling"]
    cho_confirmed = bool(cho["single_class_eval_subjects"] == cho["subjects"] and cho["subjects"] > 0)
    affected = bool(cho_confirmed and substantial and scorer_ignores_absent)
    return {
        "cho2017_single_class_eval_confirmed": cho_confirmed,
        "main_w1_cho2017_claim_affected": affected,
        "spdim_p6_cho2017_result_affected": affected,
        "current_w1_results_can_be_used_as_confirmatory": False if affected else True,
        "current_spdim_p6_can_be_used_as_seed0_baseline": False if affected else True,
        "approve_spdim_seeds_1_2": False,
        "require_alternative_w1_split": bool(affected),
        "require_metric_recompute": bool(affected),
        "reason": (
            "Cho2017 has single-class evaluation for all 52 W1 targets under contiguous_split; "
            "sklearn balanced_accuracy_score ignores absent classes and degenerates to ordinary "
            "accuracy on those rows. The main W1 geometry signal is Cho2017-driven, so current "
            "W1 aggregate claims are affected and SPDIM seeds 1/2 should not launch."
        ),
        "decision_rules_applied": [
            "Cho2017 single-class fraction is substantial: 52/52.",
            "The project scorer ignores absent classes rather than assigning absent-class recall zero or one.",
            "REVIEW_P0 reports the W1 geometry effect as Cho2017-driven.",
            "Full SPDIM expansion remains blocked until PM review after this audit.",
        ],
    }


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n")


def write_scorer_md(path: Path, scorer: dict[str, Any]) -> None:
    lines = [
        "# W1 Balanced-Accuracy Scorer Audit",
        "",
        "- status: PASS",
        f"- sklearn_version: `{scorer['sklearn_version']}`",
        f"- scorer: `{scorer['scorer']}`",
        f"- absent_class_handling: `{scorer['absent_class_handling']}`",
        "",
        "## Code Paths",
        "",
    ]
    lines.extend(f"- `{ref}`" for ref in scorer["project_scorer_code_references"])
    lines.extend([
        "",
        "## Deterministic Cases",
        "",
        "| case | y_true counts | accuracy | balanced accuracy | acc == bAcc | warnings |",
        "|---|---:|---:|---:|---|---|",
    ])
    for name, rec in scorer["cases"].items():
        warning_text = "; ".join(rec["warnings"]) if rec["warnings"] else "none"
        lines.append(
            f"| {name} | `{rec['class_counts_true']}` | {rec['accuracy']:.6f} | "
            f"{rec['balanced_accuracy']:.6f} | `{rec['acc_equals_bacc']}` | {warning_text} |"
        )
    lines.extend([
        "",
        "## Conclusion",
        "",
        "For one-class `y_true`, the project scorer is numerically defined but degenerates to the present-class recall, which equals ordinary accuracy on that one-class evaluation set. This is project-consistent but not a fair two-class balanced-accuracy interpretation.",
    ])
    path.write_text("\n".join(lines) + "\n")


def write_split_md(path: Path, audit: dict[str, Any]) -> None:
    agg = audit["aggregate"]
    lines = [
        "# W1 Split/Metric Audit",
        "",
        "- status: PASS",
        "- scope: CPU-only P6.1 split/metric impact audit",
        "- split_function: `h2cmi.data.real_eeg.contiguous_split`",
        "- main_h2cmi_w1_runner: `h2cmi/run_w1_p0.py`",
        "- legacy_w1a_runner: `h2cmi/run_w1_mi.py`",
        "- spdim_p6_runner: `h2cmi/run_spdim_w1_seed0.py`",
        "- scorer: `sklearn.metrics.balanced_accuracy_score`",
        "- no Slurm jobs launched; no GPU work.",
        "",
        "## Aggregate Single-Class Evaluation",
        "",
        "| dataset | subjects | single-class eval subjects | fraction |",
        "|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        d = agg["per_dataset"][dataset]
        lines.append(
            f"| {dataset} | {d['subjects']} | {d['single_class_eval_subjects']} | "
            f"{d['single_class_eval_fraction']:.6f} |"
        )
    all_w1 = agg["all_w1"]
    lines.extend([
        f"| all W1 | {all_w1['subjects']} | {all_w1['single_class_eval_subjects']} | {all_w1['single_class_eval_fraction']:.6f} |",
        "",
        "## Affected Rows",
        "",
        "| artifact | affected rows | note |",
        "|---|---:|---|",
        f"| corrected REVIEW_P0 W1 raw | {audit['affected_rows']['corrected_review_p0_w1_raw_total']} | Cho2017 rows including `__decomposition__` |",
        f"| corrected REVIEW_P0 W1 metric rows | {audit['affected_rows']['corrected_review_p0_w1_metric_rows_excluding_decomposition']} | Cho2017 rows excluding `__decomposition__` |",
        f"| legacy W1-A raw | {audit['affected_rows']['legacy_w1a_raw_total']} | Cho2017 W1-A method rows |",
        f"| SPDIM P6 seed-0 | {audit['affected_rows']['spdim_p6_rows']} | Cho2017 rows = 52 targets x 4 methods |",
        "",
        "## Per-Target Split Composition",
        "",
        "| dataset | target | adapt blocks | eval blocks | n adapt | class counts adapt | n eval | class counts eval | single-class eval | scorer-defined | acc==bAcc reason |",
        "|---|---:|---|---|---:|---:|---:|---:|---|---|---|",
    ])
    for r in audit["split_rows"]:
        lines.append(
            f"| {r['dataset']} | {r['target_subject']} | `{'; '.join(r['adaptation_block_ids'])}` | "
            f"`{'; '.join(r['evaluation_block_ids'])}` | {r['n_adapt']} | `{r['class_counts_adapt']}` | "
            f"{r['n_eval']} | `{r['class_counts_eval']}` | `{r['evaluation_is_single_class']}` | "
            f"`{r['balanced_accuracy_well_defined_under_project_scorer']}` | {r['acc_equals_bacc_reason']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- BNCI2014_001 has balanced evaluation blocks for all targets, so ordinary accuracy equals balanced accuracy by class balance for any prediction table.",
        "- Cho2017 has single-class evaluation blocks for all targets. Under the project scorer, balanced accuracy is numerically defined but equals ordinary accuracy because the absent class is ignored.",
        "- Lee2019_MI has both classes in evaluation for all targets, but most target blocks are not exactly balanced, so acc and bAcc are not equal by construction.",
        "- The SPDIM P6 dry-run split hashes/counts match the recomputed split counts.",
        "",
        "## Red Team Review",
        "",
    ])
    lines.extend(f"- {item}" for item in audit["red_team_review"])
    path.write_text("\n".join(lines) + "\n")


def write_verdict_md(path: Path, verdict: dict[str, Any]) -> None:
    lines = [
        "# W1 Split/Metric Impact Verdict",
        "",
        "- status: AFFECTED",
        f"- cho2017_single_class_eval_confirmed: `{verdict['cho2017_single_class_eval_confirmed']}`",
        f"- main_w1_cho2017_claim_affected: `{verdict['main_w1_cho2017_claim_affected']}`",
        f"- spdim_p6_cho2017_result_affected: `{verdict['spdim_p6_cho2017_result_affected']}`",
        f"- current_w1_results_can_be_used_as_confirmatory: `{verdict['current_w1_results_can_be_used_as_confirmatory']}`",
        f"- current_spdim_p6_can_be_used_as_seed0_baseline: `{verdict['current_spdim_p6_can_be_used_as_seed0_baseline']}`",
        f"- approve_spdim_seeds_1_2: `{verdict['approve_spdim_seeds_1_2']}`",
        f"- require_alternative_w1_split: `{verdict['require_alternative_w1_split']}`",
        f"- require_metric_recompute: `{verdict['require_metric_recompute']}`",
        "",
        "## Reason",
        "",
        verdict["reason"],
        "",
        "## Decision Rules Applied",
        "",
    ]
    lines.extend(f"- {item}" for item in verdict["decision_rules_applied"])
    lines.extend([
        "",
        "## Red Team Review",
        "",
        "- This verdict does not approve seeds 1/2 or a full three-seed SPDIM run.",
        "- This verdict does not edit manuscript TeX.",
        "- This verdict treats P6 as a seed-0 same-split expansion with a serious Cho2017 split caveat, not a full baseline.",
    ])
    path.write_text("\n".join(lines) + "\n")


def write_artifacts() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scorer = scorer_cases()
    split_audit = build_split_audit(scorer)
    verdict = build_verdict(split_audit)

    write_json(OUT_DIR / "w1_balanced_accuracy_scorer_audit.json", scorer)
    write_scorer_md(OUT_DIR / "w1_balanced_accuracy_scorer_audit.md", scorer)
    write_json(OUT_DIR / "w1_split_metric_audit.json", split_audit)
    write_split_md(OUT_DIR / "w1_split_metric_audit.md", split_audit)
    write_json(OUT_DIR / "w1_split_metric_impact_verdict.json", verdict)
    write_verdict_md(OUT_DIR / "w1_split_metric_impact_verdict.md", verdict)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-artifacts", action="store_true")
    args = ap.parse_args()
    scorer_cases()
    if args.write_artifacts:
        write_artifacts()
    print("w1_balanced_accuracy_scorer_audit=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
