"""Project A — combine per-dataset audited summaries into ONE chance-normalized digest.

Raw balanced accuracy is NOT comparable across datasets with different class counts (a 4-class 0.40
and a binary 0.40 mean opposite things vs chance). So this combiner REFUSES to pool raw bAcc across
datasets and instead pools chance-normalized excess-over-chance metrics
(`(bAcc - 1/K)/(1 - 1/K)`), which are 0 at chance and 1 at perfect for any K. Within-dataset raw
bAcc is preserved for reference. Target metrics stay oracle/evaluation-only; not a SOTA claim.

  python -m h2cmi.observability.combine_summaries \
      --inputs a_summary.json b_summary.json --out-json combined.json --out-md combined.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load(p) -> dict:
    return json.loads(Path(p).read_text())


def _mean(vals) -> Optional[float]:
    v = [x for x in vals if isinstance(x, (int, float))]
    return round(statistics.mean(v), 4) if v else None


def _harm_rate(vals) -> Optional[float]:
    v = [x for x in vals if isinstance(x, (int, float))]
    return round(sum(1 for x in v if x < 0) / len(v), 4) if v else None


def combine(summaries: List[dict]) -> Dict[str, Any]:
    """Build the combined multi-dataset digest from a list of per-dataset summary dicts."""
    per_dataset: Dict[str, Any] = {}
    all_ok_runs: List[dict] = []
    nclass_values = set()
    for s in summaries:
        agg = s.get("aggregate", {})
        val = s.get("validation", {})
        ds = s.get("dataset") or agg.get("dataset") or f"dataset_{len(per_dataset)}"
        ok_runs = [r for r in s.get("runs", []) if r.get("status") == "ok"]
        nc = agg.get("n_classes")
        nclass_values.add(nc)
        per_dataset[ds] = {
            "n_classes": nc,
            "chance_bacc": agg.get("chance_bacc"),
            "n_runs": agg.get("n_runs"), "n_ok": agg.get("n_ok"), "n_skipped": agg.get("n_skipped"),
            "all_valid": val.get("all_valid"),
            "all_forbidden_violations_empty": agg.get("all_forbidden_violations_empty"),
            "all_target_metrics_oracle_only": agg.get("all_target_metrics_oracle_only"),
            "all_target_metrics_identifiable_null": agg.get("all_target_metrics_identifiable_null"),
            "all_prior_claims_compliant": agg.get("all_prior_claims_compliant"),
            "no_unknown_estimands": agg.get("no_unknown_estimands"),
            "missing_cells": agg.get("missing_cells", []),
            # within-dataset raw bAcc kept for reference (NOT pooled across datasets)
            "mean_strict_dg_bacc": agg.get("mean_strict_dg_bacc"),
            # the cross-dataset-comparable numbers
            "mean_strict_dg_bacc_excess_norm": agg.get("mean_strict_dg_bacc_excess_norm"),
            "mean_offline_tta_gain_bacc_norm": agg.get("mean_offline_tta_gain_bacc_norm"),
            "offline_tta_harm_rate": (agg.get("overall") or {}).get("offline_tta_harm_rate"),
        }
        all_ok_runs.extend(ok_runs)

    known_nc = {n for n in nclass_values if isinstance(n, int)}
    mixed_n_classes = len(known_nc) > 1

    overall_normalized = {
        "n_ok": len(all_ok_runs),
        "mean_strict_dg_bacc_excess_norm":
            _mean([r.get("strict_dg_bacc_excess_norm") for r in all_ok_runs]),
        "mean_online_tta_bacc_excess_norm":
            _mean([r.get("online_tta_bacc_excess_norm") for r in all_ok_runs]),
        "mean_offline_tta_gain_bacc_norm":
            _mean([r.get("offline_tta_gain_bacc_norm") for r in all_ok_runs]),
        "offline_tta_harm_rate": _harm_rate([r.get("offline_tta_gain_bacc") for r in all_ok_runs]),
    }

    # A dataset with NO ok runs (e.g. cache-absent -> all legal skips) asserts no target metrics, so
    # its (fail-closed) boundary flags are neutral, NOT a violation — exclude it from the boolean
    # roll-up. It must still be VALID (a legal all-skip grid is valid), so all_datasets_valid uses
    # every dataset.
    active = [pd for pd in per_dataset.values() if (pd.get("n_ok") or 0) > 0]

    def _flag_active(key):
        vals = [pd.get(key) for pd in active]
        return all(v is True for v in vals) if vals else False

    out = {
        "project": "Project A", "step": "Step 10",
        "scope": "multi-dataset audited expansion; not SOTA",
        "datasets": sorted(per_dataset),
        "n_datasets": len(per_dataset),
        "n_datasets_with_ok_runs": len(active),
        "per_dataset": per_dataset,
        "overall_normalized": overall_normalized,
        # HARD RULE: raw bAcc is never pooled across datasets (always suppressed; certainly so when
        # class counts differ). Cross-dataset comparison uses normalized excess only.
        "raw_bacc_overall_suppressed": True,
        "mixed_n_classes": mixed_n_classes,
        "any_dataset_missing_cells": any(bool(pd["missing_cells"]) for pd in per_dataset.values()),
        "any_forbidden_violations":
            any(pd["all_forbidden_violations_empty"] is False for pd in active),
        "all_datasets_valid": all(pd.get("all_valid") is True for pd in per_dataset.values())
                              if per_dataset else False,
        "all_target_metrics_identifiable_null": _flag_active("all_target_metrics_identifiable_null"),
        "all_target_metrics_oracle_only": _flag_active("all_target_metrics_oracle_only"),
        "all_prior_claims_compliant": _flag_active("all_prior_claims_compliant"),
        "no_unknown_estimands": _flag_active("no_unknown_estimands"),
        "claim_boundary": (
            "Cross-dataset aggregates use chance-normalized excess ((bAcc-1/K)/(1-1/K)); raw bAcc is "
            "never pooled across datasets with different class counts. Target metrics remain "
            "oracle/evaluation-only. Not a SOTA claim."),
    }
    return out


def write_md(combined: Dict[str, Any], path) -> str:
    lines = [f"# {combined['step']} — multi-dataset audited summary (chance-normalized)", "",
             f"Scope: {combined['scope']}. Target metrics are oracle/evaluation-only.", "",
             f"- datasets: **{', '.join(combined['datasets'])}**  ·  n_datasets: "
             f"**{combined['n_datasets']}**",
             f"- mixed n_classes: **{combined['mixed_n_classes']}**  ·  raw-bAcc overall suppressed: "
             f"**{combined['raw_bacc_overall_suppressed']}**",
             f"- all datasets valid: **{combined['all_datasets_valid']}**  ·  any missing cells: "
             f"**{combined['any_dataset_missing_cells']}**  ·  any forbidden violations: "
             f"**{combined['any_forbidden_violations']}**",
             f"- all target metrics identifiable=null: "
             f"**{combined['all_target_metrics_identifiable_null']}**", "",
             "## Overall (chance-normalized — the cross-dataset numbers)", "",
             f"- mean strict-DG excess-norm: **{combined['overall_normalized']['mean_strict_dg_bacc_excess_norm']}**",
             f"- mean offline-TTA gain-norm: **{combined['overall_normalized']['mean_offline_tta_gain_bacc_norm']}**",
             f"- offline-TTA harm-rate: **{combined['overall_normalized']['offline_tta_harm_rate']}**  "
             f"(over **{combined['overall_normalized']['n_ok']}** ok runs)", "",
             "## Per dataset", "",
             "| dataset | K | n_ok | n_skip | valid | id=null | raw bAcc (within) | strict "
             "excess-norm | offline gain-norm | harm-rate | missing |",
             "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|"]
    for ds in combined["datasets"]:
        pd = combined["per_dataset"][ds]
        lines.append(
            f"| {ds} | {pd['n_classes']} | {pd['n_ok']} | {pd['n_skipped']} | {pd['all_valid']} | "
            f"{pd['all_target_metrics_identifiable_null']} | {_fmt(pd['mean_strict_dg_bacc'])} | "
            f"{_fmt(pd['mean_strict_dg_bacc_excess_norm'])} | "
            f"{_fmt(pd['mean_offline_tta_gain_bacc_norm'])} | {_fmt(pd['offline_tta_harm_rate'])} | "
            f"{len(pd['missing_cells'])} |")
    lines += ["", "> " + combined["claim_boundary"]]
    text = "\n".join(lines) + "\n"
    Path(path).write_bytes(text.encode("utf-8"))               # force LF (hygiene; py3.9-safe)
    return text


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Combine per-dataset audited summaries (normalized)")
    ap.add_argument("--inputs", nargs="+", required=True, help="per-dataset summary JSON paths")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)

    summaries = [_load(p) for p in args.inputs]
    combined = combine(summaries)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_bytes((json.dumps(combined, indent=2) + "\n").encode("utf-8"))
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_md(combined, args.out_md)

    o = combined["overall_normalized"]
    print(f"datasets={combined['datasets']} n_datasets={combined['n_datasets']} "
          f"mixed_n_classes={combined['mixed_n_classes']} "
          f"all_datasets_valid={combined['all_datasets_valid']} "
          f"any_missing={combined['any_dataset_missing_cells']} "
          f"any_forbidden={combined['any_forbidden_violations']} "
          f"overall_strict_excess_norm={o['mean_strict_dg_bacc_excess_norm']} "
          f"overall_offline_gain_norm={o['mean_offline_tta_gain_bacc_norm']} "
          f"overall_harm_rate={o['offline_tta_harm_rate']}")
    # combined digest is descriptive; exit non-zero only if a constituent dataset failed validation
    return 0 if (combined["all_datasets_valid"] and not combined["any_forbidden_violations"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
