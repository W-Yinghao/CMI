"""Project A — read audited run directories into a TRACKED summary digest.

Each audited run dir (produced by `h2cmi.run_real_audited`) holds `raw_results.json`,
`observability_report.json/.md`, `run_manifest.json` (kept in gitignored `results/`). This
module extracts a compact per-run summary + aggregate that CAN be committed
(`notes/project_A_observability/results_summaries/`), so a reviewer can check every run's claim
boundary WITHOUT reading the raw training outputs.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import Estimand

# estimand kinds the audit treats as reportable TARGET metrics (must be oracle/eval-only)
_TARGET_METRIC_ESTIMANDS = {"balanced_accuracy", "target_gain", "target_risk"}
# the full known estimand vocabulary — an unknown/relabelled estimand is itself a red flag
KNOWN_ESTIMANDS = {e.value for e in Estimand}
# prior-claim statuses that are compliant (a rejected-but-flagged or identified-under-TU1 prior)
_COMPLIANT_PRIOR_STATUS = {"identified_TU1", "rejected_conclusion_false", "not_emitted"}


def _load_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def _dig(d: Optional[dict], *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _prior_status(claims: List[dict]) -> str:
    pri = [c for c in claims if c.get("estimand") == "target_prior"]
    if not pri:
        return "not_emitted"
    c = pri[0]
    if c.get("allowed"):
        return "identified_TU1"
    if c.get("conclusion") is True:                      # a rejected prior asserted as a conclusion
        return "rejected_conclusion_true"                # = an overclaim (non-compliant)
    if c.get("conclusion") is False:
        return "rejected_conclusion_false"               # reported-not-identified (compliant)
    return "rejected_unknown_conclusion"                 # missing conclusion flag (non-compliant)


def load_run(run_dir) -> Dict[str, Any]:
    """Compact, committable per-run summary + claim-boundary flags."""
    run_dir = Path(run_dir)
    manifest = _load_json(run_dir / "run_manifest.json") or {}
    report = _load_json(run_dir / "observability_report.json") or {}
    raw = _load_json(run_dir / "raw_results.json") or {}
    status = manifest.get("status", "unknown")
    out: Dict[str, Any] = {
        "run_dir": run_dir.name, "status": status, "dataset": manifest.get("dataset"),
        "target_subject": manifest.get("target_subject"), "seed": manifest.get("seed"),
        "align_factor": manifest.get("align_factor"), "skip_reason": manifest.get("skip_reason"),
    }
    if status != "ok":
        return out

    claims = report.get("claims", [])
    target_metrics = [c for c in claims if c.get("estimand") in _TARGET_METRIC_ESTIMANDS]
    out.update({
        "n_source_trials": manifest.get("n_source_trials"),
        "n_target_trials": manifest.get("n_target_trials"),
        "align_degenerate": manifest.get("alignment_factor_degenerate"),
        "strict_dg_bacc": _dig(raw, "strict_dg", "balanced_acc"),
        "offline_tta_gain_bacc": _dig(raw, "offline_tta", "delta_adapt", "d_balanced_acc"),
        "online_tta_bacc": _dig(raw, "online_tta", "balanced_acc"),
        "n_claims": _dig(report, "summary", "n_claims"),
        "n_allowed": _dig(report, "summary", "n_allowed"),
        "n_rejected": _dig(report, "summary", "n_rejected"),
        "forbidden_claims_violated": report.get("forbidden_claims_violated", []),
        # claim-boundary flags (the point of the digest)
        "all_r0_r1_target_metrics_oracle_only": bool(target_metrics) and all(
            c.get("reportable_metric") and c.get("identifiable_estimand") is None
            and c.get("oracle_fields_used_for_validation_only") for c in target_metrics),
        "all_target_metrics_identifiable_null": bool(target_metrics) and all(
            c.get("identifiable_estimand") is None for c in target_metrics),
        "target_prior_claim_status": _prior_status(claims),
        # any claim carrying an unknown/relabelled estimand is a red flag
        "unknown_estimands": sorted({c.get("estimand") for c in claims
                                     if c.get("estimand") not in KNOWN_ESTIMANDS}),
    })
    return out


def index_runs(root) -> List[Dict[str, Any]]:
    """Load every run subdirectory (one that holds a run_manifest.json) under `root`."""
    root = Path(root)
    dirs = sorted(p.parent for p in root.glob("*/run_manifest.json"))
    return [load_run(d) for d in dirs]


def _stats(vals) -> Dict[str, Any]:
    """Basic descriptive stats (mean/std/min/max) over the numeric values — NO inferential stats."""
    v = [x for x in vals if isinstance(x, (int, float))]
    if not v:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {"n": len(v), "mean": round(statistics.mean(v), 4),
            "std": round(statistics.pstdev(v), 4) if len(v) > 1 else 0.0,
            "min": round(min(v), 4), "max": round(max(v), 4)}


def _harm_rate(rs) -> Optional[float]:
    """P(offline-TTA gain < 0) over the completed runs in `rs`."""
    g = [r.get("offline_tta_gain_bacc") for r in rs
         if isinstance(r.get("offline_tta_gain_bacc"), (int, float))]
    return round(sum(1 for x in g if x < 0) / len(g), 4) if g else None


def _group_stats(ok: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in ok:
        groups.setdefault(str(r.get(key)), []).append(r)
    return {k: {"n_ok": len(rs),
                "strict_dg_bacc": _stats([x.get("strict_dg_bacc") for x in rs]),
                "offline_tta_gain_bacc": _stats([x.get("offline_tta_gain_bacc") for x in rs]),
                "online_tta_bacc": _stats([x.get("online_tta_bacc") for x in rs]),
                "offline_tta_harm_rate": _harm_rate(rs)}
            for k, rs in sorted(groups.items())}


def aggregate(runs: List[Dict[str, Any]], expected_targets=None,
              expected_seeds=None) -> Dict[str, Any]:
    ok = [r for r in runs if r["status"] == "ok"]
    skipped = [r for r in runs if r["status"] != "ok"]
    gains = [r.get("offline_tta_gain_bacc") for r in ok
             if isinstance(r.get("offline_tta_gain_bacc"), (int, float))]

    def _mean(key):
        vals = [r[key] for r in ok if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    agg = {
        "n_runs": len(runs), "n_ok": len(ok), "n_skipped": len(skipped),
        "mean_strict_dg_bacc": _mean("strict_dg_bacc"),
        "mean_offline_tta_gain_bacc": _mean("offline_tta_gain_bacc"),
        "mean_online_tta_bacc": _mean("online_tta_bacc"),
        "all_forbidden_violations_empty": all(not r.get("forbidden_claims_violated")
                                              for r in ok) if ok else False,
        "all_target_metrics_oracle_only": all(r.get("all_r0_r1_target_metrics_oracle_only")
                                              for r in ok) if ok else False,
        "all_target_metrics_identifiable_null": all(r.get("all_target_metrics_identifiable_null")
                                                    for r in ok) if ok else False,
        "all_prior_claims_compliant": all(r.get("target_prior_claim_status") in
                                          _COMPLIANT_PRIOR_STATUS for r in ok) if ok else False,
        "no_unknown_estimands": all(not r.get("unknown_estimands") for r in ok) if ok else False,
        # descriptive statistical layer (mean/std/min/max + harm-rate; no inferential stats)
        "overall": {
            "strict_dg_bacc": _stats([r.get("strict_dg_bacc") for r in ok]),
            "offline_tta_gain_bacc": _stats(gains),
            "online_tta_bacc": _stats([r.get("online_tta_bacc") for r in ok]),
            "offline_tta_harm_rate": round(sum(1 for g in gains if g < 0) / len(gains), 4)
                                     if gains else None,
            "n_offline_tta_gain_negative": sum(1 for g in gains if g < 0),
            "n_offline_tta_gain_positive": sum(1 for g in gains if g > 0),
            "n_offline_tta_gain_zero": sum(1 for g in gains if g == 0),
        },
        "per_target": _group_stats(ok, "target_subject"),
        "per_seed": _group_stats(ok, "seed"),
    }
    # expected-grid completeness (a MISSING cell — no dir at all — is not a legal skip)
    if expected_targets is not None and expected_seeds is not None:
        expected = {(str(t), str(s)) for t in expected_targets for s in expected_seeds}
        present = {(str(r.get("target_subject")), str(r.get("seed"))) for r in runs}
        agg["expected_runs"] = len(expected)
        agg["missing_cells"] = [{"target": t, "seed": s} for t, s in sorted(expected - present)]
    return agg


def build_summary(root, project="Project A", step="Step 8", dataset="BNCI2014_001",
                  expected_targets=None, expected_seeds=None) -> Dict[str, Any]:
    runs = index_runs(root)
    return {
        "project": project, "step": step, "dataset": dataset, "root": str(root),
        "n_runs": len(runs), "runs": runs,
        "aggregate": aggregate(runs, expected_targets, expected_seeds),
        "claim_boundary": ("These are evaluation-only target metrics, not R0/R1 identifiable "
                           "target risk or gain."),
    }


def write_summary_md(summary: Dict[str, Any], path) -> str:
    a = summary["aggregate"]
    lines = [f"# {summary['step']} {summary['dataset']} audited mini-grid summary", "",
             "Scope: interface + audited-report validation — **not a SOTA claim**; target "
             "metrics are oracle/evaluation-only.", "",
             f"- runs: **{a['n_runs']}**  ·  ok: **{a['n_ok']}**  ·  skipped: **{a['n_skipped']}**",
             f"- all forbidden-violations empty: **{a['all_forbidden_violations_empty']}**",
             f"- all target metrics oracle-only: **{a['all_target_metrics_oracle_only']}**",
             f"- all target metrics identifiable=null: **{a['all_target_metrics_identifiable_null']}**",
             f"- all prior claims compliant: **{a['all_prior_claims_compliant']}**  ·  "
             f"no unknown estimands: **{a['no_unknown_estimands']}**",
             f"- mean strict-DG bAcc: **{a['mean_strict_dg_bacc']}**  ·  mean offline-TTA gain: "
             f"**{a['mean_offline_tta_gain_bacc']}**  ·  offline-TTA harm-rate: "
             f"**{a['overall']['offline_tta_harm_rate']}**",
             f"- expected runs: **{a.get('expected_runs', '—')}**  ·  missing cells: "
             f"**{len(a['missing_cells']) if 'missing_cells' in a else '—'}**"
             + (f" {a['missing_cells']}" if a.get('missing_cells') else ""), "",
             "| target | seed | status | strict bAcc | offline gain | online bAcc | claims | "
             "violations | prior claim | metrics id=null |",
             "|---|---:|---|---:|---:|---:|---:|---:|---|---|"]
    for r in summary["runs"]:
        if r["status"] != "ok":
            lines.append(f"| {r.get('target_subject')} | {r.get('seed')} | "
                         f"⚠️ {r['status']} ({r.get('skip_reason','')}) | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {r['target_subject']} | {r['seed']} | ✅ ok | {_fmt(r['strict_dg_bacc'])} | "
            f"{_fmt(r['offline_tta_gain_bacc'])} | {_fmt(r['online_tta_bacc'])} | {r['n_claims']} | "
            f"{len(r['forbidden_claims_violated'])} | {r['target_prior_claim_status']} | "
            f"{r['all_target_metrics_identifiable_null']} |")
    # per-target descriptive block (stability across seeds without inferential stats)
    lines.append("")
    lines.append("## Per-target (mean ± std over seeds)")
    lines.append("")
    lines.append("| target | n_ok | strict-DG bAcc | offline-TTA gain | harm-rate |")
    lines.append("|---|---:|---|---|---:|")
    for t, st in summary["aggregate"].get("per_target", {}).items():
        s1, s2 = st["strict_dg_bacc"], st["offline_tta_gain_bacc"]
        lines.append(f"| {t} | {st['n_ok']} | {_fmt(s1['mean'])} ± {_fmt(s1['std'])} "
                     f"[{_fmt(s1['min'])},{_fmt(s1['max'])}] | {_fmt(s2['mean'])} ± {_fmt(s2['std'])} "
                     f"| {_fmt(st['offline_tta_harm_rate'])} |")
    lines.append("")
    lines.append("> " + summary["claim_boundary"])
    text = "\n".join(lines) + "\n"
    Path(path).write_bytes(text.encode("utf-8"))            # force LF (hygiene; py3.9-safe)
    return text


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"
