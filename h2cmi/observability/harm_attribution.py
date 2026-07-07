"""Project A Step 12 — offline-TTA harm attribution table.

Reads the gitignored audited run directories (Step 9 / Step 10 raw outputs) and extracts, per run,
a STRICTLY regime-separated feature record:

  * r0_features    — source-only diagnostics (source leakage, source pseudo-target gain);
  * r1_features    — target-UNLABELED diagnostics (target prior estimate, TTA transform/density
                     geometry, prediction disagreement) — computed from target X, never target y;
  * oracle_fields  — target metrics computed WITH oracle target labels (identity/adapt bAcc, the
                     offline-TTA gain, and the harm label). These are EVALUATION-ONLY: they are the
                     retrospective outcome to be explained, NEVER a predictor feature.

The oracle offline-TTA gain is NOT identifiable under R0/R1 (TOS-1 / TU-2); this table is a
retrospective attribution substrate, not an identifiability claim. Requested diagnostics that the
raw outputs do not contain are recorded in `missing_diagnostics` (reason-coded, never silently 0).

  python -m h2cmi.observability.harm_attribution --roots <dir> [<dir> ...] \
      --out-json step12_harm_attribution_table.json --out-md step12_harm_attribution_table.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .result_index import _load_json, write_json_lf, write_text_lf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# oracle-only keys (need target labels) — a hard denylist for any predictor feature matrix
ORACLE_KEYS = {"strict_dg_bacc", "offline_tta_bacc_identity", "offline_tta_bacc_adapt",
               "offline_tta_gain_bacc", "target_gain_bacc", "target_harmed", "offline_tta_harmed"}
# reviewer-requested R1 diagnostics the current raw outputs do NOT provide (recorded, not faked)
_UNAVAILABLE_R1 = ["tta_confidence_mean", "tta_entropy_mean", "target_support_proxy",
                   "target_marginal_shift_proxy"]


def _dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _num(x) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)) else None


def _mean(vals) -> Optional[float]:
    v = [x for x in vals if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))]
    return round(sum(v) / len(v), 6) if v else None


def _entropy(p) -> Optional[float]:
    if not isinstance(p, list) or not p:
        return None
    return round(-sum(x * math.log(x + 1e-12) for x in p if isinstance(x, (int, float))), 6)


def _l1_uniform(p) -> Optional[float]:
    if not isinstance(p, list) or not p:
        return None
    u = 1.0 / len(p)
    return round(sum(abs(x - u) for x in p if isinstance(x, (int, float))), 6)


def _r1_from_domains(raw) -> Dict[str, Optional[float]]:
    pi = _dig(raw, "offline_tta", "per_domain_pi_T") or {}
    diag = _dig(raw, "offline_tta", "per_domain_tta_diagnostics") or {}
    ents = [_entropy(v) for v in pi.values()]
    l1s = [_l1_uniform(v) for v in pi.values()]

    def dmean(field):
        return _mean([_dig(v, field) for v in diag.values()])

    return {
        "target_prior_entropy_hat": _mean(ents),
        "target_prior_shift_l1_hat": _mean(l1s),
        "tta_prior_shift_mean": dmean("prior_shift"),
        "tta_transform_norm_mean": dmean("transform_norm"),
        "tta_condition_number_mean": dmean("condition_number"),
        "tta_delta_density_nll_mean": dmean("delta_density_nll"),
        "tta_pred_disagreement_mean": dmean("pred_disagreement"),
    }


def extract_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    manifest = _load_json(run_dir / "run_manifest.json") or {}
    if manifest.get("status") != "ok":
        return None
    raw = _load_json(run_dir / "raw_results.json") or {}

    gain = _num(_dig(raw, "offline_tta", "delta_adapt", "d_balanced_acc"))
    oracle = {
        "strict_dg_bacc": _num(_dig(raw, "strict_dg", "balanced_acc")),
        "offline_tta_bacc_identity": _num(_dig(raw, "offline_tta", "identity", "balanced_acc")),
        "offline_tta_bacc_adapt": _num(_dig(raw, "offline_tta", "adapt", "balanced_acc")),
        "offline_tta_gain_bacc": gain,
        "target_gain_bacc": gain,
        "target_harmed": (gain < 0) if gain is not None else None,
    }
    r0 = {
        "source_leakage_site_I_hat": _num(_dig(raw, "leakage", "site", "I_hat")),
        "source_leakage_subject_I_hat": _num(_dig(raw, "leakage", "subject", "I_hat")),
        "source_leakage_session_I_hat": _num(_dig(raw, "leakage", "session", "I_hat")),
        "source_cond_dom_acc_subject": _num(_dig(raw, "leakage", "subject", "cond_dom_acc")),
        "source_mean_pseudo_gain": _num(_dig(raw, "gate_info", "mean_pseudo_gain")),
    }
    r1 = _r1_from_domains(raw)
    missing = [k for k in _UNAVAILABLE_R1] + [k for k, v in {**r0, **r1}.items() if v is None]

    return {
        "dataset": manifest.get("dataset"), "target_subject": manifest.get("target_subject"),
        "seed": manifest.get("seed"), "n_classes": manifest.get("n_classes"),
        "offline_tta_harmed": oracle["target_harmed"],
        "r0_features": r0, "r1_features": r1, "oracle_fields": oracle,
        "missing_diagnostics": sorted(set(missing)),
        "claim_boundary": {
            "oracle_target_gain_identifiable": False,
            "used_for_retrospective_evaluation_only": True,
            "note": "offline-TTA gain is oracle/evaluation-only (TOS-1/TU-2); not R0/R1 identifiable.",
        },
    }


def build_table(roots: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for root in roots:
        for mp in sorted(Path(root).glob("*/run_manifest.json")):
            rec = extract_run(mp.parent)
            if rec is not None:
                rows.append(rec)
    return rows


def write_md(rows, path) -> str:
    n = len(rows)
    harmed = sum(1 for r in rows if r["offline_tta_harmed"])
    lines = ["# Step 12 — offline-TTA harm attribution table", "",
             "Per-run regime-separated diagnostics. **Oracle gain/harm are evaluation-only "
             "(TOS-1/TU-2 non-identifiable), never predictor features.**", "",
             f"- runs: **{n}**  ·  harmed (gain<0): **{harmed}**  ·  harm-rate: "
             f"**{round(harmed / n, 4) if n else None}**",
             f"- feature regimes: R0 source-only · R1 target-unlabeled · oracle evaluation-only", "",
             "| dataset | tgt | seed | K | harmed | gain | src_leak_subj | src_pseudo_gain | "
             "tgt_prior_ent | tgt_prior_l1 | tta_transform | tta_disagree |",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        o, a, b = r["oracle_fields"], r["r0_features"], r["r1_features"]
        lines.append(
            f"| {r['dataset']} | {r['target_subject']} | {r['seed']} | {r['n_classes']} | "
            f"{'Y' if r['offline_tta_harmed'] else 'n'} | {_f(o['offline_tta_gain_bacc'])} | "
            f"{_f(a['source_leakage_subject_I_hat'])} | {_f(a['source_mean_pseudo_gain'])} | "
            f"{_f(b['target_prior_entropy_hat'])} | {_f(b['target_prior_shift_l1_hat'])} | "
            f"{_f(b['tta_transform_norm_mean'])} | {_f(b['tta_pred_disagreement_mean'])} |")
    miss = sorted({m for r in rows for m in r["missing_diagnostics"]})
    lines += ["", f"Missing/unavailable diagnostics (reason-coded, not faked): **{miss or 'none'}**",
              "", "> Oracle target gain is used ONLY as the retrospective outcome to explain; it is "
              "not an R0/R1 identified quantity and never a predictor feature."]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def _f(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A Step 12 harm-attribution table")
    ap.add_argument("--roots", nargs="+", required=True, help="audited run-dir roots to scan")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)

    rows = build_table(args.roots)
    n = len(rows)
    harmed = sum(1 for r in rows if r["offline_tta_harmed"])
    table = {
        "project": "Project A", "step": "Step 12", "scope": "TTA harm attribution; not SOTA",
        "n_runs": n, "n_harmed": harmed, "harm_rate": round(harmed / n, 4) if n else None,
        "oracle_denylist": sorted(ORACLE_KEYS),
        "claim_boundary": ("Oracle offline-TTA gain/harm are evaluation-only (TOS-1/TU-2 "
                           "non-identifiable under R0/R1) and must never appear as a predictor "
                           "feature. R0/R1 features are the only admissible predictors."),
        "runs": rows,
    }
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        write_json_lf(args.out_json, table)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_md(rows, args.out_md)
    print(f"harm_attribution runs={n} harmed={harmed} "
          f"harm_rate={round(harmed / n, 4) if n else None} "
          f"missing_kinds={sorted({m for r in rows for m in r['missing_diagnostics']})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
