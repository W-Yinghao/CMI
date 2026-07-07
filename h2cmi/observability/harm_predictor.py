"""Project A Step 12 — retrospective offline-TTA harm predictor (R0 vs R1).

Reads the harm-attribution table and asks a SCIENTIFIC sanity question: can source-only (R0)
diagnostics retrospectively predict which audited cells offline-TTA harmed, and does adding
target-unlabeled (R1) diagnostics help? Uses a low-freedom, interpretable model (standardized
logistic regression) with leave-one-(dataset, target-subject)-out CV.

HARD boundaries:
  * the oracle harm label is the PREDICTION TARGET only — it never enters the feature matrix
    (enforced against `harm_attribution.ORACLE_KEYS`);
  * a positive result is an EMPIRICAL RETROSPECTIVE predictor, NOT target-gain/harm identifiability
    (TOS-1/TU-2 stand); balanced accuracy is reported against the 0.5 majority baseline.

  python -m h2cmi.observability.harm_predictor --input step12_harm_attribution_table.json \
      --out-json step12_harm_predictor_summary.json --out-md step12_harm_predictor_summary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .harm_attribution import ORACLE_KEYS
from .result_index import _load_json, write_json_lf, write_text_lf

_RETRO_CLAIM = ("empirical retrospective predictor over audited cells; NOT target-gain/harm "
                "identifiability (TOS-1/TU-2 hold). Oracle harm is the target, never a feature.")


def _feature_names(rows, groups) -> List[str]:
    """Numeric feature columns present (non-null in >=1 row, not constant), from the given groups."""
    names = set()
    for r in rows:
        for g in groups:
            names |= {k for k, v in r[g].items() if isinstance(v, (int, float))}
    keep = []
    for n in sorted(names):
        assert n not in ORACLE_KEYS, f"oracle key {n} must never be a feature"
        vals = [_get(r, n) for r in rows if _get(r, n) is not None]
        if len(vals) >= 2 and len(set(vals)) > 1:            # drop all-null / constant columns
            keep.append(n)
    return keep


def _get(row, name) -> Optional[float]:
    for g in ("r0_features", "r1_features"):
        if name in row.get(g, {}):
            v = row[g][name]
            return float(v) if isinstance(v, (int, float)) else None
    return None


def _matrix(rows, names) -> List[List[float]]:
    # median-impute missing per column (train+test median is fine for a retrospective sanity check)
    med = {}
    for n in names:
        vals = sorted(v for v in (_get(r, n) for r in rows) if v is not None)
        med[n] = vals[len(vals) // 2] if vals else 0.0
    return [[(_get(r, n) if _get(r, n) is not None else med[n]) for n in names] for r in rows]


def _evaluate(rows, names) -> Dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(_matrix(rows, names), dtype=float)
    y = np.asarray([1 if r["offline_tta_harmed"] else 0 for r in rows], dtype=int)
    groups = np.asarray([f"{r['dataset']}::{r['target_subject']}" for r in rows])
    if not names or len(set(y)) < 2:
        return {"n_features": len(names), "features": names, "balanced_acc_harm_prediction": None,
                "auc": None, "note": "degenerate (no features or single-class outcome)",
                "claim": _RETRO_CLAIM}

    oof = np.full(len(y), -1.0)                               # out-of-fold P(harm)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(set(y[tr])) < 2:                              # single-class train fold -> predict prior
            oof[te] = y[tr].mean()
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    yhat = (oof >= 0.5).astype(int)
    try:
        auc = round(float(roc_auc_score(y, oof)), 4)
    except ValueError:
        auc = None
    return {"n_features": len(names), "features": names,
            "balanced_acc_harm_prediction": round(float(balanced_accuracy_score(y, yhat)), 4),
            "auc": auc, "claim": _RETRO_CLAIM}


def build_summary(table: Dict[str, Any]) -> Dict[str, Any]:
    rows = [r for r in table.get("runs", []) if r.get("offline_tta_harmed") is not None]
    n = len(rows)
    harmed = sum(1 for r in rows if r["offline_tta_harmed"])
    f0_names = _feature_names(rows, ["r0_features"])
    f1_names = _feature_names(rows, ["r0_features", "r1_features"])
    r0 = _evaluate(rows, f0_names)
    r1 = _evaluate(rows, f1_names)

    def _beats(b):                                            # > 0.5 majority baseline?
        v = b.get("balanced_acc_harm_prediction")
        b["beats_majority_baseline"] = (v is not None and v > 0.5)
        return b["beats_majority_baseline"]

    r0_beats, r1_beats = _beats(r0), _beats(r1)
    delta = (None if r0["balanced_acc_harm_prediction"] is None or r1["balanced_acc_harm_prediction"] is None
             else round(r1["balanced_acc_harm_prediction"] - r0["balanced_acc_harm_prediction"], 4))
    # honest verdict: below/at 0.5 = NO retrospective harm signal (not "a predictor")
    verdict = ("no_retrospective_harm_signal_above_baseline" if not (r0_beats or r1_beats)
               else ("r1_target_unlabeled_adds_retrospective_signal" if r1_beats and not r0_beats
                     else "retrospective_signal_present"))
    return {
        "project": "Project A", "step": "Step 12", "scope": "retrospective harm prediction; not SOTA",
        "n_runs": n, "n_harmed": harmed, "harm_rate": round(harmed / n, 4) if n else None,
        "majority_baseline_balanced_acc": 0.5,               # always-predict-harm -> bAcc 0.5
        "cv": "leave-one-(dataset,target_subject)-out logistic regression (class_weight=balanced)",
        "feature_sets": {"R0_source_only": r0, "R1_target_unlabeled": r1},
        "r1_minus_r0_balanced_acc_delta": delta,
        "any_predictor_beats_majority_baseline": bool(r0_beats or r1_beats),
        "verdict": verdict,
        "n_minority_class": min(harmed, n - harmed),         # power caveat (tiny minority = noisy LOTO)
        "oracle_never_a_feature": all(n not in ORACLE_KEYS for n in f1_names),
        "note_F2_k": ("F2(k) = F1 + a k-label target slice is studied on the controlled simulator in "
                      "minimal_paired.py, not here (real runs store no per-trial target labels)."),
        "claim_boundary": _RETRO_CLAIM,
    }


def write_md(s: Dict[str, Any], path) -> str:
    def blk(key):
        b = s["feature_sets"][key]
        return (f"- **{key}**: {b['n_features']} features · harm-pred balanced-acc "
                f"**{b['balanced_acc_harm_prediction']}** · AUC **{b['auc']}** · beats-baseline "
                f"**{b.get('beats_majority_baseline')}**")
    lines = ["# Step 12 — retrospective offline-TTA harm predictor", "",
             f"Scope: {s['scope']}. {s['claim_boundary']}", "",
             f"- runs: **{s['n_runs']}** · harm-rate **{s['harm_rate']}** · majority-baseline "
             f"balanced-acc **{s['majority_baseline_balanced_acc']}** · minority-class n "
             f"**{s['n_minority_class']}** (LOTO is noisy at this power)",
             f"- CV: {s['cv']}", "", blk("R0_source_only"), blk("R1_target_unlabeled"),
             f"- **R1 − R0 balanced-acc delta: {s['r1_minus_r0_balanced_acc_delta']}**",
             f"- **verdict: {s['verdict']}** · any predictor beats baseline: "
             f"**{s['any_predictor_beats_majority_baseline']}**",
             f"- oracle never a feature: **{s['oracle_never_a_feature']}**", "",
             f"> {s['note_F2_k']}", "",
             "> Balanced-acc near 0.5 = no retrospective signal beyond the majority class; any lift "
             "is an empirical retrospective predictor, not identifiability."]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A Step 12 retrospective harm predictor")
    ap.add_argument("--input", required=True, help="step12_harm_attribution_table.json")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)

    table = _load_json(Path(args.input))
    if table is None:
        raise SystemExit(f"could not read {args.input}")
    summary = build_summary(table)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        write_json_lf(args.out_json, summary)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_md(summary, args.out_md)
    fs = summary["feature_sets"]
    print(f"harm_predictor n={summary['n_runs']} harm_rate={summary['harm_rate']} "
          f"R0_bAcc={fs['R0_source_only']['balanced_acc_harm_prediction']} "
          f"R1_bAcc={fs['R1_target_unlabeled']['balanced_acc_harm_prediction']} "
          f"delta={summary['r1_minus_r0_balanced_acc_delta']} "
          f"oracle_never_feature={summary['oracle_never_a_feature']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
