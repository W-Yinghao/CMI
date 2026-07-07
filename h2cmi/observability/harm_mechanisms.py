"""Project A Step 18 — TTA harm-mechanism decomposition (oracle / evaluation-only).

Step 17 showed accuracy-gain and balanced-accuracy-gain coincide on the current class-balanced grid, so
the estimand is not the lever. This module opens the harm up mechanistically: for each audited run it
decomposes the identity->adapt change into gain/loss of correct trials and per-class recall deltas, and
records the identity->adapt prediction transition matrix. The point is to see whether offline-TTA harm
is GLOBAL or CLASS-SPECIFIC, and whether some runs help one class while hurting another (which makes the
gain sign prior-dependent — see prior_stress.py).

Every quantity here uses the oracle per-trial target labels (y_true), so it is an
ORACLE / EVALUATION-ONLY mechanism decomposition. It identifies nothing under R0/R1; it makes no
adaptation claim; it never enters a deployable selector.

  python -m h2cmi.observability.harm_mechanisms --roots <dir> ... \
      --out-json step18_harm_mechanisms.json --out-md step18_harm_mechanisms.md
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .result_index import _load_json, write_json_lf, write_text_lf

_EPS = 1e-9


def _per_run(pt, manifest) -> Dict[str, Any]:
    import numpy as np
    y = np.asarray(pt["y_true"]); ip = np.asarray(pt["identity_pred"]); ap = np.asarray(pt["adapt_pred"])
    classes = list(np.unique(y)); K = len(classes)
    n = len(y)
    id_ok = (ip == y); ad_ok = (ap == y)
    lost = float((id_ok & ~ad_ok).mean())               # identity correct -> adapt wrong
    gained = float((~id_ok & ad_ok).mean())             # identity wrong  -> adapt correct
    unchanged_ok = float((id_ok & ad_ok).mean())
    unchanged_wrong = float((~id_ok & ~ad_ok).mean())
    acc_gain = float(ad_ok.mean() - id_ok.mean())
    per_class: Dict[str, Any] = {}
    recall_deltas = []
    for c in classes:
        m = (y == c); nc = int(m.sum())
        id_rec = float((ip[m] == c).mean()); ad_rec = float((ap[m] == c).mean())
        delta = ad_rec - id_rec
        recall_deltas.append(delta)
        wrong_ad = ap[m][ap[m] != c]                    # where adapt is wrong on true-class c
        dom = None
        if len(wrong_ad):
            j = Counter(int(x) for x in wrong_ad).most_common(1)[0][0]
            dom = f"{int(c)}->{j}"
        per_class[str(int(c))] = {
            "n": nc, "identity_recall": round(id_rec, 6), "adapt_recall": round(ad_rec, 6),
            "recall_delta": round(delta, 6),
            "lost_correct_rate": round(float((id_ok[m] & ~ad_ok[m]).mean()), 6),
            "gained_correct_rate": round(float((~id_ok[m] & ad_ok[m]).mean()), 6),
            "dominant_wrong_transition": dom}
    bacc_gain = float(np.mean(recall_deltas)) if recall_deltas else 0.0
    # identity->adapt prediction transition matrix (counts), classes in sorted order
    idx = {int(c): i for i, c in enumerate(classes)}
    tm = [[0] * K for _ in range(K)]
    for a, b in zip(ip, ap):
        tm[idx[int(a)]][idx[int(b)]] += 1
    # per true class: how many predictions moved, and to correct vs wrong
    tcpt = {}
    for c in classes:
        m = (y == c); moved = (ip[m] != ap[m])
        tcpt[str(int(c))] = {"n_moved": int(moved.sum()),
                             "moved_to_correct": int((moved & ad_ok[m]).sum()),
                             "moved_to_wrong": int((moved & ~ad_ok[m]).sum())}
    min_d, max_d = min(recall_deltas), max(recall_deltas)
    mixed = bool(min_d < -_EPS and max_d > _EPS)
    return {
        "dataset": manifest.get("dataset"), "target_subject": manifest.get("target_subject"),
        "seed": manifest.get("seed"), "n_classes": K, "n_trials": n,
        "identity_accuracy": round(float(id_ok.mean()), 6), "adapt_accuracy": round(float(ad_ok.mean()), 6),
        "accuracy_gain": round(acc_gain, 6),
        "identity_bacc": round(float(np.mean([per_class[str(int(c))]["identity_recall"] for c in classes])), 6),
        "adapt_bacc": round(float(np.mean([per_class[str(int(c))]["adapt_recall"] for c in classes])), 6),
        "bacc_gain": round(bacc_gain, 6),
        "lost_correct_rate": round(lost, 6), "gained_correct_rate": round(gained, 6),
        "unchanged_correct_rate": round(unchanged_ok, 6), "unchanged_wrong_rate": round(unchanged_wrong, 6),
        "net_gain_from_gain_loss": round(gained - lost, 6),
        "per_class": per_class,
        "prediction_transition_matrix_identity_to_adapt": tm,
        "transition_matrix_class_order": [int(c) for c in classes],
        "true_class_prediction_transition": tcpt,
        "harm_channel_summary": {
            "worst_class_by_recall_delta": int(classes[int(np.argmin(recall_deltas))]),
            "worst_class_recall_delta": round(min_d, 6),
            "best_class_by_recall_delta": int(classes[int(np.argmax(recall_deltas))]),
            "best_class_recall_delta": round(max_d, 6),
            "mixed_class_effects": mixed,
            "prior_dependent_possible": mixed},
        "oracle_evaluation_only": True,
        "claim_boundary": "oracle target labels used for mechanism decomposition only; not R0/R1 identifiable",
    }


def _extract(roots) -> List[Dict[str, Any]]:
    rows = []
    for root in roots:
        for mp in sorted(Path(root).glob("*/run_manifest.json")):
            m = _load_json(mp) or {}
            if m.get("status") != "ok":
                continue
            pt = (_load_json(mp.parent / "raw_results.json") or {}).get("per_trial_oracle_predictions") or {}
            if not (pt.get("y_true") and pt.get("identity_pred") and pt.get("adapt_pred")):
                continue
            rows.append(_per_run(pt, m))
    return rows


def build_summary(roots) -> Dict[str, Any]:
    import numpy as np
    rows = _extract(roots)
    n = len(rows)
    mixed = [r for r in rows if r["harm_channel_summary"]["mixed_class_effects"]]
    worst_by_ds: Dict[str, Any] = {}
    for ds in sorted({r["dataset"] for r in rows}):
        ws = [r["harm_channel_summary"]["worst_class_by_recall_delta"] for r in rows if r["dataset"] == ds]
        worst_by_ds[str(ds)] = {"most_common_worst_class": Counter(ws).most_common(1)[0][0] if ws else None,
                                "worst_class_histogram": dict(Counter(ws))}
    # dominant identity->adapt off-diagonal transitions aggregated over runs (by true-class label pair)
    dom = Counter(r["per_class"][c]["dominant_wrong_transition"]
                  for r in rows for c in r["per_class"]
                  if r["per_class"][c]["dominant_wrong_transition"])
    return {
        "project": "Project A", "step": "Step 18",
        "scope": "TTA harm-mechanism decomposition (oracle/evaluation-only); not SOTA",
        "n_runs": n,
        "mean_lost_correct_rate": round(float(np.mean([r["lost_correct_rate"] for r in rows])), 6) if n else None,
        "mean_gained_correct_rate": round(float(np.mean([r["gained_correct_rate"] for r in rows])), 6) if n else None,
        "mean_net_gain": round(float(np.mean([r["net_gain_from_gain_loss"] for r in rows])), 6) if n else None,
        "mean_accuracy_gain": round(float(np.mean([r["accuracy_gain"] for r in rows])), 6) if n else None,
        "runs_with_mixed_class_effects": len(mixed),
        "fraction_runs_with_mixed_class_effects": round(len(mixed) / n, 4) if n else None,
        "fraction_prior_dependent_possible": round(
            sum(r["harm_channel_summary"]["prior_dependent_possible"] for r in rows) / n, 4) if n else None,
        "worst_classes_by_dataset": worst_by_ds,
        "dominant_transition_patterns": [{"transition": k, "count": v} for k, v in dom.most_common(8)],
        "runs": rows,
        "oracle_labels_used_only_for_mechanism_and_evaluation": True,
        "claim_boundary_ok": True,
        "claim_boundary": ("harm-mechanism decomposition is oracle/evaluation-only; it identifies no "
                           "target functional under R0/R1 and makes no adaptation or SOTA claim."),
    }


def write_md(s: Dict[str, Any], path) -> str:
    lines = ["# Step 18 — TTA harm-mechanism decomposition", "",
             f"Scope: {s['scope']}. {s['claim_boundary']}", "",
             f"- runs: **{s['n_runs']}** · mean lost-correct **{s['mean_lost_correct_rate']}** · mean "
             f"gained-correct **{s['mean_gained_correct_rate']}** · mean net gain **{s['mean_net_gain']}**",
             f"- runs with mixed class effects: **{s['runs_with_mixed_class_effects']}** "
             f"(**{s['fraction_runs_with_mixed_class_effects']}**) · prior-dependent-possible fraction "
             f"**{s['fraction_prior_dependent_possible']}**", "",
             "| dataset | most-common worst class | worst-class histogram |", "|---|---|---|"]
    for ds, v in s["worst_classes_by_dataset"].items():
        lines.append(f"| {ds} | {v['most_common_worst_class']} | {v['worst_class_histogram']} |")
    lines += ["", "Dominant identity->adapt wrong transitions (true-class -> predicted):", ""]
    lines += [f"- `{d['transition']}` ×{d['count']}" for d in s["dominant_transition_patterns"]]
    lines += ["", "> " + s["claim_boundary"]]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A Step 18 harm-mechanism decomposition")
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)
    s = build_summary(args.roots)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        write_json_lf(args.out_json, s)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_md(s, args.out_md)
    print(f"harm_mechanisms n_runs={s['n_runs']} mean_lost={s['mean_lost_correct_rate']} "
          f"mean_gained={s['mean_gained_correct_rate']} mean_net_gain={s['mean_net_gain']} "
          f"mixed_frac={s['fraction_runs_with_mixed_class_effects']} "
          f"prior_dep_frac={s['fraction_prior_dependent_possible']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
