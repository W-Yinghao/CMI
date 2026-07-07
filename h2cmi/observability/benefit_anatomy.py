"""Project A Step 16 — oracle-only benefit anatomy.

Step 15 showed no deployable minimal-label policy safely adapts. This module asks WHERE the beneficial
cells are and whether they are stable, using the ORACLE full-target gain. Everything here is
oracle/evaluation-only: it is NOT observable under R0/R1 and is used only to characterise the problem,
never as a deployable signal or a predictor feature.

  python -m h2cmi.observability.benefit_anatomy --roots <dir> ... --out-json ... --out-md ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from .result_index import _load_json, write_json_lf, write_text_lf

_EPS = 0.005                                                   # |bAcc gain| <= EPS -> near-zero


def _bacc(y, pred) -> float:
    import numpy as np
    y = np.asarray(y); pred = np.asarray(pred)
    recalls = [float((pred[y == c] == c).mean()) for c in np.unique(y) if (y == c).any()]
    return float(np.mean(recalls)) if recalls else 0.0


def _extract(root) -> List[Dict[str, Any]]:
    rows = []
    for mp in sorted(Path(root).glob("*/run_manifest.json")):
        manifest = _load_json(mp) or {}
        if manifest.get("status") != "ok":
            continue
        pt = (_load_json(mp.parent / "raw_results.json") or {}).get("per_trial_oracle_predictions") or {}
        if not (pt.get("y_true") and pt.get("identity_pred") and pt.get("adapt_pred")):
            continue
        y, ip, ap = pt["y_true"], pt["identity_pred"], pt["adapt_pred"]
        import numpy as np
        id_bacc, ad_bacc = _bacc(y, ip), _bacc(y, ap)
        gain_bacc = round(ad_bacc - id_bacc, 6)
        gain_acc = round(float((np.asarray(ap) == np.asarray(y)).mean()
                               - (np.asarray(ip) == np.asarray(y)).mean()), 6)
        gbin = "benefit" if gain_bacc > _EPS else ("harm" if gain_bacc < -_EPS else "near_zero")
        rows.append({"dataset": manifest.get("dataset"), "target_subject": manifest.get("target_subject"),
                     "seed": manifest.get("seed"), "n_trials": len(y),
                     "identity_bacc": round(id_bacc, 6), "adapt_bacc": round(ad_bacc, 6),
                     "oracle_gain_bacc": gain_bacc, "oracle_gain_acc": gain_acc,
                     "beneficial": gain_bacc > _EPS, "harmful": gain_bacc < -_EPS,
                     "near_zero": abs(gain_bacc) <= _EPS, "gain_abs": round(abs(gain_bacc), 6),
                     "gain_bin": gbin, "oracle_only": True})
    return rows


def _dist(vals) -> Dict[str, Any]:
    import numpy as np
    v = np.asarray([x for x in vals if isinstance(x, (int, float))], dtype=float)
    if not len(v):
        return {"mean": None, "std": None, "min": None, "max": None, "q10": None, "q50": None, "q90": None}
    return {"mean": round(float(v.mean()), 6), "std": round(float(v.std()), 6),
            "min": round(float(v.min()), 6), "max": round(float(v.max()), 6),
            "q10": round(float(np.percentile(v, 10)), 6), "q50": round(float(np.percentile(v, 50)), 6),
            "q90": round(float(np.percentile(v, 90)), 6)}


def build_summary(roots: List[str]) -> Dict[str, Any]:
    rows = [r for root in roots for r in _extract(root)]
    n = len(rows)
    nb = sum(r["beneficial"] for r in rows)
    nh = sum(r["harmful"] for r in rows)
    nz = sum(r["near_zero"] for r in rows)
    # accuracy-gain benefit rate too: the harm-control / sequential policies optimise per-trial
    # ACCURACY gain, so report the benefit rate on THAT estimand alongside the bAcc one.
    nb_acc = sum(1 for r in rows if r["oracle_gain_acc"] > 0)

    def _grp(key):
        g: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            g.setdefault(str(r[key]), []).append(r)
        return g

    per_dataset = {k: {"n": len(rs), "n_beneficial": sum(x["beneficial"] for x in rs),
                       "benefit_rate": round(sum(x["beneficial"] for x in rs) / len(rs), 4)}
                   for k, rs in _grp("dataset").items()}
    per_seed = {k: {"n": len(rs), "benefit_rate": round(sum(x["beneficial"] for x in rs) / len(rs), 4)}
                for k, rs in sorted(_grp("seed").items())}
    # per (dataset, target): sign consistency across seeds
    tgt: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        tgt.setdefault(f"{r['dataset']}:{r['target_subject']}", []).append(r)
    per_target = {}
    for k, rs in sorted(tgt.items()):
        signs = {("benefit" if x["beneficial"] else ("harm" if x["harmful"] else "near_zero")) for x in rs}
        per_target[k] = {"n_seeds": len(rs), "benefit_count": sum(x["beneficial"] for x in rs),
                         "sign_consistent": len(signs) == 1,
                         "mean_gain_bacc": round(sum(x["oracle_gain_bacc"] for x in rs) / len(rs), 6)}
    n_consistent = sum(1 for v in per_target.values() if v["sign_consistent"])
    return {
        "project": "Project A", "step": "Step 16", "scope": "oracle-only benefit anatomy; not SOTA",
        "n_runs": n, "n_beneficial": nb, "n_harmful": nh, "n_near_zero": nz,
        "benefit_rate": round(nb / n, 4) if n else None,
        "benefit_estimand": "balanced_accuracy_gain (eps 0.005); policies use accuracy_gain",
        "n_beneficial_acc": nb_acc, "benefit_rate_acc": round(nb_acc / n, 4) if n else None,
        "per_dataset": per_dataset, "per_target": per_target, "per_seed": per_seed,
        "target_sign_consistency_rate": round(n_consistent / len(per_target), 4) if per_target else None,
        "gain_distribution_bacc": _dist([r["oracle_gain_bacc"] for r in rows]),
        "beneficial_gain_distribution_bacc": _dist([r["oracle_gain_bacc"] for r in rows if r["beneficial"]]),
        "runs": rows,
        "claim_boundary": ("oracle-only benefit anatomy; NOT deployment-observable under R0/R1; used "
                           "only to characterise the problem, never as a predictor or deployable signal."),
    }


def write_md(s: Dict[str, Any], path) -> str:
    bd = s["beneficial_gain_distribution_bacc"]
    lines = ["# Step 16 — oracle-only benefit anatomy", "",
             f"Scope: {s['scope']}. {s['claim_boundary']}", "",
             f"- runs: **{s['n_runs']}** · beneficial **{s['n_beneficial']}** · harmful "
             f"**{s['n_harmful']}** · near-zero **{s['n_near_zero']}** · benefit-rate **{s['benefit_rate']}**",
             f"- per dataset benefit-rate: **{ {k: v['benefit_rate'] for k, v in s['per_dataset'].items()} }**",
             f"- target sign-consistency rate (same sign across seeds): "
             f"**{s['target_sign_consistency_rate']}**",
             f"- beneficial gain dist (bAcc): mean **{bd['mean']}** q10 **{bd['q10']}** q50 **{bd['q50']}** "
             f"q90 **{bd['q90']}** max **{bd['max']}**", "",
             "| dataset:target | n_seeds | benefit_count | sign_consistent | mean_gain_bacc |",
             "|---|---:|---:|---|---:|"]
    for k, v in s["per_target"].items():
        lines.append(f"| {k} | {v['n_seeds']} | {v['benefit_count']} | {v['sign_consistent']} | "
                     f"{v['mean_gain_bacc']} |")
    lines += ["", "> " + s["claim_boundary"]]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A Step 16 oracle-only benefit anatomy")
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
    print(f"benefit_anatomy n_runs={s['n_runs']} benefit_rate={s['benefit_rate']} "
          f"n_beneficial={s['n_beneficial']} target_sign_consistency={s['target_sign_consistency_rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
