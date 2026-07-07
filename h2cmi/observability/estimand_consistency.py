"""Project A Step 17 — estimand-consistent harm control (accuracy gain vs balanced-accuracy gain).

Step 16 surfaced a mismatch: the minimal-label policies decide on paired per-trial correctness deltas
(ordinary ACCURACY gain), while earlier audits reported BALANCED accuracy. This module evaluates
harm-control policies under BOTH estimands, kept strictly separate, under two sampling contracts:
  * iid label slice   — natural for accuracy; the bAcc estimate is undefined when a class is absent
    from a small slice (bacc_slice_status = missing_class -> the policy abstains);
  * class-balanced calibration slice (contract C13) — elicits ~k/K labels per class; abstains when
    k < n_classes (under_class_budget).

Hard rule (tests enforce): accuracy-gain control and bAcc-gain control are DIFFERENT target functionals;
a policy licensed for one is never reported as controlling the other. k=0 abstains (R1 non-identifiable);
k>0 is an R2 labeled slice under a sampling contract, not full-target identification.

  python -m h2cmi.observability.estimand_consistency --roots <dir> ... \
      --ks 0 1 2 4 8 16 32 64 128 256 full --repeats 500 --taus 0.0 0.01 0.02 0.05 \
      --sampling iid class_balanced --out-json ... --out-md ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .result_index import _load_json, write_json_lf, write_text_lf

_Z = 1.96
_POLICIES = ["plugin_sign", "ci_adapt_only", "ci_three_way"]
_ESTIMANDS = ["accuracy_gain", "balanced_accuracy_gain"]
_C13 = "C13"                                                    # class-balanced calibration design
_BENEFIT_EPS = 0.005                                            # Step-16 material-benefit threshold (for the magnitude probe)


def _load_runs(roots):
    import numpy as np
    runs = []
    for root in roots:
        for mp in sorted(Path(root).glob("*/run_manifest.json")):
            m = _load_json(mp) or {}
            if m.get("status") != "ok":
                continue
            pt = (_load_json(mp.parent / "raw_results.json") or {}).get("per_trial_oracle_predictions") or {}
            if not (pt.get("y_true") and pt.get("identity_pred") and pt.get("adapt_pred")):
                continue
            y = np.asarray(pt["y_true"]); ip = np.asarray(pt["identity_pred"]); ap = np.asarray(pt["adapt_pred"])
            d = (ap == y).astype(float) - (ip == y).astype(float)   # per-trial ACCURACY delta
            classes = np.unique(y)
            acc_gain = float(d.mean())
            recall_diffs = [float((ap[y == c] == c).mean() - (ip[y == c] == c).mean()) for c in classes]
            bacc_gain = float(np.mean(recall_diffs))
            counts = [int((y == c).sum()) for c in classes]
            runs.append({"y": y, "d": d, "classes": classes,
                         "idx_by_class": {int(c): np.where(y == c)[0] for c in classes},
                         "acc_gain": round(acc_gain, 6), "bacc_gain": round(bacc_gain, 6),
                         "acc_benef": acc_gain > 0, "bacc_benef": bacc_gain > 0,
                         "class_balanced": bool(min(counts) == max(counts))})
    return runs


def _sample(r, sampling, k, rng):
    import numpy as np
    n = len(r["d"]); classes = r["classes"]; kk = n if k == "full" else min(int(k), n)
    if sampling == "iid":
        if kk == 0:
            return None, None
        idx = rng.choice(n, kk, replace=False)
        return r["y"][idx], idx
    per = kk // len(classes)                                    # class_balanced (C13)
    if per < 1:
        return None, None                                      # under_class_budget
    idx = np.concatenate([rng.choice(r["idx_by_class"][int(c)], min(per, len(r["idx_by_class"][int(c)])),
                                     replace=False) for c in classes])
    return r["y"][idx], idx


def _acc_ci(d_slice, z):
    kk = len(d_slice); gh = float(d_slice.mean())
    se = float(d_slice.std(ddof=1)) / (kk ** 0.5) if kk >= 2 else 0.0
    return gh, (-1.0, 1.0) if kk < 2 else (gh - z * se, gh + z * se)


def _bacc_ci(r, idx, z):
    import numpy as np
    ys = r["y"][idx]; ds = r["d"][idx]
    if len(np.unique(ys)) < len(r["classes"]):
        return None, None, "missing_class"
    means, var_terms = [], []
    for c in r["classes"]:
        dc = ds[ys == c]
        means.append(float(dc.mean()))
        var_terms.append(float(dc.var(ddof=1)) / len(dc) if len(dc) >= 2 else 0.0)
    K = len(r["classes"]); gh = float(np.mean(means))
    se = (sum(var_terms) / (K * K)) ** 0.5
    return gh, (gh - z * se, gh + z * se), "ok"


def _decide(policy, gh, lo, hi, tau) -> str:
    if policy == "plugin_sign":
        return "adapt" if gh > tau else "identity"
    if policy == "ci_adapt_only":
        return "adapt" if lo > tau else "abstain"
    if policy == "ci_three_way":
        if lo > tau:
            return "adapt"
        if hi < -tau:
            return "identity"
        return "abstain"
    raise ValueError(policy)


def _blank():
    return {"adapt": 0, "identity": 0, "abstain": 0, "adapt_on_harm": 0, "not_adapt_on_benef": 0,
            "correct_nonabstain": 0, "nonabstain": 0, "total": 0, "harm_total": 0, "benef_total": 0,
            "missing_class": 0}


def build_summary(roots, ks, taus, repeats, samplings, seed=0, z=_Z) -> Dict[str, Any]:
    import numpy as np
    runs = _load_runs(roots)
    rng = np.random.default_rng(int(seed))
    acc = {(e, s, str(k), t, p): _blank()
           for e in _ESTIMANDS for s in samplings for k in ks for t in taus for p in _POLICIES}

    for r in runs:
        best = {"accuracy_gain": "adapt" if r["acc_benef"] else "identity",
                "balanced_accuracy_gain": "adapt" if r["bacc_benef"] else "identity"}
        harm = {"accuracy_gain": not r["acc_benef"], "balanced_accuracy_gain": not r["bacc_benef"]}
        for s in samplings:
            for k in ks:
                for _ in range(repeats):
                    if k == 0:
                        est_ci = {e: (0.0, -1.0, 1.0, "k0") for e in _ESTIMANDS}
                    else:
                        _ys, idx = _sample(r, s, k, rng)
                        if idx is None:
                            est_ci = {e: (None, None, None, "under_class_budget") for e in _ESTIMANDS}
                        else:
                            gh_a, (lo_a, hi_a) = _acc_ci(r["d"][idx], z)
                            gh_b, ci_b, st_b = _bacc_ci(r, idx, z)
                            est_ci = {"accuracy_gain": (gh_a, lo_a, hi_a, "ok"),
                                      "balanced_accuracy_gain":
                                          (gh_b, ci_b[0] if ci_b else None, ci_b[1] if ci_b else None, st_b)}
                    for e in _ESTIMANDS:
                        gh, lo, hi, status = est_ci[e]
                        for t in taus:
                            for p in _POLICIES:
                                a = acc[(e, s, str(k), t, p)]
                                a["total"] += 1
                                a["harm_total"] += int(harm[e]); a["benef_total"] += int(not harm[e])
                                if status in ("missing_class", "under_class_budget", "k0") or gh is None:
                                    a["abstain"] += 1
                                    if status == "missing_class":
                                        a["missing_class"] += 1
                                    continue
                                action = _decide(p, gh, lo, hi, t)
                                a[action] += 1
                                if harm[e] and action == "adapt":
                                    a["adapt_on_harm"] += 1
                                if (not harm[e]) and action != "adapt":
                                    a["not_adapt_on_benef"] += 1
                                if action != "abstain":
                                    a["nonabstain"] += 1
                                    if action == best[e]:
                                        a["correct_nonabstain"] += 1

    cells = [_finalize(e, s, k, t, p, acc[(e, s, k, t, p)]) for (e, s, k, t, p) in acc]
    cross = _cross_estimand(runs)
    return {
        "project": "Project A", "step": "Step 17",
        "scope": "estimand-consistent harm control (accuracy vs balanced-accuracy); not SOTA",
        "n_runs": len(runs), "ks": [str(k) for k in ks], "taus": taus, "samplings": samplings,
        "policies": _POLICIES, "estimands": _ESTIMANDS, "repeats": repeats,
        "class_balanced_requires_contract": _C13,
        "accuracy_policy_controls_bacc": False,               # HARD: different target functionals
        **cross, "cells": cells,
        "claim_boundary_ok": True, "r2_iid_sampling_contract_required": True,
        "claim_boundary": ("Accuracy-gain and balanced-accuracy-gain are DIFFERENT target functionals; "
                           "a policy licensed for one does NOT control the other. k>0 is an R2 labeled "
                           "slice under a sampling contract (class-balanced bAcc requires C13); NOT R1 "
                           "target-gain identifiability."),
    }


def _finalize(e, s, k, t, p, a) -> Dict[str, Any]:
    tot = max(1, a["total"]); adapt = a["adapt"]
    return {
        "estimand": e, "sampling": s, "k": k, "tau": t, "policy": p,
        "requires_contract": _C13 if (s == "class_balanced" and e == "balanced_accuracy_gain") else None,
        "adaptation_coverage": round(adapt / tot, 4),
        "decision_coverage": round((adapt + a["identity"]) / tot, 4),
        "abstention_rate": round(a["abstain"] / tot, 4),
        "missing_class_rate": round(a["missing_class"] / tot, 4),
        "harm_rate_among_adapt_decisions": round(a["adapt_on_harm"] / adapt, 4) if adapt else None,
        "missed_benefit_rate": round(a["not_adapt_on_benef"] / a["benef_total"], 4) if a["benef_total"] else None,
        "conditional_action_accuracy": round(a["correct_nonabstain"] / a["nonabstain"], 4) if a["nonabstain"] else None,
    }


def _cross_estimand(runs) -> Dict[str, Any]:
    import numpy as np
    n = len(runs)
    acc_b = sum(r["acc_benef"] for r in runs)
    bacc_b = sum(r["bacc_benef"] for r in runs)
    agree = sum(1 for r in runs if (r["acc_gain"] > 0) == (r["bacc_gain"] > 0))
    a_ben_b_harm = sum(1 for r in runs if r["acc_gain"] > 0 and r["bacc_gain"] < 0)
    b_ben_a_harm = sum(1 for r in runs if r["bacc_gain"] > 0 and r["acc_gain"] < 0)
    # threshold/magnitude probe: reproduce the Step-16 material-benefit rates (eps=0.005) so we can tell
    # a SIGN disagreement from a MAGNITUDE difference from an inconsistent-THRESHOLD reporting artifact.
    acc_b_eps = sum(1 for r in runs if r["acc_gain"] > _BENEFIT_EPS)
    bacc_b_eps = sum(1 for r in runs if r["bacc_gain"] > _BENEFIT_EPS)
    ag = np.asarray([r["acc_gain"] for r in runs]) if n else np.asarray([0.0])
    bg = np.asarray([r["bacc_gain"] for r in runs]) if n else np.asarray([0.0])
    max_gap = round(float(np.max(np.abs(ag - bg))), 6) if n else 0.0
    all_balanced = bool(n and all(r["class_balanced"] for r in runs))
    identical = bool(max_gap <= 1e-9)                            # per-run acc_gain == bacc_gain everywhere
    if a_ben_b_harm + b_ben_a_harm > 0:
        rel = "sign_disagreement"
    elif identical:
        rel = "identical_on_grid"                               # e.g. class-balanced targets -> bAcc == accuracy
    else:
        rel = "sign_agree_magnitude_differs"
    if identical:
        expl = (f"On this grid all {n} target sets are class-balanced ({all_balanced}); accuracy-gain == "
                f"balanced-accuracy-gain per run (max |diff| = {max_gap}). The Step-16 0.1481-vs-0.0926 "
                f"gap was a THRESHOLD artifact — accuracy benefit thresholded at eps=0 vs bAcc at "
                f"eps={_BENEFIT_EPS}. At a shared threshold the rates coincide: eps=0 -> "
                f"{round(acc_b / n, 4)} (acc) / {round(bacc_b / n, 4)} (bAcc); eps={_BENEFIT_EPS} -> "
                f"{round(acc_b_eps / n, 4)} / {round(bacc_b_eps / n, 4)}.")
    elif rel == "sign_agree_magnitude_differs":
        expl = (f"accuracy-gain and balanced-accuracy-gain agree on sign for every run but differ in "
                f"magnitude (max |diff| = {max_gap}); the eps={_BENEFIT_EPS} material-benefit rates split "
                f"{round(acc_b_eps / n, 4)} (acc) vs {round(bacc_b_eps / n, 4)} (bAcc).")
    else:
        expl = (f"accuracy-gain and balanced-accuracy-gain disagree on SIGN for "
                f"{a_ben_b_harm + b_ben_a_harm} run(s); they are genuinely different functionals here.")
    return {"accuracy_benefit_rate": round(acc_b / n, 4) if n else None,
            "bacc_benefit_rate": round(bacc_b / n, 4) if n else None,
            "cross_estimand_sign_agreement": round(agree / n, 4) if n else None,
            "runs_accuracy_benefit_bacc_harm": a_ben_b_harm,
            "runs_bacc_benefit_accuracy_harm": b_ben_a_harm,
            "benefit_eps": _BENEFIT_EPS,
            "accuracy_material_benefit_rate_eps": round(acc_b_eps / n, 4) if n else None,
            "bacc_material_benefit_rate_eps": round(bacc_b_eps / n, 4) if n else None,
            "mean_accuracy_gain": round(float(ag.mean()), 6) if n else None,
            "mean_bacc_gain": round(float(bg.mean()), 6) if n else None,
            "median_accuracy_gain": round(float(np.median(ag)), 6) if n else None,
            "median_bacc_gain": round(float(np.median(bg)), 6) if n else None,
            "max_abs_gain_difference": max_gap,
            "all_targets_class_balanced": all_balanced,
            "estimands_identical_on_grid": identical,
            "estimand_relationship": rel,
            "estimand_gap_is_sign_disagreement": bool(a_ben_b_harm + b_ben_a_harm > 0),
            "estimand_gap_is_magnitude_only": bool(rel == "sign_agree_magnitude_differs"),
            "step16_gap_explanation": expl}


def write_md(s: Dict[str, Any], path) -> str:
    lines = ["# Step 17 — estimand-consistent harm control", "",
             f"Scope: {s['scope']}. {s['claim_boundary']}", "",
             f"- runs: **{s['n_runs']}** · accuracy-benefit-rate **{s['accuracy_benefit_rate']}** · "
             f"bAcc-benefit-rate **{s['bacc_benefit_rate']}** · sign-agreement "
             f"**{s['cross_estimand_sign_agreement']}**",
             f"- runs accuracy-benefit but bAcc-harm: **{s['runs_accuracy_benefit_bacc_harm']}** · "
             f"bAcc-benefit but accuracy-harm: **{s['runs_bacc_benefit_accuracy_harm']}**",
             f"- accuracy policy controls bAcc: **{s['accuracy_policy_controls_bacc']}** · "
             f"class-balanced bAcc requires **{s['class_balanced_requires_contract']}**", "",
             "| estimand | sampling | k | policy | adapt_cov | harm@adapt | missing_class | missed_benefit |",
             "|---|---|---|---|---:|---:|---:|---:|"]
    for c in s["cells"]:
        if c["tau"] == (s["taus"][0] if s["taus"] else 0.0) and c["policy"] == "ci_three_way" \
                and c["k"] in ("32", "full"):
            lines.append(f"| {c['estimand']} | {c['sampling']} | {c['k']} | {c['policy']} | "
                         f"{c['adaptation_coverage']} | {c['harm_rate_among_adapt_decisions']} | "
                         f"{c['missing_class_rate']} | {c['missed_benefit_rate']} |")
    lines += ["", "> Table shows ci_three_way at k in {32, full}, first tau; full grid in the JSON.",
              "> " + s["claim_boundary"]]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def _parse_ks(vals):
    return [("full" if str(v).lower() == "full" else int(v)) for v in vals]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A Step 17 estimand-consistent harm control")
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--ks", nargs="+", default=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, "full"])
    ap.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05])
    ap.add_argument("--repeats", type=int, default=500)
    ap.add_argument("--sampling", nargs="+", default=["iid", "class_balanced"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)
    s = build_summary(args.roots, _parse_ks(args.ks), args.taus, args.repeats, args.sampling, args.seed)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        write_json_lf(args.out_json, s)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_md(s, args.out_md)
    print(f"estimand_consistency n_runs={s['n_runs']} acc_benefit={s['accuracy_benefit_rate']} "
          f"bacc_benefit={s['bacc_benefit_rate']} sign_agreement={s['cross_estimand_sign_agreement']} "
          f"acc_ben_bacc_harm={s['runs_accuracy_benefit_bacc_harm']} "
          f"bacc_ben_acc_harm={s['runs_bacc_benefit_accuracy_harm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
