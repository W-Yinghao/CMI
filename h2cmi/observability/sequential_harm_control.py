"""Project A Step 16 — sequential / budgeted label-acquisition harm-control policies.

Step 15 evaluated fixed-k one-shot policies. This module evaluates SEQUENTIAL policies that acquire
target labels batch by batch (up to a budget) and stop as soon as the paired-gain CI is decisive.

Boundary discipline (hard, tests enforce):
  * a sequential label-based policy uses NO labels to license adaptation at budget 0 / before its
    first batch -> it abstains (R1 non-identifiable);
  * k > 0 is an R2 labeled slice under an iid sampling contract, not full-target identification;
  * `oracle_full_target` is an evaluation-only upper bound (uses all target labels), NEVER deployable;
  * `budget = full` for a deployable policy is a full-label CALIBRATION policy (marked calibration
    burden = full), still not the oracle policy.

  python -m h2cmi.observability.sequential_harm_control --roots <dir> ... \
      --budgets 8 16 32 64 128 256 512 full --batch-size 8 --repeats 500 --taus 0.0 0.01 0.02 0.05 \
      --out-json ... --out-md ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .result_index import _load_json, write_json_lf, write_text_lf

_Z = 1.96
_SEQ_POLICIES = ["seq_ci_three_way", "seq_ci_adapt_only", "seq_plugin_confirm"]   # deployable
_HARM_CONSTRAINT = 0.05
_MIN_COVERAGE = 0.05
_CLAIM = ("R2 sequential labeled-slice policy under an iid sampling contract; NOT R1 target-gain "
          "identifiability; oracle_full_target is an evaluation-only upper bound, not deployable.")


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
            d = (ap == y).astype(float) - (ip == y).astype(float)
            fg = float(d.mean())
            runs.append({"d": d, "full_gain": fg, "full_harm": fg < 0, "full_benef": fg > 0,
                         "adapt_acc": float((ap == y).mean()), "identity_acc": float((ip == y).mean())})
    return runs


def _trajectory(d_perm, batch, z) -> List[Tuple[int, float, float, float]]:
    import numpy as np
    n = len(d_perm)
    csum = np.cumsum(d_perm); csq = np.cumsum(d_perm * d_perm)
    checks = []
    j = batch
    while j <= n:
        m = csum[j - 1] / j
        var = max((csq[j - 1] - j * m * m) / (j - 1), 0.0) if j >= 2 else 0.0
        se = (var / j) ** 0.5
        lo, hi = (-1.0, 1.0) if j < 2 else (m - z * se, m + z * se)
        checks.append((j, float(m), float(lo), float(hi)))
        j += batch
    return checks


def _seq_decide(policy, checks, tau, cap) -> Tuple[str, int]:
    prev_pos = prev_neg = False
    last = 0
    for labels, gh, lo, hi in checks:
        if labels > cap:
            break
        last = labels
        if policy in ("seq_ci_three_way", "seq_ci_adapt_only"):
            if lo > tau:
                return "adapt", labels
            if policy == "seq_ci_three_way" and hi < -tau:
                return "identity", labels
        elif policy == "seq_plugin_confirm":
            pos, neg = gh > tau, gh < -tau
            if pos and prev_pos:
                return "adapt", labels
            if neg and prev_neg:
                return "identity", labels
            prev_pos, prev_neg = pos, neg
    return "abstain", last                                     # budget exhausted -> abstain


def build_summary(roots, budgets, taus, repeats, batch=8, seed=0, z=_Z) -> Dict[str, Any]:
    import numpy as np
    runs = _load_runs(roots)
    rng = np.random.default_rng(int(seed))
    n_harm = sum(r["full_harm"] for r in runs); n_benef = sum(r["full_benef"] for r in runs)

    def _blank():
        return {"adapt": 0, "identity": 0, "abstain": 0, "adapt_on_harm": 0, "not_adapt_on_harm": 0,
                "not_adapt_on_benef": 0, "correct_nonabstain": 0, "nonabstain": 0, "labels": [],
                "total": 0, "harm_total": 0, "benef_total": 0}
    acc = {(p, str(b), t): _blank() for p in _SEQ_POLICIES for b in budgets for t in taus}

    for r in runs:
        d, n = r["d"], len(r["d"])
        oracle_best = "adapt" if r["full_gain"] > 0 else "identity"
        for _ in range(repeats):
            checks = _trajectory(d[rng.permutation(n)], batch, z)
            for b in budgets:
                cap = n if str(b) == "full" else min(int(b), n)
                for t in taus:
                    for p in _SEQ_POLICIES:
                        a = acc[(p, str(b), t)]
                        action, labels = _seq_decide(p, checks, t, cap)
                        a["total"] += 1; a[action] += 1; a["labels"].append(labels)
                        a["harm_total"] += int(r["full_harm"]); a["benef_total"] += int(r["full_benef"])
                        if r["full_harm"] and action == "adapt":
                            a["adapt_on_harm"] += 1
                        if r["full_harm"] and action != "adapt":
                            a["not_adapt_on_harm"] += 1
                        if r["full_benef"] and action != "adapt":
                            a["not_adapt_on_benef"] += 1
                        if action != "abstain":
                            a["nonabstain"] += 1
                            if action == oracle_best:
                                a["correct_nonabstain"] += 1

    cells = [_finalize(p, b, t, acc[(p, b, t)]) for (p, b, t) in acc]
    best = _select_best(cells)
    oref = _oracle_reference(runs, n_harm, n_benef)
    return {
        "project": "Project A", "step": "Step 16",
        "scope": "sequential label-acquisition harm-control (R2); not SOTA",
        "n_runs": len(runs), "budgets": [str(b) for b in budgets], "taus": taus, "batch_size": batch,
        "repeats": repeats, "sequential_policies": _SEQ_POLICIES,
        "harm_constraint": _HARM_CONSTRAINT, "min_coverage": _MIN_COVERAGE,
        "always_adapt_harm_rate": round(n_harm / len(runs), 4) if runs else None,
        "best_sequential_policy": best,
        "oracle_reference": oref,
        "oracle_policy_selected_as_deployable": False,
        "cells": cells,
        "claim_boundary_ok": True, "r2_iid_sampling_contract_required": True,
        "claim_boundary": _CLAIM,
    }


def _finalize(p, b, t, a) -> Dict[str, Any]:
    import numpy as np
    tot = max(1, a["total"]); adapt = a["adapt"]
    harm_rate = round(a["adapt_on_harm"] / adapt, 4) if adapt else None
    labels = a["labels"] or [0]
    return {
        "policy": p, "budget": b, "tau": t, "deployable": True,
        "adaptation_coverage": round(adapt / tot, 4),
        "decision_coverage": round((adapt + a["identity"]) / tot, 4),
        "abstention_rate": round(a["abstain"] / tot, 4),
        "mean_labels_used": round(float(np.mean(labels)), 2),
        "median_labels_used": float(np.median(labels)),
        "harm_rate_among_adapt_decisions": harm_rate,
        "prevented_harm_rate_vs_always_adapt":
            round(a["not_adapt_on_harm"] / a["harm_total"], 4) if a["harm_total"] else None,
        "missed_benefit_rate":
            round(a["not_adapt_on_benef"] / a["benef_total"], 4) if a["benef_total"] else None,
        "conditional_action_accuracy":
            round(a["correct_nonabstain"] / a["nonabstain"], 4) if a["nonabstain"] else None,
        "meets_harm_constraint_0_05": (harm_rate is not None and harm_rate <= _HARM_CONSTRAINT),
        "calibration_burden": "full" if b == "full" else "partial",
        "claim_boundary": _CLAIM,
    }


def _oracle_reference(runs, n_harm, n_benef) -> Dict[str, Any]:
    # eval-only upper bound: adapt beneficial, identity harmful -> harm 0, coverage = benefit rate
    n = len(runs)
    return {"adaptation_coverage": round(n_benef / n, 4) if n else None,
            "harm_rate_among_adapt_decisions": 0.0, "prevented_harm_rate_vs_always_adapt": 1.0,
            "missed_benefit_rate": 0.0, "labels_used": "full",
            "note": "full-label upper bound; NOT deployable"}


def _select_best(cells) -> Dict[str, Any]:
    elig = [c for c in cells if c["deployable"] and c["adaptation_coverage"] >= _MIN_COVERAGE
            and c["meets_harm_constraint_0_05"]]
    if not elig:
        return {"policy": None, "reason": f"no sequential policy meets harm<=0.05 and coverage>={_MIN_COVERAGE}"}
    best = min(elig, key=lambda c: (c["mean_labels_used"], -c["adaptation_coverage"],
                                    c["missed_benefit_rate"] if c["missed_benefit_rate"] is not None else 1.0))
    return {"policy": best["policy"], "budget": best["budget"], "tau": best["tau"],
            "mean_labels_used": best["mean_labels_used"], "adaptation_coverage": best["adaptation_coverage"],
            "harm_rate_among_adapt_decisions": best["harm_rate_among_adapt_decisions"],
            "missed_benefit_rate": best["missed_benefit_rate"]}


def write_md(s: Dict[str, Any], path) -> str:
    b = s["best_sequential_policy"]
    lines = ["# Step 16 — sequential label-acquisition harm-control", "",
             f"Scope: {s['scope']}. {s['claim_boundary']}", "",
             f"- runs: **{s['n_runs']}** · budgets **{s['budgets']}** · batch **{s['batch_size']}** · "
             f"repeats **{s['repeats']}** · always-adapt harm **{s['always_adapt_harm_rate']}**",
             f"- best sequential deployable policy: **{b.get('policy')}**"
             + (f" (budget {b.get('budget')}, tau {b.get('tau')}, mean-labels {b.get('mean_labels_used')}, "
                f"adapt-cov {b.get('adaptation_coverage')}, harm {b.get('harm_rate_among_adapt_decisions')})"
                if b.get('policy') else f" — {b.get('reason')}"),
             f"- oracle reference (eval-only): adapt-cov {s['oracle_reference']['adaptation_coverage']}, "
             f"harm {s['oracle_reference']['harm_rate_among_adapt_decisions']}", "",
             "| policy | budget | tau | adapt_cov | mean_labels | harm@adapt | meets<=0.05 | missed_benefit |",
             "|---|---|---:|---:|---:|---:|---|---:|"]
    tau0 = s["taus"][0] if s["taus"] else 0.0
    for c in s["cells"]:
        if c["tau"] == tau0:
            lines.append(f"| {c['policy']} | {c['budget']} | {c['tau']} | {c['adaptation_coverage']} | "
                         f"{c['mean_labels_used']} | {c['harm_rate_among_adapt_decisions']} | "
                         f"{c['meets_harm_constraint_0_05']} | {c['missed_benefit_rate']} |")
    lines += ["", f"> Table shows tau={tau0}; full grid in the JSON.", "> " + s["claim_boundary"]]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def _parse_budgets(vals):
    return [("full" if str(v).lower() == "full" else int(v)) for v in vals]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A Step 16 sequential harm-control")
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--budgets", nargs="+", default=[8, 16, 32, 64, 128, 256, 512, "full"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=500)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)
    s = build_summary(args.roots, _parse_budgets(args.budgets), args.taus, args.repeats, args.batch_size, args.seed)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        write_json_lf(args.out_json, s)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_md(s, args.out_md)
    b = s["best_sequential_policy"]
    print(f"sequential_harm_control n_runs={s['n_runs']} best_policy={b.get('policy')} "
          f"best_budget={b.get('budget')} mean_labels={b.get('mean_labels_used')} "
          f"adapt_cov={b.get('adaptation_coverage')} harm={b.get('harm_rate_among_adapt_decisions')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
