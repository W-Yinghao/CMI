"""Project A Step 15 — coverage-aware harm-control policies under minimal labels.

Reads Step-13 run dirs with `per_trial_oracle_predictions` and evaluates R2 minimal-label policies
that choose per target: adapt / identity / abstain. The point (from Step 14) is that a small labeled
slice gives HIGH-PRECISION but LOW-COVERAGE harm-sign calls, so a policy should adapt only when the
evidence is decisive and abstain / stay identity otherwise.

Boundary discipline (hard):
  * k = 0 is R1 non-identifiable: a label-based policy MUST abstain (it cannot license adapt from a
    target quantity it cannot observe);
  * k > 0 is an R2 labeled slice under an iid SAMPLING contract; the policy estimates a labeled-slice
    gain, NOT full target risk;
  * `oracle_full_target` is an evaluation-only upper bound and is NEVER selected as a deployable
    policy;
  * oracle target labels are used only inside the R2 slice / for evaluation, never as an R0/R1 feature.

  python -m h2cmi.observability.harm_control --roots <dir> ... --ks 0 1 2 4 8 16 32 64 128 256 \
      --repeats 500 --taus 0.0 0.01 0.02 0.05 --out-json ... --out-md ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from .result_index import _load_json, write_json_lf, write_text_lf

_Z = 1.96
_DEPLOYABLE = ["always_identity", "always_adapt", "plugin_sign", "ci_adapt_only_abstain",
               "ci_adapt_only_identity", "ci_three_way"]
# label-BASED deployable policies (exclude the trivial always_* benchmarks) — the real "can minimal
# labels safely select adaptation?" question lives here
_LABEL_POLICIES = ["plugin_sign", "ci_adapt_only_abstain", "ci_adapt_only_identity", "ci_three_way"]
_ALL_POLICIES = _DEPLOYABLE + ["oracle_full_target"]           # oracle = evaluation-only upper bound
_HARM_CONSTRAINT = 0.05
_CLAIM = ("R2 labeled-slice policy under an iid sampling contract; oracle full gain used only for "
          "evaluation; NOT R1 target-gain identifiability; oracle_full_target is not deployable.")


def _load_runs(roots: List[str]):
    import numpy as np
    runs = []
    for root in roots:
        for mp in sorted(Path(root).glob("*/run_manifest.json")):
            manifest = _load_json(mp) or {}
            if manifest.get("status") != "ok":
                continue
            pt = (_load_json(mp.parent / "raw_results.json") or {}).get("per_trial_oracle_predictions") or {}
            if not (pt.get("y_true") and pt.get("identity_pred") and pt.get("adapt_pred")):
                continue
            y = np.asarray(pt["y_true"]); ip = np.asarray(pt["identity_pred"]); ap = np.asarray(pt["adapt_pred"])
            d = (ap == y).astype(float) - (ip == y).astype(float)
            fg = float(d.mean())
            runs.append({"d": d, "full_gain": fg, "full_harm": fg < 0, "full_benef": fg > 0,
                         "identity_acc": float((ip == y).mean()), "adapt_acc": float((ap == y).mean()),
                         "oracle_best": "adapt" if fg > 0 else "identity"})
    return runs


def _decide(policy, k, gain_hat, ci_low, ci_high, tau, full_gain) -> str:
    if policy == "always_identity":
        return "identity"
    if policy == "always_adapt":
        return "adapt"
    if policy == "oracle_full_target":
        return "adapt" if full_gain > 0 else "identity"
    if k == 0:
        return "abstain"                                       # R1 non-identifiable for label policies
    if policy == "plugin_sign":
        return "adapt" if gain_hat > tau else "identity"
    if policy == "ci_adapt_only_abstain":
        return "adapt" if ci_low > tau else "abstain"
    if policy == "ci_adapt_only_identity":
        return "adapt" if ci_low > tau else "identity"
    if policy == "ci_three_way":
        if ci_low > tau:
            return "adapt"
        if ci_high < -tau:
            return "identity"
        return "abstain"
    raise ValueError(policy)


def _slice_ci(d_slice, z, method, B, rng):
    import numpy as np
    k = len(d_slice)
    gh = float(d_slice.mean())
    if k < 2:
        return gh, -1.0, 1.0                                   # not decisive at k<2
    if method == "bootstrap":
        means = np.array([rng.choice(d_slice, k, replace=True).mean() for _ in range(B)])
        return gh, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    se = float(d_slice.std(ddof=1)) / (k ** 0.5)               # paired normal-approx CI (default)
    return gh, gh - z * se, gh + z * se


def _blank(policy, k, tau):
    return {"policy": policy, "k": k, "tau": tau, "adapt": 0, "identity": 0, "abstain": 0,
            "adapt_on_harm": 0, "not_adapt_on_harm": 0, "not_adapt_on_benef": 0,
            "correct_nonabstain": 0, "nonabstain": 0, "oracle_acc_sum": 0.0, "total": 0,
            "harm_total": 0, "benef_total": 0}


def build_summary(roots, ks, taus, repeats, seed=0, z=_Z, ci_method="normal", bootstrap_B=200) -> Dict[str, Any]:
    import numpy as np
    runs = _load_runs(roots)
    rng = np.random.default_rng(int(seed))
    acc = {(p, k, t): _blank(p, k, t) for p in _ALL_POLICIES for k in ks for t in taus}

    for r in runs:
        d, n = r["d"], len(r["d"])
        for k in ks:
            for _ in range(repeats):
                if k == 0:
                    gh, lo, hi = 0.0, -1.0, 1.0
                else:
                    gh, lo, hi = _slice_ci(d[rng.choice(n, min(k, n), replace=False)], z, ci_method, bootstrap_B, rng)
                for t in taus:
                    for p in _ALL_POLICIES:
                        a = acc[(p, k, t)]
                        action = _decide(p, k, gh, lo, hi, t, r["full_gain"])
                        a["total"] += 1
                        a["harm_total"] += int(r["full_harm"]); a["benef_total"] += int(r["full_benef"])
                        a[action] += 1
                        if r["full_harm"] and action == "adapt":
                            a["adapt_on_harm"] += 1
                        if r["full_harm"] and action != "adapt":
                            a["not_adapt_on_harm"] += 1
                        if r["full_benef"] and action != "adapt":
                            a["not_adapt_on_benef"] += 1
                        if action != "abstain":
                            a["nonabstain"] += 1
                            a["oracle_acc_sum"] += r["adapt_acc"] if action == "adapt" else r["identity_acc"]
                            if action == r["oracle_best"]:
                                a["correct_nonabstain"] += 1

    cells = [_finalize_cell(a) for a in acc.values()]
    best = _select_best(cells)
    # interpretability: the oracle (full-label) upper bound + the best a deployable policy can do
    oref = next((c for c in cells if c["policy"] == "oracle_full_target"), {})
    lbl_adapt = [c for c in cells if c["policy"] in _LABEL_POLICIES and c["adaptation_coverage"] > 0]
    ci_attempt = max(lbl_adapt, key=lambda c: c["adaptation_coverage"]) if lbl_adapt else None
    return {
        "project": "Project A", "step": "Step 15",
        "scope": "coverage-aware harm-control policies (R2 minimal-label); not SOTA",
        "n_runs": len(runs), "ks": ks, "taus": taus, "repeats": repeats, "ci_method": ci_method,
        "policies": _ALL_POLICIES, "deployable_policies": _DEPLOYABLE,
        "harm_constraint": _HARM_CONSTRAINT,
        "always_adapt_harm_rate": _always_adapt_harm(cells),
        "best_deployable_policy": best,
        "oracle_reference": {"adaptation_coverage": oref.get("adaptation_coverage"),
                             "harm_rate_among_adapt_decisions": oref.get("harm_rate_among_adapt_decisions"),
                             "prevented_harm_rate_vs_always_adapt": oref.get("prevented_harm_rate_vs_always_adapt"),
                             "missed_benefit_rate": oref.get("missed_benefit_rate"),
                             "note": "full-label upper bound; NOT deployable"},
        "best_deployable_ci_attempt": None if ci_attempt is None else {
            "policy": ci_attempt["policy"], "k": ci_attempt["k"], "tau": ci_attempt["tau"],
            "adaptation_coverage": ci_attempt["adaptation_coverage"],
            "harm_rate_among_adapt_decisions": ci_attempt["harm_rate_among_adapt_decisions"]},
        "cells": cells,
        "claim_boundary_ok": True,
        "r2_iid_sampling_contract_required": True,
        "oracle_policy_selected_as_deployable": False,
        "claim_boundary": _CLAIM,
    }


def _finalize_cell(a) -> Dict[str, Any]:
    tot = max(1, a["total"])
    adapt = a["adapt"]
    return {
        "policy": a["policy"], "k": a["k"], "tau": a["tau"], "deployable": a["policy"] in _DEPLOYABLE,
        "adaptation_coverage": round(adapt / tot, 4),
        "decision_coverage": round((adapt + a["identity"]) / tot, 4),
        "abstention_rate": round(a["abstain"] / tot, 4),
        "harm_rate_among_adapt_decisions": round(a["adapt_on_harm"] / adapt, 4) if adapt else None,
        "prevented_harm_rate_vs_always_adapt":
            round(a["not_adapt_on_harm"] / a["harm_total"], 4) if a["harm_total"] else None,
        "missed_benefit_rate": round(a["not_adapt_on_benef"] / a["benef_total"], 4) if a["benef_total"] else None,
        "conditional_action_accuracy":
            round(a["correct_nonabstain"] / a["nonabstain"], 4) if a["nonabstain"] else None,
        "expected_oracle_acc_of_chosen_action":
            round(a["oracle_acc_sum"] / a["nonabstain"], 4) if a["nonabstain"] else None,
        "claim_boundary": _CLAIM,
    }


def _always_adapt_harm(cells) -> Optional[float]:
    for c in cells:
        if c["policy"] == "always_adapt":
            return c["harm_rate_among_adapt_decisions"]
    return None


def _select_best(cells) -> Dict[str, Any]:
    # predeclared rule: among DEPLOYABLE cells that actually adapt with harm<=0.05, maximize
    # adaptation_coverage, tie-break minimize missed_benefit_rate. Oracle is never eligible.
    elig = [c for c in cells if c["deployable"] and c["adaptation_coverage"] > 0
            and c["harm_rate_among_adapt_decisions"] is not None
            and c["harm_rate_among_adapt_decisions"] <= _HARM_CONSTRAINT]
    if not elig:
        return {"policy": None, "reason": "no deployable policy meets the harm<=0.05 constraint "
                "while adapting (adaptation_coverage>0)"}
    best = max(elig, key=lambda c: (c["adaptation_coverage"], -(c["missed_benefit_rate"] or 1.0)))
    return {"policy": best["policy"], "k": best["k"], "tau": best["tau"],
            "adaptation_coverage": best["adaptation_coverage"],
            "decision_coverage": best["decision_coverage"],
            "harm_rate_among_adapt_decisions": best["harm_rate_among_adapt_decisions"],
            "prevented_harm_rate_vs_always_adapt": best["prevented_harm_rate_vs_always_adapt"],
            "missed_benefit_rate": best["missed_benefit_rate"]}


def write_md(s: Dict[str, Any], path) -> str:
    b = s["best_deployable_policy"]
    lines = ["# Step 15 — coverage-aware harm-control policies", "",
             f"Scope: {s['scope']}. {s['claim_boundary']}", "",
             f"- runs: **{s['n_runs']}** · policies **{len(s['policies'])}** · ks **{s['ks']}** · "
             f"taus **{s['taus']}** · repeats **{s['repeats']}** · CI **{s['ci_method']}**",
             f"- always-adapt harm-rate: **{s['always_adapt_harm_rate']}** · harm constraint "
             f"**{s['harm_constraint']}**",
             f"- best deployable policy: **{b.get('policy')}**"
             + (f" (k={b.get('k')}, tau={b.get('tau')}, adapt-coverage {b.get('adaptation_coverage')}, "
                f"harm-among-adapt {b.get('harm_rate_among_adapt_decisions')}, prevented-harm "
                f"{b.get('prevented_harm_rate_vs_always_adapt')}, missed-benefit "
                f"{b.get('missed_benefit_rate')})" if b.get('policy') else f" — {b.get('reason')}"), "",
             "| policy | k | tau | adapt_cov | decision_cov | abstain | harm@adapt | prevented_harm | "
             "missed_benefit | cond_acc |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    # show a compact slice: each policy at a mid k and the smallest tau
    show_k = s["ks"][len(s["ks"]) // 2] if s["ks"] else 0
    show_tau = s["taus"][0] if s["taus"] else 0.0
    for c in s["cells"]:
        if c["k"] == show_k and c["tau"] == show_tau:
            lines.append(f"| {c['policy']} | {c['k']} | {c['tau']} | {c['adaptation_coverage']} | "
                         f"{c['decision_coverage']} | {c['abstention_rate']} | "
                         f"{c['harm_rate_among_adapt_decisions']} | {c['prevented_harm_rate_vs_always_adapt']} | "
                         f"{c['missed_benefit_rate']} | {c['conditional_action_accuracy']} |")
    lines += ["", f"> Table shows k={show_k}, tau={show_tau}; full grid in the JSON.",
              "> " + s["claim_boundary"]]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A Step 15 coverage-aware harm-control policies")
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05])
    ap.add_argument("--repeats", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ci-method", default="normal", choices=["normal", "bootstrap"])
    ap.add_argument("--bootstrap-B", type=int, default=200)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)
    s = build_summary(args.roots, args.ks, args.taus, args.repeats, args.seed,
                      ci_method=args.ci_method, bootstrap_B=args.bootstrap_B)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        write_json_lf(args.out_json, s)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_md(s, args.out_md)
    b = s["best_deployable_policy"]
    print(f"harm_control n_runs={s['n_runs']} always_adapt_harm={s['always_adapt_harm_rate']} "
          f"best_policy={b.get('policy')} best_k={b.get('k')} best_tau={b.get('tau')} "
          f"adapt_cov={b.get('adaptation_coverage')} harm@adapt={b.get('harm_rate_among_adapt_decisions')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
