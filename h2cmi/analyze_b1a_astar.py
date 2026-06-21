"""Stage-B1b A*: the pre-registered, one-shot falsification test for a target-only router, using
the NESTED canonical-site null (run_nested_null.py) to calibrate eligibility and the EXISTING
real-target signals (run_b1a_router_signals.py) to evaluate it.

Calibration (review):
  * NOT exact conformal -- a nested source-calibrated empirical-null test.
  * OTHER-SEED: a target unit of seed s is judged with nested nulls from seeds != s only.
  * Per action a, from the other-seed null scores: q50_a, q90_a -> null-standardized excess
        Z_a(S) = (S - q50_a) / (q90_a - q50_a + eps).
  * Empirical MAX-NULL (handles multi-action dependence, no Bonferroni): for each null unit
        M_j = max_a Z_a(S_{a,j});  tau_s = Q0.9({M_j: other seeds}).
    This directly targets the frozen 10% false-adaptation criterion.

Two routers ONLY (no C, no SPD/rotation/CMI, no learned gate):
  N1  nested-null only:    eligible iff max_a Z_a > tau; select argmax_a Z_a; else identity.
  N2  nested-null + B veto: as N1 but the chosen action must also EXCEED the other-seed nested-null
      q90 of direction-cosine, effect-to-noise and prediction-stability (B is a VETO, not ranking).

Pass criteria are frozen; the N1-vs-N2 selection rule is fixed here BEFORE running. Aggregation
clusters by data seed (the 60 nested units are NOT 60 independent repeats).

  python -m h2cmi.analyze_b1a_astar \
      --real results/h2cmi/b1a_router_signals_standard.jsonl results/h2cmi/b1a_router_signals_hard.jsonl \
      --nested results/h2cmi/nested_null_standard.jsonl results/h2cmi/nested_null_hard.jsonl \
      --out results/h2cmi/b1a_astar.report.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

IDENTITY = "identity"
ACTIONS = ("pooled_empirical_diag", "gen_oneshot_diag", "gen_iterative_diag")
NULL_SCENARIOS = ("population_null", "matched_domain_null")
EPS = 1e-8
PASS = dict(false_adapt_max=0.10, hardnull_harm_max=0.10, hardnull_dbacc_min=-0.005,
            disagreement_max=0.02, nonnull_harm_max=0.20, top1_min=0.50, regret_max=0.02, coverage_min=0.25)


def _fin(x):
    return x is not None and x == x


def _q(v, p):
    v = np.asarray([x for x in v if _fin(x)], float)
    return float(np.quantile(v, p)) if len(v) else float("nan")


def _nested_units(nested_rows):
    u = defaultdict(dict)
    for r in nested_rows:
        u[(r["data_seed"], r["excluded_site_pair"], r["pseudo_target_site"])][r["action"]] = r
    return u


def _calibration(nested_units, seeds_use):
    """Per-action q50/q90 of evidence + q90 of B signals, and the max-null tau, from seeds_use."""
    units = [vm for (s, p, ps), vm in nested_units.items() if s in seeds_use]
    cal = {}
    for a in ACTIONS:
        ev = [vm[a]["raw_evidence_score"] for vm in units if a in vm]
        cal[a] = dict(q50=_q(ev, 0.5), q90=_q(ev, 0.9),
                      veto_cos=_q([vm[a]["transform_direction_cosine"] for vm in units if a in vm], 0.9),
                      veto_etn=_q([vm[a]["transform_effect_to_noise_ratio"] for vm in units if a in vm], 0.9),
                      veto_pred=_q([1.0 - vm[a]["crossfit_prediction_disagreement"] for vm in units if a in vm], 0.9))
    Z = lambda a, S: (S - cal[a]["q50"]) / (cal[a]["q90"] - cal[a]["q50"] + EPS) if _fin(S) else float("-inf")
    M = [max(Z(a, vm[a]["raw_evidence_score"]) for a in ACTIONS if a in vm) for vm in units]
    tau = _q(M, 0.9)
    return cal, tau, Z


def route(real_vm, cal, tau, Z, router) -> str:
    z = {a: Z(a, real_vm[a]["evidence_target"]) for a in ACTIONS if a in real_vm}
    elig = {a: v for a, v in z.items() if v > tau}
    if router == "N2":
        elig = {a: v for a, v in elig.items()
                if real_vm[a].get("transform_direction_cosine", -9) > cal[a]["veto_cos"]
                and real_vm[a].get("transform_effect_to_noise_ratio", -9) > cal[a]["veto_etn"]
                and (1.0 - real_vm[a].get("crossfit_prediction_disagreement", 1.0)) > cal[a]["veto_pred"]}
    return max(elig, key=elig.get) if elig else IDENTITY


def _real_units(real_rows):
    u = defaultdict(dict)
    for r in real_rows:
        u[(r.get("difficulty", "standard"), r["scenario"], r["data_seed"], r["target_site"])][r["action"]] = r
    return u


def _router_metrics(real_units, nested_units, router) -> dict:
    seeds = sorted({k[2] for k in real_units})
    rows = []
    for (diff, scen, seed, site), vm in real_units.items():
        others = [s for s in {kk[0] for kk in nested_units} if s != seed]   # other-seed nested
        if not others:
            others = sorted({kk[0] for kk in nested_units})
        cal, tau, Z = _calibration(nested_units, set(others))
        sel = route(vm, cal, tau, Z, router)
        idf = next(iter(vm.values()))["identity_bacc_uniform"]
        ido = next(iter(vm.values()))["identity_grouped_oof_bacc"]
        d_full = 0.0 if sel == IDENTITY else (vm[sel]["bacc_uniform"] - idf)
        d_oof = 0.0 if sel == IDENTITY else (vm[sel]["grouped_oof_bacc"] - ido if _fin(vm[sel].get("grouped_oof_bacc")) else float("nan"))
        cand = {IDENTITY: idf}; cand.update({a: vm[a]["bacc_uniform"] for a in vm if _fin(vm[a].get("bacc_uniform"))})
        oa = max(cand, key=cand.get)
        rows.append(dict(diff=diff, scen=scen, seed=seed, site=site, sel=sel, adapt=sel != IDENTITY,
                         d_full=d_full, d_oof=d_oof, is_null=scen in NULL_SCENARIOS,
                         top1=sel == oa, regret=cand[oa] - cand.get(sel, idf),
                         dis=vm[sel].get("crossfit_prediction_disagreement") if sel != IDENTITY else 0.0))
    nulls = [r for r in rows if r["is_null"]]; shift = [r for r in rows if not r["is_null"]]
    def m(xs, k, default=float("nan")):
        v = [x[k] for x in xs if _fin(x[k])]
        return float(np.mean(v)) if v else default
    res = dict(n_units=len(rows), adaptation_rate=m(rows, "adapt"), coverage=m(rows, "adapt"),
               false_adaptation_rate_null=m(nulls, "adapt"),
               harm_rate_full=float(np.mean([r["d_full"] < -1e-9 for r in rows if _fin(r["d_full"])])) if rows else float("nan"),
               nonnull_harm_rate=float(np.mean([r["d_full"] < -1e-9 for r in shift])) if shift else float("nan"),
               mean_dbacc_full_shift=m(shift, "d_full"), top1_oracle_full=m(rows, "top1"),
               mean_regret_full=m(rows, "regret"),
               mean_selected_disagreement=float(np.mean([r["dis"] for r in rows if r["adapt"] and _fin(r["dis"])]) or 0.0)
               if any(r["adapt"] for r in rows) else 0.0,
               hardnull_harm_full=float(np.mean([r["d_full"] < -1e-9 for r in nulls])) if nulls else float("nan"),
               hardnull_mean_dbacc_full=m(nulls, "d_full"),
               selection_frequency_shift={a: sum(r["sel"] == a for r in shift) for a in (IDENTITY,) + ACTIONS})
    return res


def _pass(std, hard) -> dict:
    out = {}
    if std:
        out.update(std_false_adapt=std["false_adaptation_rate_null"] <= PASS["false_adapt_max"],
                   shift_utility=std["mean_dbacc_full_shift"] > 0, nonnull_harm=std["nonnull_harm_rate"] <= PASS["nonnull_harm_max"],
                   top1=std["top1_oracle_full"] >= PASS["top1_min"], regret=std["mean_regret_full"] <= PASS["regret_max"],
                   coverage=std["coverage"] >= PASS["coverage_min"])
    if hard:
        out.update(hard_false_adapt=hard["false_adaptation_rate_null"] <= PASS["false_adapt_max"],
                   hard_harm=hard["hardnull_harm_full"] <= PASS["hardnull_harm_max"],
                   hard_dbacc=hard["hardnull_mean_dbacc_full"] >= PASS["hardnull_dbacc_min"],
                   hard_disagreement=hard["mean_selected_disagreement"] <= PASS["disagreement_max"])
    out["ALL"] = bool(out) and all(out.values())
    return out


def analyze(real_rows, nested_rows) -> dict:
    real_u = _real_units(real_rows)
    nested_by_diff = defaultdict(list)
    for r in nested_rows:
        nested_by_diff[r.get("difficulty", "standard")].append(r)
    rep = {"pass_criteria": PASS, "by_router": {}}
    for router in ("N1", "N2"):
        per = {}
        for diff in sorted({k[0] for k in real_u}):
            ru = {k: v for k, v in real_u.items() if k[0] == diff}
            nu = _nested_units(nested_by_diff.get(diff, []))
            if not nu:
                continue
            per[diff] = _router_metrics(ru, nu, router)
        per["pass"] = _pass(per.get("standard"), per.get("hard"))
        rep["by_router"][router] = per
    rep["decision"] = _decision(rep)
    return rep


def _decision(rep) -> dict:
    n1, n2 = rep["by_router"].get("N1", {}), rep["by_router"].get("N2", {})
    p1, p2 = n1.get("pass", {}).get("ALL", False), n2.get("pass", {}).get("ALL", False)
    g1 = n1.get("standard", {}).get("mean_dbacc_full_shift", float("nan"))
    g2 = n2.get("standard", {}).get("mean_dbacc_full_shift", float("nan"))
    if not p1 and not p2:
        return dict(outcome="A_STAR_FAIL -> pivot to B (metadata-conditioned)", chosen=None)
    if p1 and not p2:
        return dict(outcome="A_STAR_PASS", chosen="N1")
    if p2 and not p1:
        return dict(outcome="A_STAR_PASS", chosen="N2")
    chosen = "N2" if (_fin(g1) and _fin(g2) and g2 >= g1 - 0.01) else "N1"
    return dict(outcome="A_STAR_PASS", chosen=chosen)


def _load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", nargs="+", required=True)
    ap.add_argument("--nested", nargs="+", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rep = analyze(_load(args.real), _load(args.nested))
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=2)
    for R, per in rep["by_router"].items():
        print(f"=== {R} (pass={per['pass'].get('ALL')}) ===")
        for diff in ("standard", "hard"):
            if diff in per:
                d = per[diff]
                print(f"  {diff}: false_adapt_null={d['false_adaptation_rate_null']:.2f} cov={d['coverage']:.2f} "
                      f"ΔbAcc_shift={d['mean_dbacc_full_shift']:+.3f} harm={d['nonnull_harm_rate'] if d['nonnull_harm_rate']==d['nonnull_harm_rate'] else float('nan'):.2f} "
                      f"top1={d['top1_oracle_full']:.2f} regret={d['mean_regret_full']:.3f} "
                      f"hard_harm={d['hardnull_harm_full'] if d['hardnull_harm_full']==d['hardnull_harm_full'] else float('nan'):.2f}")
    print(f"DECISION: {rep['decision']}")
    if args.out:
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
