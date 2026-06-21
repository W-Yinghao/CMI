"""Analysis for the shift-grid JSONL (review §"分析", Stage-A corrected).

Corrections over v1:

1. The statistical cluster is the SEED, not (seed, target_site). Within a seed the five LOSO
   folds share 3/4 of their source training data, so (seed,site) units are NOT independent.
   We average the five sites within each seed to a seed-level delta, then bootstrap / sign-flip
   over seeds. Site-level spread is reported only as heterogeneity. (Unit-level numbers are
   kept under ``exploratory`` and clearly labelled.)

2. Four cell means M0..M3 and FIVE contrasts are reported, not just the interaction:
       strict_cmi   = M1 - M0          (CMI effect with no TTA)
       tta_no_cmi   = M2 - M0          (TTA effect without CMI)
       tta_with_cmi = M3 - M1          (TTA effect with CMI)
       cmi_under_tta= M3 - M2          (CMI effect under TTA)
       interaction  = (M3-M1)-(M2-M0)
   plus  m3_minus_max = M3 - max(M0,M1,M2)  (does the full combo beat ALL its parts?).
   A positive interaction does NOT imply M3>M2; both are required to claim "CMI helps TTA".

3. ``P>0`` is renamed ``bootstrap_mass_above_zero`` (it is NOT a p-value). A two-sided
   sign-flip permutation p over seeds is reported alongside (exact for small seed counts;
   with 3 seeds the minimum two-sided p is 0.25 -> screening cannot be confirmatory).

4. Prior-type scenarios are summarised with Delta NLL / Brier / ECE / prior-L1, since balanced
   accuracy deliberately discounts class proportion and is not the primary prior-adaptation metric.

5. Canonical scenario names: no_shift->population_null, concept->conditional_rotation,
   cov_concept->cov_conditional_rotation (old JSONL is mapped on load). The oracle ceiling uses
   ``oracle_supervised_oof`` when present; otherwise the transductive ``oracle_supervised`` is
   used and flagged (its delta_bacc equals oracle_labels by construction -> not a held-out ceiling).
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict

import numpy as np

EPS_HELP = 0.02
GAIN_TOL = 0.10
ADAPT_WARN = 0.50
SAFETY_HARM_GAP = 0.05      # min harm-rate reduction to claim a CMI safety effect

SCENARIO_ALIASES = {
    "no_shift": "population_null",
    "concept": "conditional_rotation",
    "cov_concept": "cov_conditional_rotation",
}


def canon(s: str) -> str:
    return SCENARIO_ALIASES.get(s, s)


def load(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                r["scenario"] = canon(r["scenario"])
                rows.append(r)
    return rows


def _by(rows, **eq):
    return [r for r in rows if all(r.get(k) == v for k, v in eq.items())]


def _unit_map(rows, scenario, method, cmi):
    return {(r["data_seed"], r["target_site"]): r
            for r in _by(rows, scenario=scenario, method=method, cmi=cmi)}


# ----------------------------------------------------------------- cells & contrasts
def cells(rows, scenario) -> dict:
    """(seed,site) -> {M0,M1,M2,M3} absolute balanced-accuracy cells."""
    m = {(meth, cmi): _unit_map(rows, scenario, meth, cmi)
         for meth in ("identity", "tta") for cmi in ("off", "on")}
    units = set(m[("identity", "off")])
    for k in m:
        units &= set(m[k])
    out = {}
    for u in sorted(units):
        out[u] = dict(M0=m[("identity", "off")][u]["adapted_bacc"],
                      M1=m[("identity", "on")][u]["adapted_bacc"],
                      M2=m[("tta", "off")][u]["adapted_bacc"],
                      M3=m[("tta", "on")][u]["adapted_bacc"])
    return out


CONTRASTS = {
    "strict_cmi":    lambda c: c["M1"] - c["M0"],
    "tta_no_cmi":    lambda c: c["M2"] - c["M0"],
    "tta_with_cmi":  lambda c: c["M3"] - c["M1"],
    "cmi_under_tta": lambda c: c["M3"] - c["M2"],
    "interaction":   lambda c: (c["M3"] - c["M1"]) - (c["M2"] - c["M0"]),
    "m3_minus_max":  lambda c: c["M3"] - max(c["M0"], c["M1"], c["M2"]),
}


def _seed_cluster(per_unit: dict, n_boot=2000, seed=0) -> dict:
    """Seed-clustered summary: average sites within a seed, then bootstrap/sign-flip seeds."""
    by_seed = defaultdict(list)
    for (s, t), v in per_unit.items():
        by_seed[s].append(v)
    seeds = sorted(by_seed)
    delta = np.array([float(np.mean(by_seed[s])) for s in seeds])      # seed-level deltas
    n = len(delta)
    if n == 0:
        return dict(seed_mean=float("nan"), n_seeds=0)
    obs = float(delta.mean())
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(delta, n, replace=True).mean() for _ in range(n_boot)]) if n > 1 \
        else np.array([obs])
    # exact two-sided sign-flip permutation over seeds (symmetry null)
    perm = [float(np.mean([e * d for e, d in zip(sgn, delta)]))
            for sgn in itertools.product([-1, 1], repeat=n)]
    p_two = float(np.mean([abs(m) >= abs(obs) - 1e-12 for m in perm]))
    by_site = defaultdict(list)
    for (s, t), v in per_unit.items():
        by_site[t].append(v)
    return dict(seed_mean=obs, n_seeds=n,
                seed_ci_lo=float(np.quantile(boots, 0.025)),
                seed_ci_hi=float(np.quantile(boots, 0.975)),
                bootstrap_mass_above_zero=float((boots > 0).mean()),
                signflip_p_two_sided=p_two,
                per_seed={int(s): float(np.mean(by_seed[s])) for s in seeds},
                per_site_heterogeneity={int(t): float(np.mean(v)) for t, v in by_site.items()})


def scenario_contrasts(rows, scenario, n_boot, seed) -> dict:
    cl = cells(rows, scenario)
    out = {"n_units": len(cl), "n_seeds": len({s for s, _ in cl})}
    for name, fn in CONTRASTS.items():
        per_unit = {u: fn(c) for u, c in cl.items()}
        out[name] = _seed_cluster(per_unit, n_boot, seed)
    # harm rate (unit-level, exploratory) of the TTA arms
    d_off = [c["M2"] - c["M0"] for c in cl.values()]
    d_on = [c["M3"] - c["M1"] for c in cl.values()]
    out["harm_rate_off"] = float(np.mean(np.array(d_off) < 0)) if d_off else float("nan")
    out["harm_rate_on"] = float(np.mean(np.array(d_on) < 0)) if d_on else float("nan")
    return out


# ----------------------------------------------------------------- oracle table
def _mean_delta(rows, scenario, method):
    vals = [r["delta_bacc"] for r in _by(rows, scenario=scenario, method=method)]
    return float(np.mean(vals)) if vals else float("nan")


def _mean_field(rows, scenario, method, field):
    vals = [r[field] for r in _by(rows, scenario=scenario, method=method)
            if field in r and r[field] == r[field]]
    return float(np.mean(vals)) if vals else float("nan")


def _adapt_rate(rows, scenario, method="tta"):
    vals = [bool(r.get("adapted")) for r in _by(rows, scenario=scenario, method=method)]
    return float(np.mean(vals)) if vals else float("nan")


def scenario_aggregate(rows, scenario) -> dict:
    has_oof = bool(_by(rows, scenario=scenario, method="oracle_supervised_oof"))
    sup_method = "oracle_supervised_oof" if has_oof else "oracle_supervised"
    return dict(
        scenario=scenario,
        d_tta=_mean_delta(rows, scenario, "tta"),
        d_prior=_mean_delta(rows, scenario, "oracle_prior"),
        d_labels=_mean_delta(rows, scenario, "oracle_labels"),
        d_sup=_mean_delta(rows, scenario, sup_method),
        sup_is_oof=has_oof,
        g_unsup=_mean_field(rows, scenario, "tta", "crossfit_evidence_gain"),
        g_sup=_mean_field(rows, scenario, "oracle_supervised", "crossfit_supervised_gain"),
        adapt_rate=_adapt_rate(rows, scenario, "tta"),
        n=len(_by(rows, scenario=scenario, method="tta")),
    )


def decide(agg: dict) -> tuple[str, str]:
    s = agg["scenario"]
    d_tta, d_prior, d_labels, d_sup, g_sup = (agg["d_tta"], agg["d_prior"], agg["d_labels"],
                                              agg["d_sup"], agg["g_sup"])
    sup_note = "" if agg["sup_is_oof"] else " (transductive proxy; OOF ceiling needs a re-run)"
    if s == "population_null":
        if agg["adapt_rate"] > ADAPT_WARN and d_tta < EPS_HELP:
            return ("population_null_adapts",
                    f"adapts on population_null (rate={agg['adapt_rate']:.2f}, Δ={d_tta:+.3f}); "
                    "expected -- unseen-subject random effect IS fittable. Use matched_domain_null "
                    "to calibrate the evidence threshold.")
        return ("population_null_ok", f"population_null low harm (Δ={d_tta:+.3f})")
    if s == "matched_domain_null":
        if agg["adapt_rate"] > ADAPT_WARN or d_tta < -EPS_HELP:
            return ("rollback_loose",
                    f"adapts on matched_domain_null (rate={agg['adapt_rate']:.2f}, Δ={d_tta:+.3f}) "
                    "-> evidence/rollback threshold too loose (true identity-null)")
        return ("rollback_ok", f"matched_domain_null rolls back (Δ={d_tta:+.3f})")
    if d_tta > EPS_HELP:
        return ("unsup_helps", f"unsupervised TTA helps (Δ={d_tta:+.3f})")
    if not (d_sup > EPS_HELP):
        if g_sup > GAIN_TOL:
            return ("density_perp_boundary",
                    f"supervised evidence improves (g={g_sup:+.2f}) but accuracy does not "
                    f"(Δ_sup={d_sup:+.3f}){sup_note} -> density ⟂ decision boundary")
        return ("family_insufficient",
                f"even supervised does not help (Δ_sup={d_sup:+.3f}, g_sup={g_sup:+.2f}){sup_note} "
                "-> diagonal family / density geometry insufficient")
    if d_prior > EPS_HELP:
        return ("prior_bottleneck", f"oracle prior recovers (Δ_prior={d_prior:+.3f}) vs unsup "
                                    f"(Δ={d_tta:+.3f}) -> prior estimation is the bottleneck")
    if d_labels > EPS_HELP:
        return ("responsibilities_bottleneck",
                f"oracle labels recover (Δ_labels={d_labels:+.3f}) vs unsup (Δ={d_tta:+.3f}) "
                "-> EM responsibilities are the bottleneck")
    return ("supervised_only",
            f"only direct supervised transform recovers (Δ_sup={d_sup:+.3f}){sup_note}")


def delta_metrics(rows, scenario) -> dict:
    """Δ(TTA − identity) on NLL/Brier/ECE + prior-L1, pooled over units/cmi (prior emphasis)."""
    tta = {(r["data_seed"], r["target_site"], r["cmi"]): r for r in _by(rows, scenario=scenario, method="tta")}
    idn = {(r["data_seed"], r["target_site"], r["cmi"]): r for r in _by(rows, scenario=scenario, method="identity")}
    keys = set(tta) & set(idn)
    if not keys:
        return {}
    d_nll = np.mean([tta[k]["nll"] - idn[k]["nll"] for k in keys])
    d_brier = np.mean([tta[k]["brier"] - idn[k]["brier"] for k in keys])
    d_ece = np.mean([tta[k]["ece"] - idn[k]["ece"] for k in keys])
    pl1 = [tta[k]["prior_l1_error"] for k in keys if "prior_l1_error" in tta[k]]
    return dict(d_nll=float(d_nll), d_brier=float(d_brier), d_ece=float(d_ece),
                prior_l1=float(np.mean(pl1)) if pl1 else float("nan"))


def cmi_safety_note(rows, scenarios) -> dict:
    real = [s for s in scenarios if s not in ("population_null", "matched_domain_null")]
    on = np.array([r["delta_bacc"] for s in real for r in _by(rows, scenario=s, method="tta", cmi="on")])
    off = np.array([r["delta_bacc"] for s in real for r in _by(rows, scenario=s, method="tta", cmi="off")])
    if len(on) == 0 or len(off) == 0:
        return dict(note="insufficient data")
    hr_on, hr_off = float((on < 0).mean()), float((off < 0).mean())
    gain_gap = float(on.mean() - off.mean())
    improves = (hr_off - hr_on) >= SAFETY_HARM_GAP and abs(gain_gap) < EPS_HELP
    return dict(harm_rate_on=hr_on, harm_rate_off=hr_off, mean_gain_gap=gain_gap,
                cmi_improves_safety=bool(improves),
                note=("CMI lowers harm rate >=%.2f at matched gain -> tentative 'safety' signal"
                      % SAFETY_HARM_GAP if improves
                      else f"no clear CMI safety effect (harm {hr_off:.2f}->{hr_on:.2f}, "
                           f"gap requires >= {SAFETY_HARM_GAP})"))


def analyze(path: str, n_boot=2000, seed=0) -> dict:
    rows = load(path)
    scenarios = sorted({r["scenario"] for r in rows})
    rep = dict(source=path, n_rows=len(rows), scenarios=scenarios,
               cluster="seed", contrasts={}, oracle_table={}, decisions={}, delta_metrics={})
    for s in scenarios:
        rep["contrasts"][s] = scenario_contrasts(rows, s, n_boot, seed)
        agg = scenario_aggregate(rows, s)
        rep["oracle_table"][s] = agg
        code, msg = decide(agg)
        rep["decisions"][s] = dict(code=code, message=msg)
        rep["delta_metrics"][s] = delta_metrics(rows, s)
    rep["cmi_safety"] = cmi_safety_note(rows, scenarios)
    return rep


def _print(rep: dict):
    print(f"\n=== shift-grid analysis ({rep['n_rows']} rows, cluster={rep['cluster']}) "
          f"{rep['source']} ===")
    print("\n-- seed-clustered contrasts (mean over seeds; sign-flip p two-sided; mass>0) --")
    print(f"{'scenario':<24} {'TTA(noCMI)':>11} {'TTA(CMI)':>10} {'interaction':>12} "
          f"{'p':>5} {'mass':>5} {'M3-max':>8} {'harm off/on':>12}")
    for s, c in rep["contrasts"].items():
        i, t0, t1, mx = c["interaction"], c["tta_no_cmi"], c["tta_with_cmi"], c["m3_minus_max"]
        print(f"{s:<24} {t0['seed_mean']:>+11.3f} {t1['seed_mean']:>+10.3f} "
              f"{i['seed_mean']:>+12.3f} {i['signflip_p_two_sided']:>5.2f} "
              f"{i['bootstrap_mass_above_zero']:>5.2f} {mx['seed_mean']:>+8.3f} "
              f"{c['harm_rate_off']:>5.2f}/{c['harm_rate_on']:<5.2f}")
    print("\n-- oracle decision table (mean ΔbAcc) --")
    for s, a in rep["oracle_table"].items():
        d = rep["decisions"][s]
        print(f"{s:<24} d_tta={a['d_tta']:>+.3f} d_prior={a['d_prior']:>+.3f} "
              f"d_lab={a['d_labels']:>+.3f} d_sup={a['d_sup']:>+.3f}"
              f"{'(oof)' if a['sup_is_oof'] else '(tx)'}  [{d['code']}]")
    print("\n-- prior-sensitive Δ metrics (TTA − identity) --")
    print(f"{'scenario':<24} {'ΔNLL':>8} {'ΔBrier':>8} {'ΔECE':>8} {'priorL1':>8}")
    for s, m in rep["delta_metrics"].items():
        if m:
            print(f"{s:<24} {m['d_nll']:>+8.3f} {m['d_brier']:>+8.3f} {m['d_ece']:>+8.3f} "
                  f"{m['prior_l1']:>8.3f}")
    print(f"\n-- CMI safety: {rep['cmi_safety'].get('note')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rep = analyze(args.inp, args.n_boot, args.seed)
    _print(rep)
    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2, default=float)
        print("\nsaved ->", args.out)


if __name__ == "__main__":
    main()
