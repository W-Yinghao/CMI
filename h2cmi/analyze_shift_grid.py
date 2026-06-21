"""Analysis for the shift-grid JSONL (review §"分析"). FROZEN before the GPU screening so
the statistical protocol cannot be tuned post-hoc.

Two products:

1. Paired interaction bootstrap. The statistical unit is ``(data_seed, target_site)``; the
   headline per scenario is the CMI x TTA interaction
       I = mean_unit [ Delta_TTA(CMI on) - Delta_TTA(CMI off) ]
   with a cluster bootstrap over units (mean, 95% CI, P(I>0)). Also reports per-arm mean
   TTA delta and harm rate (fraction of units with Delta_TTA < 0).

2. Oracle decision table. Per scenario, the mean accuracy gain of identity / tta /
   oracle_prior / oracle_labels / oracle_supervised plus the held-out evidence gains, run
   through a fixed decision tree that localises a negative unsupervised gain to: prior
   estimation, EM responsibilities, transform-family/density-geometry, or density vs
   decision-boundary mismatch -- exactly the review's interpretation logic.

Pure stdlib + numpy. Reads only the JSONL; writes a JSON report and prints tables.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

EPS_HELP = 0.02          # accuracy gain (bAcc) counted as "helps"
GAIN_TOL = 0.10          # held-out evidence improvement counted as "improvable"
ADAPT_WARN = 0.50        # no_shift adaptation-rate warning threshold


def load(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _by(rows, **eq):
    return [r for r in rows if all(r.get(k) == v for k, v in eq.items())]


def _unit_map(rows, scenario, method, cmi):
    """(data_seed, target_site) -> row, for a (scenario, method, cmi) slice."""
    out = {}
    for r in _by(rows, scenario=scenario, method=method, cmi=cmi):
        out[(r["data_seed"], r["target_site"])] = r
    return out


def _cluster_bootstrap(values: np.ndarray, n_boot=2000, alpha=0.05, seed=0) -> dict:
    if len(values) == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"),
                    p_gt0=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(n_boot)])
    return dict(mean=float(values.mean()),
                lo=float(np.quantile(boots, alpha / 2)),
                hi=float(np.quantile(boots, 1 - alpha / 2)),
                p_gt0=float((boots > 0).mean()), n=int(len(values)))


def paired_interaction(rows, scenario, n_boot=2000, seed=0) -> dict:
    on = _unit_map(rows, scenario, "tta", "on")
    off = _unit_map(rows, scenario, "tta", "off")
    units = sorted(set(on) & set(off))
    inter = np.array([on[u]["delta_bacc"] - off[u]["delta_bacc"] for u in units])
    d_on = np.array([on[u]["delta_bacc"] for u in units])
    d_off = np.array([off[u]["delta_bacc"] for u in units])
    return dict(
        interaction=_cluster_bootstrap(inter, n_boot, seed=seed),
        delta_tta_on=_cluster_bootstrap(d_on, n_boot, seed=seed + 1),
        delta_tta_off=_cluster_bootstrap(d_off, n_boot, seed=seed + 2),
        harm_rate_on=float((d_on < 0).mean()) if len(d_on) else float("nan"),
        harm_rate_off=float((d_off < 0).mean()) if len(d_off) else float("nan"),
        n_units=len(units),
    )


def _mean_delta(rows, scenario, method):
    vals = [r["delta_bacc"] for r in _by(rows, scenario=scenario, method=method)]
    return float(np.mean(vals)) if vals else float("nan")


def _mean_field(rows, scenario, method, field):
    vals = [r[field] for r in _by(rows, scenario=scenario, method=method)
            if field in r and r[field] == r[field]]      # drop NaN
    return float(np.mean(vals)) if vals else float("nan")


def _adapt_rate(rows, scenario, method="tta"):
    vals = [bool(r.get("adapted")) for r in _by(rows, scenario=scenario, method=method)]
    return float(np.mean(vals)) if vals else float("nan")


def scenario_aggregate(rows, scenario) -> dict:
    return dict(
        scenario=scenario,
        d_tta=_mean_delta(rows, scenario, "tta"),
        d_prior=_mean_delta(rows, scenario, "oracle_prior"),
        d_labels=_mean_delta(rows, scenario, "oracle_labels"),
        d_sup=_mean_delta(rows, scenario, "oracle_supervised"),
        g_unsup=_mean_field(rows, scenario, "tta", "crossfit_evidence_gain"),
        g_sup=_mean_field(rows, scenario, "oracle_supervised", "crossfit_supervised_gain"),
        adapt_rate=_adapt_rate(rows, scenario, "tta"),
        n=len(_by(rows, scenario=scenario, method="tta")),
    )


def decide(agg: dict) -> tuple[str, str]:
    """Localise the (lack of) unsupervised gain to a cause. Returns (code, message)."""
    s = agg["scenario"]
    d_tta, d_prior, d_labels, d_sup = agg["d_tta"], agg["d_prior"], agg["d_labels"], agg["d_sup"]
    g_sup = agg["g_sup"]
    if s == "no_shift":
        if agg["adapt_rate"] > ADAPT_WARN and d_tta < EPS_HELP:
            return ("rollback_loose",
                    f"adapts under no_shift (rate={agg['adapt_rate']:.2f}, Δ={d_tta:+.3f}) "
                    "-> evidence/rollback threshold may be too loose")
        return ("ok_rollback", f"no_shift: low harm (Δ={d_tta:+.3f}, adapt_rate={agg['adapt_rate']:.2f})")
    if d_tta > EPS_HELP:
        return ("unsup_helps", f"unsupervised TTA helps (Δ={d_tta:+.3f})")
    if not (d_sup > EPS_HELP):                          # even labels+supervised can't help
        if g_sup > GAIN_TOL:
            return ("density_perp_boundary",
                    f"supervised evidence improves (g={g_sup:+.2f}) but accuracy does not "
                    f"(Δ_sup={d_sup:+.3f}) -> density ⟂ decision boundary")
        return ("family_insufficient",
                f"even supervised does not help (Δ_sup={d_sup:+.3f}, g_sup={g_sup:+.2f}) "
                "-> diagonal transform family / density geometry insufficient")
    # d_sup helps but unsupervised does not -> isolate which oracle recovers it
    if d_prior > EPS_HELP:
        return ("prior_bottleneck", f"oracle prior recovers (Δ_prior={d_prior:+.3f}) "
                                    f"vs unsup (Δ={d_tta:+.3f}) -> prior estimation is the bottleneck")
    if d_labels > EPS_HELP:
        return ("responsibilities_bottleneck",
                f"oracle labels recover (Δ_labels={d_labels:+.3f}) vs unsup (Δ={d_tta:+.3f}) "
                "-> EM responsibilities are the bottleneck")
    return ("supervised_only",
            f"only the direct supervised transform recovers (Δ_sup={d_sup:+.3f}); "
            "prior/responsibility oracles do not -> soft-assignment geometry")


def cmi_safety_note(rows, scenarios) -> dict:
    """Pooled across REAL-shift scenarios: does CMI lower the harm rate at similar gain?"""
    real = [s for s in scenarios if s != "no_shift"]
    on = [r["delta_bacc"] for s in real for r in _by(rows, scenario=s, method="tta", cmi="on")]
    off = [r["delta_bacc"] for s in real for r in _by(rows, scenario=s, method="tta", cmi="off")]
    on, off = np.array(on), np.array(off)
    if len(on) == 0 or len(off) == 0:
        return dict(note="insufficient data")
    hr_on, hr_off = float((on < 0).mean()), float((off < 0).mean())
    gain_gap = float(on.mean() - off.mean())
    improves = hr_on < hr_off - 1e-9 and abs(gain_gap) < EPS_HELP
    return dict(harm_rate_on=hr_on, harm_rate_off=hr_off, mean_gain_gap=gain_gap,
                cmi_improves_safety=bool(improves),
                note=("CMI lowers harm rate at similar mean gain -> 'CMI improves adaptation safety'"
                      if improves else "no clear CMI safety effect"))


def analyze(path: str, n_boot=2000, seed=0) -> dict:
    rows = load(path)
    scenarios = sorted({r["scenario"] for r in rows})
    report = dict(source=path, n_rows=len(rows), scenarios=scenarios,
                  interaction={}, oracle_table={}, decisions={})
    for s in scenarios:
        report["interaction"][s] = paired_interaction(rows, s, n_boot, seed)
        agg = scenario_aggregate(rows, s)
        report["oracle_table"][s] = agg
        code, msg = decide(agg)
        report["decisions"][s] = dict(code=code, message=msg)
    report["cmi_safety"] = cmi_safety_note(rows, scenarios)
    return report


def _print(report: dict):
    print(f"\n=== shift-grid analysis ({report['n_rows']} rows, {report['source']}) ===")
    print("\n-- paired CMI×TTA interaction  [Δ_TTA(on) − Δ_TTA(off)], unit=(seed,site) --")
    print(f"{'scenario':<12} {'n':>3} {'inter':>8} {'95% CI':>18} {'P>0':>5} "
          f"{'ΔTTAoff':>8} {'ΔTTAon':>8} {'harm off/on':>12}")
    for s, it in report["interaction"].items():
        i = it["interaction"]
        print(f"{s:<12} {it['n_units']:>3} {i['mean']:>+8.3f} "
              f"[{i['lo']:>+6.3f},{i['hi']:>+6.3f}] {i['p_gt0']:>5.2f} "
              f"{it['delta_tta_off']['mean']:>+8.3f} {it['delta_tta_on']['mean']:>+8.3f} "
              f"{it['harm_rate_off']:>5.2f}/{it['harm_rate_on']:<5.2f}")
    print("\n-- oracle decision table (mean ΔbAcc; g=held-out evidence gain) --")
    print(f"{'scenario':<12} {'d_tta':>7} {'d_prior':>8} {'d_lab':>7} {'d_sup':>7} "
          f"{'g_unsup':>8} {'g_sup':>7}  diagnosis")
    for s, a in report["oracle_table"].items():
        d = report["decisions"][s]
        print(f"{s:<12} {a['d_tta']:>+7.3f} {a['d_prior']:>+8.3f} {a['d_labels']:>+7.3f} "
              f"{a['d_sup']:>+7.3f} {a['g_unsup']:>+8.2f} {a['g_sup']:>+7.2f}  "
              f"[{d['code']}] {d['message']}")
    cs = report["cmi_safety"]
    print(f"\n-- CMI safety: {cs.get('note')}")
    if "harm_rate_on" in cs:
        print(f"   harm_rate off={cs['harm_rate_off']:.2f} on={cs['harm_rate_on']:.2f} "
              f"mean_gain_gap={cs['mean_gain_gap']:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="shift-grid JSONL")
    ap.add_argument("--out", default="", help="optional JSON report path")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    report = analyze(args.inp, args.n_boot, args.seed)
    _print(report)
    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=float)
        print("\nsaved ->", args.out)


if __name__ == "__main__":
    main()
