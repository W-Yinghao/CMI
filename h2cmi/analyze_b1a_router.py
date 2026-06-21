"""Stage-B1b-1: per-unit cross-fitted evidence ROUTER over the deployable (label-free) actions,
evaluated on the existing B1a grids -- no GPU, no method change.

For each unit (difficulty, scenario, seed, target_site) the router scores each deployable action
by its grouped (per-target-subject LOSO) cross-fitted evidence gain over identity, and applies a
FIXED rule (no learned gate, no target labels, no tuned threshold):

    scores["identity"] = 0.0
    selected = argmax_a scores[a]
    if scores[selected] <= 0: selected = "identity"

It then reports the held-out OUTCOME of that choice (grouped OOF bAcc vs identity), the harm /
false-adaptation rates, the action regret vs the oracle-best deployable action, and whether the
evidence signal actually predicts accuracy gain. This turns the "safety gate" into an
interpretable cross-fitted model-selection rule and tests it per (seed x site) unit -- aggregate
identity-optimality does NOT imply every unit routes to identity.

  python -m h2cmi.analyze_b1a_router --in results/h2cmi/b1a_standard.jsonl \
      results/h2cmi/b1a_hard_null.jsonl --out results/h2cmi/b1a_router.report.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from h2cmi.analyze_b1a_grid import _cluster_bootstrap

IDENTITY = "identity"
DEPLOYABLE = ("identity", "pooled_empirical_diag", "gen_oneshot_diag", "gen_iterative_diag")
NULL_SCENARIOS = ("population_null", "matched_domain_null")
EVID = "grouped_crossfit_evidence_gain"
OOF = "grouped_oof_bacc"
FULL = "bacc_uniform"           # full-target refit (the actually-deployed output, review §lens)


def _load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def _finite(x):
    return x is not None and x == x


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan"), int(len(x))
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return (float((rx * ry).sum() / denom) if denom > 0 else float("nan")), int(len(x))


def route_units(rows, deployable=DEPLOYABLE) -> list[dict]:
    units = defaultdict(dict)                                   # (diff,scen,seed,site) -> {variant: row}
    for r in rows:
        units[(r.get("difficulty", "standard"), r["scenario"], r["data_seed"], r["target_site"])][r["variant"]] = r
    out = []
    for (diff, scen, seed, site), vmap in units.items():
        if IDENTITY not in vmap:
            continue
        scores = {IDENTITY: 0.0}
        for a in deployable:
            if a == IDENTITY or a not in vmap:
                continue
            g = vmap[a].get(EVID)
            scores[a] = float(g) if _finite(g) else float("-inf")
        selected = max(scores, key=scores.get)
        if scores.get(selected, 0.0) <= 0.0:                   # fixed rollback rule
            selected = IDENTITY

        # Selection is by the label-free OOF signal; the OUTCOME is measured two ways (review
        # §lens): OOF (fold-fitted transform's cross-subject generalisation) AND full-refit
        # (bacc_uniform = the action refit on the FULL target = the actually-deployed output).
        def _metric(metric):
            idv = vmap[IDENTITY].get(metric)
            selv = vmap[selected].get(metric)
            cand = {a: vmap[a].get(metric) for a in deployable if a in vmap and _finite(vmap[a].get(metric))}
            oa = max(cand, key=cand.get) if cand else IDENTITY
            d = (selv - idv) if (_finite(selv) and _finite(idv)) else float("nan")
            reg = (cand.get(oa, idv) - selv) if (_finite(cand.get(oa)) and _finite(selv)) else float("nan")
            return d, reg, oa
        d_oof, reg_oof, oa_oof = _metric(OOF)
        d_full, reg_full, oa_full = _metric(FULL)
        out.append(dict(difficulty=diff, scenario=scen, data_seed=seed, target_site=site,
                        selected=selected, adapted=(selected != IDENTITY), is_null=(scen in NULL_SCENARIOS),
                        d_router=d_oof, d_router_oof=d_oof, d_router_full=d_full,
                        regret_oof=reg_oof, regret_full=reg_full,
                        top1_agree=(selected == oa_oof), top1_agree_oof=(selected == oa_oof),
                        top1_agree_full=(selected == oa_full)))
    return out


def _by_seed(units, key, pred=lambda u: True):
    d = defaultdict(list)
    for u in units:
        if pred(u) and _finite(u.get(key)):
            d[u["data_seed"]].append(u[key])
    return d


def router_report(rows, *, deployable=DEPLOYABLE, n_boot=10000) -> dict:
    units = route_units(rows, deployable)
    rep = {"deployable": list(deployable), "n_units": len(units), "by_difficulty": {}}
    # evidence-gain vs accuracy-gain Spearman (per unit x non-identity action)
    ex, ey = [], []
    bydiff = defaultdict(list)
    for r in rows:
        bydiff[r.get("difficulty", "standard")].append(r)
    for r in rows:
        if r["variant"] in deployable and r["variant"] != IDENTITY:
            key = (r.get("difficulty", "standard"), r["scenario"], r["data_seed"], r["target_site"])
            ex.append(r.get(EVID)); ey.append(r.get(OOF))
    for diff in sorted({u["difficulty"] for u in units}):
        du = [u for u in units if u["difficulty"] == diff]
        shift = [u for u in du if not u["is_null"]]
        nulls = [u for u in du if u["is_null"]]
        # seed-clustered ΔbAcc on SHIFT scenarios (where adaptation can help), OOF + full-refit
        mean, lo, hi, ns = _cluster_bootstrap(_by_seed(shift, "d_router_oof"), n_boot=n_boot)
        fmean, flo, fhi, _ = _cluster_bootstrap(_by_seed(shift, "d_router_full"), n_boot=n_boot)
        rmean, rlo, rhi, _ = _cluster_bootstrap(_by_seed(shift, "regret_oof"), n_boot=n_boot)
        d_all = np.array([u["d_router_oof"] for u in du if _finite(u["d_router_oof"])])
        dfull_all = np.array([u["d_router_full"] for u in du if _finite(u["d_router_full"])])
        # evidence/accuracy spearman within this difficulty
        ix = [r.get(EVID) for r in bydiff[diff] if r["variant"] in deployable and r["variant"] != IDENTITY]
        iy = [r.get(OOF) for r in bydiff[diff] if r["variant"] in deployable and r["variant"] != IDENTITY]
        # accuracy-GAIN vs identity per unit for spearman
        gx, gy = [], []
        umap = defaultdict(dict)
        for r in bydiff[diff]:
            umap[(r["scenario"], r["data_seed"], r["target_site"])][r["variant"]] = r
        for u, vm in umap.items():
            if IDENTITY not in vm:
                continue
            idb = vm[IDENTITY].get(OOF)
            for a in deployable:
                if a == IDENTITY or a not in vm:
                    continue
                gx.append(vm[a].get(EVID))
                gy.append((vm[a].get(OOF) - idb) if (_finite(vm[a].get(OOF)) and _finite(idb)) else None)
        rho, npair = _spearman([v for v in gx], [v if _finite(v) else np.nan for v in gy])
        freq = defaultdict(lambda: defaultdict(int))
        for u in du:
            freq[u["scenario"]][u["selected"]] += 1
        rep["by_difficulty"][diff] = dict(
            n_units=len(du),
            delta_bacc_router_shift_oof=dict(mean=mean, ci_lo=lo, ci_hi=hi, n_seeds=ns),
            delta_bacc_router_shift_full=dict(mean=fmean, ci_lo=flo, ci_hi=fhi, n_seeds=ns),
            action_regret_shift_oof=dict(mean=rmean, ci_lo=rlo, ci_hi=rhi),
            harm_rate_oof=float(np.mean(d_all < -1e-9)) if len(d_all) else float("nan"),
            harm_rate_full=float(np.mean(dfull_all < -1e-9)) if len(dfull_all) else float("nan"),
            mean_delta_bacc_full_all=float(np.nanmean([u["d_router_full"] for u in du])),
            false_adaptation_rate_null=float(np.mean([u["adapted"] for u in nulls])) if nulls else float("nan"),
            adaptation_rate_shift=float(np.mean([u["adapted"] for u in shift])) if shift else float("nan"),
            top1_oracle_agreement_oof=float(np.mean([u["top1_agree_oof"] for u in du])),
            top1_oracle_agreement_full=float(np.mean([u["top1_agree_full"] for u in du])),
            evidence_accuracy_spearman=rho, spearman_n=npair,
            selection_frequency={s: dict(v) for s, v in freq.items()})
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    rep = router_report(_load(args.inp), n_boot=args.n_boot)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
    for diff, d in rep["by_difficulty"].items():
        do, df = d["delta_bacc_router_shift_oof"], d["delta_bacc_router_shift_full"]
        print(f"[{diff}] n={d['n_units']}  ΔbAcc(oof)={do['mean']:+.3f}[{do['ci_lo']:+.3f},{do['ci_hi']:+.3f}] "
              f"ΔbAcc(full)={df['mean']:+.3f}  harm(oof/full)={d['harm_rate_oof']:.2f}/{d['harm_rate_full']:.2f}  "
              f"false_adapt_null={d['false_adaptation_rate_null']:.2f}  adapt_shift={d['adaptation_rate_shift']:.2f}  "
              f"top1(oof/full)={d['top1_oracle_agreement_oof']:.2f}/{d['top1_oracle_agreement_full']:.2f}  "
              f"evid~acc_rho={d['evidence_accuracy_spearman']:.2f}")
    if args.out:
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
