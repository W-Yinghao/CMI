"""REVIEW_P0 analyzer (section E). Averages technical source seeds WITHIN each target unit first, then
applies the predeclared PRIMARY contrast per panel with the correct bootstrap (W1 stratified-within-
dataset; W2 subject; V2P subject-cluster). 10,000 percentile bootstrap, deterministic seed. Reports
"negative-change rate" at 0/-0.01/-0.02. Holm only within explicitly declared confirmatory families.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

NB = 10000
BRANCHES = ["identity_uniform", "identity_joint_prior", "joint_geometry_uniform",
            "joint_geometry_joint_prior", "fixed_iterative_geometry_uniform",
            "fixed_reference_oneshot_uniform", "pooled_uniform", "latent_im_diag_uniform",
            "source_recolored_ea"]


def _load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def _neg_rates(d):
    d = np.asarray(d, float)
    return {"lt_0": float((d < 0).mean()), "lt_-0.01": float((d < -0.01).mean()), "lt_-0.02": float((d < -0.02).mean())}


def _boot(vals, seed=0):
    v = np.asarray([x for x in vals if x == x], float)
    if len(v) < 2:
        return dict(mean=float(v.mean()) if len(v) else float("nan"), ci=[float("nan")] * 2, n=len(v))
    rng = np.random.default_rng(seed)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(NB)]
    return dict(mean=float(v.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                excludes_0=bool(np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0), n=len(v))


def _strat_boot(by_strata, seed=0):
    """Stratified bootstrap: resample units within each stratum (dataset), pool, mean."""
    keys = list(by_strata); rng = np.random.default_rng(seed)
    allv = np.concatenate([np.asarray(by_strata[k], float) for k in keys]) if keys else np.array([])
    if len(allv) < 2:
        return dict(mean=float(allv.mean()) if len(allv) else float("nan"), ci=[float("nan")] * 2, n=len(allv))
    bs = []
    for _ in range(NB):
        m = np.concatenate([np.asarray(by_strata[k], float)[rng.integers(0, len(by_strata[k]), len(by_strata[k]))] for k in keys])
        bs.append(m.mean())
    return dict(mean=float(allv.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                excludes_0=bool(np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0), n=len(allv))


def _cluster_boot(cluster_vals, seed=0):
    keys = list(cluster_vals); rng = np.random.default_rng(seed)
    if len(keys) < 2:
        return dict(mean=float("nan"), ci=[float("nan")] * 2, n=0)
    arrs = {k: np.asarray(v, float) for k, v in cluster_vals.items()}
    allv = np.concatenate([arrs[k] for k in keys])
    bs = []
    for _ in range(NB):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        bs.append(np.concatenate([arrs[k] for k in pick]).mean())
    return dict(mean=float(allv.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                excludes_0=bool(np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0), n=len(allv))


def _unit_bacc(rows, keyfn):
    """unit -> branch -> mean bAcc over seeds. Also returns decomposition means per unit."""
    acc = defaultdict(lambda: defaultdict(list)); dec = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("provenance_fail"):
            continue
        u = keyfn(r)
        if r["branch"] == "__decomposition__":
            for k in ("G", "P", "interaction", "full_joint_delta", "prior_m_step_geometry"):
                if k in r:
                    dec[u][k].append(r[k])
        elif "bacc" in r:
            acc[u][r["branch"]].append(r["bacc"])
    ub = {u: {b: float(np.mean(v)) for b, v in d.items()} for u, d in acc.items()}
    ud = {u: {k: float(np.mean(v)) for k, v in d.items()} for u, d in dec.items()}
    return ub, ud


def analyze_w1(rows):
    rows = [r for r in rows if r.get("panel") == "W1_P0"]
    ub, ud = _unit_bacc(rows, lambda r: (r["dataset"], r["target_subject"]))
    by_ds_primary = defaultdict(list); by_ds_jgp = defaultdict(list)
    primary, jvp = [], []
    for (ds, subj), b in ub.items():
        if "fixed_iterative_geometry_uniform" in b and "joint_geometry_uniform" in b:
            c = b["fixed_iterative_geometry_uniform"] - b["joint_geometry_uniform"]
            by_ds_primary[ds].append(c); primary.append(c)
        if "joint_geometry_uniform" in b and "pooled_uniform" in b:
            jvp.append(b["joint_geometry_uniform"] - b["pooled_uniform"]); by_ds_jgp[ds].append(b["joint_geometry_uniform"] - b["pooled_uniform"])
    branch_means = {br: float(np.mean([b[br] for b in ub.values() if br in b])) for br in BRANCHES}
    dec_means = {k: float(np.mean([d[k] for d in ud.values() if k in d])) for k in ("G", "P", "interaction", "full_joint_delta", "prior_m_step_geometry")}
    return dict(n_units=len(ub), branch_mean_bacc=branch_means, decomposition_mean=dec_means,
                PRIMARY_fixed_iter_minus_joint_geom=_strat_boot(by_ds_primary),
                joint_geom_minus_pooled=_strat_boot(by_ds_jgp),
                negative_change_rate_primary=_neg_rates(primary),
                per_dataset_primary={ds: _boot(v) for ds, v in by_ds_primary.items()})


def analyze_w2(rows, protocol):
    rows = [r for r in rows if r.get("panel") == "W2_P0" and r.get("protocol") == protocol]
    ub, ud = _unit_bacc(rows, lambda r: r["target_subject"])
    prim, diag = [], []
    for subj, b in ub.items():
        if "joint_geometry_uniform" in b and "identity_uniform" in b:
            prim.append(b["joint_geometry_uniform"] - b["identity_uniform"])
        if "joint_geometry_joint_prior" in b and "joint_geometry_uniform" in b:
            diag.append(b["joint_geometry_joint_prior"] - b["joint_geometry_uniform"])
    branch_means = {br: float(np.mean([b[br] for b in ub.values() if br in b])) for br in BRANCHES}
    dec_means = {k: float(np.mean([d[k] for d in ud.values() if k in d])) for k in ("G", "P", "interaction", "full_joint_delta", "prior_m_step_geometry")}
    return dict(n_units=len(ub), protocol=protocol, branch_mean_bacc=branch_means, decomposition_mean=dec_means,
                PRIMARY_joint_geom_minus_identity=_boot(prim),
                decision_prior_diagnostic=_boot(diag),
                negative_change_rate_primary=_neg_rates(prim))


def analyze_v2p(rows):
    rows = [r for r in rows if r.get("panel") == "V2PW"]
    # average seeds within (pair, subject) unit, cluster by subject
    by_unit = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("provenance_fail") or r.get("ratio") != "__geom__":
            continue
        u = (r["pair"], r["subject"])
        for k in ("logscale_disp_q075", "logscale_disp_q025", "translation_disp_q075",
                  "translation_disp_q025", "embed_disp_q075", "embed_disp_q025"):
            if k in r:
                by_unit[(u, r["estimator"])][k].append(r[k])
    # mean disp per (unit, estimator) = mean of q075 & q025 logscale displacement
    disp = defaultdict(dict)   # estimator -> unit -> scalar displacement (logscale, averaged over seeds + directions)
    for (u, est), d in by_unit.items():
        vals = d.get("logscale_disp_q075", []) + d.get("logscale_disp_q025", [])
        disp[est][u] = float(np.mean(vals)) if vals else float("nan")
    out = {}
    for est in disp:
        clus = defaultdict(list)
        for (pair, subj), v in disp[est].items():
            clus[subj].append(v)
        out[est] = _cluster_boot(clus)
    # fixed-reference vs pooled (paired by unit)
    fr = disp.get("fixed_reference_oneshot", {}); pl = disp.get("pooled", {})
    common = set(fr) & set(pl)
    clus_d = defaultdict(list)
    for (pair, subj) in common:
        clus_d[subj].append(fr[(pair, subj)] - pl[(pair, subj)])
    return dict(n_units=len(set(u for est in disp for u in disp[est])),
                PRIMARY_fixed_reference_displacement=out.get("fixed_reference_oneshot"),
                fixed_reference_minus_pooled_displacement=_cluster_boot(clus_d),
                displacement_by_estimator={e: out[e] for e in out})


def _holm(pvals_named):
    """Holm step-down on a dict name->p (within a declared confirmatory family)."""
    items = sorted(pvals_named.items(), key=lambda kv: kv[1]); m = len(items); out = {}
    for i, (name, p) in enumerate(items):
        out[name] = min(1.0, p * (m - i))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, nargs="+")
    ap.add_argument("--out", default="results/h2cmi/review_p0.report.json")
    args = ap.parse_args()
    rows = _load(args.inp)
    rep = {"n_rows": len(rows),
           "W1": analyze_w1(rows),
           "W2_primary": analyze_w2(rows, "primary"),
           "W2_secondary": analyze_w2(rows, "secondary"),
           "V2P_weighted": analyze_v2p(rows)}
    json.dump(rep, open(args.out, "w"), indent=2, default=str)

    def line(name, d):
        if not d or d.get("mean") != d.get("mean"):
            return f"  {name:46s} (no data)"
        ex = d.get("excludes_0")
        return f"  {name:46s} {d['mean']:+.4f} [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] n={d['n']}" + ("  *CI-excl-0" if ex else "")
    print("================= REVIEW_P0 RESULTS =================")
    W1 = rep["W1"]; print(f"[W1] n_units={W1['n_units']}  branch mean bAcc:")
    for br in BRANCHES:
        print(f"    {br:34s} {W1['branch_mean_bacc'].get(br, float('nan')):.4f}")
    print(f"  decomposition (mean): {W1['decomposition_mean']}")
    print(line("W1 PRIMARY fixed_iter - joint_geom (uniform)", W1["PRIMARY_fixed_iter_minus_joint_geom"]))
    print(line("W1 joint_geom - pooled (uniform)", W1["joint_geom_minus_pooled"]))
    print(f"  W1 negative-change rate (primary): {W1['negative_change_rate_primary']}")
    for tag in ("W2_primary", "W2_secondary"):
        d = rep[tag]; print(f"[{tag}] n_units={d['n_units']}")
        print(line(f"{tag} PRIMARY joint_geom - identity (uniform)", d["PRIMARY_joint_geom_minus_identity"]))
        print(line(f"{tag} decision-prior diagnostic", d["decision_prior_diagnostic"]))
        print(f"  decomposition (mean): {d['decomposition_mean']}")
    V = rep["V2P_weighted"]; print(f"[V2P_weighted] n_units={V['n_units']}")
    print(line("V2P PRIMARY fixed-reference displacement", V["PRIMARY_fixed_reference_displacement"]))
    print(line("V2P fixed-reference - pooled displacement", V["fixed_reference_minus_pooled_displacement"]))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
