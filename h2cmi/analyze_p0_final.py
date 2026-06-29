"""REVIEW_P0 corrected analyzer (finalizer #2A). Operates on the complete, integrity-verified raw rows
(all @278fc85). Produces every predeclared table; interpretation happens only in REVIEW_P0_RESULTS.

Hard constraints honored:
- average technical source seeds WITHIN a unit (scalar metrics only) BEFORE bootstrap;
- NEVER average latent (a,b) vectors across seeds -- compute scalar norms per seed, then average;
- W1 stratified-within-dataset bootstrap; W2 subject bootstrap; V2P subject-CLUSTER bootstrap with
  cluster key (dataset, subject);
- W1 PRIMARY = fixed_iterative_geometry_uniform - joint_geometry_uniform; W2 PRIMARY =
  joint_geometry_uniform - identity_uniform; W2 fixed_iter - joint_geom is SECONDARY (labeled);
- V2P PRIMARY = evaluation-embedding displacement; FRSC-vs-pooled and FRSC-vs-oracle are a 2-contrast
  Holm family; slope divisor = 2*ln(3);
- balanced-accuracy primary always uniform decision prior; report runner vs analyzer commit separately.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import defaultdict

import numpy as np

NB = 10000
RUNNER_COMMIT = "278fc85"
LOG3x2 = 2.0 * math.log(3.0)
BRANCHES = ["identity_uniform", "identity_joint_prior", "joint_geometry_uniform",
            "joint_geometry_joint_prior", "fixed_iterative_geometry_uniform",
            "fixed_reference_oneshot_uniform", "pooled_uniform", "latent_im_diag_uniform",
            "source_recolored_ea"]


def _analyzer_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _load(paths):
    rows = []
    for p in paths:
        for l in open(p):
            l = l.strip()
            if l:
                r = json.loads(l)
                c = r.get("commit")
                if not r.get("provenance_fail") and (c is None or c[:len(RUNNER_COMMIT)] == RUNNER_COMMIT):
                    rows.append(r)
    return rows


def _neg(d):
    d = np.asarray(d, float)
    return {"lt_0": float((d < 0).mean()), "lt_-0.01": float((d < -0.01).mean()),
            "lt_-0.02": float((d < -0.02).mean()), "n": int(len(d))}


def _boot(vals, seed=0):
    v = np.asarray([x for x in vals if x == x], float)
    if len(v) < 2:
        return dict(mean=float(v.mean()) if len(v) else float("nan"), ci=[float("nan")] * 2, n=int(len(v)))
    rng = np.random.default_rng(seed)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(NB)]
    return dict(mean=float(v.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                excludes_0=bool(np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0), n=int(len(v)))


def _strat_boot(by_stratum, seed=0):
    keys = [k for k in by_stratum if len(by_stratum[k])]
    if not keys:
        return dict(mean=float("nan"), ci=[float("nan")] * 2, n=0)
    arrs = {k: np.asarray(by_stratum[k], float) for k in keys}
    allv = np.concatenate([arrs[k] for k in keys])
    rng = np.random.default_rng(seed)
    bs = [np.concatenate([arrs[k][rng.integers(0, len(arrs[k]), len(arrs[k]))] for k in keys]).mean() for _ in range(NB)]
    return dict(mean=float(allv.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                excludes_0=bool(np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0), n=int(len(allv)))


def _macro_boot(by_stratum, seed=0):
    """Dataset-equal macro: bootstrap subjects within each dataset, average per-dataset means equally."""
    keys = [k for k in by_stratum if len(by_stratum[k])]
    if not keys:
        return dict(mean=float("nan"), ci=[float("nan")] * 2, n=0)
    arrs = {k: np.asarray(by_stratum[k], float) for k in keys}
    macro = float(np.mean([arrs[k].mean() for k in keys]))
    rng = np.random.default_rng(seed)
    bs = [float(np.mean([arrs[k][rng.integers(0, len(arrs[k]), len(arrs[k]))].mean() for k in keys])) for _ in range(NB)]
    return dict(mean=macro, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                excludes_0=bool(np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0), n=len(keys))


def _cluster_boot(cluster_vals, seed=0):
    keys = [k for k in cluster_vals if len(cluster_vals[k])]
    if len(keys) < 2:
        return dict(mean=float("nan"), ci=[float("nan")] * 2, n=0)
    arrs = {k: np.asarray(cluster_vals[k], float) for k in keys}
    allv = np.concatenate([arrs[k] for k in keys])
    rng = np.random.default_rng(seed)
    bs = [np.concatenate([arrs[k] for k in (keys[i] for i in rng.integers(0, len(keys), len(keys)))]).mean() for _ in range(NB)]
    return dict(mean=float(allv.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                excludes_0=bool(np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0), n=int(len(allv)))


def _holm(named_excl):
    """named_excl: {name: bootstrap-dict}. Report Holm-style ordering note (CI-based family control)."""
    return {k: {"excludes_0": v.get("excludes_0"), "rank": i}
            for i, (k, v) in enumerate(sorted(named_excl.items(), key=lambda kv: -abs(kv[1].get("mean", 0) or 0)))}


# ---- per-unit seed-averaged branch bAcc + decomposition (scalars) ----
def _unit_scalars(rows, keyfn):
    bacc = defaultdict(lambda: defaultdict(list)); dec = defaultdict(lambda: defaultdict(list))
    for r in rows:
        u = keyfn(r)
        if r.get("branch") == "__decomposition__":
            for k in ("G", "P", "interaction", "full_joint_delta", "residual", "prior_m_step_geometry"):
                if k in r:
                    dec[u][k].append(r[k])
        elif "bacc" in r and "branch" in r:
            bacc[u][r["branch"]].append(r["bacc"])
    ub = {u: {b: float(np.mean(v)) for b, v in d.items()} for u, d in bacc.items()}
    ud = {u: {k: float(np.mean(v)) for k, v in d.items()} for u, d in dec.items()}
    return ub, ud


def analyze_w1(rows):
    rows = [r for r in rows if r.get("panel") == "W1_P0"]
    ub, ud = _unit_scalars(rows, lambda r: (r["dataset"], r["target_subject"]))
    contrasts = {"PRIMARY_fixed_iter_minus_joint_geom": ("fixed_iterative_geometry_uniform", "joint_geometry_uniform"),
                 "G_joint_geom_minus_identity": ("joint_geometry_uniform", "identity_uniform"),
                 "P_identity_jointprior_minus_identity": ("identity_joint_prior", "identity_uniform"),
                 "joint_geom_minus_pooled": ("joint_geometry_uniform", "pooled_uniform")}
    out = {"unit": "target subject (seeds averaged within unit)", "n_units": len(ub)}
    for name, (a, b) in contrasts.items():
        strat = defaultdict(list)
        for (ds, s), v in ub.items():
            if a in v and b in v:
                strat[ds].append(v[a] - v[b])
        out[name] = {"subject_weighted": _strat_boot(strat), "dataset_equal_macro": _macro_boot(strat),
                     "per_dataset": {ds: _boot(strat[ds]) for ds in strat}}
    # interaction + decomposition identity (means over units)
    out["interaction_mean"] = float(np.mean([d["interaction"] for d in ud.values() if "interaction" in d]))
    out["decomposition_residual_max"] = float(np.max([abs(d.get("residual", 0.0)) for d in ud.values()])) if ud else 0.0
    # leave-one-dataset-out on the PRIMARY contrast
    prim = defaultdict(list)
    for (ds, s), v in ub.items():
        if "fixed_iterative_geometry_uniform" in v and "joint_geometry_uniform" in v:
            prim[ds].append(v["fixed_iterative_geometry_uniform"] - v["joint_geometry_uniform"])
    out["primary_leave_one_dataset_out"] = {f"drop_{drop}": _strat_boot({d: prim[d] for d in prim if d != drop})
                                            for drop in prim}
    # per-method negative-change rate vs identity (every method)
    out["negative_change_rate_vs_identity"] = {}
    for m in BRANCHES:
        if m == "identity_uniform":
            continue
        d = [v[m] - v["identity_uniform"] for v in ub.values() if m in v and "identity_uniform" in v]
        out["negative_change_rate_vs_identity"][m] = _neg(d)
    out["branch_mean_bacc"] = {m: float(np.mean([v[m] for v in ub.values() if m in v])) for m in BRANCHES}
    return out


def analyze_w2(rows, protocol):
    rows = [r for r in rows if r.get("panel") == "W2_P0" and r.get("protocol") == protocol]
    ub, ud = _unit_scalars(rows, lambda r: r["target_subject"])
    def subj_contrast(a, b):
        return _boot([v[a] - v[b] for v in ub.values() if a in v and b in v])
    out = {"protocol": protocol, "unit": "subject (seeds averaged within unit)", "n_units": len(ub),
           "PRIMARY_G_joint_geom_minus_identity": subj_contrast("joint_geometry_uniform", "identity_uniform"),
           "P_identity_jointprior_minus_identity": subj_contrast("identity_joint_prior", "identity_uniform"),
           "decision_prior_diagnostic_jointprior_minus_jointgeom": subj_contrast("joint_geometry_joint_prior", "joint_geometry_uniform"),
           "SECONDARY_fixed_iter_minus_joint_geom": subj_contrast("fixed_iterative_geometry_uniform", "joint_geometry_uniform"),
           "interaction_mean": float(np.mean([d["interaction"] for d in ud.values() if "interaction" in d])) if ud else float("nan"),
           "decomposition_residual_max": float(np.max([abs(d.get("residual", 0.0)) for d in ud.values()])) if ud else 0.0,
           "negative_change_rate_vs_identity": {}}
    for m in BRANCHES:
        if m == "identity_uniform":
            continue
        d = [v[m] - v["identity_uniform"] for v in ub.values() if m in v and "identity_uniform" in v]
        out["negative_change_rate_vs_identity"][m] = _neg(d)
    out["branch_mean_bacc"] = {m: float(np.mean([v[m] for v in ub.values() if m in v])) for m in BRANCHES}
    return out


def analyze_v2p(rows):
    rows = [r for r in rows if r.get("panel") == "V2PW"]
    # collect per (pair, subject, seed, estimator) the a,b vectors at each ratio + the embed disps
    ab = defaultdict(dict)         # (pair,subj,seed,est) -> ratio -> (a,b)
    embed = defaultdict(dict)      # (pair,subj,seed,est) -> {q075,q025}
    for r in rows:
        est = r.get("estimator"); ratio = r.get("ratio")
        if est is None:
            continue
        key = (r["pair"], r["subject"], r["seed"], est)
        if ratio in ("q0.50", "q0.75", "q0.25") and "a" in r and r["a"] is not None:
            ab[key][ratio] = (np.asarray(r["a"], float), np.asarray(r["b"], float))
        elif ratio == "__geom__":
            embed[key] = {"q075": r.get("embed_disp_q075"), "q025": r.get("embed_disp_q025")}
    # per-seed scalar geometry metrics, then average within (pair,subject) unit, cluster by (dataset,subject)
    metrics = ["embedding_disp", "logscale_disp", "translation_disp", "slope"]
    per_unit = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # (pair,subj)->est->metric->[per-seed scalars]
    for (pair, subj, seed, est), byr in ab.items():
        if not all(q in byr for q in ("q0.50", "q0.75", "q0.25")):
            continue
        a0, b0 = byr["q0.50"]; a3, b3 = byr["q0.75"]; a1, b1 = byr["q0.25"]
        ls = 0.5 * (np.linalg.norm(a3 - a0) + np.linalg.norm(a1 - a0))
        tr = 0.5 * (np.linalg.norm(b3 - b0) + np.linalg.norm(b1 - b0))
        slope = np.linalg.norm(np.concatenate([a3 - a1, b3 - b1])) / LOG3x2   # CORRECT divisor 2*ln3
        em = embed.get((pair, subj, seed, est), {})
        ev = [x for x in (em.get("q075"), em.get("q025")) if x is not None]
        u = (pair, subj)
        per_unit[u][est]["logscale_disp"].append(float(ls))
        per_unit[u][est]["translation_disp"].append(float(tr))
        per_unit[u][est]["slope"].append(float(slope))
        if ev:
            per_unit[u][est]["embedding_disp"].append(float(np.mean(ev)))
    # average seeds within unit -> scalar per (unit, est, metric)
    unit_scalar = defaultdict(lambda: defaultdict(dict))
    for u, ed in per_unit.items():
        for est, md in ed.items():
            for m in metrics:
                if md.get(m):
                    unit_scalar[est][m][u] = float(np.mean(md[m]))
    def dataset_of(pair):
        return pair.split(":")[0]
    def cluster(est, m):
        cl = defaultdict(list)
        for (pair, subj), v in unit_scalar.get(est, {}).get(m, {}).items():
            cl[(dataset_of(pair), subj)].append(v)
        return _cluster_boot(cl)
    n_units = len(set(u for est in unit_scalar for m in unit_scalar[est] for u in unit_scalar[est][m]))
    n_clusters = len(set((dataset_of(p), s) for est in unit_scalar for m in unit_scalar[est] for (p, s) in unit_scalar[est][m]))
    out = {"unit": "(pair,session) target nested in subject; cluster=(dataset,subject)",
           "n_units": n_units, "n_unique_dataset_subject_clusters": n_clusters,
           "PRIMARY_displacement_embedding": {est: cluster(est, "embedding_disp") for est in unit_scalar},
           "logscale_displacement": {est: cluster(est, "logscale_disp") for est in unit_scalar},
           "translation_displacement": {est: cluster(est, "translation_disp") for est in unit_scalar},
           "slope_2log3": {est: cluster(est, "slope") for est in unit_scalar}}
    # paired contrasts (embedding disp), cluster-bootstrapped: FRSC - pooled, FRSC - oracle (Holm family)
    def paired(estA, estB, metric="embedding_disp"):
        A = unit_scalar.get(estA, {}).get(metric, {}); B = unit_scalar.get(estB, {}).get(metric, {})
        cl = defaultdict(list)
        for u in set(A) & set(B):
            cl[(dataset_of(u[0]), u[1])].append(A[u] - B[u])
        return _cluster_boot(cl)
    fam = {"FRSC_minus_pooled": paired("fixed_reference_oneshot", "pooled"),
           "FRSC_minus_oracle": paired("fixed_reference_oneshot", "oracle_label_conditional")}
    out["confirmatory_family_FRSC"] = fam
    out["holm_family_order"] = _holm(fam)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, nargs="+")
    ap.add_argument("--out", default="results/h2cmi/review_p0.report.json")
    args = ap.parse_args()
    rows = _load(args.inp)
    rep = {"marker": "REVIEW_P0_RESULTS", "runner_commit": RUNNER_COMMIT,
           "analyzer_commit": _analyzer_commit(), "n_rows": len(rows),
           "W1": analyze_w1(rows), "W2_primary": analyze_w2(rows, "primary"),
           "W2_secondary": analyze_w2(rows, "secondary"), "V2P_weighted": analyze_v2p(rows)}
    json.dump(rep, open(args.out, "w"), indent=2, default=str)
    # STRUCTURAL confirmation ONLY (no scientific interpretation here):
    print(f"analyzer_commit={rep['analyzer_commit']} rows={rep['n_rows']}")
    print(f"W1 units={rep['W1']['n_units']} resid_max={rep['W1']['decomposition_residual_max']:.2e}")
    print(f"W2pri units={rep['W2_primary']['n_units']} resid_max={rep['W2_primary']['decomposition_residual_max']:.2e}")
    print(f"V2P units={rep['V2P_weighted']['n_units']} clusters={rep['V2P_weighted']['n_unique_dataset_subject_clusters']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
