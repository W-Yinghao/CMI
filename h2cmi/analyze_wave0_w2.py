"""WAVE 0 / W0.1 analyzer. Aggregates the deterministic eval-only W2 re-evaluation:
(a) re-confirms the G/P/Interaction decomposition + primary/secondary contrasts (subject bootstrap,
    seeds averaged within unit) and compares to the frozen terminal report;
(b) aggregates the now-ADMISSIBLE per-stage recall / confusion (per (subject,seed) row-normalize ->
    per subject average -> overall average) for all 9 branches;
(c) attests determinism (self-replay bit-reproducibility, recorded separately).
Interpretation stays out of here; that goes in the results doc."""
from __future__ import annotations

import glob
import json
from collections import defaultdict

import numpy as np

NB = 10000
STAGES = ["W", "N1", "N2", "N3", "REM"]
BRANCHES = ["identity_uniform", "identity_joint_prior", "joint_geometry_uniform",
            "joint_geometry_joint_prior", "fixed_iterative_geometry_uniform",
            "fixed_reference_oneshot_uniform", "pooled_uniform", "latent_im_diag_uniform",
            "source_recolored_ea"]


def _load(proto):
    rows = []
    for f in glob.glob(f"results/h2cmi/wave0_w2det/p0w2det_{proto}_*.jsonl"):
        if "scratch" in f:
            continue
        for l in open(f):
            if l.strip():
                rows.append(json.loads(l))
    return rows


def _boot(vals, seed=0):
    v = np.asarray([x for x in vals if x == x], float)
    if len(v) < 2:
        return dict(mean=float(v.mean()) if len(v) else float("nan"), ci=[float("nan")] * 2, n=int(len(v)))
    rng = np.random.default_rng(seed)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(NB)]
    return dict(mean=float(v.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                excludes_0=bool(np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0), n=int(len(v)))


def _unit(rows):
    bacc = defaultdict(lambda: defaultdict(list)); dec = defaultdict(lambda: defaultdict(list))
    for r in rows:
        u = r["target_subject"]
        if r.get("branch") == "__decomposition__":
            for k in ("G", "P", "interaction", "full_joint_delta", "residual"):
                if k in r:
                    dec[u][k].append(r[k])
        elif "bacc" in r and r.get("branch"):
            bacc[u][r["branch"]].append(r["bacc"])
    ub = {u: {b: float(np.mean(v)) for b, v in d.items()} for u, d in bacc.items()}
    ud = {u: {k: float(np.mean(v)) for k, v in d.items()} for u, d in dec.items()}
    return ub, ud


def _confusion(rows):
    """per (subject,seed) row-normalize -> per subject average over seeds -> overall average over subjects."""
    out = {}
    for b in BRANCHES:
        by_subj = defaultdict(list); rec_by_subj = defaultdict(list)
        for r in rows:
            if r.get("branch") == b and "confusion" in r:
                C = np.asarray(r["confusion"], float)
                by_subj[r["target_subject"]].append(C / np.clip(C.sum(1, keepdims=True), 1, None))
                if r.get("per_stage_recall"):
                    rec_by_subj[r["target_subject"]].append(r["per_stage_recall"])
        if not by_subj:
            continue
        per_subj = [np.mean(v, 0) for v in by_subj.values()]
        overall = np.mean(per_subj, 0)
        recall = [float(np.nanmean([np.nanmean([s[c] for s in seeds]) for seeds in rec_by_subj.values()]))
                  for c in range(5)] if rec_by_subj else None
        out[b] = dict(n_subjects=len(per_subj), aggregate_rownorm_confusion=overall.round(4).tolist(),
                      per_stage_recall=(None if recall is None else [round(x, 4) for x in recall]))
    return out


def analyze(proto):
    rows = _load(proto)
    ub, ud = _unit(rows)
    def c(a, b):
        return _boot([v[a] - v[b] for v in ub.values() if a in v and b in v])
    out = dict(protocol=proto, n_units=len(ub),
               PRIMARY_G_joint_geom_minus_identity=c("joint_geometry_uniform", "identity_uniform"),
               P_identity_jointprior_minus_identity=c("identity_joint_prior", "identity_uniform"),
               decision_prior_diagnostic=c("joint_geometry_joint_prior", "joint_geometry_uniform"),
               SECONDARY_fixed_iter_minus_joint_geom=c("fixed_iterative_geometry_uniform", "joint_geometry_uniform"),
               interaction_mean=float(np.mean([d["interaction"] for d in ud.values() if "interaction" in d])) if ud else float("nan"),
               decomposition_residual_max=float(np.max([abs(d.get("residual", 0.0)) for d in ud.values()])) if ud else 0.0,
               branch_mean_bacc={m: float(np.mean([v[m] for v in ub.values() if m in v])) for m in BRANCHES},
               per_stage_confusion=_confusion(rows))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/h2cmi/wave0_w2.report.json")
    args = ap.parse_args()
    rep = dict(marker="WAVE0_W2_DETERMINISTIC", design="eval-only reuse of frozen terminal bundles (763bf49d)",
               W2_primary=analyze("primary"), W2_secondary=analyze("secondary"))
    # terminal comparison (re-confirmation delta)
    try:
        term = json.load(open("h2cmi/results/review_p0.report.json"))["W2_primary"]
        rep["terminal_comparison"] = {
            "G_terminal": term["PRIMARY_G_joint_geom_minus_identity"]["mean"],
            "G_wave0_det": rep["W2_primary"]["PRIMARY_G_joint_geom_minus_identity"]["mean"],
            "P_terminal": term["P_identity_jointprior_minus_identity"]["mean"],
            "P_wave0_det": rep["W2_primary"]["P_identity_jointprior_minus_identity"]["mean"]}
    except Exception as e:
        rep["terminal_comparison"] = f"unavailable: {e}"
    json.dump(rep, open(args.out, "w"), indent=2, default=str)
    p = rep["W2_primary"]
    print(f"[W0.1] primary units={p['n_units']} resid={p['decomposition_residual_max']:.1e}")
    print(f"  G={p['PRIMARY_G_joint_geom_minus_identity']['mean']:+.4f} "
          f"P={p['P_identity_jointprior_minus_identity']['mean']:+.4f} "
          f"fixed_iter-joint={p['SECONDARY_fixed_iter_minus_joint_geom']['mean']:+.4f}")
    print(f"  confusion admissible for {len(p['per_stage_confusion'])} branches")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
