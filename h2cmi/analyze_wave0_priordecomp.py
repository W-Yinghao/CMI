"""WAVE 0 prior-decomposition analyzer (frozen W0.3_MECH_APPENDUM). Five outputs:
1. main table P_cross / P_same / P_cross-P_same / |P_same|/|P_cross|;
2. three-part decomposition (metric-mismatch + transfer + estimation) for crossnight & samesession;
3. per-stage recall under {Unif, rho_E, rho_A, pi_J};
4. prior-quality diagnostics (TV/KL/min/piJ-rhoA);
5. directionality P_AB - P_BA (samesession).
Subject cluster; P_same_s = mean over seeds+directions; P_cross_s = mean over seeds. 10k bootstrap.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict

import numpy as np

NB = 10000
STAGES = ["W", "N1", "N2", "N3", "REM"]


def _load(proto):
    rows = []
    for f in glob.glob(f"results/h2cmi/wave0_priordecomp/pd_{proto}_*.jsonl"):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r.get("marker") == "PRIORDECOMP" and "P_J" in r:
                    rows.append(r)
    return rows


def _subject_mean(rows, field):
    """subject -> mean over all its rows (seeds [+ directions])."""
    by = defaultdict(list)
    for r in rows:
        by[int(r["target_subject"])].append(float(r[field]))
    return {s: float(np.mean(v)) for s, v in by.items()}


def _cluster_boot_vals(subj_vals, seed=0):
    vals = np.array([subj_vals[s] for s in subj_vals], float)
    if len(vals) < 2:
        return dict(mean=float(vals.mean()) if len(vals) else float("nan"), ci=[float("nan")] * 2, n=len(vals))
    rng = np.random.default_rng(seed)
    bs = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(NB)]
    return dict(mean=float(vals.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                excludes_0=bool(np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0), n=len(vals))


def _decomp_table(rows):
    out = {}
    for f in ("P_J", "metric_mismatch", "transfer", "estimation"):
        out[f] = _cluster_boot_vals(_subject_mean(rows, f))
    # exactness check
    resid = max(abs(float(r["residual"])) for r in rows) if rows else 0.0
    out["max_residual"] = resid
    # consistency vs main run
    out["max_abs_consistency_B_piJ_vs_main"] = max(abs(float(r.get("consistency_B_piJ_vs_main", 0))) for r in rows) if rows else 0.0
    return out


def _recall_table(rows):
    out = {}
    for key in ("recall_unif", "recall_rhoE", "recall_rhoA", "recall_piJ"):
        per_stage = []
        for c in range(len(STAGES)):
            sv = defaultdict(list)
            for r in rows:
                if r.get(key) and r[key][c] == r[key][c]:
                    sv[int(r["target_subject"])].append(r[key][c])
            subj_mean = {s: float(np.mean(v)) for s, v in sv.items()}
            per_stage.append(round(float(np.mean(list(subj_mean.values()))), 4) if subj_mean else None)
        out[key] = dict(zip(STAGES, per_stage))
    return out


def _prior_quality(rows):
    out = {}
    for f in ("TV_piJ_rhoA", "TV_piJ_rhoE", "KL_rhoA_piJ", "min_piJ"):
        out[f] = _cluster_boot_vals(_subject_mean(rows, f))
    # stage-specific piJ - rhoA
    stage = []
    for c in range(len(STAGES)):
        sv = defaultdict(list)
        for r in rows:
            sv[int(r["target_subject"])].append(r["piJ_minus_rhoA"][c])
        subj_mean = {s: float(np.mean(v)) for s, v in sv.items()}
        stage.append(round(float(np.mean(list(subj_mean.values()))), 4))
    out["piJ_minus_rhoA_by_stage"] = dict(zip(STAGES, stage))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/h2cmi/wave0_priordecomp.report.json")
    args = ap.parse_args()
    cross = _load("crossnight"); same = _load("samesession")
    Pc = _subject_mean(cross, "P_J"); Ps = _subject_mean(same, "P_J")
    common = sorted(set(Pc) & set(Ps))
    dP = {s: Pc[s] - Ps[s] for s in common}
    # ratio R = |mean P_same| / |mean P_cross| with bootstrap
    vs = np.array([Ps[s] for s in common]); vc = np.array([Pc[s] for s in common])
    rng = np.random.default_rng(0); ratios = []
    for _ in range(NB):
        idx = rng.integers(0, len(common), len(common))
        mc = vc[idx].mean()
        if abs(mc) > 1e-9:
            ratios.append(abs(vs[idx].mean()) / abs(mc))
    R = dict(mean=float(abs(vs.mean()) / abs(vc.mean())) if abs(vc.mean()) > 1e-9 else float("nan"),
             ci=[float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))] if ratios else [float("nan")] * 2,
             n=len(common))
    # directionality for samesession
    ab = _subject_mean([r for r in same if r["direction"] == "AB"], "P_J")
    ba = _subject_mean([r for r in same if r["direction"] == "BA"], "P_J")
    dir_c = sorted(set(ab) & set(ba))
    dir_delta = _cluster_boot_vals({s: ab[s] - ba[s] for s in dir_c})
    rep = dict(marker="WAVE0_PRIORDECOMP", n_subjects_cross=len(Pc), n_subjects_same=len(Ps),
               main_table=dict(P_cross=_cluster_boot_vals(Pc), P_same=_cluster_boot_vals(Ps),
                               P_cross_minus_same=_cluster_boot_vals(dP), ratio_same_over_cross=R),
               decomposition_crossnight=_decomp_table(cross), decomposition_samesession=_decomp_table(same),
               per_stage_recall_samesession=_recall_table(same), per_stage_recall_crossnight=_recall_table(cross),
               prior_quality_samesession=_prior_quality(same), prior_quality_crossnight=_prior_quality(cross),
               directionality_P_AB_minus_BA=dir_delta)
    json.dump(rep, open(args.out, "w"), indent=2, default=str)
    m = rep["main_table"]
    print(f"[PRIORDECOMP] cross_subj={len(Pc)} same_subj={len(Ps)}")
    print(f"  P_cross={m['P_cross']['mean']:+.4f} {m['P_cross']['ci']}")
    print(f"  P_same ={m['P_same']['mean']:+.4f} {m['P_same']['ci']}")
    print(f"  P_cross-P_same={m['P_cross_minus_same']['mean']:+.4f} {m['P_cross_minus_same']['ci']}")
    print(f"  R=|P_same|/|P_cross|={m['ratio_same_over_cross']['mean']:.3f} {m['ratio_same_over_cross']['ci']}")
    dc = rep["decomposition_samesession"]
    print(f"  W0.3 split: metric_mismatch={dc['metric_mismatch']['mean']:+.4f} transfer={dc['transfer']['mean']:+.4f} estimation={dc['estimation']['mean']:+.4f} (resid {dc['max_residual']:.1e}, consistency {dc['max_abs_consistency_B_piJ_vs_main']:.1e})")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
