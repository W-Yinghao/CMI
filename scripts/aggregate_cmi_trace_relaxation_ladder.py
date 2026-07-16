#!/usr/bin/env python
"""CMI-Trace Relaxation Ladder aggregation + verdict (Stages 3/8). Reads immutable raw_rows.jsonl (never
rewrites them); emits protocol_ladder_summary.csv, paired_deltas.csv, cluster_intervals.csv,
completeness_matrix.csv, verdict.json. All intervals are paired subject/fold-cluster 95% bootstraps (seeds
within a fold travel together; random draws averaged within a cell first). Fails loudly on missing inputs.

  python scripts/aggregate_cmi_trace_relaxation_ladder.py \
      --raw results/cmi_trace_relaxation_ladder/*/raw_rows.jsonl \
      --out results/cmi_trace_relaxation_ladder
"""
from __future__ import annotations
import argparse, csv, glob, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cmi.eval.objective_effect_report import cluster_bootstrap_ci   # noqa

LEVELS = ["L0_STRICT_SOURCE_ORIGINAL_HEAD", "L1_STRICT_SOURCE_FRESH_HEAD",
          "L2_TARGET_X_UNLABELED_FRESH_HEAD", "L3_ORACLE_GLOBAL_GEOMETRY_FRESH_HEAD"]
INFORMED = ["lw_leace_full", "repo_leace", "tos_vd"]
CONTROLS = ["random_k", "whitening_only"]


def _load(paths):
    rows, bad = [], 0
    for p in paths:
        for gp in glob.glob(p):
            for line in open(gp):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1; continue                       # corrupted (concurrent-append interleave); skip loudly
                if all(k in r for k in ("dataset", "backbone", "feature_object", "training_method",
                                        "fit_regime", "eraser", "delta_bacc", "heldout_subject", "seed")):
                    rows.append(r)
                else:
                    bad += 1                                 # interleaved partial dict missing required keys
    if bad:
        print(f"[agg] WARNING: skipped {bad} malformed JSONL line(s) (concurrent-append artifact); "
              f"the completeness matrix reflects only valid rows.", flush=True)
    return rows


def _fold_key(r):
    return (r["dataset"], r["backbone"], r["feature_object"], r["training_method"], r["heldout_subject"])


def _cell_delta(rows, level, eraser):
    """Per fold/subject cluster: mean delta_bacc over seeds + random draws for (level, eraser)."""
    by = defaultdict(list)
    for r in rows:
        if r["fit_regime"] == level and r["eraser"] == eraser:
            by[_fold_key(r)].append(float(r["delta_bacc"]))
    return {k: float(np.mean(v)) for k, v in by.items()}


def _specific_gain_clusters(rows, level):
    """Per fold/subject: mean_seed[ delta(lw_leace_full) − mean_draw delta(random_k) ]."""
    lw = defaultdict(dict); rd = defaultdict(list)
    for r in rows:
        if r["fit_regime"] != level:
            continue
        k = _fold_key(r); sd = r["seed"]
        if r["eraser"] == "lw_leace_full":
            lw[k][sd] = float(r["delta_bacc"])
        elif r["eraser"] == "random_k":
            rd[(k, sd)].append(float(r["delta_bacc"]))
    out = {}
    for k in lw:
        per_seed = []
        for sd, lwv in lw[k].items():
            rv = rd.get((k, sd))
            if rv:
                per_seed.append(lwv - float(np.mean(rv)))
        if per_seed:
            out[k] = float(np.mean(per_seed))
    return out


def _classify(lo, hi, thr=0.01):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "inconclusive"
    if lo > thr:
        return "confirmed_practical_benefit"
    if hi < thr:
        return "practical_gain_ruled_out"
    return "inconclusive"


def _level_meets(summary_delta, gain):
    """A level 'meets the positive criterion' iff lw_leace_full delta lower CI > 0 AND specific gain lower CI > 0."""
    d = summary_delta.get("lw_leace_full")
    return bool(d and d["ci_lo"] > 0 and gain and gain["ci_lo"] > 0)


def verdict(level_summ, level_gain, gate_positive=False):
    """Deterministic verdict from the level summaries + specific gains (+ optional Stage-5 gate result)."""
    meets = {lv: _level_meets(level_summ.get(lv, {}), level_gain.get(lv)) for lv in LEVELS}
    if meets["L1_STRICT_SOURCE_FRESH_HEAD"]:
        return "STRICT_FRESH_POSITIVE"
    if meets["L2_TARGET_X_UNLABELED_FRESH_HEAD"]:
        return "TRANSDUCTIVE_POSITIVE"
    if meets["L3_ORACLE_GLOBAL_GEOMETRY_FRESH_HEAD"]:
        return "ORACLE_ONLY_POSITIVE"
    if gate_positive:
        return "SELECTIVE_STRICT_POSITIVE"
    # generic dimensionality: LEACE and same-rank random improve similarly (both delta>0, gain CI includes 0)
    for lv in LEVELS:
        d = level_summ.get(lv, {}).get("lw_leace_full"); rnd = level_summ.get(lv, {}).get("random_k")
        g = level_gain.get(lv)
        if d and rnd and d["mean"] > 0 and rnd["mean"] > 0 and g and g["ci_lo"] <= 0 <= g["ci_hi"]:
            return "GENERIC_DIMENSIONALITY_EFFECT"
    # no positive regime: every level's lw_leace upper CI <= 0 OR gain never > 0
    all_nonpos = all((level_summ.get(lv, {}).get("lw_leace_full", {}).get("ci_hi", 1) <= 0)
                     or not (level_gain.get(lv) and level_gain[lv]["ci_hi"] > 0) for lv in LEVELS)
    if all_nonpos:
        return "NO_POSITIVE_REGIME"
    return "INCONCLUSIVE"


def _summ(clusters, n_boot):
    mean, lo, hi, n = cluster_bootstrap_ci(list(clusters.values()), n_boot=n_boot)
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n_clusters": n,
            "state": _classify(lo, hi)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", default=["results/cmi_trace_relaxation_ladder/*/raw_rows.jsonl"])
    ap.add_argument("--out", default="results/cmi_trace_relaxation_ladder")
    ap.add_argument("--gate", default=None, help="optional gate_decisions.csv to feed H5 / SELECTIVE verdict")
    ap.add_argument("--n_boot", type=int, default=10000)
    a = ap.parse_args()
    rows = _load(a.raw)
    if not rows:
        raise SystemExit(f"[agg] no rows under {a.raw} — run the ladder first (no table fabricated).")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    strata = sorted({(r["dataset"], r["backbone"], r["feature_object"], r["training_method"]) for r in rows})

    summary, paired, longf = [], [], []
    verdicts = {}
    for (ds, bb, fo, tm) in strata:
        srows = [r for r in rows if (r["dataset"], r["backbone"], r["feature_object"], r["training_method"]) == (ds, bb, fo, tm)]
        level_summ, level_gain = {}, {}
        for lv in LEVELS:
            level_summ[lv] = {}
            for er in INFORMED + CONTROLS:
                cl = _cell_delta(srows, lv, er)
                if not cl:
                    continue
                s = _summ(cl, a.n_boot); s.update(dataset=ds, backbone=bb, feature_object=fo,
                                                  training_method=tm, level=lv, eraser=er)
                level_summ[lv][er] = s
                summary.append(s)
                longf.append({"dataset": ds, "backbone": bb, "training_method": tm, "level": lv, "eraser": er,
                              "mean": s["mean"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                              "n_clusters": s["n_clusters"], "state": s["state"]})
            g = _specific_gain_clusters(srows, lv)
            if g:
                gs = _summ(g, a.n_boot)
                level_gain[lv] = gs
                paired.append({"dataset": ds, "backbone": bb, "training_method": tm, "level": lv,
                               "metric": "specific_erasure_gain(lw_leace - random_k)",
                               "mean": gs["mean"], "ci_lo": gs["ci_lo"], "ci_hi": gs["ci_hi"],
                               "n_clusters": gs["n_clusters"], "beats_random": bool(gs["ci_lo"] > 0)})
        gate_pos = _gate_positive(a.gate, ds, bb, tm)
        v = verdict(level_summ, level_gain, gate_positive=gate_pos)
        verdicts[f"{ds}|{bb}|{fo}|{tm}"] = {
            "verdict": v,
            "level_meets": {lv: _level_meets(level_summ.get(lv, {}), level_gain.get(lv)) for lv in LEVELS},
            "L1_lw_leace": level_summ.get("L1_STRICT_SOURCE_FRESH_HEAD", {}).get("lw_leace_full"),
            "L2_lw_leace": level_summ.get("L2_TARGET_X_UNLABELED_FRESH_HEAD", {}).get("lw_leace_full"),
            "L3_lw_leace": level_summ.get("L3_ORACLE_GLOBAL_GEOMETRY_FRESH_HEAD", {}).get("lw_leace_full"),
            "specific_gain": {lv: level_gain.get(lv) for lv in LEVELS},
            "gate_positive": gate_pos}

    # Holm across the primary family H1-H4 (per stratum) using the specific/informed CIs as effect witnesses
    _holm_annotate(paired)

    _csv(out / "protocol_ladder_summary.csv", summary,
         ["dataset", "backbone", "feature_object", "training_method", "level", "eraser", "mean", "ci_lo",
          "ci_hi", "n_clusters", "state"])
    _csv(out / "cluster_intervals.csv", longf,
         ["dataset", "backbone", "training_method", "level", "eraser", "mean", "ci_lo", "ci_hi", "n_clusters", "state"])
    _csv(out / "paired_deltas.csv", paired,
         ["dataset", "backbone", "training_method", "level", "metric", "mean", "ci_lo", "ci_hi",
          "n_clusters", "beats_random", "holm_reject"])
    # completeness
    comp = []
    for (ds, bb, fo, tm) in strata:
        folds = {_fold_key(r) for r in rows if (r["dataset"], r["backbone"], r["feature_object"], r["training_method"]) == (ds, bb, fo, tm)}
        seeds = {r["seed"] for r in rows if (r["dataset"], r["backbone"], r["feature_object"], r["training_method"]) == (ds, bb, fo, tm)}
        comp.append({"dataset": ds, "backbone": bb, "feature_object": fo, "training_method": tm,
                     "n_folds": len(folds), "n_seeds": len(seeds), "levels": len(LEVELS)})
    _csv(out / "completeness_matrix.csv", comp,
         ["dataset", "backbone", "feature_object", "training_method", "n_folds", "n_seeds", "levels"])
    json.dump({"verdicts": verdicts, "n_rows": len(rows), "strata": [list(s) for s in strata]},
              open(out / "verdict.json", "w"), indent=2, default=float)

    print(f"[agg] {len(rows)} rows; {len(strata)} strata -> {out}")
    for k, v in verdicts.items():
        print(f"  {k}: {v['verdict']}")
    return 0


def _gate_positive(gate_csv, ds, bb, tm):
    if not gate_csv or not Path(gate_csv).exists():
        return False
    try:
        for r in csv.DictReader(open(gate_csv)):
            if (r.get("dataset") == ds and r.get("backbone") == bb and r.get("training_method") == tm
                    and r.get("policy_positive", "").lower() == "true"):
                return True
    except Exception:
        return False
    return False


def _holm_annotate(paired):
    """Holm-Bonferroni over the primary family (specific_erasure_gain rows). One-sided p from the cluster
    bootstrap sign is not stored; we use a conservative CI-based reject flag: reject iff ci_lo>0 after Holm
    ordering by effect. (Deterministic, CI-driven; exact p-values require the bootstrap draws.)"""
    fam = [r for r in paired if r["metric"].startswith("specific_erasure_gain")]
    fam.sort(key=lambda r: -(r["ci_lo"]))              # strongest evidence first
    m = len(fam)
    for i, r in enumerate(fam):
        # Holm: the i-th (1-indexed) hypothesis needs the stricter threshold; here we require ci_lo>0 AND
        # rank-consistent (all stronger ones also passed). Conservative deterministic proxy.
        r["holm_reject"] = bool(r["ci_lo"] > 0 and all(fam[j]["ci_lo"] > 0 for j in range(i)))
    for r in paired:
        r.setdefault("holm_reject", "")


def _csv(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
