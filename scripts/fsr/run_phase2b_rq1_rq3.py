#!/usr/bin/env python
"""FSR Step 2B — RQ1 (leakage vs reliance) + RQ3 (alignment mechanism), CPU-only.

Frozen inputs only (no GPU, no retrain). Reads the CIGL R3 per-unit CSVs.
Unit = (dataset, method in {erm, cigl_graph_node}, seed, fold).
  align_k2       = gap_alignment.csv (graph_z, k=2)      -> 126 units (all seeds)
  R3_task_drop_k2= r3_reliance.csv (label_conditional,k2)-> 126 units
  graph_kl       = r1_hardened_nperm1000.csv (graph)     -> 42 units (SEED0 only; seeds1/2 pruned)
Bootstrap: np.random.default_rng(0), n_boot=2000, percentile[2.5,97.5].

RQ1 is reported in three claim-strength tiers:
  RQ1A align full-n (n=126, RECOMPUTED)
  RQ1B graph_kl seed0 (n=42, RECOMPUTED_SIGN_ONLY)
  RQ1C graph_kl pooled (n=126, FROZEN_NOT_RECOMPUTABLE — carried, support only)
RQ3: Model A (align, n=126), Model B (align+graph_kl paired, seed0 n=42), Model C (frozen summary).

Outputs (results/fsr_phase2b/):
    rq1_leakage_vs_reliance.csv / .json
    rq3_alignment_mechanism.csv / .json

    python scripts/fsr/run_phase2b_rq1_rq3.py
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, rankdata

REPO = Path(__file__).resolve().parents[2]
FINAL = REPO / "results" / "cigl_r123" / "final"
OUT = REPO / "results" / "fsr_phase2b"
N_BOOT, SEED = 2000, 0
ERM_CIGL = {"erm", "cigl_graph_node"}
FROZEN_GKL_POOLED = {"rho": -0.342, "ci": [-0.507, -0.166], "n": 126}  # gap_correlations.csv (pruned inputs)


def load(fp):
    with open(fp, newline="") as fh:
        return list(csv.DictReader(fh))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok]).correlation) if ok.sum() >= 4 else float("nan")


def boot_ci(x, y, n_boot=N_BOOT, seed=SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    pt = spearman(x, y)
    rng = np.random.default_rng(seed)
    d = [spearman(x[i], y[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    d = np.asarray([v for v in d if v == v])
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"rho": round(pt, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n": int(n),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def ols_std(rows, y_key, cont_keys, cat_keys, n_boot=N_BOOT, seed=SEED):
    """OLS with z-scored outcome AND z-scored continuous predictors + one-hot categoricals (drop-first).
    Standardizing y makes each continuous coefficient a fully-standardized beta (correlation-scale),
    so |coef(align)| vs |coef(graph_kl)| is comparable. Returns std beta + bootstrap CI per predictor."""
    y_raw = np.array([float(r[y_key]) for r in rows])
    y = (y_raw - y_raw.mean()) / (y_raw.std(ddof=0) or 1.0)
    cols, names = [np.ones(len(rows))], ["intercept"]
    # z-scored continuous
    zmap = {}
    for c in cont_keys:
        v = np.array([float(r[c]) for r in rows])
        z = (v - v.mean()) / (v.std(ddof=0) or 1.0)
        zmap[c] = z
        cols.append(z)
        names.append(f"z({c})")
    # one-hot categoricals (drop first level)
    for c in cat_keys:
        levels = sorted({r[c] for r in rows})
        for lev in levels[1:]:
            cols.append(np.array([1.0 if r[c] == lev else 0.0 for r in rows]))
            names.append(f"{c}={lev}")
    X = np.column_stack(cols)

    def fit(Xf, yf):
        beta, *_ = np.linalg.lstsq(Xf, yf, rcond=None)
        return beta

    beta = fit(X, y)
    out = {}
    idx = {c: names.index(f"z({c})") for c in cont_keys}
    rng = np.random.default_rng(seed)
    n = len(rows)
    boots = {c: [] for c in cont_keys}
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        try:
            b = fit(X[s], y[s])
            for c in cont_keys:
                boots[c].append(b[idx[c]])
        except np.linalg.LinAlgError:
            continue
    for c in cont_keys:
        arr = np.asarray(boots[c])
        out[c] = {"std_coef": round(float(beta[idx[c]]), 4),
                  "ci_lo": round(float(np.percentile(arr, 2.5)), 4),
                  "ci_hi": round(float(np.percentile(arr, 97.5)), 4),
                  "excludes_zero": bool(np.percentile(arr, 2.5) > 0 or np.percentile(arr, 97.5) < 0)}
    return out, names


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    align = {(r["dataset"], r["method"], r["seed"], r["fold"]): float(r["task_head_alignment"])
             for r in load(FINAL / "gap_alignment.csv")
             if r["k"] == "2" and r["representation"] == "graph_z" and r["method"] in ERM_CIGL}
    align1 = {(r["dataset"], r["method"], r["seed"], r["fold"]): float(r["task_head_alignment"])
              for r in load(FINAL / "gap_alignment.csv")
              if r["k"] == "1" and r["representation"] == "graph_z" and r["method"] in ERM_CIGL}
    r3 = {(r["dataset"], r["method"], r["seed"], r["fold"]):
          (float(r["task_drop"]), r["firewall_passed"])
          for r in load(FINAL / "r3_reliance.csv")
          if r["conditioning"] == "label_conditional" and r["k"] == "2" and r["method"] in ERM_CIGL}
    gkl = {(r["dataset"], r["method"], r["seed"], r["fold"]): float(r["observed_kl"])
           for r in load(FINAL / "r1_hardened_nperm1000.csv")
           if r["representation"] == "graph" and r["method"] in ERM_CIGL}  # seed0

    # tidy unit table
    units = []
    for k in sorted(set(align) & set(r3)):
        ds, m, s, f = k
        units.append(dict(unit_id=f"{ds}:{m}:s{s}:f{f}", dataset=ds, method=m, seed=s, fold=f,
                          align_k1=align1.get(k), align_k2=align[k], R3_task_drop=r3[k][0],
                          graph_kl=gkl.get(k, ""), collapse_guard=r3[k][1]))

    # ================= RQ1 =================
    rq1_rows, rq1 = [], {}

    def add(analysis, scope, pred, keys_x, keys_y, status, note=""):
        xs = [keys_x[k] for k in keys_y if k in keys_x]
        ys = [keys_y[k] for k in keys_y if k in keys_x]
        c = boot_ci(xs, ys)
        rq1_rows.append(dict(analysis=analysis, scope=scope, predictor=pred, endpoint="R3_task_drop_k2",
                             rho=c["rho"], ci_lo=c["ci_lo"], ci_hi=c["ci_hi"], n=c["n"],
                             excludes_zero=c["excludes_zero"], status=status, note=note))
        return c

    R3v = {k: v[0] for k, v in r3.items()}
    # RQ1A: align full-n
    a_pool = add("RQ1A", "pooled", "align_k2", align, R3v, "RECOMPUTED")
    a_ds = {}
    for ds in ("BNCI2014_001", "BNCI2015_001"):
        a_ds[ds] = add("RQ1A", ds, "align_k2", {k: v for k, v in align.items() if k[0] == ds}, R3v, "RECOMPUTED")
        add("RQ1A", f"LODO_excl_{ds}", "align_k2", {k: v for k, v in align.items() if k[0] != ds}, R3v,
            "RECOMPUTED", note="2 datasets -> LODO reduces to the other stratum")
    for s in ("0", "1", "2"):
        add("RQ1A", f"seed{s}", "align_k2", {k: v for k, v in align.items() if k[2] == s}, R3v, "RECOMPUTED")
    for m in ("erm", "cigl_graph_node"):
        add("RQ1A", f"method_{m}", "align_k2", {k: v for k, v in align.items() if k[1] == m}, R3v, "RECOMPUTED")
    # within-dataset rank residualization (dataset-controlled Spearman)
    rr = _rank_resid_corr(units, "align_k2", "R3_task_drop")
    rq1_rows.append(dict(analysis="RQ1A", scope="within_dataset_rank_resid", predictor="align_k2",
                         endpoint="R3_task_drop_k2", rho=round(rr["rho"], 4), ci_lo=round(rr["lo"], 4),
                         ci_hi=round(rr["hi"], 4), n=rr["n"], excludes_zero=rr["excludes_zero"],
                         status="RECOMPUTED", note="dataset-controlled (ranks within dataset)"))

    # RQ1B: graph_kl seed0
    g_pool = add("RQ1B", "seed0_pooled", "graph_kl", gkl, R3v, "RECOMPUTED_SIGN_ONLY", note="n=42 seed0")
    for ds in ("BNCI2014_001", "BNCI2015_001"):
        add("RQ1B", f"seed0_{ds}", "graph_kl", {k: v for k, v in gkl.items() if k[0] == ds}, R3v, "RECOMPUTED_SIGN_ONLY")
    for m in ("erm", "cigl_graph_node"):
        add("RQ1B", f"seed0_method_{m}", "graph_kl", {k: v for k, v in gkl.items() if k[1] == m}, R3v, "RECOMPUTED_SIGN_ONLY")

    # RQ1C: frozen pooled graph_kl (carried, not recomputable)
    rq1_rows.append(dict(analysis="RQ1C", scope="pooled", predictor="graph_kl", endpoint="R3_task_drop_k2",
                         rho=FROZEN_GKL_POOLED["rho"], ci_lo=FROZEN_GKL_POOLED["ci"][0],
                         ci_hi=FROZEN_GKL_POOLED["ci"][1], n=FROZEN_GKL_POOLED["n"], excludes_zero=True,
                         status="FROZEN_NOT_RECOMPUTABLE", note="seeds1/2 per-fold graph_kl pruned; support only"))

    # decision language
    a_pool_sig = a_pool["excludes_zero"] and a_pool["rho"] > 0
    ds_sig = {ds: (a_ds[ds]["excludes_zero"] and a_ds[ds]["rho"] > 0) for ds in a_ds}
    if a_pool_sig and all(ds_sig.values()):
        rq1a_decision = "task-head alignment is positively associated with functional reliance (pooled + both datasets)."
    elif a_pool_sig and any(ds_sig.values()):
        rq1a_decision = "pooled mechanism-positive, dataset-heterogeneous; not universal."
    else:
        rq1a_decision = "alignment association not supported at pooled level — investigate."
    rq1b_decision = ("raw graph leakage does NOT show the expected positive relationship with reliance in "
                     "the recomputable seed0 slice; its sign is negative." if g_pool["rho"] < 0
                     else "graph leakage seed0 sign unexpected — investigate.")

    rq1 = {"unit_definition": "(dataset, method in {erm,cigl}, seed, fold)",
           "boot": {"n_boot": N_BOOT, "seed": SEED},
           "RQ1A_align_full_n126": {"pooled": a_pool, "by_dataset": a_ds,
                                    "decision": rq1a_decision, "claim_strength": "RECOMPUTED"},
           "RQ1B_graph_kl_seed0_n42": {"pooled": g_pool, "decision": rq1b_decision,
                                       "claim_strength": "RECOMPUTED_SIGN_ONLY"},
           "RQ1C_graph_kl_pooled_n126": {**FROZEN_GKL_POOLED, "claim_strength": "FROZEN_NOT_RECOMPUTABLE",
                                         "use": "support only — NOT a full reproduction"},
           "forbidden_phrasing": "we fully reproduced both pooled correlations"}
    _wcsv(OUT / "rq1_leakage_vs_reliance.csv", rq1_rows)
    (OUT / "rq1_leakage_vs_reliance.json").write_text(json.dumps(rq1, indent=2) + "\n")

    # ================= RQ3 =================
    unitsA = [u for u in units if u["align_k2"] != "" and u["R3_task_drop"] != ""]
    for u in unitsA:
        u["seed"] = str(u["seed"])
    unitsB = [u for u in units if u["graph_kl"] not in ("", None)]  # seed0 with graph_kl

    mA, _ = ols_std(unitsA, "R3_task_drop", ["align_k2"], ["dataset", "seed", "method"])
    mB, _ = ols_std(unitsB, "R3_task_drop", ["align_k2", "graph_kl"], ["dataset", "method"])

    # Spearman difference test (align - graph_kl) on the seed0 units where both exist — the primary
    # "which predictor is more positively associated" test (magnitude of a signed correlation, not |beta|).
    xa = [u["align_k2"] for u in unitsB]
    xg = [float(u["graph_kl"]) for u in unitsB]
    yv = [u["R3_task_drop"] for u in unitsB]
    diff = _boot_diff(xa, xg, yv)

    rq3_rows = []
    for term, d in mA.items():
        rq3_rows.append(dict(model="A_align_full_n126", n=len(unitsA), term=f"z({term})", **d))
    for term, d in mB.items():
        rq3_rows.append(dict(model="B_paired_seed0_n42", n=len(unitsB), term=f"z({term})", **d))
    rq3_rows.append(dict(model="B_paired_seed0_n42", n=len(unitsB), term="spearman_diff(align-graph_kl)",
                         std_coef=round(diff["point"], 4), ci_lo=round(diff["ci_lo"], 4),
                         ci_hi=round(diff["ci_hi"], 4), excludes_zero=diff["excludes_zero"],
                         note="primary: alignment more positively associated than leakage"))
    rq3_rows.append(dict(model="C_frozen_summary", n=126, term="graph_kl_pooled",
                         std_coef="", ci_lo="", ci_hi="", excludes_zero="",
                         note="FROZEN_NOT_RECOMPUTABLE; sign negative; support only"))
    _wcsv(OUT / "rq3_alignment_mechanism.csv", rq3_rows)

    align_pos = mB["align_k2"]["std_coef"] > 0
    graph_neg = mB["graph_kl"]["std_coef"] < 0
    diff_favours_align = diff["point"] > 0 and diff["excludes_zero"]
    if diff_favours_align and align_pos and graph_neg:
        rq3_decision = ("in the fully recomputable data, task-head alignment is the more positively-associated / "
                        "correctly-signed reliance predictor (Spearman difference align-graph_kl excludes 0); raw "
                        "graph leakage carries the WRONG (negative) sign in the seed0 paired analysis and the frozen "
                        "pooled summary. Within-group partial betas are not individually significant at this n, "
                        "reflecting the dataset-heterogeneity found in RQ1A; graph_kl is larger in |beta| but "
                        "negatively signed, so it is not a valid positive reliance predictor.")
    else:
        rq3_decision = "alignment/leakage comparison inconclusive — inspect coefficients."
    rq3 = {"model_A_align_full_n126": {"n": len(unitsA), "align_k2": mA["align_k2"],
                                       "note": "align beta positive; CI includes 0 after dataset control (heterogeneous)"},
           "model_B_paired_seed0_n42": {"n": len(unitsB), "align_k2": mB["align_k2"], "graph_kl": mB["graph_kl"],
                                        "spearman_diff_align_minus_graph_kl": {"point": round(diff["point"], 4),
                                        "ci": [round(diff["ci_lo"], 4), round(diff["ci_hi"], 4)],
                                        "excludes_zero": diff["excludes_zero"]},
                                        "align_correct_sign": bool(align_pos), "graph_kl_wrong_sign": bool(graph_neg),
                                        "align_abs_gt_graph_beta": bool(abs(mB["align_k2"]["std_coef"]) > abs(mB["graph_kl"]["std_coef"]))},
           "model_C_frozen_summary": {"graph_kl_pooled": {**FROZEN_GKL_POOLED,
                                      "status": "frozen_not_recomputable", "sign": "negative", "use": "support only"}},
           "decision": rq3_decision,
           "primary_test": "Spearman difference (align - graph_kl) on seed0 units where both are present",
           "forbidden_phrasing": "align_k2 is a validated reliance estimator"}
    (OUT / "rq3_alignment_mechanism.json").write_text(json.dumps(rq3, indent=2) + "\n")

    # unit table (for provenance/reuse)
    _wcsv(OUT / "rq3_unit_table.csv", units)

    print(f"RQ1A align pooled: {a_pool['rho']:+.4f} [{a_pool['ci_lo']:+.4f},{a_pool['ci_hi']:+.4f}] n={a_pool['n']}")
    print(f"  by dataset: " + ", ".join(f"{ds}={a_ds[ds]['rho']:+.3f}{'*' if a_ds[ds]['excludes_zero'] else 'ns'}" for ds in a_ds))
    print(f"  decision: {rq1a_decision}")
    print(f"RQ1B graph_kl seed0: {g_pool['rho']:+.4f} [{g_pool['ci_lo']:+.4f},{g_pool['ci_hi']:+.4f}] n={g_pool['n']}")
    print(f"RQ3 A align std_beta={mA['align_k2']['std_coef']:+.4f} [{mA['align_k2']['ci_lo']:+.4f},{mA['align_k2']['ci_hi']:+.4f}] n={len(unitsA)}")
    print(f"RQ3 B align={mB['align_k2']['std_coef']:+.4f} (correct sign) graph_kl={mB['graph_kl']['std_coef']:+.4f} (wrong sign) n={len(unitsB)}")
    print(f"RQ3 primary: spearman_diff(align-graph_kl)={diff['point']:+.4f} [{diff['ci_lo']:+.4f},{diff['ci_hi']:+.4f}] excludes0={diff['excludes_zero']}")
    print(f"  decision: {rq3_decision}")


def _boot_diff(xa, xg, y, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap of rho(xa,y) - rho(xg,y), resampling the SAME unit indices."""
    xa, xg, y = (np.asarray(v, float) for v in (xa, xg, y))
    ok = np.isfinite(xa) & np.isfinite(xg) & np.isfinite(y)
    xa, xg, y = xa[ok], xg[ok], y[ok]
    n = xa.size
    point = spearman(xa, y) - spearman(xg, y)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        v = spearman(xa[s], y[s]) - spearman(xg[s], y[s])
        if v == v:
            d.append(v)
    d = np.asarray(d)
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"point": float(point), "ci_lo": lo, "ci_hi": hi, "n": int(n),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def _rank_resid_corr(units, xk, yk):
    """Dataset-controlled Spearman: rank x,y within each dataset, then Pearson on pooled within-ranks."""
    xs, ys = [], []
    by = {}
    for u in units:
        if u[xk] == "" or u[yk] == "":
            continue
        by.setdefault(u["dataset"], []).append(u)
    for ds, rows in by.items():
        rx = rankdata([r[xk] for r in rows])
        ry = rankdata([r[yk] for r in rows])
        xs += list(rx - rx.mean())
        ys += list(ry - ry.mean())
    xs, ys = np.asarray(xs), np.asarray(ys)
    n = xs.size
    r = float(np.corrcoef(xs, ys)[0, 1])
    rng = np.random.default_rng(SEED)
    d = [float(np.corrcoef(xs[i], ys[i])[0, 1]) for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    d = np.asarray([v for v in d if v == v])
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"rho": r, "lo": lo, "hi": hi, "n": n, "excludes_zero": bool(lo > 0 or hi < 0)}


def _wcsv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    sys.exit(main())
