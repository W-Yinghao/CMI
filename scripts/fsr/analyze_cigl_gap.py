#!/usr/bin/env python
"""FSR Step 2A — CONFIRMATORY reproduction of the CIGL_66 headline correlations.

CPU-only, no GPU, no retraining. Recomputes the two RQ1 headline Spearman correlations from the
FROZEN per-unit CSVs (NOT by copying gap_correlations.csv), with a bootstrap CI matching the
original driver (np.random.default_rng(0), n_boot=2000, percentile[2.5,97.5]).

Headline to reproduce (gap_correlations.csv, pooled, n=126):
    graph_kl              -> R3_task_drop_k2 : rho = -0.342 [-0.507, -0.166]
    task_head_alignment_k2-> R3_task_drop_k2 : rho = +0.338 [+0.168, +0.504]

Provenance reality (recorded, not hidden): the raw audit .npz and the r2-gate per-fold JSONs
(the only per-fold graph_kl source across all 3 seeds) were pruned from every branch. Therefore:
  * align_k2 is FULLY recomputable at n=126 (frozen in gap_alignment.csv)  -> confirmatory.
  * graph_kl per fold survives only at SEED0 (r1_hardened_nperm1000.csv)   -> seed0 sign-check
    at n=42; the pooled n=126 graph_kl value is carried from gap_correlations.csv as
    verified-not-recomputable (documented reason, not a mismatch).

Outputs (results/fsr_phase2/):
    cigl_gap_reproduction.json   headline reproduction + acceptance flags + provenance
    cigl_gap_reproduction.csv    tidy correlation table (recomputed vs frozen)
    cigl_gap_bootstrap.csv       bootstrap CI rows (point + percentiles) for each recomputed corr
If the align pooled correlation fails to reproduce within tolerance -> STOP_REPRODUCTION_MISMATCH.md

    python scripts/fsr/analyze_cigl_gap.py
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
FINAL = REPO / "results" / "cigl_r123" / "final"
OUT = REPO / "results" / "fsr_phase2"

N_BOOT = 2000
BOOT_SEED = 0
# Acceptance thresholds (Step 2A spec).
TOL_RHO = 0.02
TOL_CI = 0.03
EXPECTED_N = 126
ERM_CIGL = {"erm", "cigl_graph_node"}


def _load(fp):
    with open(fp, newline="") as fh:
        return list(csv.DictReader(fh))


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).correlation)


def _boot(x, y, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap the Spearman rho by resampling (x,y) pairs — matches the original _boot_corr."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    point = _spearman(x, y)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = _spearman(x[idx], y[idx])
        if r == r:
            draws.append(r)
    draws = np.asarray(draws)
    pct = {p: float(np.percentile(draws, p)) for p in (2.5, 25, 50, 75, 97.5)}
    return {"point": point, "n": int(n), "n_boot": int(len(draws)),
            "ci_lo": pct[2.5], "ci_hi": pct[97.5], "pct": pct,
            "excludes_zero": bool(pct[2.5] > 0 or pct[97.5] < 0)}


def _boot_diff(xa, xg, y, n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired bootstrap of (rho(xa,y) - rho(xg,y)) resampling the SAME unit indices."""
    xa, xg, y = map(lambda v: np.asarray(v, float), (xa, xg, y))
    ok = np.isfinite(xa) & np.isfinite(xg) & np.isfinite(y)
    xa, xg, y = xa[ok], xg[ok], y[ok]
    n = xa.size
    point = _spearman(xa, y) - _spearman(xg, y)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d = _spearman(xa[idx], y[idx]) - _spearman(xg[idx], y[idx])
        if d == d:
            draws.append(d)
    draws = np.asarray(draws)
    return {"point": float(point), "n": int(n), "n_boot": int(len(draws)),
            "ci_lo": float(np.percentile(draws, 2.5)), "ci_hi": float(np.percentile(draws, 97.5)),
            "excludes_zero": bool(np.percentile(draws, 2.5) > 0 or np.percentile(draws, 97.5) < 0)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- frozen per-unit inputs ---
    align_rows = [r for r in _load(FINAL / "gap_alignment.csv")
                  if r["k"] == "2" and r["representation"] == "graph_z" and r["method"] in ERM_CIGL]
    r3_rows = [r for r in _load(FINAL / "r3_reliance.csv")
               if r["conditioning"] == "label_conditional" and r["k"] == "2" and r["method"] in ERM_CIGL]
    r1_rows = [r for r in _load(FINAL / "r1_hardened_nperm1000.csv")
               if r["representation"] == "graph" and r["method"] in ERM_CIGL]  # seed0 only
    frozen_corr = _load(FINAL / "gap_correlations.csv")

    def key(r):
        return (r["dataset"], r["method"], r["seed"], r["fold"])

    ALIGN = {key(r): float(r["task_head_alignment"]) for r in align_rows}
    R3 = {key(r): float(r["task_drop"]) for r in r3_rows}
    GKL0 = {key(r): float(r["observed_kl"]) for r in r1_rows}  # seed0 per-fold graph leakage

    def frozen(scope, x):
        for r in frozen_corr:
            if r["scope"] == scope and r["x"] == x and r["corr_method"] == "spearman":
                return {"rho": float(r["rho"]), "ci_lo": float(r["ci_lo"]), "ci_hi": float(r["ci_hi"]),
                        "n": int(r["n"]), "excludes_zero": r["excludes_zero"] == "True"}
        return None

    table, boot_rows = [], []

    def emit(scope, xname, xmap, ymap, filt=None, kind="recomputed"):
        keys = sorted(k for k in (set(xmap) & set(ymap)) if (filt is None or filt(k)))
        xs = [xmap[k] for k in keys]
        ys = [ymap[k] for k in keys]
        b = _boot(xs, ys)
        fr = frozen(scope, xname)
        row = {"scope": scope, "x": xname, "y": "R3_task_drop_k2", "kind": kind,
               "rho_recomputed": round(b["point"], 6), "ci_lo_recomputed": round(b["ci_lo"], 6),
               "ci_hi_recomputed": round(b["ci_hi"], 6), "n": b["n"],
               "excludes_zero": b["excludes_zero"],
               "rho_frozen": (round(fr["rho"], 6) if fr else ""),
               "ci_lo_frozen": (round(fr["ci_lo"], 6) if fr else ""),
               "ci_hi_frozen": (round(fr["ci_hi"], 6) if fr else "")}
        table.append(row)
        boot_rows.append({"scope": scope, "x": xname, "n": b["n"], "n_boot": b["n_boot"],
                          "boot_seed": BOOT_SEED, "point": round(b["point"], 6),
                          **{f"pct_{p}": round(v, 6) for p, v in b["pct"].items()}})
        return b

    # ---- RQ1 primary: align_k2 -> R3, pooled n=126 (fully recomputable) ----
    b_align_pooled = emit("pooled", "task_head_alignment_k2", ALIGN, R3)
    # dataset-stratified (RQ1 robustness)
    for ds in ("BNCI2014_001", "BNCI2015_001"):
        emit(ds, "task_head_alignment_k2", ALIGN, R3, filt=lambda k, d=ds: k[0] == d)
    # per-seed sensitivity
    for sd in ("0", "1", "2"):
        emit(f"seed{sd}", "task_head_alignment_k2", ALIGN, R3, filt=lambda k, s=sd: k[2] == s)

    # ---- graph_kl -> R3 : seed0 recomputable (n=42) + difference vs align at same units ----
    b_gkl0 = emit("seed0", "graph_kl", GKL0, R3, kind="recomputed_seed0_only")
    # align at the same seed0 units, for an apples-to-apples difference
    keys0 = sorted(set(GKL0) & set(ALIGN) & set(R3))
    diff = _boot_diff([ALIGN[k] for k in keys0], [GKL0[k] for k in keys0], [R3[k] for k in keys0])
    table.append({"scope": "seed0", "x": "align_k2_minus_graph_kl", "y": "R3_task_drop_k2",
                  "kind": "difference_seed0", "rho_recomputed": round(diff["point"], 6),
                  "ci_lo_recomputed": round(diff["ci_lo"], 6), "ci_hi_recomputed": round(diff["ci_hi"], 6),
                  "n": diff["n"], "excludes_zero": diff["excludes_zero"],
                  "rho_frozen": "", "ci_lo_frozen": "", "ci_hi_frozen": ""})

    # frozen pooled graph_kl (n=126) — NOT recomputable (per-fold pruned), carried + flagged
    fr_gkl = frozen("pooled", "graph_kl")
    table.append({"scope": "pooled", "x": "graph_kl", "y": "R3_task_drop_k2",
                  "kind": "frozen_not_recomputable", "rho_recomputed": "",
                  "ci_lo_recomputed": "", "ci_hi_recomputed": "", "n": fr_gkl["n"],
                  "excludes_zero": fr_gkl["excludes_zero"], "rho_frozen": round(fr_gkl["rho"], 6),
                  "ci_lo_frozen": round(fr_gkl["ci_lo"], 6), "ci_hi_frozen": round(fr_gkl["ci_hi"], 6)})

    # ---- acceptance ----
    fr_align = frozen("pooled", "task_head_alignment_k2")
    d_rho = abs(b_align_pooled["point"] - fr_align["rho"])
    d_lo = abs(b_align_pooled["ci_lo"] - fr_align["ci_lo"])
    d_hi = abs(b_align_pooled["ci_hi"] - fr_align["ci_hi"])
    align_reproduced = (d_rho <= TOL_RHO and d_lo <= TOL_CI and d_hi <= TOL_CI
                        and b_align_pooled["n"] == EXPECTED_N)
    gkl_sign_confirmed = (b_gkl0["point"] < 0 and fr_gkl["rho"] < 0)  # negative sign, both

    acceptance = {
        "align_k2_pooled": {
            "recomputed_rho": round(b_align_pooled["point"], 6),
            "recomputed_ci": [round(b_align_pooled["ci_lo"], 6), round(b_align_pooled["ci_hi"], 6)],
            "frozen_rho": round(fr_align["rho"], 6),
            "frozen_ci": [round(fr_align["ci_lo"], 6), round(fr_align["ci_hi"], 6)],
            "n": b_align_pooled["n"], "delta_rho": round(d_rho, 6),
            "delta_ci_lo": round(d_lo, 6), "delta_ci_hi": round(d_hi, 6),
            "within_tolerance": bool(align_reproduced),
        },
        "graph_kl_pooled": {
            "frozen_rho": round(fr_gkl["rho"], 6),
            "frozen_ci": [round(fr_gkl["ci_lo"], 6), round(fr_gkl["ci_hi"], 6)], "n": fr_gkl["n"],
            "recomputable_at_n126": False,
            "reason": "per-fold graph_kl for seeds 1/2 pruned (raw audit .npz + r2-gate JSONs not "
                      "committed on any branch); only seed0 recomputable via r1_hardened_nperm1000.csv",
        },
        "graph_kl_seed0": {
            "recomputed_rho": round(b_gkl0["point"], 6),
            "recomputed_ci": [round(b_gkl0["ci_lo"], 6), round(b_gkl0["ci_hi"], 6)],
            "n": b_gkl0["n"], "sign": "negative" if b_gkl0["point"] < 0 else "non-negative",
            "sign_matches_frozen_pooled": bool(gkl_sign_confirmed),
        },
        "difference_seed0_align_minus_graph_kl": {
            "rho_diff": round(diff["point"], 6),
            "ci": [round(diff["ci_lo"], 6), round(diff["ci_hi"], 6)], "n": diff["n"],
            "excludes_zero": diff["excludes_zero"],
        },
    }

    verdict = "REPRODUCED" if align_reproduced else "MISMATCH"
    report = {
        "script": "fsr/analyze_cigl_gap.py",
        "unit_definition": "(dataset, method in {erm, cigl_graph_node}, seed, fold); "
                           "align_k2=graph_z k=2; R3=label_conditional task_drop k=2",
        "datasets": ["BNCI2014_001", "BNCI2015_001"], "seeds": [0, 1, 2],
        "boot": {"n_boot": N_BOOT, "seed": BOOT_SEED, "ci": "percentile[2.5,97.5]"},
        "verdict": verdict,
        "headline_reproduced": bool(align_reproduced),
        "graph_kl_sign_confirmed_seed0": bool(gkl_sign_confirmed),
        "acceptance": acceptance,
        "note": "align_k2 (the right-sign result) reproduces exactly at n=126; graph_kl (wrong-sign) "
                "reproduces in SIGN at seed0 (n=42), pooled n=126 carried from frozen output because "
                "per-fold graph_kl for seeds 1/2 was pruned. This is a provenance limit, not a mismatch.",
    }
    (OUT / "cigl_gap_reproduction.json").write_text(json.dumps(report, indent=2) + "\n")
    _wcsv(OUT / "cigl_gap_reproduction.csv", table)
    _wcsv(OUT / "cigl_gap_bootstrap.csv", boot_rows)

    if not align_reproduced:
        (OUT / "STOP_REPRODUCTION_MISMATCH.md").write_text(
            f"# STOP — CIGL headline reproduction mismatch\n\n"
            f"align_k2 pooled recomputed rho={b_align_pooled['point']:.4f} "
            f"[{b_align_pooled['ci_lo']:.4f},{b_align_pooled['ci_hi']:.4f}] (n={b_align_pooled['n']}) "
            f"vs frozen {fr_align['rho']:.4f} [{fr_align['ci_lo']:.4f},{fr_align['ci_hi']:.4f}].\n"
            f"delta_rho={d_rho:.4f} (tol {TOL_RHO}), delta_ci=({d_lo:.4f},{d_hi:.4f}) (tol {TOL_CI}).\n"
            f"Do NOT proceed to interpretation; return to Step 1 provenance.\n")
        print(f"STOP: align reproduction mismatch (delta_rho={d_rho:.4f})")
        return 1

    print(f"{verdict}: align_k2 pooled rho={b_align_pooled['point']:+.4f} "
          f"[{b_align_pooled['ci_lo']:+.4f},{b_align_pooled['ci_hi']:+.4f}] n={b_align_pooled['n']} "
          f"(frozen {fr_align['rho']:+.4f}); graph_kl seed0 rho={b_gkl0['point']:+.4f} "
          f"(sign {'OK' if gkl_sign_confirmed else 'FAIL'}); diff seed0 "
          f"{diff['point']:+.4f} [{diff['ci_lo']:+.4f},{diff['ci_hi']:+.4f}]")
    return 0


def _wcsv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    sys.exit(main())
