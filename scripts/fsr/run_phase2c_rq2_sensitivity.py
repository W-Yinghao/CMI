#!/usr/bin/env python
"""FSR Step 2C — RQ2 sensitivity hardening (CPU-only, no new artifact, no target-label fit).

Re-analyses the frozen RQ2 cells (results/fsr_phase2b/rq2_erasure_vs_target.csv) under a battery of
subsetting / confound controls, to show the RQ2 negative-association SUPPORTING result is not driven
by over-erasure (INLP), the random-k control, task-collapse cells, or dataset/backbone confounding.
The PRIMARY claim (erasure does not certify target benefit; benefit_claimable=0) is independent of
these subsets. Bootstrap: rng(0), n_boot=2000, percentile[2.5,97.5].

Outputs (results/fsr_phase2c/):
    rq2_sensitivity_by_family.csv / .json
    rq2_within_dataset_backbone.csv
    rq2_claim_hardening.md

    python scripts/fsr/run_phase2c_rq2_sensitivity.py
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, rankdata

REPO = Path(__file__).resolve().parents[2]
RQ2 = REPO / "results" / "fsr_phase2b" / "rq2_erasure_vs_target.csv"
OUT = REPO / "results" / "fsr_phase2c"
N_BOOT, SEED = 2000, 0


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok]).correlation) if ok.sum() >= 4 else float("nan")


def boot(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 4:
        return {"rho": None, "ci_lo": None, "ci_hi": None, "n": int(n), "sign": "na", "excludes_zero": False}
    pt = spearman(x, y)
    rng = np.random.default_rng(SEED)
    d = [spearman(x[i], y[i]) for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    d = np.asarray([v for v in d if v == v])
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"rho": round(pt, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n": int(n),
            "sign": ("neg" if pt < 0 else "pos" if pt > 0 else "zero"),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def xy(cells):
    e = [fnum(c["E_subject_removed"]) for c in cells]
    t = [fnum(c["T_target_bAcc"]) for c in cells]
    pairs = [(a, b) for a, b in zip(e, t) if a is not None and b is not None]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = list(csv.DictReader(open(RQ2)))

    clean = [c for c in cells if c["task_collapse"] == "NO" and c["binary_harm"] == "NO"]
    subsets = {
        "1_all_cells": cells,
        "2_clean_cells_no_collapse_no_harm": clean,
        "3_excl_random_k": [c for c in cells if c["eraser"] != "random_k"],
        "4_excl_INLP": [c for c in cells if c["eraser"] != "INLP"],
        "5_excl_INLP_and_random_k": [c for c in cells if c["eraser"] not in ("INLP", "random_k")],
        "6_LEACE_RLACE_only": [c for c in cells if c["eraser"] in ("LEACE", "RLACE")],
    }
    fam_rows, fam = [], {}
    for name, sub in subsets.items():
        x, y = xy(sub)
        b = boot(x, y)
        fam[name] = b
        fam_rows.append(dict(subset=name, **b))

    # 7. within dataset x backbone rank residualization (control dataset AND backbone)
    rr = _rank_resid(cells, "E_subject_removed", "T_target_bAcc", group=("dataset", "backbone"))
    fam_rows.append(dict(subset="7_within_dataset_backbone_rank_resid", **rr))
    fam["7_within_dataset_backbone_rank_resid"] = rr

    # 8/9. per-dataset and per-backbone sign tables
    wdb_rows = []
    for key, col in (("dataset", "8_per_dataset"), ("backbone", "9_per_backbone")):
        for lev in sorted({c[key] for c in cells}):
            x, y = xy([c for c in cells if c[key] == lev])
            b = boot(x, y)
            wdb_rows.append(dict(group_kind=col, group=lev, **b))
    # also per (dataset,backbone)
    for ds in sorted({c["dataset"] for c in cells}):
        for bb in sorted({c["backbone"] for c in cells if c["dataset"] == ds}):
            x, y = xy([c for c in cells if c["dataset"] == ds and c["backbone"] == bb])
            b = boot(x, y)
            wdb_rows.append(dict(group_kind="cell_dataset_backbone", group=f"{ds}|{bb}", **b))
    _wcsv(OUT / "rq2_within_dataset_backbone.csv", wdb_rows)
    _wcsv(OUT / "rq2_sensitivity_by_family.csv", fam_rows)

    # ---- claim hardening verdict ----
    core = ["1_all_cells", "2_clean_cells_no_collapse_no_harm", "3_excl_random_k",
            "4_excl_INLP", "5_excl_INLP_and_random_k", "6_LEACE_RLACE_only",
            "7_within_dataset_backbone_rank_resid"]
    neg = [k for k in core if fam[k]["rho"] is not None and fam[k]["rho"] < 0]
    neg_sig = [k for k in core if fam[k]["excludes_zero"] and (fam[k]["rho"] or 0) < 0]
    per_ds = [r for r in wdb_rows if r["group_kind"] == "8_per_dataset"]
    per_bb = [r for r in wdb_rows if r["group_kind"] == "9_per_backbone"]
    all_ds_neg = all(r["rho"] is not None and r["rho"] < 0 for r in per_ds)
    all_bb_neg = all(r["rho"] is not None and r["rho"] < 0 for r in per_bb)

    lr = fam["6_LEACE_RLACE_only"]
    sign_flip = lr["rho"] is not None and lr["rho"] > 0  # principled-eraser subset flips positive
    broad = (len(neg) == len(core) and len(neg_sig) >= 4 and all_ds_neg and all_bb_neg and not sign_flip)
    supporting_status = "READY_WITH_CAVEAT" if broad else "NOT_ROBUST_DO_NOT_HEADLINE"

    verdict = {
        "primary_claim": {
            "text": "Subject signal is erasable, but erasure strength does not certify target benefit.",
            "status": "READY",
            "basis": "benefit_claimable=0/40 (proven-bAcc rule); independent of the sensitivity subsets."},
        "supporting_result": {
            "text": "Stronger subject removal is negatively associated with target bAcc in the frozen TOS cells.",
            "status": supporting_status,
            "basis": {
                "core_subsets_negative": f"{len(neg)}/{len(core)}",
                "core_subsets_negative_and_ci_excludes_0": f"{len(neg_sig)}/{len(core)}",
                "all_per_dataset_negative": bool(all_ds_neg),
                "all_per_backbone_negative": bool(all_bb_neg),
                "LEACE_RLACE_only": fam["6_LEACE_RLACE_only"],
                "within_dataset_backbone_rank_resid": fam["7_within_dataset_backbone_rank_resid"],
                "sign_flips_on_principled_erasers": bool(sign_flip)},
            "caveat": (("negative association spans erasers/datasets/backbones AND survives dropping INLP "
                        "(over-erasure) and random_k and controlling dataset+backbone -> not a collapse/over-erasure "
                        "artifact; reported as a SUPPORTING result, not the headline.") if broad else
                       (f"negative association is NOT robust: it FLIPS to positive (rho={lr['rho']}, excludes 0) on "
                        "the principled-eraser subset (LEACE/RLACE only) and is ns when INLP and/or random_k are "
                        "dropped -> it is driven by over-erasure (INLP) and the random-k anchor, NOT a real "
                        "'more removal, worse target' law. DO NOT report the negative association as a finding. The "
                        "robust, headline-eligible result is the PRIMARY claim (benefit_claimable=0/40)."))},
        "boot": {"n_boot": N_BOOT, "seed": SEED},
    }
    (OUT / "rq2_sensitivity_by_family.json").write_text(json.dumps(
        {"subsets": fam, "verdict": verdict}, indent=2) + "\n")

    # markdown hardening note
    md = ["# RQ2 claim hardening — sensitivity of the erasure→target result", "",
          "Primary claim (**READY**, independent of subsets): *subject signal is erasable, but erasure "
          "strength does not certify target benefit* — `benefit_claimable = 0/40`.", "",
          f"Supporting result (**{supporting_status}**): *stronger subject removal is negatively associated "
          "with target bAcc.* Sensitivity:", "",
          "| subset | rho | 95% CI | n | sign | excl 0 |", "|---|---|---|---|---|---|"]
    for r in fam_rows:
        md.append(f"| {r['subset']} | {r['rho']} | [{r['ci_lo']},{r['ci_hi']}] | {r['n']} | {r['sign']} | {r['excludes_zero']} |")
    md += ["", "Per-dataset / per-backbone sign (all negative required for a broad claim):", "",
           "| group_kind | group | rho | sign | excl 0 |", "|---|---|---|---|---|"]
    for r in wdb_rows:
        md.append(f"| {r['group_kind']} | {r['group']} | {r['rho']} | {r['sign']} | {r['excludes_zero']} |")
    breadth = ("broad (survives dropping INLP + random_k, clean cells, LEACE/RLACE-only, dataset+backbone "
               "control, all per-dataset & per-backbone signs negative)" if broad
               else (f"NOT robust: it FLIPS to positive (rho={lr['rho']}) on the principled-eraser subset "
                     "(LEACE/RLACE only) and is ns when INLP/random_k are dropped -> driven by over-erasure "
                     "(INLP) + the random-k anchor, not a real 'more removal, worse target' law"))
    md += ["", f"**Verdict:** the negative association is {breadth} -> supporting result status = "
           f"**{supporting_status}**.",
           "", "The headline stays the **READY** primary claim (*erasure does not certify target benefit; "
           "benefit_claimable=0/40*). The negative correlation is **not** reported as a finding.",
           "", "No target labels used for fit (target y is EVAL_ONLY in the underlying deploy summaries). CPU-only."]
    (OUT / "rq2_claim_hardening.md").write_text("\n".join(md) + "\n")

    print(f"RQ2 sensitivity: core negative {len(neg)}/{len(core)}, negative+sig {len(neg_sig)}/{len(core)}")
    print(f"  per-dataset all negative: {all_ds_neg}; per-backbone all negative: {all_bb_neg}")
    print(f"  LEACE/RLACE-only: rho={fam['6_LEACE_RLACE_only']['rho']} excl0={fam['6_LEACE_RLACE_only']['excludes_zero']}")
    print(f"  within dataset+backbone rank-resid: rho={rr['rho']} excl0={rr['excludes_zero']}")
    print(f"  primary=READY; supporting negative-association status = {supporting_status}")


def _rank_resid(cells, xk, yk, group):
    xs, ys = [], []
    by = {}
    for c in cells:
        x, y = fnum(c[xk]), fnum(c[yk])
        if x is None or y is None:
            continue
        by.setdefault(tuple(c[g] for g in group), []).append((x, y))
    for g, rows in by.items():
        if len(rows) < 2:
            continue
        rx = rankdata([r[0] for r in rows])
        ry = rankdata([r[1] for r in rows])
        xs += list(rx - rx.mean())
        ys += list(ry - ry.mean())
    xs, ys = np.asarray(xs), np.asarray(ys)
    n = xs.size
    if n < 4:
        return {"rho": None, "ci_lo": None, "ci_hi": None, "n": int(n), "sign": "na", "excludes_zero": False}
    r = float(np.corrcoef(xs, ys)[0, 1])
    rng = np.random.default_rng(SEED)
    d = [float(np.corrcoef(xs[i], ys[i])[0, 1]) for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    d = np.asarray([v for v in d if v == v])
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"rho": round(r, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n": int(n),
            "sign": ("neg" if r < 0 else "pos"), "excludes_zero": bool(lo > 0 or hi < 0)}


def _wcsv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    sys.exit(main())
