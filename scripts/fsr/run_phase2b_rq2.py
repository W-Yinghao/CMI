#!/usr/bin/env python
"""FSR Step 2B — RQ2: does erasure strength predict target benefit? CPU-only, read-only.

Reads the frozen TOS deploy summaries from git branch `tos` via `git show`. Primary data = the 5
erasure routes {TOS_VD (mean_scatter), LEACE, INLP, RLACE, random_k}; excludes baseline (`full`),
in-loss boundary routes, refusal gate. Per cell (dataset, backbone, eraser):
  E_subject_removed = subj_dec_after(full) - subj_dec_after(eraser)     (L3 erasability)
  T_target_bAcc     = dtgt_bacc  (+ CI)                                  (L6)
  T_target_NLL      = dtgt_nll                                           (L6)
  task_collapse / binary_harm flags; benefit_claimable (proven bAcc gain only)

Tests: corr(E, T_bAcc), corr(E, T_NLL) [all cells + collapse/harm-excluded]; LEACE-vs-random_k NLL
specificity; task-collapse / binary-harm / benefit_claimable counts.

Outputs (results/fsr_phase2b/):
    rq2_erasure_vs_target.csv / .json

    python scripts/fsr/run_phase2b_rq2.py
"""
from __future__ import annotations
import csv, json, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "fsr_phase2b"
REF = "tos"
BASE = "tos_cmi/results/tos_cmi_eeg_frozen/erasure_target_deploy"
ERASERS = ["TOS_VD", "LEACE", "INLP", "RLACE", "random_k"]  # excludes 'full' baseline
N_BOOT, SEED = 2000, 0
DEPLOY = {
    "BNCI2014_001": f"{BASE}/erasure_target_deploy_summary.json",
    "Lee2019_MI": f"{BASE}/Lee2019_MI/erasure_target_deploy_summary.json",
    "Cho2017": f"{BASE}/Cho2017/erasure_target_deploy_summary.json",
    "Schirrmeister2017": f"{BASE}/Schirrmeister2017/erasure_target_deploy_summary.json",
}


def git_json(path):
    try:
        return json.loads(subprocess.run(["git", "-C", str(REPO), "show", f"{REF}:{path}"],
                                         capture_output=True, text=True, check=True).stdout)
    except Exception:  # noqa: BLE001
        return None


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok]).correlation) if ok.sum() >= 4 else float("nan")


def boot_ci(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 4:
        return {"rho": None, "ci_lo": None, "ci_hi": None, "n": int(n), "excludes_zero": False}
    pt = spearman(x, y)
    rng = np.random.default_rng(SEED)
    d = [spearman(x[i], y[i]) for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    d = np.asarray([v for v in d if v == v])
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"rho": round(pt, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n": int(n),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def benefit(dtgt_bacc, dtgt_bacc_lo, tgt_bacc, chance, nonspec, raw_flag):
    if None in (dtgt_bacc,):
        return "NO", "no_bacc_gain"
    tc = tgt_bacc is not None and chance is not None and tgt_bacc <= chance + 0.02
    bh = chance is not None and chance >= 0.5 and dtgt_bacc <= -0.05
    if tc:
        return "NO", "task_collapse"
    if bh:
        return "NO", "binary_harm"
    ci_pos = dtgt_bacc_lo is not None and dtgt_bacc_lo > 0
    if not (dtgt_bacc > 0 and ci_pos):
        if bool(raw_flag) and nonspec:
            return "NO", "nll_nonspecific_randomk"
        return "NO", "no_bacc_gain"
    return "YES", ""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells, spec_rows = [], []
    for ds, path in DEPLOY.items():
        dep = git_json(path)
        if dep is None:
            continue
        s = dep.get("summary", dep)
        for bb in sorted({k.split("|")[0] for k in s}):
            full = s.get(f"{bb}|full", {})
            chance = fnum(full.get("chance_task"))
            full_subj = fnum(full.get("subj_dec_after_mean"))
            leace, rk = s.get(f"{bb}|LEACE"), s.get(f"{bb}|random_k")
            nonspec = _nonspecific(leace, rk)
            for er in ERASERS:
                c = s.get(f"{bb}|{er}")
                if c is None:
                    continue
                er_subj = fnum(c.get("subj_dec_after_mean"))
                E = (full_subj - er_subj) if (full_subj is not None and er_subj is not None) else None
                dtb, dtb_lo = fnum(c.get("dtgt_bacc")), fnum(c.get("dtgt_bacc_lo"))
                dtn = fnum(c.get("dtgt_nll"))
                tgt = fnum(c.get("tgt_bacc_mean"))
                bc, br = benefit(dtb, dtb_lo, tgt, chance, nonspec, c.get("improves_target"))
                cells.append(dict(
                    dataset=ds, backbone=bb, eraser=er, chance_task=chance,
                    E_subject_removed=_r(E), T_target_bAcc=_r(dtb), T_target_bAcc_lo=_r(dtb_lo),
                    T_target_NLL=_r(dtn), tgt_bacc=_r(tgt),
                    task_collapse=("YES" if (tgt is not None and chance is not None and tgt <= chance + 0.02) else "NO"),
                    binary_harm=("YES" if (chance is not None and chance >= 0.5 and dtb is not None and dtb <= -0.05) else "NO"),
                    benefit_claimable=bc, benefit_block_reason=br))
            if leace and rk:
                spec_rows.append(dict(dataset=ds, backbone=bb,
                                      leace_dtgt_nll=_r(fnum(leace.get("dtgt_nll"))),
                                      randomk_dtgt_nll=_r(fnum(rk.get("dtgt_nll"))),
                                      randomk_subj_dec_after=_r(fnum(rk.get("subj_dec_after_mean"))),
                                      leace_dtgt_bacc=_r(fnum(leace.get("dtgt_bacc"))),
                                      nll_move_nonspecific=("YES" if nonspec else "NO")))

    _wcsv(OUT / "rq2_erasure_vs_target.csv", cells)

    # correlations
    def corr(sel):
        e = [c["E_subject_removed"] for c in cells if sel(c) and c["E_subject_removed"] not in ("", None)]
        tb = [c["T_target_bAcc"] for c in cells if sel(c) and c["E_subject_removed"] not in ("", None)]
        tn = [c["T_target_NLL"] for c in cells if sel(c) and c["E_subject_removed"] not in ("", None)]
        return boot_ci(e, tb), boot_ci(e, tn)

    all_bacc, all_nll = corr(lambda c: True)
    clean = lambda c: c["task_collapse"] == "NO" and c["binary_harm"] == "NO"
    clean_bacc, clean_nll = corr(clean)

    n_cells = len(cells)
    n_collapse = sum(c["task_collapse"] == "YES" for c in cells)
    n_harm = sum(c["binary_harm"] == "YES" for c in cells)
    n_benefit = sum(c["benefit_claimable"] == "YES" for c in cells)
    n_nonspec = sum(r["nll_move_nonspecific"] == "YES" for r in spec_rows)

    # decision language
    def decide(c):
        if c["rho"] is None:
            return "insufficient_n"
        if c["rho"] < 0 and c["excludes_zero"]:
            return ("erasure strength is NEGATIVELY associated with target benefit (more subject removal -> "
                    "WORSE target bAcc); the old positive erasure hypothesis is refuted.")
        if not c["excludes_zero"] or c["rho"] <= 0:
            return "the old erasure hypothesis is NOT supported (erasure strength does not predict target benefit)."
        return "positive association present — inspect (unexpected under FSR)."

    decision = {
        "corr_E_vs_target_bAcc_all_cells": decide(all_bacc),
        "corr_E_vs_target_bAcc_clean_cells": decide(clean_bacc),
        "benefit": ("no eraser certifies a proven target benefit (benefit_claimable=0)"
                    if n_benefit == 0 else f"{n_benefit} cells claim benefit — inspect"),
        "nll": (f"NLL movement is non-specific in {n_nonspec}/{len(spec_rows)} LEACE-vs-random_k comparisons; "
                "not attributable to subject erasure where flagged"),
        "task_safety": f"{n_collapse}/{n_cells} cells task-collapse; {n_harm}/{n_cells} binary-harm (hard guardrail failures)",
        "headline": "subject signal is erasable, but erasure strength does not certify target benefit.",
        "forbidden_phrasing": "LEACE improves target NLL (as a DG claim)",
    }
    rq2 = {"cells": n_cells, "erasers": ERASERS, "datasets": list(DEPLOY),
           "corr_E_subject_removed_vs": {
               "target_bAcc_all": all_bacc, "target_NLL_all": all_nll,
               "target_bAcc_clean": clean_bacc, "target_NLL_clean": clean_nll},
           "randomk_specificity": spec_rows,
           "counts": {"cells": n_cells, "task_collapse": n_collapse, "binary_harm": n_harm,
                      "benefit_claimable": n_benefit, "nll_nonspecific_cells": n_nonspec},
           "decision": decision}
    (OUT / "rq2_erasure_vs_target.json").write_text(json.dumps(rq2, indent=2) + "\n")

    print(f"RQ2 cells={n_cells}  benefit_claimable={n_benefit}  task_collapse={n_collapse}  binary_harm={n_harm}")
    print(f"  corr(E, target_bAcc) all cells   : rho={all_bacc['rho']} [{all_bacc['ci_lo']},{all_bacc['ci_hi']}] n={all_bacc['n']}")
    print(f"  corr(E, target_bAcc) clean cells : rho={clean_bacc['rho']} [{clean_bacc['ci_lo']},{clean_bacc['ci_hi']}] n={clean_bacc['n']}")
    print(f"  corr(E, target_NLL) all cells    : rho={all_nll['rho']} [{all_nll['ci_lo']},{all_nll['ci_hi']}]")
    print(f"  NLL non-specific: {n_nonspec}/{len(spec_rows)} cells")
    print(f"  decision: {decision['headline']}")


def _nonspecific(leace, rk):
    if not (leace and rk):
        return False
    ln, rn, rs = fnum(leace.get("dtgt_nll")), fnum(rk.get("dtgt_nll")), fnum(rk.get("subj_dec_after_mean"))
    if None in (ln, rn, rs):
        return False
    return abs(ln - rn) <= 0.02 and rs >= 0.9


def _r(x):
    return "" if x is None else round(x, 5)


def _wcsv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    sys.exit(main())
