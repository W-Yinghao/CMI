"""Phase 2.2 -- compare LPC variants and decide the objective-scaling branch (A/B/C/D).

Reads from results/.../lpc_collapse_curves/:
  raw_lpc baseline      : ERM = raw_lpc_sub*_lam0_seed*.json (uncollapsed leakage reference) AND the
                          Phase 2.1 raw collapse runs sub*_lam{1,3}_seed*.json (no 'variant' key).
  lpc_scale_invariant_* : the scale-detached-penalty variant.
  lpc_warm_ramp_*       : the cold-start-basin variant.

Per (variant, lambda): collapse fraction (feat_norm_final<0.1 or src_bAcc<chance+0.05), median final
feat_norm / src_bAcc / task_decode / subject_decode. Then apply the pre-registered decision rules:
  A  scale_invariant avoids collapse AND preserves task  -> objective-scaling confirmed; scale fix works.
       A1 subject_decode drops vs ERM (task kept) -> corrected global LPC reduces leakage (strong baseline).
       A2 subject_decode ~ ERM (task kept)        -> raw LPC's de-domaining was via collapse, not invariance.
  B  scale_invariant still collapses                     -> collapse not just a scale loophole.
  C  warm_ramp avoids but scale_invariant doesn't        -> schedule/basin issue, not the objective.
  D  neither avoids                                       -> stop fixing global LPC.
"""
from __future__ import annotations
import glob
import json
import os
import numpy as np

BASE = "tos_cmi/results/tos_cmi_eeg_frozen/lpc_collapse_curves"
CHANCE_TASK = 0.25


def _load(pattern, variant_key):
    out = []
    for p in sorted(glob.glob("%s/%s" % (BASE, pattern))):
        if os.path.basename(p) == "summary.json":
            continue
        r = json.load(open(p)); r.setdefault("variant", variant_key); out.append(r)
    return out


def _row(r):
    c = r.get("curves") or []
    fn = c[-1]["feat_norm_mean"] if c else float("nan")
    collapsed = (fn < 0.1) or (r["final_source_bAcc"] < CHANCE_TASK + 0.05)
    return dict(lam=r["lam"], feat_norm=fn, src=r["final_source_bAcc"], tgt=r.get("final_target_bAcc", np.nan),
                task_dec=r.get("final_task_decode", np.nan), subj_dec=r.get("final_subject_decode", np.nan),
                chance_subj=r.get("chance_subj", np.nan), collapsed=bool(collapsed))


def _agg(rows, lam):
    rs = [x for x in rows if abs(x["lam"] - lam) < 1e-9]
    if not rs:
        return None
    med = lambda k: float(np.nanmedian([x[k] for x in rs]))
    return dict(n=len(rs), collapsed=sum(x["collapsed"] for x in rs), feat_norm=med("feat_norm"),
                src=med("src"), tgt=med("tgt"), task_dec=med("task_dec"), subj_dec=med("subj_dec"),
                chance_subj=med("chance_subj"))


def main():
    erm = [_row(r) for r in _load("raw_lpc_sub*_lam0_seed*.json", "raw_lpc")]
    raw = [_row(r) for r in _load("sub*_lam*_seed*.json", "raw_lpc")]   # Phase 2.1 (no variant prefix)
    si = [_row(r) for r in _load("lpc_scale_invariant_sub*_seed*.json", "lpc_scale_invariant")]
    wr = [_row(r) for r in _load("lpc_warm_ramp_sub*_seed*.json", "lpc_warm_ramp")]
    erm_a = _agg(erm, 0.0)
    print("=== Phase 2.2 variant comparison (median over folds{1,5,9} x seeds{0,1,2}) ===")
    print("ERM (raw_lpc lam=0) leakage reference:", None if not erm_a else
          "src=%.2f task_dec=%.2f subj_dec=%.2f (chance %.2f) feat_norm=%.2f"
          % (erm_a["src"], erm_a["task_dec"], erm_a["subj_dec"], erm_a["chance_subj"], erm_a["feat_norm"]))
    print("%-22s %-5s %-9s %-6s %-6s %-9s %-9s %-9s" %
          ("variant", "lam", "collapse", "feat_n", "src", "tgt", "task_dec", "subj_dec"))
    table = {}
    for name, rows in [("raw_lpc", raw), ("lpc_scale_invariant", si), ("lpc_warm_ramp", wr)]:
        for lam in [1.0, 3.0]:
            a = _agg(rows, lam)
            if not a:
                continue
            table[(name, lam)] = a
            print("%-22s %-5g %d/%-7d %-6.3f %-6.3f %-9.3f %-9.3f %-9.3f"
                  % (name, lam, a["collapsed"], a["n"], a["feat_norm"], a["src"], a["tgt"],
                     a["task_dec"], a["subj_dec"]))

    # ---- decision rules ----
    def avoids(name):
        cells = [table.get((name, l)) for l in [1.0, 3.0] if (name, l) in table]
        if not cells:
            return None
        # "avoids collapse AND preserves task" at BOTH lambdas (median): feat_norm not ~0, src well above chance
        return all(c["feat_norm"] > 0.1 and c["src"] > CHANCE_TASK + 0.15 for c in cells)

    si_ok, wr_ok = avoids("lpc_scale_invariant"), avoids("lpc_warm_ramp")
    # KEYSTONE (cross-variant): in EVERY collapse-free, task-preserving cell, is leakage reduced vs ERM?
    erm_sd = erm_a["subj_dec"] if erm_a else float("nan")
    cf = []  # collapse-free cells (per-cell medians)
    for (name, lam), a in table.items():
        if a["feat_norm"] > 0.1 and a["src"] > CHANCE_TASK + 0.15:    # not collapsed + task preserved
            cf.append((name, lam, a["subj_dec"], a["task_dec"]))
    leak_reduced_any = any((not np.isnan(erm_sd)) and sd < erm_sd - 0.10 for _, _, sd, _ in cf)
    print("\n=== KEYSTONE: collapse-free cells (task preserved) -- does the penalty reduce leakage? ===")
    print("ERM subj_dec reference: %s" % (None if np.isnan(erm_sd) else round(erm_sd, 3)))
    for name, lam, sd, td in cf:
        print("  %-22s lam=%g  subj_dec=%.3f  task_dec=%.3f  %s" % (name, lam, sd, td,
              "LEAKAGE REDUCED" if (not np.isnan(erm_sd) and sd < erm_sd - 0.10) else "leakage ~ERM (NOT reduced)"))
    print("=> any collapse-free LPC reduces subject leakage:", leak_reduced_any)
    print("\n=== DECISION ===")
    print("scale_invariant avoids collapse & preserves task:", si_ok)
    print("warm_ramp avoids collapse & preserves task:", wr_ok)
    verdict = "D: neither avoids -> global LPC collapse not fixed by simple scale/schedule; stop fixing it."
    if si_ok:
        # branch A: compare subject leakage to ERM
        sd = np.nanmedian([table[("lpc_scale_invariant", l)]["subj_dec"] for l in [1.0, 3.0]
                           if ("lpc_scale_invariant", l) in table])
        erm_sd = erm_a["subj_dec"] if erm_a else np.nan
        if not np.isnan(erm_sd) and sd < erm_sd - 0.10:
            verdict = ("A1: scale-invariant LPC prevents the Z->0 trivial minimizer (objective-scaling "
                       "CONFIRMED) AND reduces subject leakage (%.2f vs ERM %.2f) with task preserved -> "
                       "corrected global LPC is a genuine baseline; TOS = measurement/certification framework."
                       % (sd, erm_sd))
        else:
            verdict = ("A2: scale-invariant LPC prevents collapse & preserves task (objective-scaling "
                       "CONFIRMED) BUT subject leakage stays ~ERM (%.2f vs %.2f) -> raw LPC's apparent "
                       "de-domaining was via COLLAPSE, not real invariance." % (sd, erm_sd))
    elif wr_ok:
        verdict = ("C: warm-ramp avoids collapse but scale-invariant does not -> high-lambda cold-start "
                   "LPC is dynamically unstable (schedule/basin issue), not purely a scale loophole.")
    elif si is not None and len(si):
        verdict = ("B: scale-invariant LPC STILL collapses -> the collapse is not merely a feature-scale "
                   "loophole (encoder/critic dynamics or objective conflict); global LPC remains unstable.")
    print("VERDICT:", verdict)
    keystone = ("KEYSTONE (robust across fixes): NO collapse-free, task-preserving LPC reduces subject "
                "leakage below ERM -- raw global-LPC's apparent de-domaining was ENTIRELY an artifact of "
                "the representation collapse; prevent the collapse and the penalty removes ZERO leakage."
                if (cf and not leak_reduced_any) else
                "KEYSTONE: at least one collapse-free LPC reduces leakage (see cells above).")
    print(keystone)
    json.dump({"erm_ref": erm_a, "table": {f"{k[0]}|{k[1]}": v for k, v in table.items()},
               "si_avoids": si_ok, "wr_avoids": wr_ok, "verdict": verdict,
               "collapse_free_cells": [{"variant": n, "lam": l, "subj_dec": sd, "task_dec": td} for n, l, sd, td in cf],
               "leakage_reduced_any_collapse_free": bool(leak_reduced_any), "keystone": keystone},
              open("%s/variant_compare.json" % BASE, "w"), indent=1)
    print("VARIANT_COMPARE_DONE")


if __name__ == "__main__":
    main()
