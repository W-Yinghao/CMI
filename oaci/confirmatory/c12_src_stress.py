"""C12 — SRC stress-replication aggregator. Reads the per-config one-fold pilot bodies (targets × τ_lse) and
answers ONE narrow question: is SRC's target-001 failure a single-fold fluke, or does source-side control
fail to transfer across BNCI2014-001 folds? Emits the 4 tables + a hard verdict:
  continue_SRC | stop_SRC_pivot_measurement_only | inconclusive_needs_one_more_fold
This is the LAST SRC round — not open-ended exploration.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

from ..artifacts.canonical_json import canonical_json_bytes

_N_CLASSES = 4                              # BNCI2014-001 motor imagery
_UNIFORM_NLL = math.log(_N_CLASSES)          # 1.386 — target NLL above this = confidently WRONG
_SEVERE_NLL_OVER_ERM = 0.5
_BACC_MARGIN = 0.0                           # SRC must be >= ERM (no accuracy loss) to count as a gain
_NLL_MARGIN = 0.0


def load_configs(indir) -> list:
    out = []
    for p in sorted(glob.glob(os.path.join(indir, "target-*-temp*.json"))):
        b = json.load(open(p))
        out.append({"target": int(b["target"]), "temp": float(b["smooth_temperature"]), "body": b})
    return out


def _cells(configs) -> list:
    """One row per (target, temp, level): ERM/OACI/SRC target worst-domain + SRC transfer diagnostics."""
    rows = []
    for c in configs:
        b = c["body"]
        for L, lv in b["levels"].items():
            erm, oaci, src = lv["ERM"], lv["OACI"], lv["SRC"]
            enll = erm["target_worst_nll"]
            snll = src["target_worst_nll"]
            blow = snll is not None and (snll > _UNIFORM_NLL or (enll is not None and snll > enll + _SEVERE_NLL_OVER_ERM))
            rows.append({
                "target": c["target"], "temp": c["temp"], "level": int(L),
                "erm_target_bacc": erm["target_worst_bacc"], "erm_target_nll": enll,
                "erm_source_guard_nll": erm["source_guard_worst_nll"],
                "src_target_bacc": src["target_worst_bacc"], "src_target_nll": snll,
                "src_source_guard_nll": src["source_guard_worst_nll"],
                "src_fallback_erm": bool(src["fallback_erm"]), "src_risk_feasible": bool(src["risk_feasible"]),
                "src_n_guard_pass": src.get("n_guard_pass"),
                "d_bacc_vs_erm": src["K2_delta_target_worst_bacc"], "d_nll_vs_erm": src["K2_delta_target_worst_nll"],
                "d_bacc_vs_oaci": (None if (src["target_worst_bacc"] is None or oaci["target_worst_bacc"] is None)
                                   else src["target_worst_bacc"] - oaci["target_worst_bacc"]),
                "target_nll_blowup": blow,
                # source improved but target didn't => non-transfer
                "source_improved_nll": (src["source_guard_worst_nll"] is not None and erm["source_guard_worst_nll"] is not None
                                        and src["source_guard_worst_nll"] <= erm["source_guard_worst_nll"] + 1e-9),
            })
    return rows


def verdict(rows) -> dict:
    active = [r for r in rows if not r["src_fallback_erm"]]
    n = len(rows)
    n_fallback = sum(1 for r in rows if r["src_fallback_erm"])
    n_blowup = sum(1 for r in rows if r["target_nll_blowup"])
    n_bacc_gain = sum(1 for r in active if r["d_bacc_vs_erm"] is not None and r["d_bacc_vs_erm"] > _BACC_MARGIN)
    n_nll_gain = sum(1 for r in active if r["d_nll_vs_erm"] is not None and r["d_nll_vs_erm"] < -_NLL_MARGIN)
    n_bacc_harm = sum(1 for r in active if r["d_bacc_vs_erm"] is not None and r["d_bacc_vs_erm"] < -1e-9)
    # non-transfer: source improved but target NLL got worse
    n_nontransfer = sum(1 for r in active if r["source_improved_nll"] and r["d_nll_vs_erm"] is not None and r["d_nll_vs_erm"] > 0)
    # blowup by temperature (does gentler tau avoid the blowup?)
    blow_by_temp = {}
    for r in rows:
        blow_by_temp.setdefault(str(r["temp"]), [0, 0])
        blow_by_temp[str(r["temp"])][0] += int(r["target_nll_blowup"]); blow_by_temp[str(r["temp"])][1] += 1

    # ---- gate ----
    pivot_reasons = []
    if n_blowup >= 1:
        pivot_reasons.append(f"target NLL blowup in {n_blowup}/{n} cells (SRC confidently wrong on target)")
    if n_fallback > n / 2:
        pivot_reasons.append(f"SRC fell back to ERM in {n_fallback}/{n} cells (no viable source-robust checkpoint)")
    if active and n_bacc_gain == 0 and n_nll_gain == 0:
        pivot_reasons.append("no active SRC cell improves target worst-domain bAcc or NLL over ERM")
    if n_nontransfer >= 1:
        pivot_reasons.append(f"source-side improvement did NOT transfer to target in {n_nontransfer} cells")

    continue_ok = (not pivot_reasons and len(active) >= 2 and n_bacc_harm == 0
                   and all(r["src_risk_feasible"] for r in rows)
                   and n_bacc_gain + n_nll_gain >= 2)
    if continue_ok:
        v = "continue_SRC"
    elif pivot_reasons:
        v = "stop_SRC_pivot_measurement_only"
    else:
        v = "inconclusive_needs_one_more_fold"
    return {"verdict": v, "pivot_reasons": pivot_reasons, "n_cells": n, "n_active": len(active),
            "n_fallback": n_fallback, "n_target_nll_blowup": n_blowup, "n_bacc_gain": n_bacc_gain,
            "n_nll_gain": n_nll_gain, "n_bacc_harm": n_bacc_harm, "n_source_improved_not_transferred": n_nontransfer,
            "blowup_by_temperature": {k: {"blowup": a, "cells": b} for k, (a, b) in blow_by_temp.items()},
            "uniform_nll_threshold": _UNIFORM_NLL}


def _f(x, nd=4):
    return "n/a" if x is None else (f"{x:+.{nd}f}" if isinstance(x, (int, float)) else str(x))


def render_md(rows, vd, configs) -> str:
    tgs = sorted({r["target"] for r in rows}); temps = sorted({r["temp"] for r in rows})
    L = [f"# C12 — SRC stress replication (BNCI2014-001 seed-0; targets {tgs}, τ_lse {temps})", "",
         "> Last SRC round. Narrow question: is target-001's SRC failure a single-fold fluke, or does "
         "source-side control fail to transfer across folds?", "",
         f"- configs loaded: **{len(configs)}** ({len(rows)} target×temp×level cells)",
         f"- **VERDICT: `{vd['verdict']}`**"]
    if vd["pivot_reasons"]:
        L += ["- pivot triggers: " + "; ".join(vd["pivot_reasons"])]
    L += ["", "## Table 1 — SRC τ=0.1 vs τ=0.3 (target worst NLL; blowup if > uniform 1.386 or > ERM+0.5)", "",
          "| target | level | ERM NLL | SRC NLL τ0.1 | blow | SRC NLL τ0.3 | blow |", "|---:|---:|---:|---:|:--:|---:|:--:|"]
    by = {(r["target"], r["level"], r["temp"]): r for r in rows}
    for t in tgs:
        for lv in sorted({r["level"] for r in rows}):
            r01 = by.get((t, lv, 0.1)); r03 = by.get((t, lv, 0.3))
            erm = (r01 or r03 or {}).get("erm_target_nll")
            L.append(f"| {t} | {lv} | {_f(erm)} | {_f(r01 and r01['src_target_nll'])} | "
                     f"{'YES' if r01 and r01['target_nll_blowup'] else '-'} | {_f(r03 and r03['src_target_nll'])} | "
                     f"{'YES' if r03 and r03['target_nll_blowup'] else '-'} |")
    L += ["", "## Table 2 — SRC target worst-domain Δ vs ERM (bAcc↑ NLL↓ better)", "",
          "| target | temp | level | Δ bAcc | Δ NLL | fallback | risk-feasible |", "|---:|---:|---:|---:|---:|:--:|:--:|"]
    for r in rows:
        L.append(f"| {r['target']} | {r['temp']} | {r['level']} | {_f(r['d_bacc_vs_erm'])} | {_f(r['d_nll_vs_erm'])} | "
                 f"{'ERM' if r['src_fallback_erm'] else '-'} | {r['src_risk_feasible']} |")
    L += ["", "## Table 3 — fallback frequency", "",
          f"- SRC fell back to ERM in **{vd['n_fallback']}/{vd['n_cells']}** cells; active (trained-ckpt) cells "
          f"**{vd['n_active']}**", ""]
    for k, v in vd["blowup_by_temperature"].items():
        L.append(f"- τ={k}: target NLL blowup in {v['blowup']}/{v['cells']} cells")
    L += ["", "## Table 4 — source-side improvement vs target transfer", "",
          "| target | temp | level | ΔsrcGuard NLL | Δtarget NLL | transferred? |", "|---:|---:|---:|---:|---:|:--:|"]
    for r in rows:
        ds = (None if (r["src_source_guard_nll"] is None or r["erm_source_guard_nll"] is None)
              else r["src_source_guard_nll"] - r["erm_source_guard_nll"])
        trans = "-" if r["src_fallback_erm"] else ("YES" if (r["d_nll_vs_erm"] is not None and r["d_nll_vs_erm"] <= 0) else "NO")
        L.append(f"| {r['target']} | {r['temp']} | {r['level']} | {_f(ds)} | {_f(r['d_nll_vs_erm'])} | {trans} |")
    L += ["", "## Interpretation", ""]
    if vd["verdict"] == "continue_SRC":
        L += ["> SRC shows consistent, feasible, no-blowup target improvement across folds -> promote SRC to a "
              "first-class method and run BNCI001 LOSO seeds[0,1,2] (C13)."]
    elif vd["verdict"] == "stop_SRC_pivot_measurement_only":
        L += ["> SRC fails to transfer across folds (see pivot triggers). Combined with C10 (leakage/oracle) and "
              "C11 (endpoint), THREE source-side interventions fail -> **STOP SRC. Pivot to measurement-only / "
              "source-target-instability (C13 memo).** Keep support-aware leakage + K1/K2 as the falsification "
              "instrument; do not build another DG control penalty."]
    else:
        L += ["> Mixed / borderline (no blowup, but not consistently better). At most ONE more fold before "
              "deciding; do not open-ended explore."]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.confirmatory.c12_src_stress")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args(argv)
    configs = load_configs(args.in_dir)
    if not configs:
        raise ValueError(f"no per-config pilot JSONs in {args.in_dir}")
    rows = _cells(configs)
    vd = verdict(rows)
    for p in (args.out_md, args.out_json):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(args.out_json, "wb") as f:
        f.write(canonical_json_bytes({"targets": sorted({r["target"] for r in rows}),
                                      "temperatures": sorted({r["temp"] for r in rows}),
                                      "cells": rows, "verdict": vd}))
    with open(args.out_md, "w") as f:
        f.write(render_md(rows, vd, configs))
    print(f"wrote {args.out_json} + {args.out_md}: verdict={vd['verdict']} "
          f"(blowup {vd['n_target_nll_blowup']}/{vd['n_cells']}, fallback {vd['n_fallback']}/{vd['n_cells']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
