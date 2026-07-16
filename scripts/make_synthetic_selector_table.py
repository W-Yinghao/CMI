#!/usr/bin/env python
"""P0.5 closeout: explicit synthetic (DGP config x selector) table resolving the apparent contradiction between
the unit test (majority-sign shortcut -> nested-prefix selector REFUSES) and the audit preview (majority-sign
shortcut -> greedy source selector RECOVERS +0.150). They use DIFFERENT selectors; this makes it explicit."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.data.spurious_task_dgp import make_spurious_task_dgp
from tos_cmi.eval.dg_identifiability import (get_candidate_basis, crossfit_target_oracle,
    nested_source_meta_multi, apply_rule_to_target_full, source_greedy_audit)

OUT = REPO / "notes" / "DG_SYNTHETIC_SELECTOR_TABLE.md"


def split(dgp):
    Z, y, d, t = dgp["Z"], dgp["y"], dgp["d"], dgp["target_dom"]; src = d != t
    return Z[src], y[src].astype(int), d[src], Z[d == t], y[d == t].astype(int)


def main():
    rows = []
    for cfg, nmin in [("majority-sign (nmin=3)", 3), ("balanced-sign (nmin=5)", 5)]:
        dgp = make_spurious_task_dgp(n_domains=12, per_domain=250, seed=1, n_minority_source=nmin,
                                     inv_strength=0.5, spur_strength=2.5, id_strength=3.0)
        Zs, ys, ds, Zt, yt = split(dgp)
        B = get_candidate_basis("cond", False, Zs, ys, ds, seed=0)
        orc = crossfit_target_oracle(Zs, ys, Zt, yt, B, seed=0, mode="greedy")
        sms = nested_source_meta_multi(Zs, ys, ds, "cond", False, seed=0, objectives=("mean_1se", "cvar25"), eps=0.01)
        ev_mean = apply_rule_to_target_full(Zs, ys, ds, Zt, yt, "cond", False, sms["mean_1se"]["k_star"], seed=0)
        ev_cvar = apply_rule_to_target_full(Zs, ys, ds, Zt, yt, "cond", False, sms["cvar25"]["k_star"], seed=0)
        au = source_greedy_audit(Zs, ys, ds, Zt, yt, B, seed=0)
        rows.append((cfg,
                     f"{orc['delta_query']:+.3f} (rand {orc['delta_query_random']:+.3f})",
                     f"k*={sms['mean_1se']['k_star']} Δ={ev_mean['delta_query']:+.3f}",
                     f"k*={sms['cvar25']['k_star']} Δ={ev_cvar['delta_query']:+.3f}",
                     f"k={au['k_src']} Δ={au['delta_src']:+.3f} (rand {au['delta_src_random']:+.3f}, align {au['alignment']:.2f})"))
    lines = ["# Synthetic DGP x selector table (P0.5 closeout — resolves the majority-shortcut apparent contradiction)",
             "",
             "cond/full basis, spurious-task DGP (n_domains=12, per_domain=250, seed=1, inv 0.5 / spur 2.5 / id 3.0).",
             "The unit test asserts the *nested-prefix* selector REFUSES a majority-sign shortcut; the audit preview",
             "showed the *greedy source* selector RECOVERS a STRONG majority shortcut. Both are correct — different",
             "selectors. Greedy source is strictly more expressive (arbitrary coordinates + no no-harm gate), so it",
             "can exploit a strong majority shortcut that helps the source average; the prefix+no-harm nested rule",
             "cannot. This is exactly why the real-EEG negative is meaningful: the *stronger* greedy source selector",
             "also fails there, so the failure is not a too-weak selector.",
             "",
             "| DGP config | greedy target oracle (existence) | nested prefix mean_1SE | nested prefix CVaR25 | greedy source audit |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines += ["",
              "Reading: BALANCED shortcut is recovered by BOTH selectors (source-visible instability).",
              "MAJORITY shortcut: nested-prefix REFUSES (correct under its no-harm gate — deletion hurts the source",
              "majority), but greedy-source RECOVERS it because with a STRONG shortcut the mis-signed minority is",
              "hurt enough that deleting improves the source-LOSO average. Neither is a bug. On real EEG neither",
              "selector recovers the confirmed ticket -> genuine source-unobservability, not selector weakness."]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[synthetic-table] wrote -> {OUT}")


if __name__ == "__main__":
    main()
