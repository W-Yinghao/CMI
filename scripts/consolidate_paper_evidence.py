#!/usr/bin/env python
"""Task 5: consolidate the deployment forest (Panel F) + information-regime ladder (Panel E) + summary stats
into one paper-facing JSON/CSV. Reads ONLY already-computed results (verdicts_3state.json + the Tier-1 smoke
summary). Data only — figures/framing are owner-controlled.
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path("tos_cmi/results/tos_cmi_eeg_frozen/erasure_target_deploy")
TIER1 = Path("tos_cmi/results/target_info/tier1_smoke/target_info_tier1_smoke_summary.json")
OUT = Path("tos_cmi/results/paper_evidence")


def panel_F():
    v = json.loads((ROOT / "verdicts_3state.json").read_text())
    return {"cells": v["cells"], "rows": v["rows"], "stats": v["stats"]}


def panel_E():
    if not TIER1.exists():
        return {"status": "tier1_smoke summary absent"}
    s = json.loads(TIER1.read_text())
    # information-regime ladder: for each deployable budget, deployable accepts (any cell clears the gate?)
    pb = s["per_budget"]
    ladder = []
    for reg, budget in [("source-only", "B0_source_only"), ("target-X(unlabeled)", "B1_unlabeled_target")]:
        acts = pb.get(budget, {})
        ladder.append({"regime": reg, "selector": budget, "accepts": int(acts.get("accept", 0)),
                       "any_cell_clears_gate": bool(acts.get("accept", 0) > 0)})
    # calibration k-curve, aggregated over worlds
    kc = {}
    for r in s.get("b2_k_curve", []):
        kc.setdefault(r["k"], {"accept": 0, "true": 0, "false": 0, "n": 0})
        kc[r["k"]]["accept"] += r["accept"]; kc[r["k"]]["true"] += r["true_accept"]
        kc[r["k"]]["false"] += r["false_accept"]; kc[r["k"]]["n"] += r["n"]
    for k in sorted(kc):
        c = kc[k]
        ladder.append({"regime": f"calibration-k={k}", "selector": "B2_k_labels_per_class",
                       "accepts": c["accept"], "true_accepts": c["true"], "false_accepts": c["false"],
                       "any_cell_clears_gate": bool(c["accept"] > 0)})
    ob = s.get("b4_oracle_by_world", {})
    return {"scope": s.get("scope"), "ladder": ladder,
            "deployable_false_accept_rate": s.get("deployable_false_accept_rate"),
            "oracle_upper_bound_by_world": {w: {"mean_audit_dbacc": d["mean_audit_dbacc"],
                                                "max_audit_dbacc": d["max_audit_dbacc"]} for w, d in ob.items()},
            "any_regime_crosses_ceiling": any(x["any_cell_clears_gate"] for x in ladder)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    F, E = panel_F(), panel_E()
    ev = {
        "panel_F_deployment_forest": F,
        "panel_E_information_regime_ladder": E,
        "headline_stats": {
            "deployment_cells": F["stats"]["n_cells"],
            "deployment_all_ruled_out": F["stats"]["paper_claim_no_source_eraser_clears_+0.01"],
            "deployment_confirmed": F["stats"]["cells_with_any_confirmed_benefit"],
            "deployment_max_src_drop_ucb": F["stats"]["max_src_drop_ucb_over_all_rows"],
            "ladder_any_regime_crosses": E.get("any_regime_crosses_ceiling"),
            "ladder_false_accept_rate": E.get("deployable_false_accept_rate"),
            "oracle_upper_bound": E.get("oracle_upper_bound_by_world"),
        },
    }
    (OUT / "paper_evidence.json").write_text(json.dumps(ev, indent=1))
    with open(OUT / "panel_F_forest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(F["rows"][0].keys())); w.writeheader()
        for r in F["rows"]:
            w.writerow(r)
    with open(OUT / "panel_E_ladder.csv", "w", newline="") as fh:
        cols = ["regime", "selector", "accepts", "any_cell_clears_gate"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in E["ladder"]:
            w.writerow(r)
    print(json.dumps(ev["headline_stats"], indent=1))
    print(f"\n-> {OUT}/paper_evidence.json + panel_{{E,F}}_*.csv")


if __name__ == "__main__":
    main()
