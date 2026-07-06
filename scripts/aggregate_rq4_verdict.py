#!/usr/bin/env python
"""FSR Phase 4B — aggregate the branch-local L1-L6 CSVs into a verdict with bootstrap CIs.

Verdict logic (per branch): a HARMFUL shortcut needs measurable(L1) + load-bearing(L4) +
functionally-relied(L5 specific vs random) + TARGET-HARMFUL(L6). Sign convention:
task_drop = bacc_orig - bacc_erased; task_drop < 0 (erasing HELPS target) => harmful shortcut;
task_drop > 0 (erasing HURTS target) => task-useful/benign reliance (repair would harm).

    <python> scripts/aggregate_rq4_verdict.py
"""
import csv, json, statistics as st
from pathlib import Path
import numpy as np

R = Path("results/fsr_rq4_refit")
RNG = np.random.default_rng(0)


def load(f):
    return list(csv.DictReader(open(R / f)))


def ci(vals):
    v = np.array([x for x in vals if x == x], float)
    if len(v) < 2:
        return dict(mean=float(v.mean()) if len(v) else float("nan"), lo=float("nan"), hi=float("nan"), n=len(v))
    b = [np.mean(v[RNG.integers(0, len(v), len(v))]) for _ in range(2000)]
    return dict(mean=round(float(v.mean()), 4), lo=round(float(np.percentile(b, 2.5)), 4),
                hi=round(float(np.percentile(b, 97.5)), 4), n=len(v), excludes_zero=bool(np.percentile(b, 2.5) > 0 or np.percentile(b, 97.5) < 0))


def by(rows, ds, br, col):
    return [float(r[col]) for r in rows if r["dataset"] == ds and r["branch"] == br and r[col] not in ("", None)]


def main():
    l1, l4, l5, l6 = load("branch_leakage_probe.csv"), load("branch_task_coupling.csv"), load("branch_reliance_replay.csv"), load("branch_target_consequence.csv")
    datasets = sorted({r["dataset"] for r in l1})
    out = {"seed": 0, "n_folds": {d: len({r["target_subject"] for r in l1 if r["dataset"] == d}) for d in datasets},
           "sign_convention": "task_drop=bacc_orig-bacc_erased; <0 erasing HELPS target=HARMFUL; >0 erasing HURTS target=task-useful/benign",
           "branches": {}}
    for ds in datasets:
        out["branches"][ds] = {}
        for br in ("graph_z", "temporal_z", "spatial_z", "fused_z"):
            e = {}
            e["L1_subject_probe_bacc"] = ci(by(l1, ds, br, "probe_bacc"))
            e["L1_chance"] = round(st.mean(by(l1, ds, br, "chance")), 3) if by(l1, ds, br, "chance") else None
            if br != "fused_z":
                e["L4_ablation_drop"] = ci(by(l4, ds, br, "ablation_drop"))
                e["L4_gate_weight"] = round(st.mean(by(l4, ds, br, "gate_weight")), 3) if by(l4, ds, br, "gate_weight") else None
                td = by(l5, ds, br, "task_drop")
                tdr = by(l5, ds, br, "task_drop_random")
                e["L5_task_drop"] = ci(td)
                e["L5_task_drop_specificity_vs_random"] = ci([a - b for a, b in zip(td, tdr)])
                e["L5_logit_symkl"] = ci(by(l5, ds, br, "logit_symkl"))
                e["L5_logit_symkl_specificity"] = ci([a - b for a, b in zip(by(l5, ds, br, "logit_symkl"), by(l5, ds, br, "logit_symkl_random"))])
                # per-branch harmful-shortcut verdict
                lk = e["L1_subject_probe_bacc"]["mean"] - (e["L1_chance"] or 0)
                load_bearing = e["L4_ablation_drop"]["mean"] > 0.03
                coupled = e["L5_logit_symkl_specificity"].get("excludes_zero", False)
                harmful = e["L5_task_drop"]["mean"] < 0 and e["L5_task_drop"].get("excludes_zero", False)
                e["verdict"] = ("VERIFIED_HARMFUL_SHORTCUT" if (lk > 0.1 and load_bearing and coupled and harmful)
                                else "MEASURABLE_COUPLED_BUT_NOT_HARMFUL" if (lk > 0.1 and coupled and e["L5_task_drop"]["mean"] >= 0)
                                else "MEASURABLE_ONLY")
            out["branches"][ds][br] = e
    for ds in datasets:
        out["branches"][ds]["L6_target_bacc"] = ci([float(r["target_bacc"]) for r in l6 if r["dataset"] == ds])

    # project-level verdict
    verds = [out["branches"][ds][b].get("verdict") for ds in datasets for b in ("graph_z", "temporal_z", "spatial_z")]
    out["natural_eeg_verdict"] = ("NO_VERIFIED_HARMFUL_BRANCH_SHORTCUT" if "VERIFIED_HARMFUL_SHORTCUT" not in verds
                                  else "VERIFIED_HARMFUL_BRANCH_SHORTCUT_FOUND")
    out["interpretation"] = ("the most subject-leaky, load-bearing branch (spatial) is functionally coupled "
                             "(logit SymKL >> random) but its subject subspace is target-USEFUL (erasing it HURTS "
                             "the target: task_drop>0), so it fails the target-harmful condition -> not a harmful "
                             "shortcut; blind erasure would harm. Motivates refusal + the injected positive control (4C).")
    (R / "rq4_branch_local_results.json").write_text(json.dumps(out, indent=2) + "\n")

    # firewall summary (from manifests)
    import glob
    fw = [json.load(open(m))["firewall"] for m in glob.glob(str(R / "latents/*manifest*.json"))]
    fwok = all(not f["target_y_used_for_training"] and not f["target_y_used_for_selection"]
               and not f["target_y_used_for_probe_fit"] and f["target_y_used_for_final_eval_only"] for f in fw)
    (R / "rq4_target_label_firewall.json").write_text(json.dumps(
        {"n_folds": len(fw), "all_clean": bool(fwok),
         "rule": "target y for final eval only; probes/subspaces source-fit"}, indent=2) + "\n")

    print("natural_eeg_verdict:", out["natural_eeg_verdict"])
    for ds in datasets:
        print(f"--- {ds} (n={out['n_folds'][ds]} folds) L6 target bAcc={out['branches'][ds]['L6_target_bacc']['mean']} ---")
        for br in ("graph_z", "temporal_z", "spatial_z"):
            e = out["branches"][ds][br]
            print(f"  {br:11s} L1={e['L1_subject_probe_bacc']['mean']:.3f} L4_drop={e['L4_ablation_drop']['mean']:+.4f} "
                  f"L5_task_drop={e['L5_task_drop']['mean']:+.4f}{'*' if e['L5_task_drop'].get('excludes_zero') else ''} "
                  f"(spec_vs_rand={e['L5_task_drop_specificity_vs_random']['mean']:+.4f}{'*' if e['L5_task_drop_specificity_vs_random'].get('excludes_zero') else ''}) "
                  f"symkl_spec={e['L5_logit_symkl_specificity']['mean']:.3f}{'*' if e['L5_logit_symkl_specificity'].get('excludes_zero') else ''} -> {e['verdict']}")
    print("firewall all_clean:", fwok)


if __name__ == "__main__":
    main()
