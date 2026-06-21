"""Stage-B2a analysis: the six comparators under the metadata substrate, the frozen acceptance
clauses, and the review-required extra diagnostics. The "should-abstain" stratum is the non-DIAG
geometry (NONE + UNSUPPORTED, where the diagonal family cannot honestly help while SPD/rotation are
frozen); the "shift-utility" stratum is DIAG_COMPATIBLE geometry. Aggregation clusters by seed.

  python -m h2cmi.analyze_b2a --in results/h2cmi/b2a_grid.jsonl --out results/h2cmi/b2a.report.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

COMPARATORS = ("identity", "always_pooled", "always_canonical", "n1_target_ranking",
               "metadata_gated", "metadata_oracle")
PASS = dict(false_adapt_max=0.10, coverage_min=0.25, nonnull_harm_max=0.20, top1_min=0.50,
            regret_max=0.02)


def _fin(x):
    return x is not None and x == x


def analyze(rows) -> dict:
    ep = defaultdict(dict)
    for r in rows:
        ep[(r["data_seed"], r["target_site"], r["episode"])][r["comparator"]] = r
    rep = {"pass_criteria": PASS, "by_comparator": {}}
    for comp in COMPARATORS:
        diag, nondiag, allv = [], [], []
        adapt, dbacc, harm, top1, regret = [], [], [], [], []
        gatepos_regret, veto_fail, missing_abst = [], [], []
        confusion = defaultdict(int)
        for key, cm in ep.items():
            if comp not in cm or "metadata_oracle" not in cm:
                continue
            r = cm[comp]; orc = cm["metadata_oracle"]
            is_diag = r["eff_geom"] == "DIAG_COMPATIBLE"
            adapt.append(r["adapted"]); dbacc.append(r["dbacc_full"])
            harm.append(r["dbacc_full"] < -1e-9)
            top1.append(r["selected_op"] == orc["selected_op"])
            regret.append(orc["selected_bacc"] - r["selected_bacc"])
            (diag if is_diag else nondiag).append(r["dbacc_full"]); allv.append(r)
            if comp == "metadata_gated":
                if r.get("metadata_gate_pass"):
                    gatepos_regret.append(orc["selected_bacc"] - r["selected_bacc"])
                if r["metadata_operator"] != "identity" and not r.get("metadata_gate_pass"):
                    veto_fail.append(1)
                else:
                    veto_fail.append(0)
                unknown = (r["geometry_compatibility"] == "UNKNOWN") or (r["prevalence_risk"] == "UNKNOWN")
                if unknown:
                    missing_abst.append(r["selected_op"] == "identity")
                if is_diag:
                    confusion[f"oracle={orc['selected_op']}|meta={r['selected_op']}"] += 1
        def fa(xs):
            return float(np.mean([cm[comp]["adapted"] for k, cm in ep.items()
                                  if comp in cm and cm[comp]["eff_geom"] != "DIAG_COMPATIBLE"])) if xs is not None else float("nan")
        res = dict(
            n_episodes=len(adapt), coverage=float(np.mean(adapt)),
            false_adaptation_rate=float(np.mean([cm[comp]["adapted"] for k, cm in ep.items()
                                                 if comp in cm and cm[comp]["eff_geom"] != "DIAG_COMPATIBLE"])),
            harm_rate=float(np.mean(harm)),
            mean_dbacc_diag=float(np.mean(diag)) if diag else float("nan"),
            mean_dbacc_all=float(np.mean(dbacc)),
            nonnull_harm_rate=float(np.mean([d < -1e-9 for d in diag])) if diag else float("nan"),
            top1_oracle=float(np.mean(top1)), mean_regret=float(np.mean(regret)))
        if comp == "metadata_gated":
            res["gate_positive_operator_regret"] = float(np.mean(gatepos_regret)) if gatepos_regret else float("nan")
            res["metadata_op_veto_fail_rate"] = float(np.mean(veto_fail)) if veto_fail else float("nan")
            res["missing_metadata_abstention_rate"] = float(np.mean(missing_abst)) if missing_abst else float("nan")
            res["pooled_vs_cc_confusion"] = dict(confusion)
            res["passes"] = dict(
                false_adapt=res["false_adaptation_rate"] <= PASS["false_adapt_max"],
                coverage=res["coverage"] >= PASS["coverage_min"],
                shift_utility=res["mean_dbacc_diag"] > 0,
                nonnull_harm=res["nonnull_harm_rate"] <= PASS["nonnull_harm_max"],
                top1=res["top1_oracle"] >= PASS["top1_min"], regret=res["mean_regret"] <= PASS["regret_max"])
            res["passes"]["ALL"] = all(res["passes"].values())
        rep["by_comparator"][comp] = res
    mg = rep["by_comparator"]["metadata_gated"]
    rep["decision"] = ("B2A_PASS -> freeze selection architecture; open pooled-SPD/rotation + confirmatory"
                       if mg.get("passes", {}).get("ALL") else
                       "B2A_FAIL -> selection architecture insufficient on dev seeds")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", default="")
    args = ap.parse_args()
    rep = analyze([json.loads(l) for l in open(args.inp) if l.strip()])
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=2)
    for comp, d in rep["by_comparator"].items():
        extra = f" PASS={d['passes']['ALL']}" if "passes" in d else ""
        print(f"  {comp:18s} cov={d['coverage']:.2f} false_adapt={d['false_adaptation_rate']:.2f} "
              f"ΔbAcc_diag={d['mean_dbacc_diag'] if _fin(d['mean_dbacc_diag']) else float('nan'):+.3f} "
              f"harm={d['harm_rate']:.2f} top1={d['top1_oracle']:.2f} regret={d['mean_regret']:.3f}{extra}")
    print(f"DECISION: {rep['decision']}")
    if args.out:
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
