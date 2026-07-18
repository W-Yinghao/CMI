"""Aggregate the Information-Regime Ladder (Track B; CPU, env c84c). Reads per-cell jsons and reports, per dataset and
inference-unit = target subject (seeds -> subject first, subject-cluster bootstrap + exact sign-flip):
  * query gain  dU_k              for k in R0,RX,R1,R2,R4,RF   (informed B_cond dictionary)
  * specificity dU_k^specific     = informed - matched-random  (subspace-specific?)
  * minimal info level k*         = first regime (info order) with LCB95(dU_k) > 0
  * subspace-specific threshold   = first regime with LCB95(dU_k) > 0 AND LCB95(dU_k^specific) > 0
  * recovery_k                    = dU_k / dU_RF   (only if LCB95(dU_RF) > 0)
  * crossfit target-oracle ceiling (context)
  * head-only calibration secondary (native vs selected-subspace query bAcc) -> cases A/B/C/D
Routes IL-A..E. Requires BOTH datasets to agree for any positive claim (they have historically reversed sign).
Refuses partial matrices. Manuscript FROZEN; only the project owner stops/redirects a scientific line.

  python scripts/aggregate_information_ladder.py --from-dir results/cmi_trace_info_ladder --expect 63
"""
from __future__ import annotations
import argparse, glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(REPO))
from scripts.aggregate_mcc import _cluster_ci

DATASETS = ["BNCI2014_001", "BNCI2015_001"]
INFO_ORDER = ["R0", "RX", "R1", "R2", "R4", "RF"]        # increasing target information
LABEL_REGIMES = ["R1", "R2", "R4", "RF"]


def _subj_ci(per_subj):
    return _cluster_ci([float(np.mean(v)) for v in per_subj.values()])


def _route(ds_stats):
    """ds_stats[ds] = {regime: {'dU':ci,'spec':ci}, '_headcase':str, '_rf_lcb':float}. Positive claims require BOTH
    datasets. Precedence: IL-R0 (source-only, unexpected) > IL-A (few-shot subspace-specific) > IL-D (unlabeled
    geometry) > IL-B (few-shot generic low-rank) > IL-E (full-cal only) > IL-C (head-only) > null."""
    dsl = list(ds_stats)
    both = len(dsl) >= 2
    def lcb(ds, reg, key): return ds_stats[ds][reg][key]["lo"]
    def any_few_specific():   # some few-shot regime with dU LCB>0 AND spec LCB>0 on BOTH datasets
        for reg in ["R1", "R2", "R4"]:
            if both and all(lcb(ds, reg, "dU") > 0 and lcb(ds, reg, "spec") > 0 for ds in dsl):
                return reg
        return None
    def any_few_gain():       # some few-shot regime with dU LCB>0 (informed) BOTH datasets (spec need not clear)
        for reg in ["R1", "R2", "R4"]:
            if both and all(lcb(ds, reg, "dU") > 0 for ds in dsl):
                return reg
        return None
    r0_pos = both and all(lcb(ds, "R0", "dU") > 0 for ds in dsl)
    rx_only = both and all(lcb(ds, "RX", "dU") > 0 for ds in dsl) and all(lcb(ds, "R0", "dU") <= 0 for ds in dsl)
    rf_only = both and all(lcb(ds, "RF", "dU") > 0 for ds in dsl) and not any_few_gain()
    head_pos = both and all(ds_stats[ds].get("_headcase") in ("B_head_only", "C_both") for ds in dsl)

    if r0_pos:
        spec = both and all(lcb(ds, "R0", "spec") > 0 for ds in dsl)
        return dict(verdict="IL-R0_SOURCE_ONLY_SELECTS" + ("_SUBSPACE_SPECIFIC" if spec else "_GENERIC"),
                    next="UNEXPECTED given the whole line: source-only (no target info) selects a query-helpful action on BOTH datasets" + (" AND beats matched-random" if spec else " but NOT beyond matched-random (generic)") + " -> re-examine the source-meta gauge; confirm on a 3rd dataset before any claim (contradicts prior source-unobservable results)")
    reg = any_few_specific()
    if reg:
        return dict(verdict="IL-A_FEWSHOT_LABELS_RESOLVE_IDENTIFIABILITY_SUBSPACE_SPECIFIC",
                    next=f"at {reg} the informed B_cond dictionary beats identity AND matched-random on BOTH datasets -> subspace EXISTS, source+target-X observability FAILS, few-shot labels resolve identifiability; THEN a few-shot subspace selector / TTE is worth building (OWNER decision, 3rd-dataset confirm required)")
    if rx_only:
        return dict(verdict="IL-D_UNLABELED_TARGET_GEOMETRY_SUFFICES",
                    next="RX (unlabeled target X) beats identity where R0 (source-only) does not -> transductive adaptation, not more source proxy")
    reg2 = any_few_gain()
    if reg2:
        return dict(verdict="IL-B_GENERIC_LOWRANK_NOT_SUBSPACE_SPECIFIC",
                    next=f"few-shot labels raise utility at {reg2} but informed == matched-random -> generic low-rank/action-search/calibration effect, NOT the subject subspace; pivot to ordinary few-shot adaptation, not basis surgery")
    if rf_only:
        return dict(verdict="IL-E_ONLY_FULL_CALIBRATION_HELPS",
                    next="only RF (all cal labels) beats identity -> task strongly subject-specific; large calibration needed; not near-calibration-free DG")
    if head_pos:
        return dict(verdict="IL-C_HEAD_CALIBRATION_NOT_SUBSPACE",
                    next="selection-only null but head-only few-shot calibration positive -> the DG bottleneck is the target-specific P(Y|Z) readout, not a deletable subspace")
    return dict(verdict="IL_NULL_NO_REGIME_HELPS",
                next="no information regime yields a replicable subspace-selection gain and head calibration does not rescue it -> the action family is wrong / earlier target-hindsight was selection optimism (OWNER decides next line)")


def _head_case(rf_hs):
    """rf_hs: {'sel_minus_native': ci, 'native_minus_source': ci} at RF (most labels). A/B/C/D by which LCB>0."""
    sel = rf_hs.get("sel_minus_native"); nat = rf_hs.get("native_minus_source")
    sel_pos = sel is not None and sel["lo"] > 0
    nat_pos = nat is not None and nat["lo"] > 0
    if sel_pos and nat_pos:
        return "C_both"
    if nat_pos and not sel_pos:
        return "B_head_only"
    if sel_pos and not nat_pos:
        return "A_selection_only"
    return "D_neither"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", default="results/cmi_trace_info_ladder"); ap.add_argument("--expect", type=int, default=63)
    a = ap.parse_args()
    d = Path(a.from_dir); cells = sorted(glob.glob(str(d / "cells" / "*.json")))
    done = sorted(glob.glob(str(d / "cells" / "*.done")))
    if len(done) < a.expect:
        print(f"[il-agg] INCOMPLETE {len(done)}/{a.expect} cells -> REFUSING."); raise SystemExit(2)
    dU = {reg: defaultdict(lambda: defaultdict(list)) for reg in INFO_ORDER}      # reg -> ds -> subj -> [seed]
    spec = {reg: defaultdict(lambda: defaultdict(list)) for reg in INFO_ORDER}
    orc = defaultdict(lambda: defaultdict(list)); orc_rand = defaultdict(lambda: defaultdict(list))
    ho_selnat = {reg: defaultdict(lambda: defaultdict(list)) for reg in LABEL_REGIMES}   # selected - native
    ho_natsrc = {reg: defaultdict(lambda: defaultdict(list)) for reg in LABEL_REGIMES}   # native - source_identity
    skipped = defaultdict(int)
    for c in cells:
        r = json.loads(Path(c).read_text())
        if r.get("status") != "ok":
            skipped[r.get("reason", "?")] += 1; continue
        ds, subj = r["dataset"], r["subject"]
        for reg in INFO_ORDER:
            dU[reg][ds][subj].append(r["informed"][reg]); spec[reg][ds][subj].append(r["specific"][reg])
        if "error" not in r.get("oracle", {}):
            orc[ds][subj].append(r["oracle"]["delta"]); orc_rand[ds][subj].append(r["oracle"]["random"])
        for reg in LABEL_REGIMES:
            h = r["head_only"].get(reg, {})
            if h and all(np.isfinite(h.get(k, np.nan)) for k in ("native", "selected", "source_identity")):
                ho_selnat[reg][ds][subj].append(h["selected"] - h["native"])
                ho_natsrc[reg][ds][subj].append(h["native"] - h["source_identity"])

    per_ds = []; ds_stats = {}
    for ds in DATASETS:
        if ds not in dU["R0"] or not dU["R0"][ds]:
            continue
        regci = {reg: dict(dU=_subj_ci(dU[reg][ds]), spec=_subj_ci(spec[reg][ds])) for reg in INFO_ORDER}
        # k* and subspace-specific threshold along the info order
        kstar = next((reg for reg in INFO_ORDER if regci[reg]["dU"]["lo"] > 0), None)
        kspec = next((reg for reg in INFO_ORDER if regci[reg]["dU"]["lo"] > 0 and regci[reg]["spec"]["lo"] > 0), None)
        rf_lcb = regci["RF"]["dU"]["lo"]
        recov = {reg: (regci[reg]["dU"]["mean"] / regci["RF"]["dU"]["mean"]) if rf_lcb > 0 and abs(regci["RF"]["dU"]["mean"]) > 1e-9 else None
                 for reg in INFO_ORDER}
        oc = _subj_ci(orc[ds]); ocr = _subj_ci(orc_rand[ds])
        hs = {reg: dict(sel_minus_native=_subj_ci(ho_selnat[reg][ds]), native_minus_source=_subj_ci(ho_natsrc[reg][ds])) for reg in LABEL_REGIMES}
        headcase = _head_case(hs["RF"])
        ds_stats[ds] = {**{reg: {"dU": regci[reg]["dU"], "spec": regci[reg]["spec"]} for reg in INFO_ORDER},
                        "_headcase": headcase, "_rf_lcb": rf_lcb}
        per_ds.append(dict(dataset=ds, n_subjects=regci["R0"]["dU"]["n"],
                           dU={reg: dict(mean=regci[reg]["dU"]["mean"], lcb=regci[reg]["dU"]["lo"], p=regci[reg]["dU"]["signflip_p"]) for reg in INFO_ORDER},
                           specific={reg: dict(mean=regci[reg]["spec"]["mean"], lcb=regci[reg]["spec"]["lo"]) for reg in INFO_ORDER},
                           minimal_info_level=kstar, subspace_specific_threshold=kspec,
                           recovery={reg: recov[reg] for reg in INFO_ORDER},
                           crossfit_oracle=dict(delta=oc["mean"], lcb=oc["lo"], random=ocr["mean"]),
                           head_only_case=headcase, head_only_sel_minus_native={reg: hs[reg]["sel_minus_native"]["mean"] for reg in LABEL_REGIMES},
                           skipped=dict(skipped)))
    route = _route(ds_stats) if len(ds_stats) >= 1 else dict(verdict="NO_DATA", next="no ok cells")
    out = dict(per_dataset=per_ds, routing=route, n_cells=len(cells), skipped=dict(skipped),
               discipline="inference unit=target subject (seeds->subj first), subject-cluster bootstrap + exact sign-flip; "
                          "positive claims require BOTH datasets; no CLOSED without 3rd-dataset confirm; manuscript FROZEN")
    json.dump(out, open(d / "info_ladder_verdict.json", "w"), indent=2, default=float)
    print(f"[il-agg] {len(cells)} cells ({sum(skipped.values())} skipped); routing={route['verdict']}")
    for s in per_ds:
        print(f"  {s['dataset']} (n={s['n_subjects']}): k*={s['minimal_info_level']} subspace-specific*={s['subspace_specific_threshold']} headcase={s['head_only_case']}")
        print("    dU      : " + " ".join(f"{reg}={s['dU'][reg]['mean']:+.4f}[{s['dU'][reg]['lcb']:+.4f}]" for reg in INFO_ORDER))
        print("    specific: " + " ".join(f"{reg}={s['specific'][reg]['mean']:+.4f}[{s['specific'][reg]['lcb']:+.4f}]" for reg in INFO_ORDER))
        print(f"    oracle-ceiling delta={s['crossfit_oracle']['delta']:+.4f}[{s['crossfit_oracle']['lcb']:+.4f}] vs random {s['crossfit_oracle']['random']:+.4f}")
    print(f"  -> {route['verdict']} : next = {route['next']}")


if __name__ == "__main__":
    main()
