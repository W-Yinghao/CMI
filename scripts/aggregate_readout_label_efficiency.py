"""Aggregate the Target Readout Calibration Ladder (CPU, env c84c). Per dataset and inference-unit = target subject
(draws already averaged in-cell; seeds -> subject first; subject-cluster bootstrap + exact sign-flip; Holm across the
7 budgets):
  dU_MAP-frozen(k), dU_MAP-fresh(k) (anchoring), dU_MAP-bias(k) (low-capacity), dGh_specific(k) (subspace specificity)
  k*_utility = min{k: LCB95[dU_MAP-frozen]>0}, k*_anchor = min{k: LCB95[dU_MAP-fresh]>0}
Datasets split into DEV (BNCI2014_001, BNCI2015_001) and CONFIRMATORY (Lee2019_MI, BNCI2014_004); external drives
status. Routes R-A..R-F. Refuses partial matrices. Manuscript FROZEN; only the owner stops/redirects a line.

  python scripts/aggregate_readout_label_efficiency.py --from-dir results/cmi_trace_readout --expect 252
"""
from __future__ import annotations
import argparse, glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(REPO))
from scripts.aggregate_mcc import _cluster_ci

DEV = ["BNCI2014_001", "BNCI2015_001"]
CONFIRM = ["Lee2019_MI", "BNCI2014_004"]
DATASETS = DEV + CONFIRM
BUDGETS = ["1", "2", "4", "8", "16", "32", "Full"]
FEW = ["1", "2", "4", "8"]
ENDPTS = ["dU_MAP_frozen", "dU_MAP_fresh", "dU_MAP_bias", "dGh_specific"]


def _subj_ci(per_subj):
    return _cluster_ci([float(np.mean(v)) for v in per_subj.values()])


def _holm(pvals):
    """Holm-Bonferroni: return dict budget->adjusted p (one-sided sign-flip p's over the 7 budgets)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); adj = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p)); adj[k] = running
    return adj


def _kstar(ci_by_k, holm_by_k, use_holm=True):
    """First budget (info order) with LCB95>0 (and Holm-adjusted sign-flip p<0.05 if use_holm)."""
    for k in BUDGETS:
        if ci_by_k[k]["lo"] > 0 and (not use_holm or holm_by_k.get(k, 1.0) < 0.05):
            return k
    return None


def _route(stats):
    """stats[ds] = {'kU':k*_utility, 'kA':k*_anchor, 'specific_any_k':bool, 'bias_ge_map':bool, 'full_frozen_pos':bool}."""
    def few(k): return k is not None and k in FEW
    dev_both_few = all(stats[ds]["both_few"] for ds in DEV if ds in stats)   # each dev: MAP>frozen AND MAP>fresh few-shot
    confirm_both = [ds for ds in CONFIRM if ds in stats and stats[ds]["both_few"]]  # SAME dataset satisfies both
    confirm_full = [ds for ds in CONFIRM if ds in stats and stats[ds]["full_frozen_pos"]]
    harm_anywhere = any(stats[ds]["harm"] for ds in stats)                   # MAP significantly worse than from-scratch anywhere
    specific = [ds for ds in stats if stats[ds]["specific_any_k"]]
    any_readout_benefit = any(stats[ds]["kA"] is not None or stats[ds]["full_frozen_pos"] for ds in stats)
    # R-E: subspace revival only if dGh_specific LCB>0 on >=2 independent datasets
    if len(specific) >= 2:
        return dict(verdict="R-E_INFORMED_SUBSPACE_BEATS_RANDOM_REVIVE", datasets_specific=specific,
                    next="dGh_specific LCB>0 on >=2 datasets -> re-activate subspace-actionability; confirm design")
    # R-A: MAP beats frozen AND fresh few-shot on EVERY dev dataset AND on >=1 confirmatory dataset (SAME dataset for
    # both conditions), with NO significant MAP-worse-than-fresh harm anywhere.
    if dev_both_few and confirm_both and not harm_anywhere:
        return dict(verdict="R-A_MAP_WINS_FEWSHOT_LABEL_EFFICIENT", confirm_both=confirm_both,
                    next="source-anchored MAP beats frozen AND fresh at k<=8 on every dev + >=1 confirmatory (same dataset), no harm -> real deployable readout-adaptation direction; fix source-prior adaptation, add stronger few-shot baselines, test bias-only vs full-direction")
    # R-C: bias/temp ~ MAP (MAP never beats bias) AND a readout benefit exists -> low-capacity calibration sufficient
    if any_readout_benefit and all(stats[ds]["bias_ge_map"] for ds in stats):
        return dict(verdict="R-C_BIAS_TEMPERATURE_SUFFICIENT",
                    next="a readout benefit exists but MAP does not beat bias/temperature -> the bottleneck is prior shift/class bias/logit scale, not a full head-direction update")
    dev_anchor_any = any(stats[ds]["kA"] is not None for ds in DEV if ds in stats)   # MAP beats fresh on >=1 dev
    # R-D: MAP anchoring works (beats from-scratch) but subspace NOT revived (<2 specific datasets) -> generic
    # readout shift. Takes precedence over R-B because anchoring IS a real readout effect; R-B is the no-anchoring,
    # full-cal-only regime. len(specific)<2 (not ==0) so a single specific dataset -- not enough to revive per R-E --
    # still routes to the generic verdict rather than falling through.
    if dev_anchor_any and len(specific) < 2:
        note = "" if not specific else f" (1 dataset {specific} shows specificity but <2 required to revive)"
        return dict(verdict="R-D_READOUT_SHIFT_GENERIC_SUBSPACE_NOT_CAUSAL",
                    next="MAP anchoring beats from-scratch (readout adaptation is real) but dGh_specific not LCB>0 on >=2 datasets -> the readout shift is GENERIC; subspace erasure is NOT causal; pursue readout label-efficiency, not subspace surgery" + note)
    # R-B: only Full calibration positive, no few-shot anchoring -> needs many labels
    dev_full_only = all((stats[ds]["kU"] == "Full") for ds in DEV if ds in stats)
    if dev_full_only and not dev_anchor_any:
        confirm = "; confirmatory replicate: " + (",".join(confirm_full) if confirm_full else "NONE (R-F risk)")
        return dict(verdict="R-B_FULL_CALIBRATION_ONLY", next="readout adaptation needs MANY labels (only Full is positive, no few-shot anchoring) -> next question is which supervision/structure lowers calibration sample complexity, NOT erasure" + confirm)
    # R-F: confirmatory datasets do not replicate the dev full-cal head gain
    if not confirm_full and any(stats[ds]["full_frozen_pos"] for ds in DEV if ds in stats):
        return dict(verdict="R-F_DEVELOPMENT_ONLY_NO_EXTERNAL_REPLICATION",
                    next="dev full-cal head gain does NOT replicate on the confirmatory datasets -> dataset/session-specific, not a general readout method")
    return dict(verdict="READOUT_NULL_OR_INCONCLUSIVE", next="no robust readout-adaptation gain; inspect per-dataset heterogeneity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", default="results/cmi_trace_readout"); ap.add_argument("--expect", type=int, default=252)
    a = ap.parse_args()
    d = Path(a.from_dir); done = sorted(glob.glob(str(d / "cells" / "*.done")))
    if len(done) < a.expect:
        print(f"[ro-agg] INCOMPLETE {len(done)}/{a.expect} cells -> REFUSING."); raise SystemExit(2)
    acc = {e: {k: defaultdict(lambda: defaultdict(list)) for k in BUDGETS} for e in ENDPTS}   # ep -> k -> ds -> subj -> [seed]
    skipped = defaultdict(int)
    for c in sorted(glob.glob(str(d / "cells" / "*.json"))):
        r = json.loads(Path(c).read_text())
        if r.get("status") != "ok":
            skipped[r.get("reason", "?")] += 1; continue
        ds, subj = r["dataset"], r["subject"]
        for k in BUDGETS:
            ek = r["endpoints"].get(k, {})
            for e in ENDPTS:
                v = ek.get(e)
                if v is not None and np.isfinite(v):
                    acc[e][k][ds][subj].append(v)

    per_ds = []; stats = {}
    for ds in DATASETS:
        if ds not in acc["dU_MAP_frozen"]["Full"] or not acc["dU_MAP_frozen"]["Full"][ds]:
            continue
        ci = {e: {k: _subj_ci(acc[e][k][ds]) for k in BUDGETS} for e in ENDPTS}
        holm = {e: _holm({k: ci[e][k]["signflip_p"] for k in BUDGETS}) for e in ENDPTS}
        kU = _kstar(ci["dU_MAP_frozen"], holm["dU_MAP_frozen"])
        kA = _kstar(ci["dU_MAP_fresh"], holm["dU_MAP_fresh"])
        specific_any = any(ci["dGh_specific"][k]["lo"] > 0 and holm["dGh_specific"][k] < 0.05 for k in BUDGETS)
        bias_ge_map = all(ci["dU_MAP_bias"][k]["lo"] <= 0 for k in BUDGETS)   # MAP never significantly beats bias
        full_frozen_pos = ci["dU_MAP_frozen"]["Full"]["lo"] > 0
        both_few = kU in FEW and kA in FEW                                    # SAME dataset: MAP>frozen AND MAP>fresh few-shot
        # NO-HARM gate must include harm vs the FROZEN head (the deployable reference): if few-shot adaptation hurts
        # on average vs just keeping the source head, it is not deployable there. (dU_MAP_fresh harm alone is a
        # strawman -- a from-scratch head is trivially bad at 1-2 labels, so it never triggers.) Directional (mean)
        # because a NEGATIVE mean few-shot utility is a deploy risk even if the CI crosses 0.
        harm_vs_fresh = any(ci["dU_MAP_fresh"][k]["hi"] < 0 for k in FEW)
        harm_vs_frozen = min(ci["dU_MAP_frozen"]["1"]["mean"], ci["dU_MAP_frozen"]["2"]["mean"]) < -0.005
        stats[ds] = dict(kU=kU, kA=kA, specific_any_k=specific_any, bias_ge_map=bias_ge_map,
                         full_frozen_pos=full_frozen_pos, both_few=both_few,
                         harm=(harm_vs_fresh or harm_vs_frozen), harm_vs_frozen=harm_vs_frozen)
        per_ds.append(dict(dataset=ds, role=("dev" if ds in DEV else "confirmatory"), n_subjects=ci["dU_MAP_frozen"]["Full"]["n"],
                           kstar_utility=kU, kstar_anchor=kA, subspace_specific_any_k=specific_any,
                           bias_sufficient=bias_ge_map, full_cal_frozen_positive=full_frozen_pos,
                           **{e: {k: dict(mean=ci[e][k]["mean"], lcb=ci[e][k]["lo"], holm_p=holm[e][k]) for k in BUDGETS} for e in ENDPTS}))
    route = _route(stats) if stats else dict(verdict="NO_DATA", next="no ok cells")
    out = dict(per_dataset=per_ds, routing=route, n_cells=len(done), skipped=dict(skipped),
               dev_datasets=DEV, confirmatory_datasets=CONFIRM,
               discipline="inference unit=target subject (draws->seed->subject), subject-cluster bootstrap + exact sign-flip, Holm across 7 budgets; "
                          "external datasets drive status; no CLOSED without external replication; manuscript FROZEN")
    json.dump(out, open(d / "readout_verdict.json", "w"), indent=2, default=float)
    print(f"[ro-agg] {len(done)} cells ({sum(skipped.values())} skipped); routing={route['verdict']}")
    for s in per_ds:
        print(f"  {s['dataset']} [{s['role']}] (n={s['n_subjects']}): k*_util={s['kstar_utility']} k*_anchor={s['kstar_anchor']} subspace-specific={s['subspace_specific_any_k']} bias-suff={s['bias_sufficient']} full-cal-frozen+={s['full_cal_frozen_positive']}")
        for e in ("dU_MAP_frozen", "dU_MAP_fresh", "dGh_specific"):
            print(f"    {e:14s}: " + " ".join(f"k{k}={s[e][k]['mean']:+.3f}[{s[e][k]['lcb']:+.3f}]" for k in BUDGETS))
    print(f"  -> {route['verdict']} : next = {route['next']}")


if __name__ == "__main__":
    main()
