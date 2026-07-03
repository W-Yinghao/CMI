"""Big-N validation report (PM template). Reads the committed analyzer outputs for one dataset --
erasure_report.json (Track G source-side, per backbone) + erasure_target_deploy summary (C12) +
validation_manifest.json -- and emits a CONFIRM / MIXED / OVERTURN readout using the PRE-REGISTERED
thresholds (see tos_cmi/notes/REAL_EEG_VALIDATION.md). Does NOT recompute; run the analyzers first.
  python -m tos_cmi.eeg.bigN_report <DATASET>
"""
from __future__ import annotations
import json
import os
import sys

RESULTS = "tos_cmi/results/tos_cmi_eeg_frozen"
PRINCIPLED = ["LEACE", "TOS_VD", "RLACE"]   # erasers that, if they helped target, would OVERTURN (not INLP/random)


def _deploy_summary(dataset):
    p = ("%s/erasure_target_deploy/erasure_target_deploy_summary.json" % RESULTS) if dataset == "BNCI2014_001" \
        else ("%s/erasure_target_deploy/%s/erasure_target_deploy_summary.json" % (RESULTS, dataset))
    return json.load(open(p)) if os.path.exists(p) else None


def _erasure_report(dataset, bb):
    p = "%s/%s_%s_LOSO/erasure_report.json" % (RESULTS, dataset, bb)
    return json.load(open(p))["aggregate"] if os.path.exists(p) else None


def _manifest(dataset):
    p = "%s/validation_manifest.json" % RESULTS
    if not os.path.exists(p):
        return {}
    return {(c["backbone"], c["seed"]): c for c in json.load(open(p)) if c["dataset"] == dataset}


def _gain_label(hi):
    if hi < 0.01:
        return "no practically meaningful gain"
    if hi <= 0.02:
        return "no statistically-supported gain (underpowered)"
    return "POSSIBLE GAIN (upper CI > +0.02)"


def _c12_verdict(S, bb):
    """CONFIRM if no principled eraser has upper CI > +0.02; OVERTURN if any has lower CI > +0.01; else MIXED."""
    rows = {m: S["summary"].get("%s|%s" % (bb, m)) for m in PRINCIPLED}
    rows = {m: r for m, r in rows.items() if r}
    if not rows:
        return "N/A (no valid folds)", {}
    overturn = any(r["dtgt_bacc_lo"] > 0.01 for r in rows.values())
    confirm = all(r["dtgt_bacc_hi"] <= 0.02 for r in rows.values())
    v = "OVERTURN" if overturn else ("CONFIRM" if confirm else "MIXED")
    return v, rows


def _c78_verdict(agg):
    """2a erasure profile: LEACE linear->~chance, task preserved, MLP residual present, INLP destroys task."""
    if not agg:
        return "DEGENERATE (no valid folds)"
    ch = agg["chance_subj"]
    leace_lin_chance = agg.get("subj_LEACE_lin", 1) <= ch + 0.06
    task_pres = agg.get("task_LEACE_lin", 0) >= agg.get("task_full_lin", 1) - 0.06
    mlp_resid = agg.get("subj_LEACE_mlp", 0) > ch + 0.05
    inlp_hurts = agg.get("task_INLP_lin", 1) < agg.get("task_full_lin", 1) - 0.08
    score = sum([leace_lin_chance, task_pres, mlp_resid, inlp_hurts])
    return "CONFIRM" if score >= 3 else ("MIXED" if score == 2 else "OVERTURN")


def _report_bb(dataset, bb, S, man):
    agg = _erasure_report(dataset, bb)
    cells = [man[(bb, s)] for s in (0, 1, 2) if (bb, s) in man]
    comp = sum(c["n_folds_completed"] for c in cells)
    exp = sum(c["n_folds_expected"] for c in cells)
    zdim = cells[0]["z_dim"] if cells and cells[0]["z_dim"] else "?"
    # metric validity from the deployment summary presence
    has_deploy = S and any(k.startswith(bb + "|") for k in S.get("summary", {}))
    validity = "VALID" if (agg and has_deploy) else ("DEGENERATE" if not agg and not has_deploy else "PARTIAL")
    print("\n--- %s / %s ---" % (dataset, bb))
    print("  folds completed/expected: %d/%d ; z_dim=%s ; metric validity: %s" % (comp, exp, zdim, validity))
    if agg:
        ch = agg["chance_subj"]
        print("  Track G (source-side): full subj lin/mlp %.2f/%.2f -> LEACE %.2f/%.2f (chance %.3f); "
              "task full %.2f -> LEACE %.2f, INLP %.2f; TOS mlp %.2f RLACE mlp %.2f"
              % (agg.get("subj_full_lin", float('nan')), agg.get("subj_full_mlp", float('nan')),
                 agg.get("subj_LEACE_lin", float('nan')), agg.get("subj_LEACE_mlp", float('nan')), ch,
                 agg.get("task_full_lin", float('nan')), agg.get("task_LEACE_lin", float('nan')),
                 agg.get("task_INLP_lin", float('nan')), agg.get("subj_TOS_VD_mlp", float('nan')),
                 agg.get("subj_RLACE_mlp", float('nan'))))
        print("  C7/C8 erasure profile: %s" % _c78_verdict(agg))
    else:
        print("  Track G: no valid folds (backbone skipped / degenerate)")
    v, rows = _c12_verdict(S, bb) if S else ("N/A", {})
    if rows:
        full = S["summary"].get("%s|full" % bb, {})
        print("  C12 deployment: full-Z tgt bAcc %.3f NLL %.3f (chance %.2f), n=%d target-subjects x seeds"
              % (full.get("tgt_bacc_mean", float('nan')), full.get("tgt_nll_mean", float('nan')),
                 full.get("chance_task", float('nan')), full.get("n", 0)))
        for m in PRINCIPLED:
            r = rows.get(m)
            if r:
                print("    %-7s dtgt_bAcc %+.3f [%+.3f,%+.3f] (subject-cluster CI) -> %s"
                      % (m, r["dtgt_bacc"], r["dtgt_bacc_lo"], r["dtgt_bacc_hi"], _gain_label(r["dtgt_bacc_hi"])))
    print("  C12 target deployment: %s" % v)
    print("  TSMNet metric validity: %s" % (validity if bb == "TSMNet" else "n/a (EEGNet)"))
    return {"backbone": bb, "validity": validity, "c78": _c78_verdict(agg) if agg else "DEGENERATE",
            "c12": v, "folds": "%d/%d" % (comp, exp)}


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "BNCI2014_004"
    S = _deploy_summary(dataset)
    man = _manifest(dataset)
    print("=" * 78)
    print("BIG-N VALIDATION READOUT: %s" % dataset)
    print("Thresholds (pre-registered): dbAcc upper CI <+0.01 no-meaningful; +0.01..+0.02 underpowered; "
          ">+0.02 = gain. Compare PAIRED delta vs full-Z, not absolute bAcc.")
    print("=" * 78)
    res = [_report_bb(dataset, bb, S, man) for bb in ("EEGNet", "TSMNet")]   # EEGNet first (PM order)
    print("\n=== ONE-LINE VERDICTS (%s) ===" % dataset)
    for r in res:
        print("  %-7s folds %s | metric %s | C7/C8 %s | C12 %s"
              % (r["backbone"], r["folds"], r["validity"], r["c78"], r["c12"]))
    print("BIGN_REPORT_DONE")


if __name__ == "__main__":
    main()
