"""Track B post-hoc audit: compare the SOURCE-ONLY gate decisions (trackB_summary) against the ACTUAL target
outcomes (erasure_target_deploy summaries). Target labels enter ONLY here, for scoring the gate -- never in the
gate itself. Emits tos_cmi/results/method_deepen/trackB/trackB_report.md.
  python -m tos_cmi.eeg.report_trackB [--tag full]
Definitions (actual target ΔbAcc paired subject-cluster CI vs full Z):
  beneficial = lower CI > +0.01 ; harmful = upper CI < -0.01 ; neither otherwise.
  false-accept = ACCEPT a non-beneficial ; false-reject = REJECT a beneficial ; missed-benefit = ABSTAIN a beneficial.
  harm-prevented = REJECT a harmful.
"""
from __future__ import annotations
import argparse
import json
import os
from tos_cmi.eeg import bigN_report as B

OUT = "tos_cmi/results/method_deepen/trackB"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="full"); a = ap.parse_args()
    suff = ("_" + a.tag) if a.tag else ""
    gate = json.load(open("%s/trackB_summary%s.json" % (OUT, suff)))["summary"]
    rows, fa, fr, mb, hp = [], 0, 0, 0, 0
    for key in sorted(gate):
        g = gate[key]; ds, bb, m = g["dataset"], g["backbone"], g["method"]
        S = B._deploy_summary(ds)
        dep = S["summary"].get("%s|%s" % (bb, m)) if S else None
        db = dep["dtgt_bacc"] if dep else float("nan")
        lo = dep["dtgt_bacc_lo"] if dep else float("nan")
        hi = dep["dtgt_bacc_hi"] if dep else float("nan")
        beneficial = dep is not None and lo > 0.01
        harmful = dep is not None and hi < -0.01
        act = g["action"]
        false_accept = act == "ACCEPT" and not beneficial
        false_reject = act == "REJECT" and beneficial
        missed = act == "ABSTAIN" and beneficial
        harm_prev = act == "REJECT" and harmful
        fa += false_accept; fr += false_reject; mb += missed; hp += harm_prev
        correct = (act == "ACCEPT" and beneficial) or (act in ("REJECT", "ABSTAIN") and not beneficial)
        rows.append(dict(ds=ds, bb=bb, m=m, act=act, tdu=g["task_drop_ucb"], blcb=g["benefit_lcb"],
                         db=db, lo=lo, hi=hi, beneficial=beneficial, harmful=harmful, correct=correct,
                         false_accept=false_accept, false_reject=false_reject, harm_prev=harm_prev))
    os.makedirs(OUT, exist_ok=True)
    L = ["# Track B source-OOD benefit gate --- post-hoc target audit\n",
         "Gate is SOURCE-ONLY; target used only here to score it. Actual = target ΔbAcc [subject-cluster CI].\n",
         "| dataset | bb | method | gate action | src task-drop UCB | src benefit LCB | **actual target ΔbAcc [CI]** | class | correct? |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cls = "beneficial" if r["beneficial"] else ("HARMFUL" if r["harmful"] else "neutral")
        L.append("| %s | %s | %s | **%s** | %+.3f | %+.3f | %+.3f [%+.3f,%+.3f] | %s | %s |"
                 % (r["ds"], r["bb"], r["m"], r["act"], r["tdu"], r["blcb"], r["db"], r["lo"], r["hi"],
                    cls, "yes" if r["correct"] else "**NO**"))
    acts = [r["act"] for r in rows]
    L += ["", "## Summary",
          "- cells: %d  (ACCEPT %d / REJECT %d / ABSTAIN %d)"
          % (len(rows), acts.count("ACCEPT"), acts.count("REJECT"), acts.count("ABSTAIN")),
          "- **false-accept (ACCEPT a non-beneficial): %d**" % fa,
          "- false-reject (REJECT a beneficial): %d" % fr,
          "- missed-benefit (ABSTAIN a beneficial): %d" % mb,
          "- **harm-prevented (REJECT a harmful): %d**" % hp,
          "- correct decisions: %d/%d" % (sum(1 for r in rows if r["correct"]), len(rows)),
          "- harmful cells actually present: %d (all should be REJECT/ABSTAIN)" % sum(1 for r in rows if r["harmful"]),
          "- beneficial cells actually present: %d" % sum(1 for r in rows if r["beneficial"])]
    open("%s/trackB_report.md" % OUT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nTRACKB_REPORT_DONE")


if __name__ == "__main__":
    main()
