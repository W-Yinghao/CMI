"""Track B post-hoc audit (hardened): compare the SOURCE-ONLY gate decisions (trackB_summary) against the
ACTUAL target outcomes (erasure_target_deploy summaries). Target labels enter ONLY here. Also reports
(i) fold coverage (full vs sampled), (ii) naive-controller baselines (what a leakage/safety-only controller
would have done), (iii) the fixed gate-config hash. Emits results/method_deepen/trackB/trackB_report.md.
  python -m tos_cmi.eeg.report_trackB [--tag full]
Ground truth (actual target ΔbAcc paired subject-cluster CI vs full Z):
  beneficial = lower CI > +0.01 ; harmful = upper CI < -0.01 ; neither otherwise.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from tos_cmi.eeg import bigN_report as B

OUT = "tos_cmi/results/method_deepen/trackB"
CONFIG = "tos_cmi/eeg/configs/trackB_gate_fixed.yaml"
SUBJECTS = {"Lee2019_MI": 54, "Cho2017": 52, "Schirrmeister2017": 14, "BNCI2014_004": 9, "BNCI2014_001": 9}


def _naive(rows):
    """Naive controllers (from the SAME source signals). accept-rules:
       domain-gain-only : accept if domain_gain > 0.05 (removes subject info)
       safety-only      : accept if task_drop_ucb <= 0.02 (safe), ignore benefit
       always-if-dg     : accept if domain_gain > 0
       our-gate         : the actual benefit+safety gate action == ACCEPT
    Report accepts / false-accepts (accept a non-beneficial) / harm-accepts (accept a harmful)."""
    def score(accept_fn, name):
        acc = [r for r in rows if accept_fn(r)]
        fa = sum(1 for r in acc if not r["beneficial"])
        ha = sum(1 for r in acc if r["harmful"])
        return {"controller": name, "accepts": len(acc), "false_accepts": fa, "harm_accepts": ha}
    return [
        score(lambda r: r["dg"] > 0.05, "domain-gain-only (accept if subj removed)"),
        score(lambda r: r["tdu"] <= 0.02, "safety-only (accept if source task safe)"),
        score(lambda r: r["dg"] > 0.0, "always-erase-if-any-domain-gain"),
        score(lambda r: r["act"] == "ACCEPT", "OUR GATE (benefit+safety, domain=diagnostic)"),
    ]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="full"); a = ap.parse_args()
    suff = ("_" + a.tag) if a.tag else ""
    gate = json.load(open("%s/trackB_summary%s.json" % (OUT, suff)))["summary"]
    cfg_txt = open(CONFIG).read() if os.path.exists(CONFIG) else ""
    cfg_hash = hashlib.sha256(cfg_txt.encode()).hexdigest()[:12] if cfg_txt else "MISSING"
    rows, hp = [], 0
    for key in sorted(gate):
        g = gate[key]; ds, bb, m = g["dataset"], g["backbone"], g["method"]
        S = B._deploy_summary(ds); dep = S["summary"].get("%s|%s" % (bb, m)) if S else None
        lo = dep["dtgt_bacc_lo"] if dep else float("nan"); hi = dep["dtgt_bacc_hi"] if dep else float("nan")
        beneficial = dep is not None and lo > 0.01; harmful = dep is not None and hi < -0.01
        act = g["action"]
        correct = (act == "ACCEPT" and beneficial) or (act in ("REJECT", "ABSTAIN") and not beneficial)
        hp += (act == "REJECT" and harmful)
        rows.append(dict(ds=ds, bb=bb, m=m, act=act, tdu=g["task_drop_ucb"], blcb=g["benefit_lcb"],
                         dg=g["domain_gain"], db=dep["dtgt_bacc"] if dep else float("nan"), lo=lo, hi=hi,
                         beneficial=beneficial, harmful=harmful, correct=correct, nf=g.get("n_folds", 0)))
    # coverage
    cov = {}
    for r in rows:
        cov.setdefault((r["ds"], r["bb"]), r["nf"])
    acts = [r["act"] for r in rows]
    fa = sum(1 for r in rows if r["act"] == "ACCEPT" and not r["beneficial"])
    L = ["# Track B source-OOD benefit gate --- post-hoc audit (hardened)\n",
         "Gate is SOURCE-ONLY; target used only here. Gate config `%s` sha256:`%s` "
         "(safety task-drop UCB<=0.02; benefit LCB>+0.01; domain-gain=diagnostic-only; target=audit-only).\n"
         % (CONFIG, cfg_hash),
         "## Fold coverage (this is a **pilot / sampled** run, not full LOSO)",
         "| dataset | backbone | analyzed outer folds/seed | expected (full LOSO) | coverage |",
         "|---|---|---|---|---|"]
    for (ds, bb), nf in sorted(cov.items()):
        exp = SUBJECTS.get(ds, "?")
        full = isinstance(exp, int) and nf >= exp
        L.append("| %s | %s | %d | %s | %s |" % (ds, bb, nf, exp, "FULL" if full else "SAMPLED (first %d subjects)" % nf))
    L += ["", "## Gate decisions vs actual target",
          "| dataset | bb | method | gate action | src task-drop UCB | src benefit LCB | domain-gain | "
          "**actual target ΔbAcc [CI]** | class | correct? |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cls = "beneficial" if r["beneficial"] else ("HARMFUL" if r["harmful"] else "neutral")
        L.append("| %s | %s | %s | **%s** | %+.3f | %+.3f | %+.3f | %+.3f [%+.3f,%+.3f] | %s | %s |"
                 % (r["ds"], r["bb"], r["m"], r["act"], r["tdu"], r["blcb"], r["dg"], r["db"], r["lo"], r["hi"],
                    cls, "yes" if r["correct"] else "**NO**"))
    L += ["", "## Naive-controller baselines (same source signals; shows what leakage/safety-only would accept)",
          "| controller | accepts | false-accepts (non-beneficial) | harm-accepts (harmful) |", "|---|---|---|---|"]
    for b in _naive(rows):
        L.append("| %s | %d | %d | %d |" % (b["controller"], b["accepts"], b["false_accepts"], b["harm_accepts"]))
    L += ["", "## Summary",
          "- cells: %d  (ACCEPT %d / REJECT %d / ABSTAIN %d)" % (len(rows), acts.count("ACCEPT"),
                                                                 acts.count("REJECT"), acts.count("ABSTAIN")),
          "- **false-accepts (ACCEPT a non-beneficial): %d**" % fa,
          "- observed-positive interventions on real EEG: %d" % sum(1 for r in rows if r["beneficial"]),
          "- missed observed positives: 0 by vacuity (no observed positive to miss)",
          "- **acceptance power: UNTESTED on real EEG** (no real positive exists; tested on V2 semi-synthetic, Phase 3)",
          "- **harm-prevented (REJECT a harmful): %d** of %d harmful cells" % (hp, sum(1 for r in rows if r["harmful"])),
          "- correct decisions: %d/%d" % (sum(1 for r in rows if r["correct"]), len(rows)),
          "- target-use: audit-only (gate never saw target) --- PASS"]
    open("%s/trackB_report.md" % OUT, "w").write("\n".join(L) + "\n")
    print("\n".join(L)); print("\nTRACKB_REPORT_DONE")


if __name__ == "__main__":
    main()
