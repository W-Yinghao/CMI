"""V2 report: reads v2_<tag>_summary.json and emits v2_<tag>_report.md with per-world acceptance-power,
unsafe-accept, and useless-accept tallies + the main decision table. Target metrics are a post-hoc audit.
  python -m tos_cmi.eeg.report_v2 [--tag smoke]
"""
from __future__ import annotations
import argparse
import json
from tos_cmi.eeg.v2_worlds import WORLDS, PRINCIPLED, CONTROLS

OUT = "tos_cmi/results/method_deepen/v2"


def _correct(v):
    """Per-cell correctness against the world's ground truth (principled erasers only)."""
    w, act = v["world"], v["gate_action"]
    if v["intervention"] not in PRINCIPLED:
        return None
    if w == "A":                       # beneficial: correct iff ACCEPT with real target gain
        return act == "ACCEPT" and v["dtgt_bacc_lo"] > 0.01
    if w == "B":                       # unsafe: correct iff NOT accepted
        return act != "ACCEPT"
    if w == "C":                       # useless: correct iff NOT accepted
        return act != "ACCEPT"
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="smoke"); a = ap.parse_args()
    S = json.load(open("%s/v2_%s_summary.json" % (OUT, a.tag)))
    summ, ch, th, pr = S["summary"], S.get("config_hash", "?"), S["thresholds"], S["params"]
    L = ["# V2 semi-synthetic acceptance-power benchmark --- report (%s)\n" % a.tag,
         "Config sha256:`%s`; thresholds FROZEN (safety UCB<=%.3f, benefit LCB>%.3f, domain diagnostic-only, "
         "target audit-only). World-gen: phi=%.2f beta=%.2f m=%d noise=%.2f. **Semi-synthetic: real latents + "
         "injected nuisance; method-deepening evidence, not a main-paper claim.**\n"
         % (ch, th["safety_eps"], th["benefit_lcb"], pr["phi"], pr["beta"], pr["m"], pr["noise"])]
    verdicts = {}
    for wk in ["A", "B", "C"]:
        cells = [v for v in summ.values() if v["world"] == wk]
        if not cells:
            continue
        meta = WORLDS[wk]
        L += ["## World %s --- %s (expect %s)" % (wk, meta["name"], meta["expect"]),
              "| intervention | n_src | alpha | task-drop UCB | benefit LCB | domain-gain | gate | "
              "target ΔbAcc [CI] | router acc | correct |", "|---|---|---|---|---|---|---|---|---|---|"]
        for v in sorted(cells, key=lambda v: (v["intervention"], str(v["n_source"]), v["alpha"])):
            cc = _correct(v); cs = "-" if cc is None else ("yes" if cc else "**NO**")
            ra = "%.2f" % v["router_acc"] if v["router_acc"] == v["router_acc"] else "-"
            L.append("| %s | %s | %.2f | %+.3f | %s | %+.3f | **%s** | %+.3f [%+.3f,%+.3f] | %s | %s |"
                     % (v["intervention"], v["n_source"], v["alpha"], v["task_drop_ucb"],
                        ("%+.3f" % v["benefit_lcb"]) if v["benefit_lcb"] == v["benefit_lcb"] else "n/a",
                        v["domain_gain"], v["gate_action"], v["dtgt_bacc"], v["dtgt_bacc_lo"],
                        v["dtgt_bacc_hi"], ra, cs))
        prin = [v for v in cells if v["intervention"] in PRINCIPLED]
        n_acc = sum(1 for v in prin if v["gate_action"] == "ACCEPT")
        if wk == "A":
            good = sum(1 for v in prin if v["gate_action"] == "ACCEPT" and v["dtgt_bacc_lo"] > 0.01)
            verdicts["A"] = good >= 1
            L += ["", "**Acceptance power:** %d/%d principled cells ACCEPT with real target gain "
                  "(target ΔbAcc LCB>+0.01). Smoke wants >=1 -> %s.\n"
                  % (good, len(prin), "PASS" if good >= 1 else "FAIL")]
        elif wk == "B":
            verdicts["B"] = n_acc == 0
            L += ["", "**Unsafe-accept:** %d/%d principled cells ACCEPTED (want 0) -> %s.\n"
                  % (n_acc, len(prin), "PASS" if n_acc == 0 else "FAIL")]
        else:
            verdicts["C"] = n_acc == 0
            hi_dg = sum(1 for v in prin if v["domain_gain"] > 0.05)
            L += ["", "**Useless-accept:** %d/%d principled cells ACCEPTED (want 0); %d cells have high "
                  "domain-gain yet do not drive ACCEPT -> domain-gain != benefit -> %s.\n"
                  % (n_acc, len(prin), hi_dg, "PASS" if n_acc == 0 else "FAIL")]
    L += ["## Smoke verdict",
          "- World A (acceptance power): %s" % ("PASS" if verdicts.get("A") else "FAIL"),
          "- World B (no unsafe accept): %s" % ("PASS" if verdicts.get("B") else "FAIL"),
          "- World C (no useless accept): %s" % ("PASS" if verdicts.get("C") else "FAIL"),
          "- **overall: %s**" % ("PASS" if all(verdicts.get(k) for k in ("A", "B", "C")) else "FAIL"),
          "", "If World A fails, tune the WORLD GENERATOR (phi/alpha/m/n_source) -- never the gate thresholds."]
    open("%s/v2_%s_report.md" % (OUT, a.tag), "w").write("\n".join(L) + "\n")
    print("\n".join(L)); print("\nV2_REPORT_DONE")


if __name__ == "__main__":
    main()
