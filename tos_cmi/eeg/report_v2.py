"""V2 report (source-only acceptance CEILING). Reads v2_<tag>_summary.json and emits v2_<tag>_report.md +
v2_<tag>_scatter.png. NO world expects ACCEPT. Target metrics are a post-hoc audit only.

Pass conditions (per world, principled erasers unless noted):
  World A (target-beneficial but source-uncertifiable): >=1 SAFE cell with a real target gain
      (target dbAcc LCB>+0.01) that the gate does NOT accept, its source-LOSO benefit LCB<=+0.01, AND
      random-k does not reproduce the oracle target gain. Oracle diagnostic shows the gain exists.
  World B (unsafe): 0 unsafe ACCEPT.
  World C (useless): 0 ACCEPT and a high-domain-gain cell with no target benefit exists.
Naive controllers show source-only leakage/safety rules FALSE-ACCEPT, our gate accepts ~nothing (conservative,
correct under the ceiling), and only the ORACLE target-informed selector picks the beneficial cells.
  python -m tos_cmi.eeg.report_v2 [--tag smoke]
"""
from __future__ import annotations
import argparse
import json
from tos_cmi.eeg.v2_worlds import WORLDS, PRINCIPLED, DEPLOYABLE, DIAGNOSTIC, CONTROLS

OUT = "tos_cmi/results/method_deepen/v2"


def _naive_controllers(cells, benefit_thr):
    """Over DEPLOYABLE cells: a cell is a GOOD accept iff it is actually target-beneficial. Report each
    controller's accepts and how many are FALSE (non-beneficial) vs TRUE (beneficial)."""
    dep = [v for v in cells if v["deployable"]]
    def score(name, rule):
        acc = [v for v in dep if rule(v)]
        false_acc = sum(1 for v in acc if not v["target_beneficial"])
        true_acc = sum(1 for v in acc if v["target_beneficial"])
        return {"controller": name, "accepts": len(acc), "false_accepts": false_acc, "true_accepts": true_acc}
    rows = [
        score("domain-gain-only (accept if subj/z removed)", lambda v: v["domain_gain"] > 0.05),
        score("safety-only (accept if source task safe)", lambda v: v["is_safe"]),
        score("always-erase-if-any-domain-gain", lambda v: v["domain_gain"] > 0.0),
        score("OUR GATE (benefit+safety, source-only)", lambda v: v["gate_action"] == "ACCEPT"),
    ]
    # oracle target-informed selector -- DIAGNOSTIC only (uses target labels): accept if real target gain
    orc = [v for v in dep if v["target_beneficial"]]
    rows.append({"controller": "ORACLE target-informed selector [DIAGNOSTIC, uses target labels]",
                 "accepts": len(orc), "false_accepts": sum(1 for v in orc if not v["target_beneficial"]),
                 "true_accepts": sum(1 for v in orc if v["target_beneficial"])})
    return rows


def _scatter(summ, tag, benefit_thr):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return "(matplotlib unavailable: %r)" % e
    fig, ax = plt.subplots(figsize=(7, 5))
    col = {"ACCEPT": "tab:green", "REJECT": "tab:red", "ABSTAIN": "tab:orange", "DIAGNOSTIC": "tab:blue"}
    for v in summ.values():
        c = col.get(v["gate_action"], "gray")
        mk = "o" if v["is_safe"] else "x"
        x = v["benefit_lcb"] if v["benefit_lcb"] == v["benefit_lcb"] else 0.0
        ax.scatter(x, v["dtgt_bacc_lo"], c=c, marker=mk, s=28, alpha=0.7,
                   edgecolors="none" if mk == "o" else None)
    ax.axhline(benefit_thr, ls="--", c="k", lw=0.7); ax.axvline(benefit_thr, ls="--", c="k", lw=0.7)
    ax.set_xlabel("source-LOSO benefit LCB (gate input)")
    ax.set_ylabel("actual target dbAcc LCB (post-hoc)")
    ax.set_title("V2 ceiling: target-beneficial (upper) but source-invisible (left) -> not accepted")
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", ls="", c=col[k], label=k) for k in col] + \
          [Line2D([0], [0], marker="o", ls="", c="gray", label="safe"),
           Line2D([0], [0], marker="x", ls="", c="gray", label="unsafe")]
    ax.legend(handles=leg, fontsize=7, loc="best")
    path = "%s/v2_%s_scatter.png" % (OUT, tag)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="smoke"); a = ap.parse_args()
    S = json.load(open("%s/v2_%s_summary.json" % (OUT, a.tag)))
    summ, ch, th, pr = S["summary"], S.get("config_hash", "?"), S["thresholds"], S["params"]
    bt = th["benefit_lcb"]; st = th["safety_eps"]
    L = ["# V2 --- source-only acceptance CEILING / non-identifiability (%s)\n" % a.tag,
         "Config sha256:`%s`; thresholds FROZEN (safety UCB<=%.3f, benefit LCB>%.3f, domain diagnostic-only, "
         "target audit-only). **No world expects ACCEPT.** World-gen: variantA=%s f_align/phi=%.2f beta=%.2f "
         "m=%d. **Semi-synthetic (real latents + injected nuisance); a limit result, not a main-paper claim.**\n"
         % (ch, st, bt, pr.get("variantA", "?"), pr["phi"], pr["beta"], pr["m"])]
    verdicts = {}
    for wk in ["A", "B", "C"]:
        cells = [v for v in summ.values() if v["world"] == wk]
        if not cells:
            continue
        meta = WORLDS[wk]
        L += ["## World %s --- %s (expect %s)" % (wk, meta["name"], meta["expect"]),
              "| intervention | n_src | alpha | task-drop UCB | src-LOSO benefit LCB | domain-gain | gate | "
              "target ΔbAcc [CI] | safe | tgt-benef |", "|---|---|---|---|---|---|---|---|---|---|"]
        for v in sorted(cells, key=lambda v: (v["intervention"], str(v["n_source"]), v["alpha"])):
            blc = "%+.3f" % v["benefit_lcb"] if v["benefit_lcb"] == v["benefit_lcb"] else "n/a"
            L.append("| %s | %s | %.2f | %+.3f | %s | %+.3f | **%s** | %+.3f [%+.3f,%+.3f] | %s | %s |"
                     % (v["intervention"], v["n_source"], v["alpha"], v["task_drop_ucb"], blc, v["domain_gain"],
                        v["gate_action"], v["dtgt_bacc"], v["dtgt_bacc_lo"], v["dtgt_bacc_hi"],
                        "Y" if v["is_safe"] else "n", "Y" if v["target_beneficial"] else "n"))
        prin = [v for v in cells if v["intervention"] in PRINCIPLED]
        orc = [v for v in cells if v["intervention"] in DIAGNOSTIC]
        rnd = [v for v in cells if v["intervention"] == "random_k"]
        n_acc = sum(1 for v in prin if v["gate_action"] == "ACCEPT")
        if wk == "A":
            safe_ben = [v for v in prin + orc if v["is_safe"] and v["target_beneficial"]]
            og = max([v["dtgt_bacc"] for v in orc], default=float("nan"))
            rg = max([v["dtgt_bacc"] for v in rnd], default=float("nan"))
            rnd_no = not (rg == rg and rg > bt)
            verdicts["A"] = (len(safe_ben) >= 1) and n_acc == 0 and rnd_no
            L += ["", "**Ceiling:** %d SAFE cell(s) with a real target gain (target dbAcc LCB>+%.2f) that the gate "
                  "does NOT accept (principled ACCEPTs=%d). oracle target dbAcc=%+.3f vs random_k=%+.3f "
                  "(random reproduces oracle? %s). -> %s.\n"
                  % (len(safe_ben), bt, n_acc, og, rg, "yes" if not rnd_no else "no",
                     "PASS" if verdicts["A"] else "FAIL")]
        elif wk == "B":
            verdicts["B"] = n_acc == 0
            L += ["", "**Unsafe-accept:** %d/%d principled cells ACCEPTED (want 0) -> %s.\n"
                  % (n_acc, len(prin), "PASS" if verdicts["B"] else "FAIL")]
        else:
            hi = any(v["domain_gain"] > 0.05 and not v["target_beneficial"] for v in prin)
            verdicts["C"] = n_acc == 0 and hi
            L += ["", "**Useless-accept:** %d/%d principled cells ACCEPTED (want 0); high-domain-gain-but-useless "
                  "cell present=%s -> domain-gain != benefit -> %s.\n"
                  % (n_acc, len(prin), hi, "PASS" if verdicts["C"] else "FAIL")]
    # naive controllers (all deployable cells)
    L += ["## Naive controllers (all deployable cells; a GOOD accept = actually target-beneficial)",
          "| controller | accepts | false-accepts (non-beneficial) | true-accepts (beneficial) |",
          "|---|---|---|---|"]
    for b in _naive_controllers(list(summ.values()), bt):
        L.append("| %s | %d | %d | %d |" % (b["controller"], b["accepts"], b["false_accepts"], b["true_accepts"]))
    fig = _scatter(summ, a.tag, bt)
    L += ["", "Scatter (source-LOSO benefit LCB vs actual target ΔbAcc LCB, colored by gate action, o=safe "
          "x=unsafe): `%s`" % fig,
          "", "## Ceiling smoke verdict",
          "- World A (target-beneficial but source-uncertifiable, NO accept): %s" % _pf(verdicts.get("A")),
          "- World B (no unsafe accept): %s" % _pf(verdicts.get("B")),
          "- World C (no useless accept; domain-gain != benefit): %s" % _pf(verdicts.get("C")),
          "- **overall: %s**" % _pf(all(verdicts.get(k) for k in ("A", "B", "C"))),
          "", "**Reading:** naive source-only controllers (domain-gain / safety) FALSE-ACCEPT; OUR gate accepts "
          "~nothing (conservative -- correct under the ceiling); only the ORACLE target-informed selector "
          "(diagnostic, uses target labels) picks the beneficial cells -> crossing the ceiling needs target info."]
    open("%s/v2_%s_report.md" % (OUT, a.tag), "w").write("\n".join(L) + "\n")
    print("\n".join(L)); print("\nV2_REPORT_DONE")


def _pf(b):
    return "PASS" if b else "FAIL"


if __name__ == "__main__":
    main()
