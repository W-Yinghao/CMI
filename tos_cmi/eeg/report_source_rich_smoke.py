"""Fork 2 Phase 1A --- source-rich smoke report + Case A/B/C/D verdict. Reads source_rich_<tag>_summary.json.
E_oracle (regime) is the DIAGNOSTIC upper bound (Prop 2 with known environments); E0 subject is the baseline
that should MISS the benefit; E2/E4/E5 are source-only DISCOVERED environments; random is the p-hacking control.
  python -m tos_cmi.eeg.report_source_rich_smoke [--tag smoke]
"""
from __future__ import annotations
import argparse
import json
from tos_cmi.eeg.source_rich_worlds import DEPLOYABLE_ENVS, DIAGNOSTIC_ENVS

OUT = "tos_cmi/results/source_rich/smoke"
DISCOVERED = ["covariance_cluster", "margin_cluster", "augmentation_shift"]


def _env_accepts(cells, bt):
    """deployable ACCEPTs in this env, split into good (target LCB>bt), HARMFUL (target hi<-0.01), and
    BENIGN-BOUNDARY (safe, target mean>0 but LCB<=bt -- an eps_coverage over-prediction, NOT a harmful accept)."""
    dep = [v for v in cells if v["intervention"] not in ("oracle_nuisance_DIAGNOSTIC",)]
    acc = [v for v in dep if v["gate_action"] == "ACCEPT"]
    good = [v for v in acc if v["dtgt_bacc_lo"] > bt]
    harmful = [v for v in acc if v["dtgt_bacc_hi"] < -0.01]
    boundary = [v for v in acc if v not in good and v not in harmful and v["dtgt_bacc"] > 0]
    return acc, good, harmful, boundary


def _scatter(summ, tag, bt):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        return "(matplotlib unavailable: %r)" % e
    fig, ax = plt.subplots(figsize=(6, 5))
    col = {"subject": "tab:gray", "oracle": "tab:blue", "covariance_cluster": "tab:green",
           "margin_cluster": "tab:red", "augmentation_shift": "tab:purple", "random": "tab:orange"}
    for v in summ.values():
        if v["intervention"] == "oracle_nuisance_DIAGNOSTIC":
            continue
        ax.scatter(v["benefit_lcb"] if v["benefit_lcb"] == v["benefit_lcb"] else 0, v["dtgt_bacc_lo"],
                   c=col.get(v["env"], "k"), s=26, alpha=0.7)
    ax.axhline(bt, ls="--", c="k", lw=0.6); ax.axvline(bt, ls="--", c="k", lw=0.6)
    ax.set_xlabel("source-LOEO benefit LCB (gate input)"); ax.set_ylabel("actual target ΔbAcc LCB")
    ax.set_title("source-rich: does the environment make benefit source-visible?")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", c=c, label=e) for e, c in col.items()], fontsize=7)
    p = "%s/source_rich_%s_scatter.png" % (OUT, tag); fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="smoke"); a = ap.parse_args()
    S = json.load(open("%s/source_rich_%s_summary.json" % (OUT, a.tag)))
    summ, bt = S["summary"], S["thresholds"]["benefit_lcb"]
    L = ["# Fork 2 Phase 1A --- source-rich smoke (Proposition 2 constructive test)\n",
         "Thresholds FROZEN (safety<=%.3f, benefit>%.3f). E_oracle=regime (DIAGNOSTIC known-environment upper "
         "bound); E0 subject should MISS; E2/E4/E5 discovered (source-only); random = control. "
         "params %s.\n" % (S["thresholds"]["safety_eps"], bt, S["params"])]
    # per-environment summary (harmful false-accepts are the real failure; benign boundary = eps_coverage)
    L += ["| env | deployable? | ACCEPTs | target-good | HARMFUL accepts | benign-boundary | best benefit LCB (interv) | best target ΔbAcc [CI] |",
          "|---|---|---|---|---|---|---|---|"]
    per_env = {}
    for env in ["oracle", "subject", "covariance_cluster", "margin_cluster", "augmentation_shift", "random"]:
        cells = [v for v in summ.values() if v["env"] == env]
        if not cells:
            continue
        acc, good, harmful, boundary = _env_accepts(cells, bt)
        dep = [v for v in cells if v["intervention"] != "oracle_nuisance_DIAGNOSTIC"]
        best = max(dep, key=lambda v: v["benefit_lcb"] if v["benefit_lcb"] == v["benefit_lcb"] else -9, default=None)
        per_env[env] = dict(acc=len(acc), good=len(good), harmful=len(harmful), boundary=len(boundary))
        L.append("| %s | %s | %d | %d | %d | %d | %s (%s) | %+.3f [%+.3f,%+.3f] |"
                 % (env, env in DEPLOYABLE_ENVS, len(acc), len(good), len(harmful), len(boundary),
                    ("%+.3f" % best["benefit_lcb"]) if best else "n/a", best["intervention"] if best else "-",
                    best["dtgt_bacc"] if best else 0, best["dtgt_bacc_lo"] if best else 0, best["dtgt_bacc_hi"] if best else 0))
    # Case verdict (HARMFUL false-accepts are the failure signal; benign boundary is eps_coverage, not failure)
    orc = per_env.get("oracle", {"good": 0, "harmful": 0, "boundary": 0})
    disc_good = sum(per_env.get(e, {}).get("good", 0) for e in DISCOVERED)
    disc_harm = sum(per_env.get(e, {}).get("harmful", 0) for e in DISCOVERED)
    rnd_good = per_env.get("random", {}).get("good", 0)
    if orc["good"] < 1:
        case = "C (E_oracle cannot safely accept a target-beneficial intervention -> source-rich WORLD construction failed; redesign world-gen, thresholds frozen)"
    elif disc_harm > 0:
        case = "D (a discovered source-only environment HARMFULLY false-accepts -> environment discovery overfits; audit, do not proceed to real EEG)"
    elif disc_good >= 1 and disc_good > rnd_good:
        case = "A (E_oracle accepts AND a discovered source-only environment recovers it, beating random -> strongest source-only positive)"
    else:
        case = "B (E_oracle accepts [Prop 2 validated with known environments] but discovered source-only environments do NOT recover it -> discovery unsolved; motivates Fork 1)"
    prop2 = orc["good"] >= 1 and orc.get("harmful", 0) == 0
    fig = _scatter(summ, a.tag, bt)
    L += ["", "## Verdict",
          "- E_oracle: target-good accepts %d ; HARMFUL accepts %d ; benign-boundary %d (eps_coverage over-prediction, safe+target-positive)"
          % (orc["good"], orc.get("harmful", 0), orc.get("boundary", 0)),
          "- discovered (E2/E4/E5): target-good accepts %d ; HARMFUL accepts %d ; random target-good %d"
          % (disc_good, disc_harm, rnd_good),
          "- E0 subject baseline should MISS: subject target-good accepts = %d (want 0 or << oracle)"
          % per_env.get("subject", {}).get("good", 0),
          "- **Case %s**" % case,
          "", "Scatter (source-LOEO benefit LCB vs target ΔbAcc LCB, colored by environment): `%s`" % fig,
          "", "Proposition 2 empirically supported? %s"
          % ("YES -- E_oracle safely accepts >=1 target-beneficial intervention with 0 HARMFUL accepts (any "
             "benign-boundary accept is the eps_coverage slack at the strongest shift, not a violation)" if prop2
             else "NO -- E_oracle did not safely accept a target-beneficial intervention")]
    open("%s/source_rich_%s_report.md" % (OUT, a.tag), "w").write("\n".join(L) + "\n")
    print("\n".join(L)); print("\nSOURCE_RICH_REPORT_DONE")


if __name__ == "__main__":
    main()
