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
    """deployable-intervention ACCEPTs in this env that are actually target-beneficial + count false-accepts."""
    dep = [v for v in cells if v["intervention"] not in ("oracle_nuisance_DIAGNOSTIC",)]
    acc = [v for v in dep if v["gate_action"] == "ACCEPT"]
    good = [v for v in acc if v["dtgt_bacc_lo"] > bt]
    false = [v for v in acc if not (v["is_safe"] and v["dtgt_bacc_lo"] > bt)]
    return acc, good, false


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
    # per-environment summary
    L += ["| env | deployable? | ACCEPTs | target-good accepts | false accepts | best benefit LCB (interv) | best target ΔbAcc [CI] |",
          "|---|---|---|---|---|---|---|"]
    per_env = {}
    for env in ["oracle", "subject", "covariance_cluster", "margin_cluster", "augmentation_shift", "random"]:
        cells = [v for v in summ.values() if v["env"] == env]
        if not cells:
            continue
        acc, good, false = _env_accepts(cells, bt)
        dep = [v for v in cells if v["intervention"] != "oracle_nuisance_DIAGNOSTIC"]
        best = max(dep, key=lambda v: v["benefit_lcb"] if v["benefit_lcb"] == v["benefit_lcb"] else -9, default=None)
        per_env[env] = dict(acc=len(acc), good=len(good), false=len(false))
        L.append("| %s | %s | %d | %d | %d | %s (%s) | %+.3f [%+.3f,%+.3f] |"
                 % (env, env in DEPLOYABLE_ENVS, len(acc), len(good), len(false),
                    ("%+.3f" % best["benefit_lcb"]) if best else "n/a", best["intervention"] if best else "-",
                    best["dtgt_bacc"] if best else 0, best["dtgt_bacc_lo"] if best else 0, best["dtgt_bacc_hi"] if best else 0))
    # Case verdict
    orc = per_env.get("oracle", {"good": 0, "false": 0})
    disc_good = sum(per_env.get(e, {}).get("good", 0) for e in DISCOVERED)
    disc_false = sum(per_env.get(e, {}).get("false", 0) for e in DISCOVERED)
    rnd_good = per_env.get("random", {}).get("good", 0)
    if orc["good"] < 1:
        case = "C (E_oracle cannot accept a target-beneficial intervention -> source-rich WORLD construction failed; redesign world-gen, thresholds frozen)"
    elif disc_false > 0:
        case = "D (a discovered source-only environment FALSE-ACCEPTs -> environment discovery overfits; audit, do not proceed to real EEG)"
    elif disc_good >= 1 and disc_good > rnd_good:
        case = "A (E_oracle accepts AND a discovered source-only environment recovers it, beating random -> strongest source-only positive)"
    else:
        case = "B (E_oracle accepts [Prop 2 validated with known environments] but discovered source-only environments do NOT recover it -> discovery unsolved; motivates Fork 1)"
    fig = _scatter(summ, a.tag, bt)
    L += ["", "## Verdict",
          "- E_oracle target-good accepts: %d ; false accepts: %d" % (orc["good"], orc.get("false", 0)),
          "- discovered (E2/E4/E5) target-good accepts: %d ; false accepts: %d ; random target-good: %d"
          % (disc_good, disc_false, rnd_good),
          "- E0 subject baseline should MISS: subject target-good accepts = %d (want 0 or << oracle)"
          % per_env.get("subject", {}).get("good", 0),
          "- **Case %s**" % case,
          "", "Scatter (source-LOEO benefit LCB vs target ΔbAcc LCB, colored by environment): `%s`" % fig,
          "", "Proposition 2 empirically supported? %s"
          % ("YES (E_oracle accepts a target-beneficial intervention safely)" if orc["good"] >= 1 and orc.get("false", 0) == 0
             else "NO (E_oracle did not safely accept a target-beneficial intervention)")]
    open("%s/source_rich_%s_report.md" % (OUT, a.tag), "w").write("\n".join(L) + "\n")
    print("\n".join(L)); print("\nSOURCE_RICH_REPORT_DONE")


if __name__ == "__main__":
    main()
