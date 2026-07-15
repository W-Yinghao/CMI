"""Phase-write step 0: result-index sanity check (NO plotting, NO compute).
Verifies every artifact path referenced by the figures exists, has the expected seed/fold/config counts
and key fields, and prints the JSON schema each figure will consume -- so we never plot from stale or
mis-shaped JSON. Run: python -m tos_cmi.paper.scripts.check_result_index
"""
from __future__ import annotations
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
R = os.path.join(ROOT, "tos_cmi", "results")
EEG = os.path.join(R, "tos_cmi_eeg_frozen")
TSM = os.path.join(EEG, "BNCI2014_001_TSMNet_LOSO")
EEGN = os.path.join(EEG, "BNCI2014_001_EEGNet_LOSO")
LPC = os.path.join(EEG, "lpc_collapse_curves")
OK, BAD = "  [ok]", "  [MISSING]"


def _exists(path, label):
    g = glob.glob(path)
    tag = OK if g else BAD
    print("%s %-44s %s (%d match)" % (tag, label, path.replace(ROOT + "/", ""), len(g)))
    return g


def _schema(path, label, list_expected=False):
    g = sorted(glob.glob(path))
    if not g:
        print("%s %-44s %s" % (BAD, label, path.replace(ROOT + "/", ""))); return None
    d = json.load(open(g[0]))
    if isinstance(d, list):
        keys = sorted(d[0].keys()) if d else []
        print("%s %-44s list[%d] keys=%s" % (OK, label, len(d), keys))
    else:
        print("%s %-44s dict keys=%s" % (OK, label, sorted(d.keys())))
        if "aggregate" in d and isinstance(d["aggregate"], dict):
            ek = d["aggregate"].get("erm:0")
            if ek:
                print("       aggregate['erm:0'] keys=%s" % sorted(ek.keys()))
    return d


def main():
    print("== FIG 3 (TSMNet collapse) — already rendered, relabel only ==")
    _exists(os.path.join(LPC, "TSMNet", "collapse_curves.png"), "fig3 source png")
    _exists(os.path.join(LPC, "TSMNet", "variant_compare.json"), "fig3 companion (variant_compare)")
    _exists(os.path.join(LPC, "TSMNet", "summary.json"), "fig3 collapse summary")

    print("\n== FIG 4 (TSMNet redundant leakage) ==")
    g = _exists(os.path.join(TSM, "ablation_report_seed*.json"), "TSMNet ablation (3 seeds)")
    print("       seeds found: %s" % sorted(p.split("seed")[-1][0] for p in g))
    _schema(os.path.join(TSM, "ablation_report_seed0.json"), "TSMNet ablation schema")

    print("\n== FIG 5 (EEGNet contrast) ==")
    g = _exists(os.path.join(EEGN, "ablation_report_seed*.json"), "EEGNet ablation (3 seeds)")
    print("       seeds found: %s" % sorted(p.split("seed")[-1][0] for p in g))
    _schema(os.path.join(EEGN, "ablation_report_seed0.json"), "EEGNet ablation schema")
    gl = _exists(os.path.join(LPC, "EEGNet", "raw_lpc_sub*_seed*.json"), "EEGNet LPC sweep (expect 108)")
    if gl:
        r = json.load(open(gl[0]))
        print("       LPC json keys=%s ; curves[-1] keys=%s" %
              (sorted(k for k in r if k != "curves"), sorted((r.get("curves") or [{}])[-1].keys())))
        lams = sorted(set(round(json.load(open(p))["lam"], 2) for p in gl))
        print("       lambdas=%s  n_files=%d (expect 27 cells x 4 lam = 108)" % (lams, len(gl)))

    print("\n== FIG 2 (synthetic certification line) ==")
    for f in ["cert_cells", "cert_table_cells", "frontier.json", "frontier_cells", "frontier_plugin",
              "frontier_deploy", "estimator_diag.json", "phase_diagram_powerfloor.json",
              "phase_diagram_smoke.json", "power_cells"]:
        p = os.path.join(R, f)
        if os.path.isdir(p):
            n = len(glob.glob(os.path.join(p, "*.json")))
            print("%s %-44s dir/ (%d json)" % (OK if n else BAD, f, n))
        elif os.path.exists(p):
            try:
                d = json.load(open(p)); k = sorted(d.keys()) if isinstance(d, dict) else "list[%d]" % len(d)
            except Exception as e:
                k = "(unreadable: %r)" % e
            print("%s %-44s %s" % (OK, f, k))
        else:
            print("%s %-44s (absent)" % (BAD, f))

    print("\n== TABLE 1 sources ==")
    _exists(os.path.join(ROOT, "tos_cmi", "notes", "PHASE2_REPORT.md"), "TSMNet narrative")
    _exists(os.path.join(ROOT, "tos_cmi", "notes", "PHASE3_BACKBONE_GENERALITY.md"), "EEGNet narrative")
    print("\nCHECK_RESULT_INDEX_DONE")


if __name__ == "__main__":
    main()
