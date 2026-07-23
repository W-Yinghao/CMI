#!/usr/bin/env python
"""Task 1 re-aggregation: apply the NEW three-condition gate (target benefit + UCB source-safety +
specificity) and the three-state verdict to the ALREADY-BANKED deployment per-fold CSVs, with NO retraining.

Reuses tos_cmi.eeg.erasure_target_deploy.aggregate() verbatim (per dataset), so the verdict logic is identical
to what future runs produce. Writes NEW artifacts (verdicts_3state.{csv,json}); committed summaries untouched.

  python scripts/reaggregate_deploy_verdicts.py
"""
from __future__ import annotations
import csv, glob, json, os, re
from collections import defaultdict

import tos_cmi.eeg.erasure_target_deploy as etd

ROOT = "tos_cmi/results/tos_cmi_eeg_frozen/erasure_target_deploy"
NUM = {"tgt_bacc", "tgt_nll", "src_bacc", "src_nll", "subj_dec_after", "chance_task"}
SRC_ERASERS = ["TOS_VD", "LEACE", "RLACE", "INLP"]   # source-subject-fitted; random_k is the control


def _load_rows():
    """Load every per-fold CSV, tagged with its dataset (parsed from the filename)."""
    by_ds = defaultdict(list)
    for p in glob.glob(f"{ROOT}/**/*_seed*.csv", recursive=True):
        name = os.path.basename(p)
        m = re.match(r"(.+)_(EEGNet|TSMNet)_seed(\d+)\.csv$", name)
        if not m:
            continue
        ds = m.group(1)
        with open(p) as fh:
            for r in csv.DictReader(fh):
                row = {k: (float(v) if k in NUM else int(v) if k in ("seed",) else v) for k, v in r.items()}
                by_ds[ds].append(row)
    return by_ds


def main():
    by_ds = _load_rows()
    rows_out, cells = [], []
    for ds in sorted(by_ds):
        etd.DATASET = ds
        etd.DIRS = {"TSMNet": "x", "EEGNet": "y"}          # aggregate() iterates keys; empty backbones skipped
        summary, paired = etd.aggregate(by_ds[ds])
        for (bb, nm), rec in summary.items():
            if nm == "full" or "verdict_3state" not in rec:
                continue
            rows_out.append({"dataset": ds, "backbone": bb, "method": nm, "n": rec["n"],
                             "tgt_delta": round(rec["dtgt_bacc"], 4), "tgt_lo": round(rec["dtgt_bacc_lo"], 4),
                             "tgt_ucb": round(rec["dtgt_bacc_hi"], 4),
                             "src_drop": round(rec["dsrc_bacc_drop"], 4),
                             "src_drop_ucb": round(rec["dsrc_bacc_drop_ucb"], 4),
                             "spec_lo": round(rec["spec_bacc_lo"], 4),
                             "src_safe_ucb02": rec["src_safe_ucb02"], "specific": rec["specific_vs_random"],
                             "verdict_3state": rec["verdict_3state"], "beneficial_3cond": rec["beneficial_3cond"]})
        # per-cell rollup over source-fitted erasers
        for bb in ("EEGNet", "TSMNet"):
            recs = {nm: summary.get((bb, nm)) for nm in SRC_ERASERS if (bb, nm) in summary}
            if not recs:
                continue
            any_conf = any(r["verdict_3state"] == "confirmed_benefit" for r in recs.values())
            any_gate = any(r["beneficial_3cond"] for r in recs.values())
            cells.append({"dataset": ds, "backbone": bb, "n_source_erasers": len(recs),
                          "any_confirmed_benefit": any_conf, "any_clears_3cond_gate": any_gate,
                          "max_src_drop_ucb": round(max(r["dsrc_bacc_drop_ucb"] for r in recs.values()), 4),
                          "per_eraser_verdict": {nm: r["verdict_3state"] for nm, r in recs.items()}})

    # paper-facing aggregate stats
    from collections import Counter
    vc = Counter(r["verdict_3state"] for r in rows_out)
    stats = {"n_cells": len(cells), "n_source_eraser_rows": len(rows_out),
             "verdict_counts_over_eraser_rows": dict(vc),
             "cells_with_any_confirmed_benefit": sum(c["any_confirmed_benefit"] for c in cells),
             "cells_clearing_full_3cond_gate": sum(c["any_clears_3cond_gate"] for c in cells),
             "max_src_drop_ucb_over_all_rows": round(max((r["src_drop_ucb"] for r in rows_out), default=float("nan")), 4),
             "paper_claim_no_source_eraser_clears_+0.01":
                 all(r["verdict_3state"] != "confirmed_benefit" for r in rows_out)}
    out = {"rows": rows_out, "cells": cells, "stats": stats}
    os.makedirs(ROOT, exist_ok=True)
    json.dump(out, open(f"{ROOT}/verdicts_3state.json", "w"), indent=1)
    with open(f"{ROOT}/verdicts_3state.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys())); w.writeheader()
        for r in rows_out:
            w.writerow(r)
    # console report
    print(f"{'dataset':<18}{'bb':<7}{'method':<8}{'tgt_delta [lo,ucb]':<26}{'src_drop_ucb':<13}{'spec_lo':<9}{'verdict':<20}{'3cond'}")
    for r in rows_out:
        print(f"{r['dataset']:<18}{r['backbone']:<7}{r['method']:<8}"
              f"{r['tgt_delta']:+.3f} [{r['tgt_lo']:+.3f},{r['tgt_ucb']:+.3f}]   "
              f"{r['src_drop_ucb']:<13.3f}{r['spec_lo']:<+9.3f}{r['verdict_3state']:<20}{r['beneficial_3cond']}")
    print("\nSTATS:", json.dumps(stats, indent=1))
    print(f"\n-> {ROOT}/verdicts_3state.{{csv,json}}")


if __name__ == "__main__":
    main()
