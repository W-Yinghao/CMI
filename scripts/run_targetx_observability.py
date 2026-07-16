#!/usr/bin/env python
"""Target-X observability audit runner (Fork 2). A6 preflight session manifest + the audit. Default --smoke
runs the PM smoke gate (1 subject/dataset, seed 0, cond, G1 only, identity/singleton/rank<=2 + firewall trace).
NO adaptation. Full audit (--full) only after the smoke is approved.

  python scripts/run_targetx_observability.py --smoke
  python scripts/run_targetx_observability.py --manifest-only
"""
from __future__ import annotations
import argparse, csv, glob, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.eeg.relaxation_ladder import feat_from_tos_dump
from tos_cmi.eval.targetx_observability import audit_fold, session_split

OUT = REPO / "results" / "cmi_trace_dg_identifiability"
DATASETS = ["BNCI2014_001", "BNCI2015_001"]


def _cells(ds, backbone, seeds):
    dd = REPO / "tos_cmi/results/tos_cmi_eeg_frozen" / f"{ds}_{backbone}_LOSO"
    return [p for p in sorted(glob.glob(str(dd / "sub*_erm_lam0_seed*.npz")))
            if any(p.endswith(f"_seed{s}.npz") for s in seeds)]


def manifest(backbone="EEGNet", seeds=("0",)):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in DATASETS:
        for cp in _cells(ds, backbone, seeds):
            f = feat_from_tos_dump(cp)
            if "session_target" not in f:
                rows.append(dict(dataset=ds, subject=f["heldout_subject"], exclusion_reason="no_session_metadata"))
                continue
            yt = np.asarray(f["y_target"]).astype(int)
            cal, qry, info = session_split(f["session_target"], yt)
            excl = "" if (cal.sum() >= 8 and qry.sum() >= 8) else "insufficient_cal_or_query_trials"
            rows.append(dict(dataset=ds, subject=f["heldout_subject"],
                             cal_sessions="|".join(map(str, info["cal_sessions"])),
                             query_sessions="|".join(map(str, info["query_sessions"])),
                             n_cal=info["n_cal"], n_query=info["n_query"],
                             class_counts_cal=dict(Counter(yt[cal].tolist())),
                             class_counts_query=dict(Counter(yt[qry].tolist())),
                             fallback_used=info["fallback_used"], exclusion_reason=excl))
    fp = OUT / "session_split_manifest.csv"
    keys = ["dataset", "subject", "cal_sessions", "query_sessions", "n_cal", "n_query",
            "class_counts_cal", "class_counts_query", "fallback_used", "exclusion_reason"]
    with open(fp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"[manifest] wrote {len(rows)} rows -> {fp}")
    for r in rows[:3] + rows[-3:]:
        print(f"   {r['dataset']} sub{r['subject']}: cal={r.get('cal_sessions')}({r.get('n_cal')}) "
              f"query={r.get('query_sessions')}({r.get('n_query')}) fallback={r.get('fallback_used')} excl='{r.get('exclusion_reason')}'")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--full", action="store_true")
    ap.add_argument("--manifest-only", action="store_true"); ap.add_argument("--backbone", default="EEGNet")
    ap.add_argument("--seeds", nargs="+", default=["0"])
    a = ap.parse_args()
    manifest(a.backbone, tuple(a.seeds))
    if a.manifest_only:
        return
    smoke = not a.full
    tag = "smoke" if smoke else "full"
    n_smoke_subj = 2                                             # F2.0d engineering smoke: 2 subjects / dataset
    fold_rows, action_rows, completeness = [], [], []
    for ds in DATASETS:
        cells = _cells(ds, a.backbone, a.seeds)
        if smoke:
            cells = cells[:n_smoke_subj]
        for cp in cells:
            f = feat_from_tos_dump(cp)
            res = audit_fold(f, seed=int(f["seed"]), family="cond", smoke=smoke,
                             n_random_per_rank=(8 if smoke else 50))
            ok = res is not None
            completeness.append(dict(dataset=ds, subject=f["heldout_subject"], seed=int(f["seed"]),
                                     status=("ok" if ok else "excluded"), reason=("" if ok else "empty_basis")))
            if not ok:
                continue
            slim = {k: res[k] for k in ("dataset", "heldout_subject", "seed", "session_info", "firewall",
                                        "n_actions", "n_informed", "n_random", "selected_action", "selected_rank",
                                        "delta_tx_macro", "delta_tx_pooled", "delta_random_macro_mean", "selector_diag")}
            fold_rows.append(slim)
            for rw in res["rows"]:                               # PRESERVE all per-action rows (P0.4)
                action_rows.append(dict(dataset=ds, subject=res["heldout_subject"], seed=res["seed"],
                                        action=rw["action"], kind=rw["kind"], rank=rw["rank"],
                                        eligible=rw["eligible"], G1=rw["scores"].get("G1"),
                                        utility_macro=rw["utility_macro"], utility_pooled=rw["utility_pooled"]))
            print(f"  {ds} sub{res['heldout_subject']}: informed={res['n_informed']} random={res['n_random']} "
                  f"selected={res['selected_action']}(rank{res['selected_rank']}) "
                  f"Δmacro={res['delta_tx_macro']:+.3f} Δpooled={res['delta_tx_pooled']:+.3f} "
                  f"Δrand_macro={res['delta_random_macro_mean']:+.3f} "
                  f"| qx_sel={res['firewall']['query_x_used_for_selection']} "
                  f"tgtgreedy={res['firewall']['target_greedy_in_action_set']} "
                  f"rand_selectable={res['firewall']['random_controls_selectable']}")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"targetx_action_rows_{tag}.jsonl", "w") as fh:
        for r in action_rows:
            fh.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    with open(OUT / f"targetx_fold_summary_{tag}.jsonl", "w") as fh:
        for r in fold_rows:
            fh.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    with open(OUT / f"targetx_completeness_matrix_{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "subject", "seed", "status", "reason"]); w.writeheader()
        [w.writerow(r) for r in completeness]
    print(f"[targetx-{tag}] {len(fold_rows)} folds, {len(action_rows)} action rows -> {OUT}")


if __name__ == "__main__":
    main()
