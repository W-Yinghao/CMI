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
    out_rows = []
    for ds in DATASETS:
        cells = _cells(ds, a.backbone, a.seeds)
        if smoke:
            cells = cells[:1]                                   # 1 subject / dataset
        for cp in cells:
            f = feat_from_tos_dump(cp)
            res = audit_fold(f, seed=int(f["seed"]), family="cond", smoke=smoke)
            if res is None:
                continue
            # strip the bulky per-action rows for the summary jsonl (keep firewall + selection + a few scores)
            slim = {k: res[k] for k in ("dataset", "heldout_subject", "seed", "session_info", "firewall",
                                        "n_actions", "selected_action", "selected_S", "delta_tx", "delta_random_mean")}
            slim["g1_top3"] = sorted(([rw["action"], rw["scores"].get("G1"), rw["utility"]] for rw in res["rows"] if rw["S"]),
                                     key=lambda t: -(t[1] or -9))[:3]
            out_rows.append(slim)
            print(f"  {ds} sub{res['heldout_subject']}: n_actions={res['n_actions']} "
                  f"selected={res['selected_action']} Δ_TX={res['delta_tx']:+.3f} "
                  f"Δ_rand={res['delta_random_mean']:+.3f} fallback={res['session_info']['fallback_used']} "
                  f"| firewall query_x_in_selection={res['firewall']['query_x_used_for_selection']} "
                  f"target_greedy_in_actions={res['firewall']['target_greedy_in_action_set']}")
    tag = "smoke" if smoke else "full"
    fp = OUT / f"targetx_observability_{tag}.jsonl"
    with open(fp, "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    print(f"[targetx-{tag}] wrote {len(out_rows)} rows -> {fp}")


if __name__ == "__main__":
    main()
