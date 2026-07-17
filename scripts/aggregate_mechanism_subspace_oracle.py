#!/usr/bin/env python
"""Aggregate the Mechanism-Subspace Oracle rows (M0.2) into per-dataset x family tri-state verdicts.

Reads results/cmi_trace_mechanism_subspace/mechanism_oracle_rows_{tag}.jsonl (written by
run_mechanism_subspace_oracle.py) and computes, per (dataset, backbone, family):
  - dU_specific = dU_source_safe - mean(MATCHED random)     [PRIMARY specificity control, P0.5]
    matched-random is used ONLY when the shared-overlap match passed fail-closed; if it FAILED, the cell falls
    back to ambient and is flagged specificity_control="AMBIENT_ONLY" (NEVER silently called task-matched).
  - q95_exceedance = 1[dU_source_safe > q95(random)]        (subject-cluster rate vs the 0.05 null)
  - subject-cluster bootstrap (10000) -> LCB95/UCB95 on the per-subject mean dU_specific.
  - one-sided Holm p across the CONFIRMATORY family (contrast_disagreement / EEGNet / both datasets).
Routing via mechanism_subspace.route_stage_result: confirmatory route A needs Holm p<0.05 + same-dataset LCB>0
+ other-dataset UCB>-0.01; secondary -> BACKBONE_SPECIFIC / READOUT_OR_ESTIMATOR_DEPENDENT / negative-ref anomaly.
NO CLOSED verdict is ever written; every non-enriched cell carries a failure record. Only the project owner may
stop a scientific line. Manuscript FROZEN.

  python scripts/aggregate_mechanism_subspace_oracle.py            # full
  python scripts/aggregate_mechanism_subspace_oracle.py --smoke    # engineering smoke (NO scientific weight)
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(REPO))
from tos_cmi.eval import mechanism_subspace as MS

OUT = REPO / "results" / "cmi_trace_mechanism_subspace"
DATASETS = ["BNCI2014_001", "BNCI2015_001"]
Q95_NULL_RATE = 0.05
N_BOOT = 10000


def _cluster_ci(per_subject_means, seed=7, n_boot=N_BOOT):
    v = np.asarray([x for x in per_subject_means if np.isfinite(x)], float)
    if not v.size:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), p_one_sided=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    b = np.array([v[rng.integers(0, v.size, v.size)].mean() for _ in range(n_boot)])
    # one-sided p for H1: mean>0  ->  fraction of bootstrap means at or below 0 (cluster-level, sign-based)
    p = float((np.sum(b <= 0.0) + 1) / (n_boot + 1))
    return dict(mean=float(v.mean()), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)),
                p_one_sided=p, n=int(v.size))


def _holm(pvals):
    """Holm step-down adjusted p-values (order preserved)."""
    idx = np.argsort(pvals); m = len(pvals); adj = [None] * m; run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(1.0, run)
    return adj


def _cell_specific(row):
    """Per-cell dU_specific + which control was used, honouring fail-closed matched control."""
    inf = row.get("dU_source_safe")
    matched = row.get("dU_random_matched"); ambient = row.get("dU_random_ambient")
    if matched:                                            # matched control passed fail-closed
        ctrl, ref = "MATCHED", np.asarray(matched, float)
    elif ambient:                                          # matched failed/absent -> ambient, FLAGGED
        ctrl, ref = "AMBIENT_ONLY", np.asarray(ambient, float)
    else:
        return None
    if inf is None or not np.isfinite(inf):
        return None
    q95 = float(np.quantile(ref, 0.95))
    return dict(dU_specific=float(inf - ref.mean()), beats_q95=bool(inf > q95), control=ctrl,
                match_verdict=(row.get("shared_overlap_match") or {}).get("verdict"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    tag = "smoke" if a.smoke else "full"
    src = OUT / f"mechanism_oracle_rows_{tag}.jsonl"
    rows = [json.loads(l) for l in open(src)]
    prov = dict(rows_file=str(src.relative_to(REPO)), rows_sha256=hashlib.sha256(open(src, "rb").read()).hexdigest()[:16],
                n_rows=len(rows), git_sha=next((r.get("git_sha") for r in rows if r.get("git_sha")), "unknown"),
                config_hash=next((r.get("config_hash") for r in rows if r.get("config_hash")), "unknown"))

    # skip / firewall audit (reason-coded losses are surfaced, never dropped silently)
    skipped = [dict(dataset=r.get("dataset"), backbone=r.get("backbone"), subject=r.get("subject"),
                    family=r.get("family"), reason=r.get("reason") or r.get("fail")) for r in rows if r.get("status") == "skipped"]

    per_cell = defaultdict(lambda: defaultdict(list))       # (bb,fam,ds) -> subject -> [dU_specific]
    win = defaultdict(lambda: defaultdict(list)); ctrl_used = defaultdict(set)
    for r in rows:
        if r.get("status") != "ok":
            continue
        cs = _cell_specific(r)
        if cs is None:
            continue
        key = (r["backbone"], r["family"], r["dataset"])
        per_cell[key][r["subject"]].append(cs["dU_specific"])
        win[key][r["subject"]].append(1.0 if cs["beats_q95"] else 0.0)
        ctrl_used[key].add(cs["control"])

    # cluster CIs per (bb, fam, ds)
    cell_ci = {}
    for key, by in per_cell.items():
        spec = _cluster_ci([np.mean(v) for v in by.values()])
        wr = _cluster_ci([np.mean(v) for v in win[key].values()])
        cell_ci[key] = dict(spec=spec, q95_exceedance=wr["mean"], q95_exceedance_lo=wr["lo"],
                            q95_exceedance_hi=wr["hi"], control=sorted(ctrl_used[key]))

    # Holm across the CONFIRMATORY family: contrast_disagreement / EEGNet / both datasets
    conf_keys = [("EEGNet", "contrast_disagreement", ds) for ds in DATASETS if ("EEGNet", "contrast_disagreement", ds) in cell_ci]
    conf_p = [cell_ci[k]["spec"]["p_one_sided"] for k in conf_keys]
    conf_adj = _holm(conf_p) if conf_p else []
    holm_by_ds = {conf_keys[i][2]: conf_adj[i] for i in range(len(conf_keys))}

    summ = []
    for key in sorted(cell_ci):
        bb, fam, ds = key; c = cell_ci[key]; spec = c["spec"]
        holm_p = holm_by_ds.get(ds, 1.0) if (bb == "EEGNet" and fam == "contrast_disagreement") else spec["p_one_sided"]
        other = [cell_ci[k]["spec"]["hi"] for k in cell_ci if k[0] == bb and k[1] == fam and k[2] != ds]
        other_ucb = min(other) if other else -1.0        # if the other dataset is missing, cannot clear route A
        route = MS.route_stage_result(spec["lo"], spec["hi"], holm_p, other_ucb, fam, bb)
        significance_state = ("SIGNIFICANT_ENRICHMENT" if holm_p < 0.05 and spec["lo"] > 0
                              else "NO_DETECTED_ENRICHMENT" if spec["hi"] <= 0 else "INCONCLUSIVE")
        interval_state = ("EXCLUDES_ZERO_POSITIVE" if spec["lo"] > 0 else
                          "EXCLUDES_POSITIVE" if spec["hi"] <= 0 else "STRADDLES_ZERO")
        practical_state = ("PRACTICAL_ENRICHMENT" if spec["mean"] >= 0.02 and spec["lo"] > 0 else
                           "NO_PRACTICAL_ENRICHMENT" if spec["hi"] < 0.02 else "PRACTICAL_ENRICHMENT_NOT_RULED_OUT")
        summ.append(dict(backbone=bb, family=fam, dataset=ds, n_subjects=spec["n"],
                         dU_specific_mean=spec["mean"], dU_specific_lcb=spec["lo"], dU_specific_ucb=spec["hi"],
                         holm_p=holm_p, one_sided_p=spec["p_one_sided"], other_dataset_ucb=other_ucb,
                         q95_exceedance_rate=c["q95_exceedance"], q95_null_rate=Q95_NULL_RATE,
                         specificity_control=c["control"], verdict=route["verdict"], next_step=route.get("next"),
                         significance_state=significance_state, interval_state=interval_state, practical_state=practical_state,
                         **{k: route[k] for k in ("failure_layer", "learned_lesson", "next_hypothesis", "next_experiment") if k in route}))

    # confirmatory decision (both datasets must be present + clear route A)
    conf_rows = [s for s in summ if s["backbone"] == "EEGNet" and s["family"] == "contrast_disagreement"]
    conf_ok_datasets = [s for s in conf_rows if s["verdict"] == "MECHANISM_ENRICHED_OVER_RANDOM"]
    confirmatory = dict(
        family="contrast_disagreement/EEGNet", datasets_present=sorted(s["dataset"] for s in conf_rows),
        route_A_datasets=sorted(s["dataset"] for s in conf_ok_datasets),
        decision=("MECHANISM_ENRICHED_OVER_RANDOM" if conf_ok_datasets else
                  "NO_DETECTED_MECHANISM_ENRICHMENT" if conf_rows and all(s["interval_state"] == "EXCLUDES_POSITIVE" for s in conf_rows) else
                  "INCONCLUSIVE"),
        note="secondary hits (DGCNN/rule/grad) do NOT unlock M2 without an independent confirmatory rerun (P0.6)")

    import csv
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(summ[0].keys()) if summ else []
    with open(OUT / f"mechanism_oracle_summary_{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow({k: r.get(k) for k in keys}) for r in summ]
    json.dump(dict(provenance=prov, confirmatory=confirmatory, per_cell=summ, skipped=skipped, n_boot=N_BOOT,
                   scope=("engineering_smoke_no_scientific_weight" if a.smoke else "M1_confirmatory"),
                   discipline="no CLOSED; graded verdicts; only the project owner stops a scientific line; manuscript FROZEN"),
              open(OUT / f"mechanism_oracle_verdict_{tag}.json", "w"), indent=2, default=float)

    print(f"[mech-agg-{tag}] {len(summ)} cells; skipped={len(skipped)}; confirmatory={confirmatory['decision']} "
          f"(route_A={confirmatory['route_A_datasets']})")
    for s in summ:
        print(f"  {s['dataset'][:11]}/{s['backbone']:6}/{s['family'][:9]}: dU_spec={s['dU_specific_mean']:+.4f}"
              f"[{s['dU_specific_lcb']:+.4f},{s['dU_specific_ucb']:+.4f}] holm={s['holm_p']:.3f} "
              f"ctrl={'/'.join(s['specificity_control'])} q95ex={s['q95_exceedance_rate']:.2f} -> {s['verdict']}")
    if skipped:
        print(f"  [skipped/reason-coded] {len(skipped)}: " + ", ".join(sorted({s['reason'] for s in skipped if s['reason']})))


if __name__ == "__main__":
    main()
