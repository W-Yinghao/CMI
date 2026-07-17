#!/usr/bin/env python
"""Aggregate the Mechanism-Subspace Oracle rows (AMENDMENT 03, shared-null conditional estimand) into per-dataset
x family tri-state verdicts.

Reads results/cmi_trace_mechanism_subspace/mechanism_oracle_rows_{tag}.jsonl (run_mechanism_subspace_oracle.py) and
computes, per (dataset, backbone, family), the SYMMETRIC specificity contrasts (P0.1):
  - dU_safe_specific = dU_informed_safe - mean(SHARED_NULL_HAAR dU_safe)   [CONFIRMATORY]
  - dU_unc_specific  = dU_informed_unc  - mean(SHARED_NULL_HAAR dU_unc)    [reported alongside]
The PRIMARY control is SHARED_NULL_HAAR (conditional randomization inside the shared-null space); if it was
degenerate/absent (primary_control != SHARED_NULL_HAAR) the cell falls back to ambient and is flagged AMBIENT_ONLY
(never silently called the primary control). Statistics: subject-cluster 10k bootstrap CI + EXACT one-sided
sign-flip permutation p over subjects (2^9/2^12 enumerated) + Holm across the two confirmatory datasets; bootstrap
p is sensitivity only. Routing (mechanism_subspace.route_stage_result) grants route A only for
contrast/EEGNet/LCB>0/Holm-signflip-p<0.05/other-UCB>-0.01 AND specificity_control==SHARED_NULL_HAAR. NO CLOSED
verdict; reason-coded skips + LOW_DOF surfaced. Only the project owner stops a scientific line. Manuscript FROZEN.

  python scripts/aggregate_mechanism_subspace_oracle.py [--smoke]
"""
from __future__ import annotations
import argparse, csv, hashlib, json
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
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), boot_p=float("nan"), signflip_p=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    b = np.array([v[rng.integers(0, v.size, v.size)].mean() for _ in range(n_boot)])
    boot_p = float((np.sum(b <= 0.0) + 1) / (n_boot + 1))                  # sensitivity only
    return dict(mean=float(v.mean()), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)),
                boot_p=boot_p, signflip_p=MS.exact_sign_flip_p(v), n=int(v.size))


def _holm(pvals):
    idx = np.argsort(pvals); m = len(pvals); adj = [None] * m; run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(1.0, run)
    return adj


def _mean(recs, key):
    xs = [r[key] for r in recs if r.get(key) is not None and np.isfinite(r[key])]
    return float(np.mean(xs)) if xs else float("nan"), xs


def _cell_specific(row):
    """Per-cell symmetric specificity vs the PRIMARY (shared-null-Haar) control; fall back to ambient (flagged)."""
    inf_safe, inf_unc = row.get("dU_informed_safe"), row.get("dU_informed_unc")
    if inf_safe is None or not np.isfinite(inf_safe):
        return None
    nh = row.get("shared_null_haar"); amb = row.get("ambient")
    if row.get("primary_control") == "SHARED_NULL_HAAR" and nh:
        ctrl, recs = "SHARED_NULL_HAAR", nh
    elif amb:
        ctrl, recs = "AMBIENT_ONLY", amb
    else:
        return None
    m_safe, safe_xs = _mean(recs, "dU_safe"); m_unc, _ = _mean(recs, "dU_unc")
    q95 = float(np.quantile(safe_xs, 0.95)) if safe_xs else float("inf")
    return dict(dU_safe_specific=float(inf_safe - m_safe),
                dU_unc_specific=(float(inf_unc - m_unc) if inf_unc is not None and np.isfinite(inf_unc) and np.isfinite(m_unc) else float("nan")),
                beats_q95=bool(inf_safe > q95), control=ctrl,
                # DICTIONARY capture = full rank-r basis; SELECTED capture = the <=3 deleted projector (named apart)
                dictionary_gdis_capture=row.get("dictionary_gdis_capture"),
                selected_safe_gdis_capture=row.get("selected_safe_gdis_capture"),
                mean_random_dictionary_gdis_capture=_mean(recs, "dictionary_gdis_capture")[0],
                mean_random_selected_gdis_capture=_mean(recs, "selected_safe_gdis_capture")[0],
                mean_subspace_overlap=_mean(recs, "subspace_overlap")[0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--from-dir", default=None); ap.add_argument("--expect", type=int, default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    if a.from_dir:                                                 # ---- M1-P: aggregate per-cell files, refuse if incomplete ----
        d = Path(a.from_dir); done = sorted(d.glob("*.done")); cells = sorted(d.glob("cell_*.jsonl"))
        tag = a.tag or "m1p"
        if a.expect is not None and len(done) < a.expect:
            print(f"[mech-agg-{tag}] INCOMPLETE: {len(done)}/{a.expect} cells done -> REFUSING to aggregate (no partial-result threshold edits). "
                  f"missing {a.expect - len(done)}.")
            sys.exit(2)
        rows = [json.loads(l) for c in cells for l in open(c)]
        prov = dict(from_dir=str(d), n_done=len(done), n_cells=len(cells), n_rows=len(rows),
                    git_sha=next((r.get("git_sha") for r in rows if r.get("git_sha")), "unknown"),
                    config_hash=next((r.get("config_hash") for r in rows if r.get("config_hash")), "unknown"),
                    feature_hashes=sorted({r.get("feature_hash") for r in rows if r.get("feature_hash")}))
    else:
        tag = "smoke" if a.smoke else "full"
        src = OUT / f"mechanism_oracle_rows_{tag}.jsonl"
        rows = [json.loads(l) for l in open(src)]
        prov = dict(rows_file=str(src.relative_to(REPO)), rows_sha256=hashlib.sha256(open(src, "rb").read()).hexdigest()[:16],
                    n_rows=len(rows), git_sha=next((r.get("git_sha") for r in rows if r.get("git_sha")), "unknown"),
                    config_hash=next((r.get("config_hash") for r in rows if r.get("config_hash")), "unknown"))
    skipped = [dict(dataset=r.get("dataset"), backbone=r.get("backbone"), subject=r.get("subject"),
                    family=r.get("family"), reason=r.get("reason") or r.get("fail")) for r in rows if r.get("status") == "skipped"]
    unexpected = [dict(dataset=r.get("dataset"), backbone=r.get("backbone"), subject=r.get("subject"),
                       family=r.get("family"), status=r.get("status")) for r in rows if r.get("status") not in ("ok", "skipped")]

    per = defaultdict(lambda: defaultdict(list)); winr = defaultdict(lambda: defaultdict(list))
    unc = defaultdict(lambda: defaultdict(list)); ctrl_used = defaultdict(set); capdiag = defaultdict(list)
    for r in rows:
        if r.get("status") != "ok":
            continue
        cs = _cell_specific(r)
        if cs is None:
            continue
        key = (r["backbone"], r["family"], r["dataset"]); s = r["subject"]
        per[key][s].append(cs["dU_safe_specific"]); unc[key][s].append(cs["dU_unc_specific"])
        winr[key][s].append(1.0 if cs["beats_q95"] else 0.0); ctrl_used[key].add(cs["control"])
        capdiag[key].append((cs["dictionary_gdis_capture"], cs["mean_random_dictionary_gdis_capture"],
                             cs["selected_safe_gdis_capture"], cs["mean_random_selected_gdis_capture"], cs["mean_subspace_overlap"]))

    cell = {}
    for key, by in per.items():
        spec = _cluster_ci([np.mean(v) for v in by.values()])
        uncc = _cluster_ci([np.mean(v) for v in unc[key].values()])
        wr = _cluster_ci([np.mean(v) for v in winr[key].values()])
        caps = np.array([t for t in capdiag[key] if all(c is not None for c in t)], float)
        cm = (lambda i: float(caps[:, i].mean()) if caps.size else float("nan"))
        cell[key] = dict(spec=spec, unc=uncc, q95=wr["mean"], control=sorted(ctrl_used[key]),
                         cap_dict_informed=cm(0), cap_dict_random=cm(1),          # FULL rank-r dictionary capture
                         cap_sel_informed=cm(2), cap_sel_random=cm(3),            # SELECTED <=3 projector capture
                         subspace_overlap=cm(4))

    # Holm across the confirmatory family (contrast/EEGNet/both datasets) using the EXACT sign-flip p
    conf_keys = [("EEGNet", "contrast_disagreement", ds) for ds in DATASETS if ("EEGNet", "contrast_disagreement", ds) in cell]
    conf_adj = _holm([cell[k]["spec"]["signflip_p"] for k in conf_keys]) if conf_keys else []
    holm_by_ds = {conf_keys[i][2]: conf_adj[i] for i in range(len(conf_keys))}

    summ = []
    for key in sorted(cell):
        bb, fam, ds = key; c = cell[key]; spec = c["spec"]
        is_conf = (bb == "EEGNet" and fam == "contrast_disagreement")
        holm_p = holm_by_ds.get(ds, 1.0) if is_conf else spec["signflip_p"]
        other = [cell[k]["spec"]["hi"] for k in cell if k[0] == bb and k[1] == fam and k[2] != ds]
        other_ucb = min(other) if other else -1.0
        base = dict(backbone=bb, family=fam, dataset=ds, n_subjects=spec["n"],
                    dU_safe_specific_mean=spec["mean"], dU_safe_specific_lcb=spec["lo"], dU_safe_specific_ucb=spec["hi"],
                    dU_unc_specific_mean=c["unc"]["mean"], signflip_p=spec["signflip_p"], holm_p=holm_p,
                    boot_p_sensitivity=spec["boot_p"], other_dataset_ucb=other_ucb, q95_exceedance_rate=c["q95"],
                    q95_null_rate=Q95_NULL_RATE, specificity_control="/".join(c["control"]),
                    gdis_capture_dict_informed=c["cap_dict_informed"], gdis_capture_dict_random=c["cap_dict_random"],
                    gdis_capture_sel_informed=c["cap_sel_informed"], gdis_capture_sel_random=c["cap_sel_random"],
                    informed_random_subspace_overlap=c["subspace_overlap"])
        if spec["n"] < 2:
            route = dict(verdict="INSUFFICIENT_CLUSTERS", next="increase_subjects_or_seeds",
                         failure_layer="insufficient_cluster_units", learned_lesson="single-cluster bootstrap degenerate",
                         next_hypothesis="need >=2 subject clusters", next_experiment="run more subjects/seeds")
            tri = ("INSUFFICIENT_CLUSTERS",) * 3
        else:
            ctrl = "SHARED_NULL_HAAR" if c["control"] == ["SHARED_NULL_HAAR"] else "AMBIENT_ONLY"
            route = MS.route_stage_result(spec["lo"], spec["hi"], holm_p, other_ucb, fam, bb, specificity_control=ctrl)
            tri = (("SIGNIFICANT_ENRICHMENT" if holm_p < 0.05 and spec["lo"] > 0 else
                    "NO_DETECTED_ENRICHMENT" if spec["hi"] <= 0 else "INCONCLUSIVE"),
                   ("EXCLUDES_ZERO_POSITIVE" if spec["lo"] > 0 else "EXCLUDES_POSITIVE" if spec["hi"] <= 0 else "STRADDLES_ZERO"),
                   ("PRACTICAL_ENRICHMENT" if spec["mean"] >= 0.02 and spec["lo"] > 0 else
                    "NO_PRACTICAL_ENRICHMENT" if spec["hi"] < 0.02 else "PRACTICAL_ENRICHMENT_NOT_RULED_OUT"))
        base.update(verdict=route["verdict"], next_step=route.get("next"), significance_state=tri[0],
                    interval_state=tri[1], practical_state=tri[2],
                    **{k: route[k] for k in ("failure_layer", "learned_lesson", "next_hypothesis", "next_experiment") if k in route})
        summ.append(base)

    conf_rows = [s for s in summ if s["backbone"] == "EEGNet" and s["family"] == "contrast_disagreement"]
    conf_ok = [s for s in conf_rows if s["verdict"] == "MECHANISM_ENRICHED_OVER_RANDOM"]
    confirmatory = dict(family="contrast_disagreement/EEGNet", datasets_present=sorted(s["dataset"] for s in conf_rows),
                        route_A_datasets=sorted(s["dataset"] for s in conf_ok),
                        decision=("MECHANISM_ENRICHED_OVER_RANDOM" if conf_ok else
                                  "NO_DETECTED_MECHANISM_ENRICHMENT" if conf_rows and all(s["interval_state"] == "EXCLUDES_POSITIVE" for s in conf_rows) else
                                  "INCONCLUSIVE"),
                        note="secondary hits do NOT unlock M2 without an independent confirmatory rerun (P0.6); conf p = EXACT sign-flip")

    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(summ[0].keys()) if summ else []
    with open(OUT / f"mechanism_oracle_summary_{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow({k: r.get(k) for k in keys}) for r in summ]
    json.dump(dict(provenance=prov, confirmatory=confirmatory, per_cell=summ, skipped=skipped, unexpected=unexpected,
                   n_boot=N_BOOT, scope=("engineering_smoke_no_scientific_weight" if a.smoke else "M1_confirmatory"),
                   discipline="no CLOSED; graded verdicts; exact sign-flip confirmatory p; only the project owner stops a scientific line; manuscript FROZEN"),
              open(OUT / f"mechanism_oracle_verdict_{tag}.json", "w"), indent=2, default=float)

    print(f"[mech-agg-{tag}] {len(summ)} cells; skipped={len(skipped)}; unexpected={len(unexpected)}; "
          f"confirmatory={confirmatory['decision']} (route_A={confirmatory['route_A_datasets']})")
    for s in summ:
        print(f"  {s['dataset'][:11]}/{s['backbone']:6}/{s['family'][:9]}: dU_safe_spec={s['dU_safe_specific_mean']:+.4f}"
              f"[{s['dU_safe_specific_lcb']:+.4f},{s['dU_safe_specific_ucb']:+.4f}] signflip_p={s['signflip_p']:.3f} holm={s['holm_p']:.3f} "
              f"capDict={s['gdis_capture_dict_informed']:.2f}/{s['gdis_capture_dict_random']:.2f} "
              f"capSel={s['gdis_capture_sel_informed']:.2f}/{s['gdis_capture_sel_random']:.2f} ctrl={s['specificity_control']} -> {s['verdict']}")
    if unexpected:
        print(f"  [UNEXPECTED STATUS] {len(unexpected)}: {sorted({str(u['status']) for u in unexpected})}")
    if skipped:
        print(f"  [skipped/reason-coded] {len(skipped)}: {sorted({str(s['reason']) for s in skipped if s['reason']})}")


if __name__ == "__main__":
    main()
