"""RW0 frozen weight audit (env icml; NO model training). For each of the 63 bundles: load the hash-verified ERM
warm-up, forward its continuation-train SOURCE split to Z, and compute the source-LOSO excess-risk RW-MCC weights +
the weight-permuted control. Writes risk_weight_rows.csv (per subject-pair), risk_weight_fold_summary.csv (per
bundle diagnostics), risk_weight_completeness.csv. Source-only characterization: target arrays are NEVER touched.
Manuscript FROZEN.

  python -m scripts.run_risk_weight_audit --device cpu --out-dir results/cmi_trace_risk_weighted_mcc
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.train.run_mcc_arms import enumerate_bundles, _warmup, _source_val_split
from tos_cmi.eeg.feature_dump import _forward_dump
from tos_cmi.train.risk_weighted_mcc import source_loso_excess_risk_weights, permute_weights, weight_hash, rw_mcc_loss


def _audit_bundle(ds, subj, seed, device, cache_dir, verify_dir, perm_seed=0):
    bb, Xtr, ytr, dtr, n_cls, Xte, yte, dataset, classes, meta_arr, warm_hash, X_shape = _warmup(
        ds, subj, seed, device, 300, 64, cache_dir)
    if verify_dir:
        prev = Path(verify_dir) / f"{ds}_sub{subj}_seed{seed}.manifest.json"
        if prev.exists():
            assert json.loads(prev.read_text()).get("warmup_hash") == warm_hash, "warm-up hash mismatch vs MCC round"
    tr_idx, _ = _source_val_split(dtr, ytr, seed=seed)                     # SAME continuation-train source as training
    Za = _forward_dump(bb, Xtr[tr_idx], device)[1]                        # warm-up source features
    ya, da = ytr[tr_idx], dtr[tr_idx]
    out = source_loso_excess_risk_weights(Za, ya, da)
    subs, pairs, w = out["subs"], out["pairs"], out["weights"]
    wp = permute_weights(w, subs, pairs, seed=perm_seed)
    # confirm true vs permuted RW-MCC losses differ (source Z only)
    Zt = torch.tensor(np.asarray(Za, float), dtype=torch.float32)
    Lt = float(rw_mcc_loss(Zt, ya, da, w)[0]); Lp = float(rw_mcc_loss(Zt, ya, da, wp)[0]) if out["status"] == "ok" else Lt
    per_subj = {s: float(sum(w[(s, p)] for p in pairs)) for s in subs}
    summary = dict(dataset=ds, subject=str(subj), seed=int(seed), warmup_hash=warm_hash, status=out["status"],
                   n_subjects=len(subs), n_pairs=len(pairs),
                   positive_weight_fraction=out["positive_weight_fraction"], effective_weight_support=out["effective_weight_support"],
                   max_weight=out["max_weight"], weight_entropy=out["weight_entropy"], winsor_threshold=out["winsor_threshold"],
                   weight_hash=weight_hash(w, subs, pairs), perm_weight_hash=weight_hash(wp, subs, pairs),
                   true_vs_perm_loss_diff=abs(Lt - Lp), max_subject_total_weight=max(per_subj.values()) if per_subj else 0.0,
                   top_subject_share=(max(per_subj.values()) / (sum(per_subj.values()) + 1e-9)) if per_subj else 0.0)
    rows = [dict(dataset=ds, subject=str(subj), seed=int(seed), src_subject=int(s), cls_a=int(a), cls_b=int(b),
                 weight=w[(s, (a, b))], excess_risk=out["r"][(s, (a, b))], l_hold=out["hold"][(s, (a, b))], l_ref=out["ref"][(s, (a, b))])
            for s in subs for (a, b) in pairs]
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu"); ap.add_argument("--out-dir", default="results/cmi_trace_risk_weighted_mcc")
    ap.add_argument("--cache-dir", default="results/cmi_trace_mcc/warmup_cache"); ap.add_argument("--verify-from", default="results/cmi_trace_mcc")
    ap.add_argument("--bundle-index", type=int, default=None)
    a = ap.parse_args()
    bundles = enumerate_bundles()
    todo = [bundles[a.bundle_index]] if a.bundle_index is not None else bundles
    outd = Path(a.out_dir); outd.mkdir(parents=True, exist_ok=True)
    summaries, rows, comp = [], [], []
    for ds, subj, seed in todo:
        try:
            s, r = _audit_bundle(ds, subj, seed, a.device, a.cache_dir, a.verify_from)
            summaries.append(s); rows.extend(r); comp.append(dict(dataset=ds, subject=str(subj), seed=seed, status="ok", weight_status=s["status"]))
            print(f"  {ds} sub{subj} s{seed}: status={s['status']} eff_support={s['effective_weight_support']:.1f} "
                  f"max_w={s['max_weight']:.2f} pos_frac={s['positive_weight_fraction']:.2f} top_subj_share={s['top_subject_share']:.2f} "
                  f"true!=perm={s['true_vs_perm_loss_diff']:.4f}", flush=True)
        except Exception as e:
            comp.append(dict(dataset=ds, subject=str(subj), seed=seed, status=f"FAIL:{type(e).__name__}", weight_status=""));
            print(f"  {ds} sub{subj} s{seed}: FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)

    def _w(fp, data, keys):
        with open(fp, "w", newline="") as fh:
            wt = csv.DictWriter(fh, fieldnames=keys); wt.writeheader(); [wt.writerow({k: d.get(k) for k in keys}) for d in data]
    if summaries:
        _w(outd / "risk_weight_fold_summary.csv", summaries, list(summaries[0].keys()))
        _w(outd / "risk_weight_rows.csv", rows, list(rows[0].keys()))
    _w(outd / "risk_weight_completeness.csv", comp, ["dataset", "subject", "seed", "status", "weight_status"])
    nz = sum(1 for s in summaries if s["status"] == "NO_POSITIVE_SOURCE_TRANSFER_GAP")
    print(f"[rw-audit] {len(summaries)}/{len(todo)} bundles; NO_POSITIVE={nz}; wrote 3 CSVs -> {outd}", flush=True)


if __name__ == "__main__":
    main()
