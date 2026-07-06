#!/usr/bin/env python
"""CIGL_68 direct-reliance CONFIRMATION (seeds 0/1/2) — dcigl_consistency_beta0.1 vs FROZEN CIGL / FCIGL-align
per (dataset, seed, fold), with hierarchical bootstrap CIs AND anti-triviality logit diagnostics. All from saved
.audit.npz + head-replay; CPU only, no retrain, no target-label fit.

Anti-triviality: dcigl directly optimizes prediction consistency under removal, so we must rule out that R3 falls
merely because logits FLATTEN. We report, on TARGET trials: mean_logit_norm, mean_margin, prediction_entropy,
target_nll, removed_vs_original_agreement, removed_target_bacc. A genuine positive: R3 task_drop ↓ WITHOUT
margin/confidence collapse.
"""
from __future__ import annotations
import argparse, csv, glob, json, re, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO)
from cmi.eval.evidence_hardening import hierarchical_bootstrap                       # noqa: E402

DATASETS = ["BNCI2014_001", "BNCI2015_001"]
SEEDS = [0, 1, 2]
# (label, method-in-filename, audit_dir) for the three lines
DCIGL = "dcigl_consistency_beta0.1"; CIGL = "cigl_graph_node"; FCIGL = "fcigl_align_eta0.01"
QTY = ["target_bacc", "graph_kl", "node_kl", "R3_task_drop_k2", "R3_task_drop_k8",
       "random_subspace_task_drop_k2", "task_head_alignment_k2", "mean_logit_norm", "mean_margin",
       "prediction_entropy", "removed_target_bacc"]


def _audit_paths(adir, ds, m, s):
    keep = []
    for p in glob.glob(str(Path(adir) / ds / "audit" / f"{ds}_fold*_{m}_seed{s}.audit.npz")):
        mt = re.search(rf"_sub[^_]+_(.+)_seed{s}\.audit\.npz$", Path(p).name)
        if mt and mt.group(1) == m:
            keep.append(p)
    return sorted(keep)


def _one(a):
    ds, label, s, p = a
    sys.path.insert(0, REPO)
    import numpy as _np
    from cmi.eval.audit_npz import load_audit_npz, head_replay_ok
    from cmi.eval.leakage_removal import evaluate_reliance
    from cmi.eval.gap_diagnostic import subject_offset_matrix, subject_subspace, task_head_alignment
    from sklearn.metrics import balanced_accuracy_score
    d = load_audit_npz(p)
    ti = _np.asarray(d.get("target_indices", []))
    if not ti.size:
        return None
    ti = ti.ravel(); tgt = int(_np.asarray(d["d"])[ti][0]); fold = int(_np.asarray(d.get("fold", -1)))
    out = dict(dataset=ds, method=label, seed=s, fold=fold, head_replay_ok=bool(head_replay_ok(d)))
    for k in (2, 8):
        out[f"R3_task_drop_k{k}"] = evaluate_reliance(d, target_domain=tgt, k=k, conditioning="label_conditional")["task_drop"]
    out["random_subspace_task_drop_k2"] = evaluate_reliance(d, target_domain=tgt, k=2, conditioning="random_subspace")["task_drop"]
    # alignment (graph_z head) + logit diagnostics on TARGET trials
    gz = _np.asarray(d["graph_z"], float); y = _np.asarray(d["y"]); dom = _np.asarray(d["d"])
    si = _np.asarray(d["source_indices"]).ravel()
    if "task_head_weight" in d:
        W = _np.asarray(d["task_head_weight"], float); b = _np.asarray(d.get("task_head_bias", 0.0), float)
        M = subject_offset_matrix(gz[si], y[si], dom[si]); out["task_head_alignment_k2"] = task_head_alignment(W, subject_subspace(M, 2))
        # TARGET-trial logit diagnostics (head-replay; source-fit k2 removal)
        lo = gz[ti] @ W.T + b                                  # original target logits (== model_logits)
        sm = _np.exp(lo - lo.max(1, keepdims=True)); sm = sm / sm.sum(1, keepdims=True)
        srt = _np.sort(lo, 1)
        out["mean_logit_norm"] = float(_np.linalg.norm(lo, axis=1).mean())
        out["mean_margin"] = float((srt[:, -1] - srt[:, -2]).mean())
        out["prediction_entropy"] = float((-(sm * _np.log(sm + 1e-12)).sum(1)).mean())
        out["target_nll"] = float((-_np.log(sm[_np.arange(len(y[ti])), y[ti]] + 1e-12)).mean())
        S = subject_subspace(M, 2); P = _np.eye(gz.shape[1]) - S.T @ S
        lo_rm = (gz[ti] @ P.T) @ W.T + b
        out["removed_vs_original_agreement"] = float((_np.argmax(lo, 1) == _np.argmax(lo_rm, 1)).mean())
        out["removed_target_bacc"] = float(balanced_accuracy_score(y[ti], _np.argmax(lo_rm, 1)))
    return out


def _perfold(dcigl_dir, cigl_dir, fcigl_dir, workers):
    tasks = []
    for ds in DATASETS:
        for s in SEEDS:
            tasks += [(ds, DCIGL, s, p) for p in _audit_paths(dcigl_dir, ds, DCIGL, s)]
            tasks += [(ds, CIGL, s, p) for p in _audit_paths(cigl_dir, ds, CIGL, s)]
            tasks += [(ds, FCIGL, s, p) for p in _audit_paths(fcigl_dir, ds, FCIGL, s)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for r in ex.map(_one, tasks) if r]
    return rows


def _attach_leak(rows, dcigl_dir, cigl_gate, fcigl_dir):
    """target/graph/node from each line's per-fold gate JSON."""
    leak = {}
    for adir, method, gate in ((dcigl_dir, DCIGL, dcigl_dir), (None, CIGL, cigl_gate), (fcigl_dir, FCIGL, fcigl_dir)):
        for ds in DATASETS:
            for jp in glob.glob(str(Path(gate) / ds / f"{ds}_fold*_{method}_seed*.json")):
                mt = re.search(rf"_fold(\d+)_{method}_seed(\d+)\.json$", Path(jp).name)
                if not mt:
                    continue
                rec = json.load(open(jp)); pr = rec.get("pareto_row")
                if pr:
                    leak[(ds, method, int(mt.group(2)), int(mt.group(1)))] = pr
    for r in rows:
        pr = leak.get((r["dataset"], r["method"], r["seed"], r["fold"]), {})
        r["target_bacc"] = _f(pr.get("target_bacc")); r["graph_kl"] = _f(pr.get("graph_kl_proxy")); r["node_kl"] = _f(pr.get("node_kl_proxy"))
    return rows


def _f(x):
    return float(x) if x is not None else float("nan")


def _boot(recs, levels, n_boot=4000):
    if len(recs) < 2:
        return None
    r = hierarchical_bootstrap(recs, value_key="value", levels=levels, n_boot=n_boot, seed=0)
    return dict(point=r["point"], lo=r["lo"], hi=r["hi"], n=r["n_records"], excludes_zero=bool(r["lo"] > 0 or r["hi"] < 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcigl_dir", default="results/cigl/direct_reliance_gate")
    ap.add_argument("--cigl_dir", default="/home/infres/yinwang/CMI_AAAI_cigl_r123/results/cigl/r2_seed0_gate")
    ap.add_argument("--fcigl_dir", default="/home/infres/yinwang/CMI_AAAI_fcigl/results/cigl/functional_gate")
    ap.add_argument("--out_dir", default="results/cigl_direct_reliance/final")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("[dconf] per-fold (dcigl + CIGL + FCIGL, seeds 0/1/2)...", flush=True)
    rows = _attach_leak(_perfold(args.dcigl_dir, args.cigl_dir, args.fcigl_dir, args.workers),
                        args.dcigl_dir, args.cigl_dir, args.fcigl_dir)
    fields = ["dataset", "method", "seed", "fold", "target_bacc", "graph_kl", "node_kl", "R3_task_drop_k2",
              "R3_task_drop_k8", "random_subspace_task_drop_k2", "task_head_alignment_k2", "mean_logit_norm",
              "mean_margin", "prediction_entropy", "target_nll", "removed_vs_original_agreement",
              "removed_target_bacc", "head_replay_ok"]
    _w(out / "direct_reliance_metrics.csv", rows, fields)
    _w(out / "direct_reliance_r3.csv", rows, ["dataset", "method", "seed", "fold", "R3_task_drop_k2", "R3_task_drop_k8", "random_subspace_task_drop_k2", "head_replay_ok"])
    _w(out / "direct_reliance_logit_diagnostics.csv", rows,
       ["dataset", "method", "seed", "fold", "mean_logit_norm", "mean_margin", "prediction_entropy", "target_nll", "removed_vs_original_agreement", "removed_target_bacc"])

    idx = {(r["dataset"], r["method"], r["seed"], r["fold"]): r for r in rows}
    ci_rows = []
    for base, quants in ((CIGL, QTY), (FCIGL, ["target_bacc", "R3_task_drop_k2", "task_head_alignment_k2"])):
        for q in quants:
            deltas = []
            for (ds, m, s, f), r in idx.items():
                if m != DCIGL:
                    continue
                b = idx.get((ds, base, s, f))
                if b and r.get(q) is not None and b.get(q) is not None and r[q] == r[q] and b[q] == b[q]:
                    deltas.append(dict(dataset=ds, seed=s, fold=f, value=float(r[q] - b[q])))
            for scope, lv, sub in (("pooled", ("dataset", "seed", "fold"), deltas),
                                   ("BNCI2014_001", ("seed", "fold"), [d for d in deltas if d["dataset"] == "BNCI2014_001"]),
                                   ("BNCI2015_001", ("seed", "fold"), [d for d in deltas if d["dataset"] == "BNCI2015_001"])):
                bt = _boot(sub, lv)
                if bt:
                    ci_rows.append(dict(comparison=f"dcigl_b0.1_minus_{base}", quantity=q, scope=scope,
                                        point=bt["point"], ci_lo=bt["lo"], ci_hi=bt["hi"], n=bt["n"], excludes_zero=bt["excludes_zero"]))
    _w(out / "direct_reliance_bootstrap_ci.csv", ci_rows, ["comparison", "quantity", "scope", "point", "ci_lo", "ci_hi", "n", "excludes_zero"])
    _w(out / "direct_reliance_vs_frozen.csv", _summary(rows), ["dataset", "method", "target_bacc", "graph_kl", "node_kl", "R3_task_drop_k2", "task_head_alignment_k2", "mean_margin", "prediction_entropy", "removed_target_bacc"])
    _manifest(out, rows, ci_rows)

    print(f"\n[dconf] wrote {out}/ ({len(rows)} fold rows, {len(ci_rows)} CI rows)")
    print("\n=== dcigl b0.1 - old CIGL : hierarchical bootstrap CIs (sig = excludes 0) ===")
    for q in ("R3_task_drop_k2", "target_bacc", "mean_margin", "prediction_entropy", "task_head_alignment_k2"):
        for scope in ("pooled", "BNCI2014_001", "BNCI2015_001"):
            r = [c for c in ci_rows if c["comparison"] == f"dcigl_b0.1_minus_{CIGL}" and c["quantity"] == q and c["scope"] == scope]
            if r:
                r = r[0]; print(f"  {q:24s} {scope:13s} {r['point']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] sig={r['excludes_zero']}")
    print("\n=== dcigl b0.1 - FCIGL R3 ===")
    for scope in ("pooled", "BNCI2015_001"):
        r = [c for c in ci_rows if c["comparison"] == f"dcigl_b0.1_minus_{FCIGL}" and c["quantity"] == "R3_task_drop_k2" and c["scope"] == scope]
        if r:
            r = r[0]; print(f"  R3_task_drop_k2 {scope:13s} {r['point']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] sig={r['excludes_zero']}")


def _summary(rows):
    out = []
    for ds in DATASETS:
        for m in (CIGL, FCIGL, DCIGL):
            g = [r for r in rows if r["dataset"] == ds and r["method"] == m]
            if g:
                out.append(dict(dataset=ds, method=m, **{q: float(np.nanmean([r.get(q, np.nan) for r in g]))
                                                         for q in ("target_bacc", "graph_kl", "node_kl", "R3_task_drop_k2", "task_head_alignment_k2", "mean_margin", "prediction_entropy", "removed_target_bacc")}))
    return out


def _w(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _manifest(out, rows, ci_rows):
    lines = ["# CIGL_68 direct-reliance confirmation manifest (seeds 0/1/2)",
             "phase: CIGL_68_DIRECT_RELIANCE_CONFIRMATION", "branch: project/cigl-direct-reliance-cmi",
             "method: dcigl_consistency_beta0.1 (gamma=0.5, k=2)", "seeds: [0, 1, 2]",
             "comparators_frozen: [cigl_graph_node, fcigl_align_eta0.01]",
             "projector: {fit: source_train_only, excludes_target: true, excludes_source_val: true, k: 2}",
             "anti_triviality: logit_norm/margin/entropy/removed_agreement/removed_target_bacc reported",
             "bootstrap: {per_dataset: seed->fold, pooled: dataset->seed->fold, n_boot: 4000}",
             f"n_fold_rows: {len(rows)}",
             "files: [direct_reliance_metrics.csv, direct_reliance_r3.csv, direct_reliance_vs_frozen.csv, direct_reliance_bootstrap_ci.csv, direct_reliance_logit_diagnostics.csv]"]
    (out / "MANIFEST.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
