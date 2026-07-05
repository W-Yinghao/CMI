#!/usr/bin/env python
"""CIGL_66 driver — measurement→control gap diagnostics over the FROZEN real-EEG artifacts. CPU only, no
retraining, no GPU. WITHIN-RUN scalars (subject-subspace spectrum + task-head alignment) computed per run, then
compared across methods on the scalars; task_drop reused from the frozen r3_reliance.csv (no recompute). Never
compares raw ERM vs CIGL coordinates.

    python scripts/analyze_cigl_gap.py --gate_dir results/cigl/r2_seed0_gate --final_dir results/cigl_r123/final
"""
from __future__ import annotations
import argparse, csv, glob, json, re, sys
from pathlib import Path
import numpy as np

REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO)
from cmi.eval.audit_npz import load_audit_npz, head_replay_ok                        # noqa: E402
from cmi.eval.gap_diagnostic import (subject_offset_matrix, spectrum_diagnostics,    # noqa: E402
                                     alignment_curve, DEFAULT_KS)
from cmi.eval.spatial_correlation import spatial_correlation                         # noqa: E402
from cmi.eval.evidence_hardening import hierarchical_bootstrap                       # noqa: E402

DATASETS = ["BNCI2014_001", "BNCI2015_001"]
METHODS = ["erm", "cigl_graph_node", "cdan"]
SEEDS = [0, 1, 2]


def _audit_paths(gate_dir, ds, m, s):
    keep = []
    for p in glob.glob(str(Path(gate_dir) / ds / "audit" / f"{ds}_fold*_{m}_seed{s}.audit.npz")):
        mt = re.search(rf"_sub[^_]+_(.+)_seed{s}\.audit\.npz$", Path(p).name)
        if mt and mt.group(1) == m:
            keep.append(p)
    return sorted(keep)


def _src(d, key):
    si = np.asarray(d["source_indices"]).ravel()
    return np.asarray(d[key])[si]


def spectrum_and_alignment(gate_dir):
    """Per (dataset, method, seed, fold): source-only subject-subspace spectrum for graph_z + node_z, and
    task-head alignment for graph_z (the head input). node_z has NO direct head -> alignment only for graph_z."""
    spec_rows, align_rows = [], []
    for ds in DATASETS:
        for m in METHODS:
            for s in SEEDS:
                for p in _audit_paths(gate_dir, ds, m, s):
                    d = load_audit_npz(p)
                    fold = int(np.asarray(d.get("fold", -1))); tsub = str(d.get("target_subject", ""))
                    y, dom = _src(d, "y"), _src(d, "d")
                    for rep in ("graph_z", "node_z"):
                        z = _src(d, rep)
                        if rep == "node_z":
                            z = z.reshape(z.shape[0], -1)
                        M = subject_offset_matrix(z, y, dom)
                        sp = spectrum_diagnostics(M)
                        spec_rows.append(dict(dataset=ds, method=m, seed=s, fold=fold, target_subject=tsub,
                                              representation=rep, total_subject_energy=sp["total_subject_energy"],
                                              effective_rank=sp["effective_rank"],
                                              **{f"top{k}_energy_fraction": sp[f"top{k}_energy_fraction"] for k in DEFAULT_KS}))
                    # alignment: only for graph_z (head consumes graph_z); requires exported linear head
                    if "task_head_weight" in d and d.get("task_head_input", "graph_z") == "graph_z":
                        Mg = subject_offset_matrix(_src(d, "graph_z"), y, dom)
                        ac = alignment_curve(Mg, np.asarray(d["task_head_weight"], float))
                        for k, a in ac.items():
                            align_rows.append(dict(dataset=ds, method=m, seed=s, fold=fold, target_subject=tsub,
                                                   representation="graph_z", k=k, task_head_alignment=a,
                                                   head_replay_ok=bool(head_replay_ok(d))))
    return spec_rows, align_rows


def _gate_fold_leakage(gate_dir):
    """(dataset, method, seed, fold) -> {graph_kl, node_kl, target_bacc}."""
    out = {}
    for ds in DATASETS:
        for jp in glob.glob(str(Path(gate_dir) / ds / f"{ds}_fold*_*_seed*.json")):
            rec = json.load(open(jp)); pr = rec.get("pareto_row")
            if not pr:
                continue
            out[(ds, rec.get("gate_label"), int(pr["seed"]), int(pr["fold"]))] = dict(
                graph_kl=pr["graph_kl_proxy"], node_kl=pr["node_kl_proxy"], target_bacc=pr["target_bacc"])
    return out


def _r3_taskdrop(final_dir):
    """(dataset, method, seed, fold) -> {k -> task_drop} for label_conditional; + subject_leak_drop k2."""
    out = {}
    fp = Path(final_dir) / "r3_reliance.csv"
    if not fp.exists():
        return out
    for r in csv.DictReader(open(fp)):
        if r["conditioning"] != "label_conditional":
            continue
        kk = (r["dataset"], r["method"], int(r["seed"]), int(r["fold"]))
        out.setdefault(kk, {})[int(r["k"])] = float(r["task_drop"])
        if int(r["k"]) == 2:
            out[kk]["subject_leak_drop"] = (float(r["subject_leak_drop"]) if r["subject_leak_drop"] not in ("", "nan") else float("nan"))
    return out


def _boot_corr(pairs, method="spearman", n_boot=2000, seed=0):
    """Bootstrap CI of a correlation by resampling the (x,y) pairs."""
    x = np.array([p[0] for p in pairs], float); y = np.array([p[1] for p in pairs], float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 4:
        return None
    point = spatial_correlation(x, y, method)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, x.size, x.size)
        r = spatial_correlation(x[idx], y[idx], method)
        if r == r:
            bs.append(r)
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (float("nan"), float("nan")))
    return dict(point=float(point), lo=float(lo), hi=float(hi), n=int(x.size), method=method)


def correlations(spec_rows, align_rows, gate_leak, r3):
    """Correlate KL / spectrum-energy / alignment against R3 task_drop (label_conditional k2), ERM+CIGL, per
    dataset + pooled. The key comparison: does alignment predict task_drop better than the KL proxy?"""
    align_k2 = {(r["dataset"], r["method"], r["seed"], r["fold"]): r["task_head_alignment"]
                for r in align_rows if r["k"] == 2}
    energy = {(r["dataset"], r["method"], r["seed"], r["fold"]): r["total_subject_energy"]
              for r in spec_rows if r["representation"] == "graph_z"}
    rows = []
    for scope in ["pooled", "BNCI2014_001", "BNCI2015_001"]:
        for xname, xget in (("graph_kl", lambda k: gate_leak.get(k, {}).get("graph_kl")),
                            ("node_kl", lambda k: gate_leak.get(k, {}).get("node_kl")),
                            ("graphz_subject_energy", lambda k: energy.get(k)),
                            ("task_head_alignment_k2", lambda k: align_k2.get(k))):
            pairs = []
            for kk, kd in r3.items():
                ds, m, s, f = kk
                if m not in ("erm", "cigl_graph_node"):
                    continue
                if scope != "pooled" and ds != scope:
                    continue
                xv = xget(kk); td = kd.get(2)
                if xv is not None and td is not None:
                    pairs.append((xv, td))
            for meth in ("spearman", "pearson"):
                c = _boot_corr(pairs, meth)
                if c:
                    rows.append(dict(scope=scope, x=xname, y="R3_task_drop_k2", corr_method=meth,
                                     rho=c["point"], ci_lo=c["lo"], ci_hi=c["hi"], n=c["n"],
                                     excludes_zero=bool(c["lo"] > 0 or c["hi"] < 0)))
    return rows


def graph_node_mismatch(final_dir):
    """Per (dataset, method): mean graph_kl/node_kl and reduction vs ERM; the head consumes graph_z, so a larger
    node-than-graph reduction is a candidate mechanism for the gap (node leakage cut, graph task-path not)."""
    rows = list(csv.DictReader(open(Path(final_dir) / "multiseed_pareto.csv")))
    agg = {}
    for r in rows:
        agg.setdefault((r["dataset"], r["method"]), []).append(r)
    mean = {k: dict(graph_kl=np.mean([float(x["graph_kl"]) for x in v]),
                    node_kl=np.mean([float(x["node_kl"]) for x in v])) for k, v in agg.items()}
    out = []
    for ds in DATASETS:
        erm = mean.get((ds, "erm"))
        for m in ("cigl_graph_node", "cdan"):
            mm = mean.get((ds, m))
            if not (erm and mm):
                continue
            g_red = erm["graph_kl"] - mm["graph_kl"]; n_red = erm["node_kl"] - mm["node_kl"]
            out.append(dict(dataset=ds, method=m, head_input="graph_z",
                            graph_kl=mm["graph_kl"], node_kl=mm["node_kl"],
                            graph_kl_reduction_vs_erm=g_red, node_kl_reduction_vs_erm=n_red,
                            node_reduction_exceeds_graph=bool(n_red > g_red),
                            node_minus_graph_reduction=float(n_red - g_red)))
    return out


def _wcsv(path, rows):
    if not rows:
        Path(path).write_text(""); return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)


def _scalar_summary(spec_rows, align_rows):
    """Cross-method comparison of the WITHIN-RUN scalars (mean over seeds/folds), per dataset."""
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for m in METHODS:
            sr = [r for r in spec_rows if r["dataset"] == ds and r["method"] == m and r["representation"] == "graph_z"]
            ar = [r for r in align_rows if r["dataset"] == ds and r["method"] == m and r["k"] == 2]
            if sr:
                out[ds][m] = dict(
                    graphz_subject_energy=float(np.mean([r["total_subject_energy"] for r in sr])),
                    graphz_effective_rank=float(np.mean([r["effective_rank"] for r in sr])),
                    graphz_top2_energy_fraction=float(np.mean([r["top2_energy_fraction"] for r in sr])),
                    task_head_alignment_k2=(float(np.mean([r["task_head_alignment"] for r in ar])) if ar else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_dir", default="results/cigl/r2_seed0_gate")
    ap.add_argument("--final_dir", default="results/cigl_r123/final")
    args = ap.parse_args()
    final = Path(args.final_dir); final.mkdir(parents=True, exist_ok=True)

    print("[gap] spectrum + alignment (within-run)...", flush=True)
    spec, align = spectrum_and_alignment(args.gate_dir)
    _wcsv(final / "gap_spectrum.csv", spec)
    _wcsv(final / "gap_alignment.csv", align)
    print("[gap] correlations...", flush=True)
    gate_leak = _gate_fold_leakage(args.gate_dir); r3 = _r3_taskdrop(args.final_dir)
    corr = correlations(spec, align, gate_leak, r3)
    _wcsv(final / "gap_correlations.csv", corr)
    print("[gap] graph-node mismatch...", flush=True)
    gnm = graph_node_mismatch(args.final_dir)
    _wcsv(final / "gap_graph_node_mismatch.csv", gnm)

    summ = _scalar_summary(spec, align)
    lines = ["# CIGL_66 gap-diagnostic summary (within-run scalars; cross-method compares scalars only)",
             f"n_spectrum_rows: {len(spec)}", f"n_alignment_rows: {len(align)}",
             "graphz_scalars_by_dataset_method:"]
    for ds in DATASETS:
        lines.append(f"  {ds}:")
        for m, v in summ[ds].items():
            al = "null" if v["task_head_alignment_k2"] is None else f"{v['task_head_alignment_k2']:.4f}"
            lines.append(f"    {m}: {{subject_energy: {v['graphz_subject_energy']:.4f}, "
                         f"effective_rank: {v['graphz_effective_rank']:.3f}, "
                         f"top2_energy_fraction: {v['graphz_top2_energy_fraction']:.4f}, "
                         f"task_head_alignment_k2: {al}}}")
    (final / "gap_diagnostic_summary.yaml").write_text("\n".join(lines) + "\n")

    print(f"\n[gap] wrote gap_spectrum({len(spec)}) gap_alignment({len(align)}) gap_correlations({len(corr)}) "
          f"gap_graph_node_mismatch({len(gnm)}) + summary")
    print("\n=== graph_z within-run scalars (ERM vs CIGL vs CDAN; alignment_k2 = head reliance on subject subspace) ===")
    for ds in DATASETS:
        print(f"  {ds}:")
        for m, v in summ[ds].items():
            al = "n/a" if v["task_head_alignment_k2"] is None else f"{v['task_head_alignment_k2']:.4f}"
            print(f"    {m:16s} subj_energy={v['graphz_subject_energy']:7.3f} erank={v['graphz_effective_rank']:5.2f} "
                  f"top2E={v['graphz_top2_energy_fraction']:.3f} align_k2={al}")
    print("\n=== correlation with R3 task_drop k2 (does alignment predict reliance better than KL?) ===")
    for r in corr:
        if r["scope"] == "pooled" and r["corr_method"] == "spearman":
            print(f"    {r['x']:24s} rho={r['rho']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] sig={r['excludes_zero']} (n={r['n']})")


if __name__ == "__main__":
    sys.exit(main())
