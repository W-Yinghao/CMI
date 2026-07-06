"""WAVE 1 / W1.geometry analyzer (frozen W1_GEOMETRY_FROZEN.md). Per perturbation:
primary falsification contrast max(full-cov) - max(diagonal-latent) BA (paired per unit, (dataset,subject)
cluster bootstrap); identity BA drop vs none; per-operator BA; secondary CORAL-FRSC and EA-CORAL.
Facts only."""
from __future__ import annotations

import glob
import json
from collections import defaultdict

import numpy as np

NB = 10000
PERTS = ["none", "reref", "gain", "dropout"]
DIAG = ["bacc_fixed_reference_oneshot_uniform", "bacc_fixed_iterative_geometry_uniform",
        "bacc_joint_geometry_uniform", "bacc_latent_im_diag_uniform", "bacc_pooled_uniform"]
FC = ["bacc_coral_latent", "bacc_source_recolored_ea"]
ALLOPS = ["bacc_identity_uniform"] + DIAG + FC


def _load():
    rows = []
    for f in glob.glob("results/h2cmi/wave1_geom/w1g_*.jsonl"):
        if "probe" in f:
            continue
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r.get("panel") == "W1GEOM" and "perturbation" in r and not r.get("provenance_fail"):
                    rows.append(r)
    return rows


def _cluster_boot(cluster_vals, seed=0):
    keys = [k for k in cluster_vals if len(cluster_vals[k])]
    if len(keys) < 2:
        return dict(mean=float("nan"), ci=[float("nan")] * 2, n=0)
    arrs = {k: np.asarray(cluster_vals[k], float) for k in keys}
    allv = np.concatenate([arrs[k] for k in keys])
    rng = np.random.default_rng(seed)
    bs = [np.concatenate([arrs[k] for k in (keys[i] for i in rng.integers(0, len(keys), len(keys)))]).mean() for _ in range(NB)]
    return dict(mean=round(float(allv.mean()), 4), ci=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                excludes_0=bool(np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0), n=int(len(keys)))


def _unit_op(rows, pert):
    """(dataset,subject) -> {op: seed-avg BA} for a perturbation."""
    tmp = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["perturbation"] == pert:
            u = (r["dataset"], int(r["subject"]))
            for op in ALLOPS:
                if op in r:
                    tmp[u][op].append(r[op])
    return {u: {op: float(np.mean(v)) for op, v in d.items()} for u, d in tmp.items()}


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/h2cmi/wave1_geom.report.json")
    args = ap.parse_args()
    rows = _load()
    units = sorted({(r["dataset"], int(r["subject"])) for r in rows})
    none_u = _unit_op(rows, "none")
    rep = dict(marker="WAVE1_GEOMETRY", n_units=len(units),
               n_units_per_dataset={ds: len({s for (d, s) in units if d == ds}) for ds in {d for d, s in units}})
    per_pert = {}
    for pert in PERTS:
        uo = _unit_op(rows, pert)
        # primary falsification contrast: max(FC) - max(DIAG), paired per unit
        cl_contrast = defaultdict(list); cl_coral_frsc = defaultdict(list); cl_ea_coral = defaultdict(list); cl_iddrop = defaultdict(list)
        for u, d in uo.items():
            if all(op in d for op in DIAG + FC):
                cl_contrast[(u[0], u[1])].append(max(d[o] for o in FC) - max(d[o] for o in DIAG))
                cl_coral_frsc[(u[0], u[1])].append(d["bacc_coral_latent"] - d["bacc_fixed_reference_oneshot_uniform"])
                cl_ea_coral[(u[0], u[1])].append(d["bacc_source_recolored_ea"] - d["bacc_coral_latent"])
            if u in none_u and "bacc_identity_uniform" in d and "bacc_identity_uniform" in none_u[u]:
                cl_iddrop[(u[0], u[1])].append(none_u[u]["bacc_identity_uniform"] - d["bacc_identity_uniform"])
        per_op = {op.replace("bacc_", ""): _cluster_boot({(u[0], u[1]): [d[op]] for u, d in uo.items() if op in d}) for op in ALLOPS}
        per_pert[pert] = dict(
            falsification_maxFC_minus_maxDIAG=_cluster_boot(cl_contrast),
            identity_BA_drop_vs_none=(None if pert == "none" else _cluster_boot(cl_iddrop)),
            coral_minus_FRSC=_cluster_boot(cl_coral_frsc), EA_minus_CORAL=_cluster_boot(cl_ea_coral),
            per_operator_BA=per_op)
    rep["per_perturbation"] = per_pert
    json.dump(rep, open(args.out, "w"), indent=2, default=str)
    print(f"[W1GEOM] units={len(units)} {rep['n_units_per_dataset']}")
    print(f"  {'perturbation':<10} {'maxFC-maxDIAG':>22} {'identity BA drop':>20} {'CORAL-FRSC':>14}")
    for pert in PERTS:
        p = per_pert[pert]; c = p["falsification_maxFC_minus_maxDIAG"]; dd = p["identity_BA_drop_vs_none"]; cf = p["coral_minus_FRSC"]
        ds = f"{dd['mean']:+.3f}{dd['ci']}" if dd else "   (anchor)"
        flag = "" if pert == "none" else (" <-- BOUNDS diag" if c.get("excludes_0") and c["mean"] > 0 else "")
        print(f"  {pert:<10} {c['mean']:+.4f}{str(c['ci']):>13}{'SIG' if c.get('excludes_0') else ' ns':>4} {ds:>20} {cf['mean']:+.3f}{flag}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
