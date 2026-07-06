"""C18-P CPU stage — recompute per-candidate SOURCE signals under each S0-S7 support mask from the persisted
GPU extraction (per-unit source logits + per-candidate Z-features + base support graphs). Forwarding was
mask-independent, so masking here is exact: subset/reweight the per-unit predictions by (domain,class) cells
and recompute the worst-domain source endpoints; subset the Z-features + rebuild a masked grouped fold plan
and re-estimate the leakage points (abstaining -> None when a masked cell family becomes non-estimable).

ONLY the 8 recomputable-under-mask features (6 worst-domain eval endpoints + 2 leakage points) are recomputed
per regime; the 4 static training-log features (R_src/balanced_err/train_surrogate/epoch) are carried at their
extracted S0 value and are EXCLUDED from S1-S7 mask claims. Target labels (tgt__) are joined POST HOC from C10
(diagnostic-only) and are NEVER masked (support degradation is source-side)."""
from __future__ import annotations

import json
import os

import numpy as np

from ..eval.calibration import fixed_bin_edges, worst_domain_ece, worst_domain_nll
from ..eval.metrics import worst_domain_bacc
from ..identifiability.signal_atlas import build_atlas, load_replay
from ..leakage.critic import CriticConfig
from ..leakage.crossfit import FrozenFeatures, make_fold_plan_from_design
from ..leakage.design import make_leakage_design
from ..leakage.errors import LeakageNonEstimableError
from ..leakage.estimate import estimate_extractable_leakage
from ..support_graph import build_support_graph, counts_from_labels
from . import masks, schema
from . import stress_plan as sp

_RECOMPUTABLE = schema.RECOMPUTABLE_UNDER_MASK
_STATIC = schema.STATIC_TRAINING_LOG_ONLY


def _load_support(path):
    z = np.load(path, allow_pickle=True)
    return build_support_graph(np.asarray(z["counts"], dtype=np.int64), int(z["m"]),
                               cell_mass=np.asarray(z["cell_mass"], dtype=np.float64),
                               reference_prior=np.asarray(z["reference_prior"], dtype=np.float64),
                               domain_names=[str(x) for x in z["domain_names"]],
                               class_names=[str(x) for x in z["class_names"]])


def load_fold_level(extract_dir, seed, target, level) -> dict:
    p = os.path.join(extract_dir, f"seed-{seed}-target-{target:03d}", f"level-{level}")
    d = {"seed": seed, "target": target, "level": level, "units": {}, "logits": {}, "featz": {}}
    for role in ("source_guard", "source_audit"):
        u = np.load(os.path.join(p, f"units-{role}.npz"), allow_pickle=True)
        d["units"][role] = {"domain_raw": [str(x) for x in u["domain_raw"]], "y": np.asarray(u["y"], dtype=int),
                            "group": [str(x) for x in u["group"]], "sample_id": [str(x) for x in u["sample_id"]]}
        d["logits"][role] = np.load(os.path.join(p, f"logits-{role}.npy"))
    for dz in ("selection", "audit"):
        fp = os.path.join(p, f"featz-{dz}.npz")
        if os.path.exists(fp):
            z = np.load(fp, allow_pickle=True)
            d["featz"][dz] = {"Z": z["Z"], "y": np.asarray(z["y"], dtype=int),
                              "d_code": np.array([int(x) for x in z["d"]]),
                              "group": [str(x) for x in z["group"]], "sample_id": [str(x) for x in z["sample_id"]]}
    d["support_source"] = _load_support(os.path.join(p, "support-source.npz"))
    ap = os.path.join(p, "support-audit.npz")
    d["support_audit"] = _load_support(ap) if os.path.exists(ap) else None
    d["config"] = json.load(open(os.path.join(p, "config.json")))
    d["cand_meta"] = json.load(open(os.path.join(p, "cand_meta.json")))
    s0p = os.path.join(p, "s0_scalars.json")                          # optional (needed only by self-check/identity)
    d["s0_scalars"] = {r["model_hash"]: r for r in json.load(open(s0p))} if os.path.exists(s0p) else {}
    return d


def _critic(cfg) -> CriticConfig:
    c = cfg["critic"]
    return CriticConfig(capacities=tuple(int(x) for x in c["capacities"]), l2_C=float(c["l2_C"]),
                        max_iter=int(c["max_iter"]), prob_floor=float(c["prob_floor"]),
                        feature_seed_base=int(c["feature_seed_base"]))


def _codes(names):
    uniq = {n: i for i, n in enumerate(sorted(set(names)))}
    return np.array([uniq[n] for n in names], dtype=int)


def _nanfloat(x):
    return float(x) if x is not None else float("nan")


def _worst_domain(logits_ci, y, domain_raw, keep, classes, edges):
    idx = np.where(keep)[0]
    if len(idx) == 0:
        return float("nan"), float("nan"), float("nan")
    lg = np.asarray(logits_ci)[idx]; yy = np.asarray(y)[idx]
    dom = _codes([domain_raw[i] for i in idx])
    pred = lg.argmax(1)
    return (_nanfloat(worst_domain_bacc(yy, pred, dom, classes, "reference")),
            _nanfloat(worst_domain_nll(lg, yy, dom)),
            _nanfloat(worst_domain_ece(lg, yy, dom, bin_edges=edges)))


def _leakage(featz, ci, name_actions, base_sg, critic, n_folds, *, seed, target, level, salt):
    if featz is None or base_sg is None or n_folds is None:
        return None
    names = [str(base_sg.domain_names[c]) for c in featz["d_code"]]     # design domain-code -> domain NAME
    keep = masks.subset_rows_by_cells(name_actions, names, featz["y"], seed=seed, target=target, level=level)
    if len(keep) == 0:
        return None
    d_code = featz["d_code"][keep]; yk = featz["y"][keep]
    grp = [str(featz["group"][i]) for i in keep]; sid = [str(featz["sample_id"][i]) for i in keep]
    mass = np.ones(len(yk), dtype=np.float64); Z = np.asarray(featz["Z"][ci])[keep]
    # support graph DERIVED from the MASKED rows (design cell_mass = per-cell row mass sum -> matches by
    # construction; abstractly-masked cell_mass would mismatch for skew/rare). Reference prior stays FIXED.
    D, C = base_sg.counts.shape
    counts = counts_from_labels(d_code, yk, n_domains=D, n_classes=C)
    cell_mass = np.zeros((D, C), dtype=np.float64); np.add.at(cell_mass, (d_code, yk), mass)
    sg = build_support_graph(counts, int(base_sg.m), cell_mass=cell_mass,
                             reference_prior=np.asarray(base_sg.reference_prior, dtype=np.float64),
                             domain_names=list(base_sg.domain_names), class_names=list(base_sg.class_names))
    try:
        design = make_leakage_design(tuple(sid), yk, d_code, grp, mass, sg)
        fp = make_fold_plan_from_design(design, sg, n_folds=int(n_folds),
                                        seed=sp._fold_seed(seed, target, level, salt))
        feat = FrozenFeatures(Z=Z, y=yk, d=d_code, group=np.array(grp), sample_mass=mass, sample_id=tuple(sid))
        return float(estimate_extractable_leakage(feat, sg, fp, critic)["L_abs"])
    except LeakageNonEstimableError:
        return None                                                    # GENUINE abstention -> H5 signal (not a bug swallow)


def recompute_candidate(fld, ci, source_na, audit_na, *, edges, classes, with_leakage=True):
    """8 recomputable source signals for candidate ci. The source-train (guard) and source-audit splits are
    DISJOINT domain sets, so each is degraded by its OWN side of the regime: source_na (built on the source
    support graph) drives source_guard + selection leakage; audit_na (built on the audit support graph) drives
    source_audit + audit leakage. Worst-domain endpoints always; leakage only when with_leakage (probe cost)."""
    seed, target, level = fld["seed"], fld["target"], fld["level"]
    out = {}
    for role, pre, na in (("source_guard", "source_guard", source_na), ("source_audit", "source_audit", audit_na)):
        u = fld["units"][role]
        keep, _ = masks.unit_keep_weight(na, u["domain_raw"], u["y"], seed=seed, target=target, level=level)
        b, n, e = _worst_domain(fld["logits"][role][ci], u["y"], u["domain_raw"], keep, classes, edges)
        out[f"{pre}_worst_bacc"], out[f"{pre}_worst_nll"], out[f"{pre}_worst_ece"] = b, n, e
    if with_leakage:
        cfg = fld["config"]; critic = _critic(cfg)
        out["selection_leakage_point"] = _leakage(fld["featz"].get("selection"), ci, source_na,
                                                  fld["support_source"], critic, cfg.get("selection_n_folds"),
                                                  seed=seed, target=target, level=level, salt="sel_leak")
        out["audit_leakage_point"] = _leakage(fld["featz"].get("audit"), ci, audit_na, fld["support_audit"],
                                              critic, cfg.get("audit_n_folds"), seed=seed, target=target,
                                              level=level, salt="audit_leak")
    else:
        out["selection_leakage_point"] = None
        out["audit_leakage_point"] = None
    return out


def _regime_name_actions(regime, sg, *, boundary_classes, seed, target, level, n_perturb):
    """(name_actions dict, RegimePlan) for one support graph; {} if sg is None."""
    if sg is None:
        return {}, None
    plan = sp.build_regime_plan(regime, sg.counts, sg.cell_mass, sg.eligible, sg.m,
                                boundary_classes=boundary_classes, seed=seed, target=target, level=level,
                                n_perturb=n_perturb)
    return masks.actions_by_name(plan, sg.domain_names), plan


def _target_labels(c10_dir) -> dict:
    """(seed,target,level,model_hash) -> tgt__ label dict, from the committed C10 atlas (unmasked)."""
    rows = build_atlas(load_replay(c10_dir))
    return {(r["seed"], r["target"], r["level"], r["model_hash"]):
            {k: r[k] for k in r if k.startswith("tgt__")} for r in rows}


def build_regime_atlas(extract_dir, c10_dir, regime, *, boundary_classes, n_perturb=2, folds=None,
                       leakage_cache=None) -> list:
    """CPU-recomputed atlas rows for ONE regime across all extracted folds/levels. Feasible-OACI candidates
    only (the C17 probe population). src__ = masked recompute (recomputable) + extracted-S0 (static); tgt__ =
    C10 labels (unmasked). leakage_cache[(seed,target,level,regime,model_hash)]=(sel,audit) (from the parallel
    precompute) avoids re-fitting the leakage probes here; when absent, leakage is recomputed inline (slow)."""
    labels = _target_labels(c10_dir)
    fold_dirs = folds if folds is not None else _list_folds(extract_dir)
    edges = fixed_bin_edges(15)
    inline_leak = leakage_cache is None
    rows = []
    for (seed, target) in fold_dirs:
        for level in _levels(extract_dir, seed, target):
            fld = load_fold_level(extract_dir, seed, target, level)
            classes = list(range(fld["logits"]["source_guard"].shape[2]))
            source_na, _ = _regime_name_actions(regime, fld["support_source"], boundary_classes=boundary_classes,
                                                seed=seed, target=target, level=level, n_perturb=n_perturb)
            audit_na, _ = _regime_name_actions(regime, fld["support_audit"], boundary_classes=boundary_classes,
                                               seed=seed, target=target, level=level, n_perturb=n_perturb)
            for ci, cm in enumerate(fld["cand_meta"]):
                if cm["is_erm"] or not cm["feasible"]:
                    continue                                           # feasible OACI only (C17 population)
                key = (seed, target, level, cm["model_hash"])
                if key not in labels:
                    continue
                rec = recompute_candidate(fld, ci, source_na, audit_na, edges=edges, classes=classes,
                                          with_leakage=inline_leak)
                if not inline_leak:
                    sel, aud = leakage_cache.get((seed, target, level, regime, cm["model_hash"]), (None, None))
                    rec["selection_leakage_point"], rec["audit_leakage_point"] = sel, aud
                row = {"seed": seed, "target": target, "level": level, "model_hash": cm["model_hash"],
                       "regime": regime, "diagnostic_only_non_deployable": True}
                for s in _RECOMPUTABLE:
                    row["src__" + s] = rec[s]
                for s in _STATIC:                                       # carried, marked static, excluded from claims
                    row["src__" + s] = cm.get(s)
                row.update(labels[key])
                rows.append(row)
    return rows


def _fold_regime_leakage(extract_dir, seed, target, level, boundary_classes, n_perturb, regimes=None):
    """Worker: all feasible-OACI candidates' (selection, audit) leakage for the requested regimes at one
    fold-level. Pure numpy/sklearn (no GPU/torch); amortizes the fold load across regimes. Returns a flat dict."""
    fld = load_fold_level(extract_dir, seed, target, level)
    critic = _critic(fld["config"]); cfg = fld["config"]
    out = {}
    for regime in (regimes if regimes is not None else schema.REGIME_ORDER):
        source_na, _ = _regime_name_actions(regime, fld["support_source"], boundary_classes=boundary_classes,
                                            seed=seed, target=target, level=level, n_perturb=n_perturb)
        audit_na, _ = _regime_name_actions(regime, fld["support_audit"], boundary_classes=boundary_classes,
                                           seed=seed, target=target, level=level, n_perturb=n_perturb)
        for cm in fld["cand_meta"]:
            if cm["is_erm"] or not cm["feasible"]:
                continue
            ci = cm["index"]
            sel = _leakage(fld["featz"].get("selection"), ci, source_na, fld["support_source"], critic,
                           cfg.get("selection_n_folds"), seed=seed, target=target, level=level, salt="sel_leak")
            aud = _leakage(fld["featz"].get("audit"), ci, audit_na, fld["support_audit"], critic,
                           cfg.get("audit_n_folds"), seed=seed, target=target, level=level, salt="audit_leak")
            out[(seed, target, level, regime, cm["model_hash"])] = (sel, aud)
    return out


def precompute_all_leakage(extract_dir, *, boundary_classes, n_perturb=2, folds=None, n_workers=8, regimes=None) -> dict:
    """Parallel (process-level) leakage precompute across all fold-levels x regimes (default all; pass
    `regimes` to restrict). Returns {(seed,target,level,regime,model_hash): (selection_leakage, audit_leakage)}."""
    fold_dirs = folds if folds is not None else _list_folds(extract_dir)
    rg = tuple(regimes) if regimes is not None else None
    tasks = [(extract_dir, s, t, level, tuple(boundary_classes), n_perturb, rg)
             for (s, t) in fold_dirs for level in _levels(extract_dir, s, t)]
    cache = {}
    if n_workers <= 1:
        for tk in tasks:
            cache.update(_fold_regime_leakage(*tk))
        return cache
    from multiprocessing import Pool
    with Pool(n_workers) as pool:
        for part in pool.starmap(_fold_regime_leakage, tasks):
            cache.update(part)
    return cache


def build_identity_atlas(extract_dir, c10_dir, *, folds=None) -> list:
    """S0 all-column atlas from the EXTRACTED authoritative S0 scalars (which matched C10) + C10 tgt__ labels.
    This is the extraction's reconstruction of the C17 atlas; the identity probe on it must reproduce 0.602
    (G2). Includes all 12 src__ signals (8 recomputable + 4 static)."""
    from ..identifiability.signal_atlas import SOURCE_SIGNALS
    labels = _target_labels(c10_dir)
    fold_dirs = folds if folds is not None else _list_folds(extract_dir)
    rows = []
    for (seed, target) in fold_dirs:
        for level in _levels(extract_dir, seed, target):
            fld = load_fold_level(extract_dir, seed, target, level)
            for cm in fld["cand_meta"]:
                if cm["is_erm"] or not cm["feasible"]:
                    continue
                key = (seed, target, level, cm["model_hash"])
                if key not in labels:
                    continue
                s0 = fld["s0_scalars"][cm["model_hash"]]
                row = {"seed": seed, "target": target, "level": level, "model_hash": cm["model_hash"],
                       "regime": "S0_full_support", "diagnostic_only_non_deployable": True}
                for s in SOURCE_SIGNALS:
                    row["src__" + s] = s0.get(s)
                row.update(labels[key])
                rows.append(row)
    return rows


def _list_folds(extract_dir) -> list:
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(extract_dir, "seed-*-target-*"))):
        b = os.path.basename(p)
        s = int(b.split("-")[1]); t = int(b.split("-")[3])
        out.append((s, t))
    return out


def _levels(extract_dir, seed, target) -> list:
    import glob
    fdir = os.path.join(extract_dir, f"seed-{seed}-target-{target:03d}")
    return sorted(int(os.path.basename(d).split("-")[1])
                  for d in glob.glob(os.path.join(fdir, "level-*")) if os.path.isdir(d))


def self_check_s0(extract_dir, seed, target, *, tol=5e-3) -> dict:
    """S0 self-consistency: recomputed (no-mask) worst-domain signals from persisted logits must equal the
    extracted S0 scalars (validates domain/class alignment of the masking pipeline)."""
    edges = fixed_bin_edges(15); checks = []
    for level in _levels(extract_dir, seed, target):
        fld = load_fold_level(extract_dir, seed, target, level)
        classes = list(range(fld["logits"]["source_guard"].shape[2]))
        for ci, cm in enumerate(fld["cand_meta"]):                     # S0 = no actions on either side
            rec = recompute_candidate(fld, ci, {}, {}, edges=edges, classes=classes, with_leakage=False)
            s0 = fld["s0_scalars"][cm["model_hash"]]
            for k in ("source_guard_worst_bacc", "source_audit_worst_bacc", "source_guard_worst_nll",
                      "source_audit_worst_nll"):
                a, b = rec[k], s0.get(k)
                am = a is None or (isinstance(a, float) and a != a); bm = b is None or (isinstance(b, float) and b != b)
                ok = (am and bm) or (not am and not bm and abs(float(a) - float(b)) <= tol)
                checks.append({"level": level, "ci": ci, "key": k, "ok": bool(ok),
                               "diff": (None if (am or bm) else abs(float(a) - float(b)))})
    return {"n": len(checks), "n_ok": sum(c["ok"] for c in checks), "all_ok": all(c["ok"] for c in checks),
            "worst_diff": max((c["diff"] for c in checks if c["diff"] is not None), default=0.0)}
