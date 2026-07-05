"""C18-P GPU stage — re-inference of the C17 CANDIDATE checkpoints (NO retraining). For each C8 fold/
level, reload the staging fold + trained trajectories and forward every candidate (ERM + feasible OACI
trajectory) on the source roles, PERSISTING the per-sample source evidence C17 never saved: per-unit
source_guard/source_audit logits (+ raw domain/class/group/id) and per-candidate frozen Z-features for the
two leakage designs. Forwarding is MASK-INDEPENDENT (the model's per-sample output does not depend on which
cells count), so we forward once on GPU here and do ALL S0-S7 mask recompute on CPU downstream.

Reuses oaci.diagnostics.candidate_replay end-to-end so the SELECTED checkpoints reproduce the stored artifact
prediction hashes (identity gate) and the S0 (no-mask) recomputed scalars reproduce the persisted C10 atlas.
This module ADDS persistence + an S0-vs-C10 identity comparison; it introduces no new training or objective.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import numpy as np

from ..diagnostics.candidate_replay import (_artifact_dir, _body, _bundle, _identity_check, candidate_records)
from ..eval.calibration import fixed_bin_edges
from ..leakage.estimate import estimate_extractable_leakage
from ..runner.audit import build_training_data_for_design
from ..runner.features import extract_frozen_features
from ..runner.metrics import evaluate_prediction_bundle
from ..runner.staged_fold import _level_contexts, load_phase_a_fold
from ..train.rng import derive_seed
from . import schema

_ROLES = ("source_guard", "source_audit", "target_audit")
_PREFIX = {"source_guard": "source_guard", "source_audit": "source_audit", "target_audit": "target"}
# roles whose per-unit predictions we persist for masked source-signal recompute (source only)
_PERSIST_ROLES = ("source_guard", "source_audit")


def _inv(domain_map) -> dict:
    return {int(v): str(k) for k, v in domain_map.items()}


def _extract_feats(ms, mh, training_data, design, fold, rk, seed_ns, extra, device):
    """Extract frozen Z on a leakage design's population (mirrors candidate_replay._leak_point's feature
    extraction EXACTLY so S0 leakage is byte-reproducible), returning the FeatureArtifact for persistence."""
    parts = (rk.run_key_hash, *extra)
    fseed = derive_seed(rk.model_seed, seed_ns, *parts)
    return extract_frozen_features(ms, mh, fold.model_factory(), training_data, design, factory_seed=fseed,
                                   chunk_size=fold.execution_config.feature_chunk_size, device=device)


def _c10_row(c10_dir, seed, target, level, model_hash):
    p = os.path.join(c10_dir, f"seed-{seed}-target-{target:03d}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    for c in d["levels"][str(level)]["candidates"]:
        if c["model_hash"] == model_hash:
            return c
    return None


def extract_level(fold, level, rk, ss, lp, plans, trained, stage1, device, *, ece_bins, tol,
                  out_dir, seed, target, c10_dir):
    fd, maps = fold.fold_data, fold.maps
    edges = fixed_bin_edges(ece_bins)
    views = {"source_guard": fd.make_role_view("source_guard", ss.source_train_idx),
             "source_audit": fd.make_role_view("source_audit"),
             "target_audit": fd.make_role_view("target_audit")}
    dmap = {"source_guard": maps.source_domain_to_index, "source_audit": maps.evaluation_domain_to_index,
            "target_audit": maps.evaluation_domain_to_index}
    sa = fold.fold_scope.source_audit
    sa_ok = sa.status == "estimable" and sa.fold_plan is not None
    audit_td = build_training_data_for_design(fd, sa.design) if sa_ok else None
    sel_ok = plans.selection_status == "estimable" and plans.selection_fold_plan is not None

    cands = candidate_records(stage1, trained, tol)
    ldir = os.path.join(out_dir, f"seed-{seed}-target-{target:03d}", f"level-{level}")
    os.makedirs(ldir, exist_ok=True)
    # persist base support graphs (source for guard-signals + selection leakage; audit for audit-signals)
    np.savez(os.path.join(ldir, "support-source.npz"), **_dump_support_graph(ss.support_graph))
    if sa_ok:
        np.savez(os.path.join(ldir, "support-audit.npz"), **_dump_support_graph(sa.support_graph))
    cc = fold.execution_config.critic
    json.dump({"critic": {"capacities": [int(c) for c in cc.capacities], "l2_C": float(cc.l2_C),
                          "max_iter": int(cc.max_iter), "prob_floor": float(cc.prob_floor),
                          "feature_seed_base": int(cc.feature_seed_base)},
               "selection_n_folds": (int(plans.selection_fold_plan.n_folds) if sel_ok else None),
               "audit_n_folds": (int(sa.fold_plan.n_folds) if sa_ok else None),
               "sel_ok": bool(sel_ok), "sa_ok": bool(sa_ok)},
              open(os.path.join(ldir, "config.json"), "w"), sort_keys=True)

    unit_meta = {}                                   # role -> per-unit domain_raw/y/group/sample_id (shared)
    logits_stack = {r: [] for r in _PERSIST_ROLES}   # role -> [n_cand][Nu,C]
    featz = {"selection": {"Z": [], "meta": None}, "audit": {"Z": [], "meta": None}}
    cand_meta, scalar_rows, s0_cmp = [], [], []

    for ci, (origin, rec, feasible, is_erm) in enumerate(cands):
        mh, ms = rec.model_hash, rec.model_state
        row = {"origin": origin, "model_hash": mh, "epoch": int(rec.epoch), "lambda": float(rec.lam),
               "R_src": float(rec.R_src), "balanced_err": float(rec.balanced_err),
               "train_surrogate": float(rec.train_surrogate), "feasible": bool(feasible), "is_erm": bool(is_erm)}
        bundles = {}
        for role in _ROLES:
            b = _bundle(ms, mh, origin, role, views[role], dmap[role], rk, fold, ss, level, device)
            bundles[role] = b
            m = evaluate_prediction_bundle(b, bin_edges=edges)
            pre = _PREFIX[role]
            row[f"{pre}_worst_bacc"] = m.worst_domain_reference_bacc
            row[f"{pre}_worst_nll"] = m.worst_domain_nll
            row[f"{pre}_worst_ece"] = m.worst_domain_ece
        # persist per-unit source predictions (mask-independent); meta stored once per role
        for role in _PERSIST_ROLES:
            b = bundles[role]
            inv = _inv(dmap[role])
            if role not in unit_meta:
                unit_meta[role] = {"domain_raw": np.array([inv[int(d)] for d in b.domain], dtype=object),
                                   "y": np.asarray(b.y, dtype=np.int64),
                                   "group": np.asarray(b.group, dtype=object),
                                   "sample_id": np.asarray(b.sample_id, dtype=object)}
            logits_stack[role].append(np.asarray(b.logits, dtype=np.float64))
        # leakage designs: extract + persist Z (for masked recompute), and compute S0 leakage point
        if sel_ok:
            f = _extract_feats(ms, mh, lp.training_data, plans.selection_design, fold, rk,
                               "selection_feature", (), device)
            row["selection_leakage_point"] = float(estimate_extractable_leakage(
                f.features, ss.support_graph, plans.selection_fold_plan, fold.execution_config.critic)["L_abs"])
            featz["selection"]["Z"].append(np.asarray(f.features.Z, dtype=np.float32))
            if featz["selection"]["meta"] is None:
                featz["selection"]["meta"] = _feat_meta(f.features)
        else:
            row["selection_leakage_point"] = None
        if sa_ok:
            f = _extract_feats(ms, mh, audit_td, sa.design, fold, rk, "audit_feature_factory", (mh,), device)
            row["audit_leakage_point"] = float(estimate_extractable_leakage(
                f.features, sa.support_graph, sa.fold_plan, fold.execution_config.critic)["L_abs"])
            featz["audit"]["Z"].append(np.asarray(f.features.Z, dtype=np.float32))
            if featz["audit"]["meta"] is None:
                featz["audit"]["meta"] = _feat_meta(f.features)
        else:
            row["audit_leakage_point"] = None

        cand_meta.append({"index": ci, "origin": origin, "model_hash": mh, "epoch": int(rec.epoch),
                          "lambda": float(rec.lam), "R_src": float(rec.R_src),
                          "balanced_err": float(rec.balanced_err), "train_surrogate": float(rec.train_surrogate),
                          "feasible": bool(feasible), "is_erm": bool(is_erm)})
        scalar_rows.append(row)
        # S0 identity comparison vs persisted C10 (feasible OACI + ERM; C10 stored them)
        c10 = _c10_row(c10_dir, seed, target, level, mh) if c10_dir else None
        if c10 is not None:
            s0_cmp.append(_compare_s0(row, c10))

    # ---- persist ----
    for role in _PERSIST_ROLES:
        um = unit_meta[role]
        np.savez(os.path.join(ldir, f"units-{role}.npz"), domain_raw=um["domain_raw"], y=um["y"],
                 group=um["group"], sample_id=um["sample_id"])
        np.save(os.path.join(ldir, f"logits-{role}.npy"), np.stack(logits_stack[role]))
    for dz, blk in featz.items():
        if blk["meta"] is not None:
            mm = blk["meta"]
            np.savez(os.path.join(ldir, f"featz-{dz}.npz"), Z=np.stack(blk["Z"]), y=mm["y"], d=mm["d"],
                     group=mm["group"], sample_id=mm["sample_id"])
    json.dump(cand_meta, open(os.path.join(ldir, "cand_meta.json"), "w"), sort_keys=True)
    # extracted S0 per-candidate source signals (authoritative; match C10) for the G2 identity probe
    json.dump(scalar_rows, open(os.path.join(ldir, "s0_scalars.json"), "w"), sort_keys=True)
    return scalar_rows, s0_cmp


def _dump_support_graph(sg) -> dict:
    return {"counts": np.asarray(sg.counts, dtype=np.int64), "cell_mass": np.asarray(sg.cell_mass, dtype=np.float64),
            "m": np.int64(sg.m), "reference_prior": np.asarray(sg.reference_prior, dtype=np.float64),
            "domain_names": np.array([str(x) for x in sg.domain_names], dtype=object),
            "class_names": np.array([str(x) for x in sg.class_names], dtype=object),
            "eligible": np.asarray(sg.eligible, dtype=bool)}


def _feat_meta(fr) -> dict:
    return {"y": np.asarray(fr.y, dtype=np.int64), "d": np.array([str(x) for x in fr.d], dtype=object),
            "group": np.array([str(x) for x in fr.group], dtype=object),
            "sample_id": np.array([str(x) for x in fr.sample_id], dtype=object)}


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and x != x)          # None OR NaN both count as "not estimated"


def _compare_s0(row, c10) -> dict:
    """Recomputed S0 (no-mask) scalars vs persisted C10; classify each within predeclared tolerance. A NaN/None
    worst-domain metric (a domain legitimately missing a class) matches C10 iff C10 is ALSO NaN/None."""
    checks = {}
    for k, tol in (("source_guard_worst_bacc", schema.S0_BACC_TOL), ("source_audit_worst_bacc", schema.S0_BACC_TOL),
                   ("source_guard_worst_nll", schema.S0_NLL_TOL), ("source_audit_worst_nll", schema.S0_NLL_TOL),
                   ("R_src", schema.S0_NLL_TOL), ("balanced_err", schema.S0_BACC_TOL),
                   ("selection_leakage_point", schema.S0_LEAK_TOL), ("audit_leakage_point", schema.S0_LEAK_TOL)):
        a, b = row.get(k), c10.get(k)
        if _missing(a) or _missing(b):
            checks[k] = {"got": (None if _missing(a) else float(a)), "c10": (None if _missing(b) else float(b)),
                         "diff": None, "ok": _missing(a) and _missing(b)}
        else:
            d = abs(float(a) - float(b))
            checks[k] = {"got": float(a), "c10": float(b), "diff": d, "ok": d <= tol}
    return {"model_hash": row["model_hash"], "is_erm": row["is_erm"], "checks": checks,
            "all_ok": all(v["ok"] for v in checks.values())}


def extract_fold(loso_root, seed, target, device, out_dir, *, require_identity=True, c10_dir=None):
    staging = os.path.join(loso_root, f"seed-{seed}", f"target-{target:03d}", "staging")
    artifact = _artifact_dir(loso_root, seed, target)
    meta = json.load(open(os.path.join(staging, "phase_a.json")))
    dataset_id, model_seed = meta["dataset_id"], meta["model_seed"]
    fold = load_phase_a_fold(staging)
    ece_bins = fold.execution_config.ece_bins
    tol = float(fold.execution_config.engine_template.numerical_tol)
    ctx = {L: (rk, ss, lp, plans) for L, rk, ss, lp, plans in _level_contexts(fold, model_seed, dataset_id)}
    out = {"seed": int(seed), "target": int(target), "artifact_dir": artifact, "loso_root": loso_root,
           "levels": {}, "identity": [], "s0_vs_c10": []}
    for level in sorted(ctx):
        rk, ss, lp, plans = ctx[level]
        t = pickle.load(open(os.path.join(staging, f"level-{level}-trained.pkl"), "rb"))
        checks = _identity_check(fold, level, rk, ss, t["trained"], t["stage1"], artifact, device)
        out["identity"].extend(checks)
        if require_identity and not all(c["match"] for c in checks):
            bad = [f"L{c['level']} {c['method']}/{c['role']}" for c in checks if not c["match"]]
            raise RuntimeError(f"IDENTITY FAIL seed-{seed}/target-{target:03d}: {bad} — replay must not be interpreted")
        rows, s0 = extract_level(fold, level, rk, ss, lp, plans, t["trained"], t["stage1"], device,
                                 ece_bins=ece_bins, tol=tol, out_dir=out_dir, seed=seed, target=target, c10_dir=c10_dir)
        out["levels"][str(level)] = {"n_candidates": len(rows)}
        out["s0_vs_c10"].extend(s0)
    fdir = os.path.join(out_dir, f"seed-{seed}-target-{target:03d}")
    os.makedirs(fdir, exist_ok=True)
    json.dump(out, open(os.path.join(fdir, "extract_manifest.json"), "w"), sort_keys=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.support_stress.replay_extract")
    ap.add_argument("--loso-root", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--c10-dir", default=None, help="C10 replay dir for the S0-vs-C10 identity comparison")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-require-identity", action="store_true")
    args = ap.parse_args(argv)
    from ..runtime.cuda import configure_cuda_determinism
    device = args.device if args.device else configure_cuda_determinism()[0]
    res = extract_fold(args.loso_root, args.seed, args.target, device, args.out_dir,
                       require_identity=not args.no_require_identity, c10_dir=args.c10_dir)
    nid = sum(1 for c in res["identity"] if c["match"])
    ncmp = len(res["s0_vs_c10"]); nok = sum(1 for c in res["s0_vs_c10"] if c["all_ok"])
    print(f"identity {nid}/{len(res['identity'])} match; S0-vs-C10 {nok}/{ncmp} candidates within tol; "
          f"levels={list(res['levels'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
