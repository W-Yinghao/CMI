"""C10b — epoch-level GPU inference replay (NO retrain / NO objective / NO schema change / NO target-in-
selection). For each C8 fold/level, reload the EXACT staging fold + trained trajectories and forward every
candidate (ERM + risk-feasible OACI trajectory) on the three roles, recomputing per-candidate source_guard /
source_audit / target worst-domain bAcc·NLL·ECE plus selection & audit leakage POINT estimates (single
deterministic grouped cross-fit — Tier 1, no bootstrap). Reuses the runner's exact predict / aggregate /
evaluate / leakage functions, so the SELECTED checkpoints reproduce the stored artifact prediction hashes.

IDENTITY GATE: replay_fold asserts the selected ERM/OACI checkpoints' source_audit + target
prediction_content_hash equal the stored C8 artifact. A mismatch raises — the replay must not be interpreted.

Target is EVALUATION-ONLY here; selection role-gating lives in selectors.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

from ..eval.calibration import fixed_bin_edges
from ..leakage.estimate import estimate_extractable_leakage
from ..runner.audit import build_training_data_for_design
from ..runner.features import extract_frozen_features
from ..runner.metrics import evaluate_prediction_bundle
from ..runner.predict import aggregate_role_to_bundle, predict_checkpoint
from ..runner.staged_fold import _level_contexts, load_phase_a_fold
from ..train.rng import derive_seed

_ROLES = ("source_guard", "source_audit", "target_audit")
# field prefix per role (target_audit -> "target" so keys match selectors.py: target_worst_bacc, etc.)
_PREFIX = {"source_guard": "source_guard", "source_audit": "source_audit", "target_audit": "target"}


def _body(p):
    d = json.load(open(p))
    return d.get("body", d)


def _artifact_dir(loso_root, seed, target):
    c = sorted(glob.glob(os.path.join(loso_root, f"seed-{seed}", f"target-{target:03d}",
                                      "artifacts", "*", "COMMITTED.json")))
    if len(c) != 1:
        raise ValueError(f"seed-{seed}/target-{target:03d}: {len(c)} committed artifacts")
    return os.path.dirname(c[0])


def candidate_records(stage1, trained, tol):
    """ERM + risk-feasible unique OACI trajectory checkpoints (dedup by model_hash). Each item:
    (origin, CheckpointRecord, feasible, is_erm)."""
    erm = trained["OACI"].train_result.erm_stage.checkpoint
    out = [("ERM", erm, True, True)]
    seen = {erm.model_hash}
    tm = trained["OACI"]
    tau = float(tm.shared_tau)
    for c in tm.train_result.trajectory:
        if c.R_src <= tau + tol and c.model_hash not in seen:
            seen.add(c.model_hash)
            out.append(("OACI", c, True, False))
    return out


def _bundle(ms, mh, method_name, role, rv, dmap, rk, fold, ss, level, device):
    fd, maps, exec_cfg = fold.fold_data, fold.maps, fold.execution_config
    fseed = derive_seed(rk.model_seed, "prediction_factory", rk.run_key_hash, role, mh)
    row = predict_checkpoint(ms, mh, fold.model_factory(), rv, factory_seed=fseed,
                             chunk_size=exec_cfg.prediction_chunk_size, device=device)
    return aggregate_role_to_bundle(
        row, rv, method_name=method_name, selected_model_hash=mh, domain_map=dmap,
        class_names=maps.class_names, model_seed=rk.model_seed,
        fold_key_hash=fold.fold_scope.fold_key.fold_key_hash, support_hash=ss.support_hash,
        split_manifest_hash=fd.split_manifest_hash, preprocess_hash=fd.preprocess_hash,
        risk_metric=exec_cfg.engine_template.metric, prob_floor=exec_cfg.prediction_prob_floor,
        deletion_level=level)


def _leak_point(ms, mh, training_data, design, support_graph, fold_plan, seed_ns, extra, fold, rk, device):
    if design is None or fold_plan is None:
        return None
    parts = (rk.run_key_hash, *extra)
    fseed = derive_seed(rk.model_seed, seed_ns, *parts)
    feat = extract_frozen_features(ms, mh, fold.model_factory(), training_data, design,
                                   factory_seed=fseed, chunk_size=fold.execution_config.feature_chunk_size,
                                   device=device)
    est = estimate_extractable_leakage(feat.features, support_graph, fold_plan, fold.execution_config.critic)
    return float(est["L_abs"])


def replay_level(fold, level, rk, ss, lp, plans, trained, stage1, device, *, ece_bins, tol):
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

    rows = []
    for origin, rec, feasible, is_erm in candidate_records(stage1, trained, tol):
        mh, ms = rec.model_hash, rec.model_state
        r = {"origin": origin, "model_hash": mh, "epoch": int(rec.epoch), "lambda": float(rec.lam),
             "R_src": float(rec.R_src), "balanced_err": float(rec.balanced_err),
             "train_surrogate": float(rec.train_surrogate), "feasible": bool(feasible), "is_erm": bool(is_erm)}
        for role in _ROLES:
            pre = _PREFIX[role]
            b = _bundle(ms, mh, origin, role, views[role], dmap[role], rk, fold, ss, level, device)
            m = evaluate_prediction_bundle(b, bin_edges=edges)
            r[f"{pre}_worst_bacc"] = m.worst_domain_reference_bacc
            r[f"{pre}_worst_nll"] = m.worst_domain_nll
            r[f"{pre}_worst_ece"] = m.worst_domain_ece
            r[f"{pre}_pred_hash"] = b.prediction_content_hash()
        r["selection_leakage_point"] = _leak_point(
            ms, mh, lp.training_data, plans.selection_design, ss.support_graph,
            plans.selection_fold_plan, "selection_feature", (), fold, rk, device) if sel_ok else None
        r["audit_leakage_point"] = _leak_point(
            ms, mh, audit_td, (sa.design if sa_ok else None), (sa.support_graph if sa_ok else None),
            (sa.fold_plan if sa_ok else None), "audit_feature_factory", (mh,), fold, rk, device) if sa_ok else None
        rows.append(r)
    return rows


def _identity_check(fold, level, rk, ss, trained, stage1, artifact_dir, device):
    """Selected ERM/OACI on source_audit + target must reproduce the stored prediction_content_hash."""
    fd, maps = fold.fold_data, fold.maps
    mmap = {}
    for tm in trained.values():
        ec = tm.train_result.erm_stage.checkpoint
        mmap[ec.model_hash] = ec.model_state
        for c in tm.train_result.trajectory:
            mmap[c.model_hash] = c.model_state
    views = {"source_audit": fd.make_role_view("source_audit"), "target_audit": fd.make_role_view("target_audit")}
    checks = []
    for method in ("ERM", "OACI"):
        mj = _body(os.path.join(artifact_dir, f"levels/level-{level:03d}", "methods", method, "method.json"))
        sel_mh = mj["selection"]["model_hash"]
        for role in ("source_audit", "target_audit"):
            stored = _body(os.path.join(artifact_dir, f"levels/level-{level:03d}", "methods", method,
                                        f"{role}.json"))["prediction_content_hash"]
            got = _bundle(mmap[sel_mh], sel_mh, method, role, views[role],
                          maps.evaluation_domain_to_index, rk, fold, ss, level, device).prediction_content_hash()
            checks.append({"method": method, "role": role, "level": level, "match": got == stored,
                           "got": got, "stored": stored})
    return checks


def replay_fold(loso_root, seed, target, device, *, require_identity=True):
    staging = os.path.join(loso_root, f"seed-{seed}", f"target-{target:03d}", "staging")
    artifact = _artifact_dir(loso_root, seed, target)
    meta = json.load(open(os.path.join(staging, "phase_a.json")))
    dataset_id, model_seed = meta["dataset_id"], meta["model_seed"]
    fold = load_phase_a_fold(staging)
    ece_bins = fold.execution_config.ece_bins
    tol = float(fold.execution_config.engine_template.numerical_tol)
    ctx = {L: (rk, ss, lp, plans) for L, rk, ss, lp, plans in _level_contexts(fold, model_seed, dataset_id)}
    out = {"seed": int(seed), "target": int(target), "artifact_dir": artifact, "levels": {}, "identity": []}
    for level in sorted(ctx):
        rk, ss, lp, plans = ctx[level]
        t = pickle.load(open(os.path.join(staging, f"level-{level}-trained.pkl"), "rb"))
        checks = _identity_check(fold, level, rk, ss, t["trained"], t["stage1"], artifact, device)
        out["identity"].extend(checks)
        if require_identity and not all(c["match"] for c in checks):
            bad = [f"L{c['level']} {c['method']}/{c['role']}" for c in checks if not c["match"]]
            raise RuntimeError(f"IDENTITY FAIL seed-{seed}/target-{target:03d}: {bad} — replay must not be interpreted")
        selected = {m: _body(os.path.join(artifact, f"levels/level-{level:03d}", "methods", m,
                                          "method.json"))["selection"]["model_hash"] for m in ("ERM", "OACI")}
        rows = replay_level(fold, level, rk, ss, lp, plans, t["trained"], t["stage1"], device,
                            ece_bins=ece_bins, tol=tol)
        out["levels"][str(level)] = {"n_candidates": len(rows), "candidates": rows, "selected": selected}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.diagnostics.candidate_replay")
    ap.add_argument("--loso-root", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default=None, help="cuda|cpu; default = auto")
    ap.add_argument("--no-require-identity", action="store_true", help="diagnostic only; never for the real sweep")
    args = ap.parse_args(argv)
    from ..runtime.cuda import configure_cuda_determinism
    if args.device:
        device = args.device
    else:
        device, _ = configure_cuda_determinism()
    res = replay_fold(args.loso_root, args.seed, args.target, device, require_identity=not args.no_require_identity)
    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, f"seed-{args.seed}-target-{args.target:03d}.json")
    with open(p, "w") as f:
        json.dump(res, f, sort_keys=True)
    nid = sum(1 for c in res["identity"] if c["match"])
    ncand = sum(v["n_candidates"] for v in res["levels"].values())
    print(f"wrote {p}: identity {nid}/{len(res['identity'])} match, {ncand} candidate-rows across "
          f"{len(res['levels'])} levels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
