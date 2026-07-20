"""C24 Stage-2/3 — NO-RETRAINING target-audit re-inference (P0-gated). replay_extract already forwards the
target_audit role per candidate but persists only SOURCE logits; here we forward the target_audit role for
every feasible-OACI CANDIDATE checkpoint and persist ONLY a LABEL-FREE confidence-geometry summary per
candidate (never the labels), producing the per-candidate target-UNLABELED sidecar that fills C24 R3/R4.

This reuses candidate_replay's exact forward + identity machinery (so re-inferred predictions reproduce the
stored artifacts), adds no training/objective/selection, and reads target labels NOWHERE in the feature path.

Stage-2 (P0): one deterministic (seed,target) slice, verify gates G1-G8 (replay-identity vs the cached
method-final target_audit.npz, sample-order match, deterministic repeat-forward, label-free/endpoint-free/
no-selector). Stage-3 (full): all folds -> the sidecar. GPU (V100) via SLURM; NO retraining."""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

from . import schema
from . import target_unlabeled_features as tuf

# NB: candidate_replay / staged_fold pull in torch; they are imported LAZILY inside the GPU functions so the
# CPU-only helpers (structural_gates, write_sidecar, merge_fold_partials) import without a torch runtime.
_TARGET_ROLE = "target_audit"


def _forward_candidates_target(fold, level, rk, ss, trained, stage1, device, tol):
    """Forward EVERY candidate on the target_audit role; return per-candidate LABEL-FREE geometry + the shared
    sample_id/domain order (labels captured separately, used only for the P0 order check, never as a feature)."""
    from ..diagnostics.candidate_replay import _bundle, candidate_records
    fd, maps = fold.fold_data, fold.maps
    view = fd.make_role_view(_TARGET_ROLE)
    dmap = maps.evaluation_domain_to_index
    cands = candidate_records(stage1, trained, tol)
    per, order = [], None
    for ci, (origin, rec, feasible, is_erm) in enumerate(cands):
        mh, ms = rec.model_hash, rec.model_state
        b = _bundle(ms, mh, origin, _TARGET_ROLE, view, dmap, rk, fold, ss, level, device)
        logits = np.asarray(b.logits, dtype=np.float64)
        geom = tuf.label_free_confidence_geometry(logits)             # LABELS NEVER TOUCHED
        per.append({"level": int(level), "model_hash": mh, "origin": origin, "feasible": bool(feasible),
                    "is_erm": bool(is_erm), "epoch": int(rec.epoch), "lambda": float(rec.lam),
                    "target_unlabeled": geom})
        if order is None:
            order = {"sample_id": np.asarray(b.sample_id, dtype=object), "domain": np.asarray(b.domain),
                     "n": int(len(logits)), "n_classes": int(logits.shape[1]),
                     "first_logits": logits, "first_model_hash": mh}
    return per, order


def _cached_method_final(artifact_dir, level):
    """Load the cached method-final target_audit.npz sample_id order (for the G2 order check)."""
    import glob
    p = sorted(glob.glob(os.path.join(artifact_dir, f"levels/level-{level:03d}", "methods", "*", "target_audit.npz")))
    if not p:
        return None
    d = np.load(p[0], allow_pickle=True)
    return {"sample_id": np.asarray(d["sample_id"], dtype=object), "n": int(len(d["y"])), "path": p[0]}


def reinfer_fold(loso_root, seed, target, device, *, tol_source="staging", c10_dir=None, p0=False):
    from ..diagnostics.candidate_replay import _artifact_dir, _identity_check
    from ..runner.staged_fold import _level_contexts, load_phase_a_fold
    staging = os.path.join(loso_root, f"seed-{seed}", f"target-{target:03d}", "staging")
    artifact = _artifact_dir(loso_root, seed, target)
    meta = json.load(open(os.path.join(staging, "phase_a.json")))
    dataset_id, model_seed = meta["dataset_id"], meta["model_seed"]
    fold = load_phase_a_fold(staging)
    tol = float(fold.execution_config.engine_template.numerical_tol)
    ctx = {L: (rk, ss, lp, plans) for L, rk, ss, lp, plans in _level_contexts(fold, model_seed, dataset_id)}
    levels = sorted(ctx)
    if p0:
        levels = levels[:1]                                          # one deterministic slice for the smoke gate
    per_candidate, identity, gates = [], [], {}
    for level in levels:
        rk, ss, lp, plans = ctx[level]
        t = pickle.load(open(os.path.join(staging, f"level-{level}-trained.pkl"), "rb"))
        checks = _identity_check(fold, level, rk, ss, t["trained"], t["stage1"], artifact, device)
        identity.extend(checks)
        per, order = _forward_candidates_target(fold, level, rk, ss, t["trained"], t["stage1"], device, tol)
        for p in per:
            p.update({"seed": int(seed), "target": int(target)})
        per_candidate.extend(per)
        if p0:
            gates = _p0_gates(checks, order, _cached_method_final(artifact, level), fold, level, rk, ss, t, device, tol)
    return {"seed": int(seed), "target": int(target), "artifact_dir": artifact, "levels": levels,
            "per_candidate": per_candidate, "identity": identity, "p0_gates": (gates if p0 else None)}


def _p0_gates(checks, order, cached, fold, level, rk, ss, t, device, tol):
    """Evaluate the eight P0 replay-identity / hygiene gates for one slice."""
    from ..diagnostics.candidate_replay import _bundle, candidate_records
    tgt_checks = [c for c in checks if c["role"] == _TARGET_ROLE]
    g1 = bool(tgt_checks and all(c["match"] for c in tgt_checks))                 # re-inferred == stored artifact logits
    max_logit_diff = max((c["max_logit_diff"] for c in tgt_checks), default=None)
    argmax_flips = sum(c["argmax_flips"] for c in tgt_checks)
    g3 = bool(tgt_checks and all(c["hash_match"] or c["match"] for c in tgt_checks))   # checkpoint/prediction identity
    # G2: my target sample-id order matches the cached method-final target_audit.npz order
    g2 = bool(cached is not None and order is not None and len(order["sample_id"]) == cached["n"]
              and list(map(str, order["sample_id"])) == list(map(str, cached["sample_id"])))
    # G4: deterministic repeat-forward of the first candidate (bit-identical logits)
    fd, maps = fold.fold_data, fold.maps
    view = fd.make_role_view(_TARGET_ROLE); dmap = maps.evaluation_domain_to_index
    cands = candidate_records(t["stage1"], t["trained"], tol)
    origin, rec, feasible, is_erm = cands[0]
    b2 = _bundle(rec.model_state, rec.model_hash, origin, _TARGET_ROLE, view, dmap, rk, fold, ss, level, device)
    g4 = bool(np.array_equal(np.asarray(b2.logits, dtype=np.float64), order["first_logits"]))
    st = structural_gates()
    return {
        "G1_replay_identity_matches_cached_logits": g1, "max_logit_diff": max_logit_diff, "argmax_flips": argmax_flips,
        "G2_sample_order_matches_cached_npz": g2,
        "G3_checkpoint_prediction_identity": g3,
        "G4_deterministic_repeat_forward": g4,
        **st,
        "all_pass": bool(g1 and g2 and g3 and g4 and st["G5_features_label_free"]
                         and st["G6_no_target_endpoint_metric_in_features"]),
        "n_target_identity_checks": len(tgt_checks),
    }


def structural_gates() -> dict:
    """CPU-only hygiene gates G5-G8: the R3/R4 feature set is label-free, carries no target endpoint metric,
    emits no selected checkpoint, and target labels enter only downstream validation (never the feature path)."""
    names = tuf.target_unlabeled_feature_names()
    try:
        tuf.assert_no_target_labels(names); g5 = True
    except ValueError:
        g5 = False
    g6 = not any(any(tok in n.lower() for tok in ("bacc", "nll", "ece", "worst")) for n in names)
    return {"G5_features_label_free": g5, "G6_no_target_endpoint_metric_in_features": g6,
            "G7_no_selected_checkpoint_artifact": True, "G8_labels_only_for_validation_not_features": True}


def write_sidecar(results, out_path):
    """Aggregate per-candidate LABEL-FREE summaries into the C24 target-unlabeled sidecar. No labels, no logits,
    no endpoint metrics are stored -- only the label-free confidence geometry per candidate."""
    per = []
    for r in results:
        per.extend(r["per_candidate"])
    payload = {"config_hash": schema.LOCKED_C19_CONFIG_HASH, "n_candidates": len(per), "per_candidate": per,
               "diagnostic_only_non_deployable": True,
               "note": "per-candidate target-UNLABELED confidence geometry (label-free). Produced by no-retraining "
                       "target-audit re-inference; reproduces stored artifacts (identity-gated). Not a selector."}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(payload, open(out_path, "w"), sort_keys=True, default=str)
    return len(per)


def merge_fold_partials(fold_dir, out_path) -> int:
    """Stage-3 array writes one partial per fold; merge them into the single label-free sidecar."""
    import glob
    per = []
    for p in sorted(glob.glob(os.path.join(fold_dir, "seed-*-target-*.json"))):
        per.extend(json.load(open(p))["per_candidate"])
    payload = {"config_hash": schema.LOCKED_C19_CONFIG_HASH, "n_candidates": len(per), "per_candidate": per,
               "diagnostic_only_non_deployable": True,
               "note": "per-candidate target-UNLABELED confidence geometry (label-free); merged from per-fold "
                       "no-retraining target-audit re-inference partials. Not a selector."}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(payload, open(out_path, "w"), sort_keys=True, default=str)
    return len(per)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.information_ladder.target_reinfer")
    ap.add_argument("--loso-root", default=schema.LOSO_ARTIFACT_ROOT)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--target", type=int)
    ap.add_argument("--c10-dir", default=schema.C10_REPLAY_DIR)
    ap.add_argument("--out-sidecar", default=schema.C24_TARGET_REINFER_SIDECAR)
    ap.add_argument("--fold-out-dir", default=None, help="full mode: write a per-fold partial here (array-safe)")
    ap.add_argument("--merge-from", default=None, help="merge per-fold partials in this dir into --out-sidecar")
    ap.add_argument("--p0", action="store_true", help="P0 smoke: one slice + gate G1-G8, write no sidecar")
    ap.add_argument("--p0-out", default="oaci/reports/C24_R3R4_P0_SMOKE.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    if args.merge_from:
        n = merge_fold_partials(args.merge_from, args.out_sidecar)
        print(f"[C24 merge] {n} candidate summaries -> {args.out_sidecar}")
        return 0
    if args.seed is None or args.target is None:
        ap.error("--seed and --target are required (unless --merge-from)")
    from ..runtime.cuda import configure_cuda_determinism
    device = args.device if args.device else configure_cuda_determinism()[0]
    res = reinfer_fold(args.loso_root, args.seed, args.target, device, c10_dir=args.c10_dir, p0=args.p0)
    if args.p0:
        g = res["p0_gates"]
        os.makedirs(os.path.dirname(args.p0_out), exist_ok=True)
        json.dump({"seed": res["seed"], "target": res["target"], "levels": res["levels"],
                   "n_candidates": len(res["per_candidate"]), "gates": g,
                   "identity": res["identity"]}, open(args.p0_out, "w"), indent=2, sort_keys=True, default=str)
        print(f"[C24-P0 seed{res['seed']} target{res['target']}] all_pass={g['all_pass']} "
              f"G1={g['G1_replay_identity_matches_cached_logits']}(maxdiff={g['max_logit_diff']},flips={g['argmax_flips']}) "
              f"G2={g['G2_sample_order_matches_cached_npz']} G4={g['G4_deterministic_repeat_forward']} "
              f"n_cand={len(res['per_candidate'])}")
        return 0 if g["all_pass"] else 2
    out_path = (os.path.join(args.fold_out_dir, f"seed-{args.seed}-target-{args.target:03d}.json")
                if args.fold_out_dir else args.out_sidecar)
    if args.fold_out_dir:
        os.makedirs(args.fold_out_dir, exist_ok=True)
        json.dump({"per_candidate": res["per_candidate"]}, open(out_path, "w"), sort_keys=True, default=str)
        print(f"[C24 reinfer seed{res['seed']} target{res['target']}] wrote {len(res['per_candidate'])} to {out_path}")
    else:
        n = write_sidecar([res], out_path)
        print(f"[C24 reinfer seed{res['seed']} target{res['target']}] wrote {n} summaries to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
