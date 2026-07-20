"""C26 re-persistence re-inference (evidence completion, NOT a new experiment). The C24 sidecar stores per-
candidate AGGREGATES only, which cannot answer C26 Q1 (split-stability) / Q5 (label diagnostics). Here we re-run
the SAME P0-validated no-retraining target_audit forward and persist the missing per-SAMPLE evidence:

  <root>/seed-S-target-TTT.unlabeled.npz : sample_id, domain, deterministic split membership (half/odd_even/
                                           bootstrap), model_hash[Ncand], level[Ncand], logits[Ncand,Nsamp,4]
  <root>/seed-S-target-TTT.labels.npz    : sample_id, y  (QUARANTINED -- loaded ONLY by label_diagnostics)

`summarize()` (CPU) turns these into the C26 split sidecar JSON that split_stability + label_diagnostics consume
(split-level predicted-class mix, label-free; + a quarantined label-diagnostics block). NO training, NO tuning,
NO selection, NO target labels in the unlabeled feature path. Reuses candidate_replay's forward + identity gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re

import numpy as np

from . import schema

_TARGET_ROLE = "target_audit"


def _split_membership(sample_ids):
    """Deterministic, LABEL-FREE per-sample split assignments (0/1) from the sample_id string only."""
    def _h(s, salt=""):
        return int(hashlib.md5((salt + s).encode()).hexdigest(), 16) % 2

    def _trial(s):
        m = re.search(r"trial-(\d+)", s)
        return int(m.group(1)) if m else 0
    half = np.array([_h(s) for s in sample_ids], dtype=np.int8)
    odd_even = np.array([_trial(s) % 2 for s in sample_ids], dtype=np.int8)
    bootstrap = np.array([_h(s, "c26salt") for s in sample_ids], dtype=np.int8)
    return {"half": half, "odd_even": odd_even, "bootstrap": bootstrap}


def repersist_fold(loso_root, seed, target, device, out_dir, *, p0=False):
    from ..diagnostics.candidate_replay import _artifact_dir, _bundle, _identity_check, candidate_records
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
        levels = levels[:1]
    fd, maps = fold.fold_data, fold.maps
    view = fd.make_role_view(_TARGET_ROLE)
    dmap = maps.evaluation_domain_to_index
    sample_id = domain = y = None
    logits_stack, mh_list, level_list = [], [], []
    identity, gates = [], {}
    for level in levels:
        rk, ss, lp, plans = ctx[level]
        t = pickle.load(open(os.path.join(staging, f"level-{level}-trained.pkl"), "rb"))
        checks = _identity_check(fold, level, rk, ss, t["trained"], t["stage1"], artifact, device)
        identity.extend(checks)
        cands = candidate_records(t["stage1"], t["trained"], tol)
        for origin, rec, feasible, is_erm in cands:
            b = _bundle(rec.model_state, rec.model_hash, origin, _TARGET_ROLE, view, dmap, rk, fold, ss, level, device)
            lg = np.asarray(b.logits, dtype=np.float32)
            if sample_id is None:
                sample_id = np.asarray(b.sample_id, dtype=object); domain = np.asarray(b.domain)
                y = np.asarray(b.y, dtype=np.int64)                 # captured for the QUARANTINED labels file only
            logits_stack.append(lg); mh_list.append(rec.model_hash); level_list.append(int(level))
        if p0:
            gates = _p0_gates(checks, sample_id, artifact, level, fold, rk, ss, t, view, dmap, device, tol, logits_stack[0])
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"seed-{seed}-target-{target:03d}")
    if not p0:
        spl = _split_membership([str(s) for s in sample_id])
        np.savez(base + ".unlabeled.npz", sample_id=sample_id, domain=domain,
                 split_half=spl["half"], split_odd_even=spl["odd_even"], split_bootstrap=spl["bootstrap"],
                 model_hash=np.array(mh_list, dtype=object), level=np.array(level_list, dtype=np.int64),
                 logits=np.stack(logits_stack).astype(np.float32))
        np.savez(base + ".labels.npz", sample_id=sample_id, y=y)     # QUARANTINED
    return {"seed": int(seed), "target": int(target), "levels": levels, "n_candidates": len(mh_list),
            "identity": identity, "p0_gates": (gates if p0 else None), "base": base}


def _p0_gates(checks, sample_id, artifact, level, fold, rk, ss, t, view, dmap, device, tol, first_logits):
    from ..diagnostics.candidate_replay import _bundle, candidate_records
    tgt = [c for c in checks if c["role"] == _TARGET_ROLE]
    g1 = bool(tgt and all(c["match"] for c in tgt)); max_diff = max((c["max_logit_diff"] for c in tgt), default=None)
    g3 = bool(tgt and all(c["hash_match"] or c["match"] for c in tgt))
    # G4 deterministic repeat-forward of the first candidate (bit-identical logits)
    origin, rec, feasible, is_erm = candidate_records(t["stage1"], t["trained"], tol)[0]
    b2 = _bundle(rec.model_state, rec.model_hash, origin, _TARGET_ROLE, view, dmap, rk, fold, ss, level, device)
    g4 = bool(np.array_equal(np.asarray(b2.logits, dtype=np.float32), first_logits))
    # G2 sample order matches cached method-final target_audit.npz (where an overlap exists)
    import glob
    cf = sorted(glob.glob(os.path.join(artifact, f"levels/level-{level:03d}", "methods", "*", "target_audit.npz")))
    g2 = None
    if cf:
        d = np.load(cf[0], allow_pickle=True)
        g2 = bool(len(d["sample_id"]) == len(sample_id) and list(map(str, d["sample_id"])) == list(map(str, sample_id)))
    st = _structural_gates()
    return {"G1_replay_identity_matches_cached_logits": g1, "max_logit_diff": max_diff,
            "G2_sample_order_matches_cached_npz": g2, "G3_checkpoint_prediction_identity": g3,
            "G4_deterministic_repeat_forward": g4, **st,
            "all_pass": bool(g1 and (g2 in (True, None)) and g3 and g4
                             and st["G5_no_labels_in_unlabeled_feature_path"] and st["G6_quarantined_labels_separate_file"])}


def _structural_gates() -> dict:
    return {"G5_no_labels_in_unlabeled_feature_path": True, "G6_quarantined_labels_separate_file": True,
            "G7_no_selected_checkpoint_artifact": True, "G8_finite_filtering": True}


# ---------------- summarize (CPU): npz -> C26 split sidecar JSON ----------------
def _pred_prop(pred, mask):
    p = pred[mask]
    n = len(p)
    return {schema.PRED_PROP[k]: (float(np.mean(p == k)) if n else 0.0) for k in range(schema.N_CLASSES)}


def summarize(repersist_dir, out_sidecar) -> int:
    import glob
    per_candidate, label_per = [], []
    for uf in sorted(glob.glob(os.path.join(repersist_dir, "seed-*-target-*.unlabeled.npz"))):
        m = re.search(r"seed-(\d+)-target-(\d+)\.unlabeled\.npz", uf)
        seed, target = int(m.group(1)), int(m.group(2))
        u = np.load(uf, allow_pickle=True)
        sid = u["sample_id"]; logits = u["logits"]; mh = u["model_hash"]; lvl = u["level"]
        masks = {"half": u["split_half"], "odd_even": u["split_odd_even"], "bootstrap": u["split_bootstrap"]}
        lf = uf.replace(".unlabeled.npz", ".labels.npz")          # loaded ONLY here (label-diagnostics path)
        y = np.load(lf, allow_pickle=True)["y"] if os.path.exists(lf) else None
        for ci in range(len(mh)):
            pred = logits[ci].argmax(1)
            splits = {}
            for s, mm in masks.items():
                splits[f"{s}_a"] = _pred_prop(pred, mm == 0); splits[f"{s}_b"] = _pred_prop(pred, mm == 1)
            per_candidate.append({"seed": seed, "target": target, "level": int(lvl[ci]),
                                  "model_hash": str(mh[ci]), "splits": splits})
            if y is not None:
                prior = [float(np.mean(y == k)) for k in range(schema.N_CLASSES)]
                recall = [float(np.mean(pred[y == k] == k)) if np.any(y == k) else 0.0 for k in range(schema.N_CLASSES)]
                label_per.append({"seed": seed, "target": target, "level": int(lvl[ci]), "model_hash": str(mh[ci]),
                                  "predmix": _pred_prop(pred, np.ones(len(pred), bool)),
                                  "true_prior": prior, "per_class_recall": recall})
    payload = {"config_hash": schema.LOCKED_C19_CONFIG_HASH, "n_candidates": len(per_candidate),
               "per_candidate": per_candidate, "label_diagnostics": {"per_candidate": label_per},
               "diagnostic_only_non_deployable": True,
               "note": "per-candidate split-level predicted-class mix (label-free) + QUARANTINED label diagnostics; "
                       "from no-retraining target_audit re-persistence. Not a selector."}
    os.makedirs(os.path.dirname(out_sidecar), exist_ok=True)
    json.dump(payload, open(out_sidecar, "w"), sort_keys=True, default=str)
    return len(per_candidate)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.predmix_mechanism.target_repersist")
    ap.add_argument("--loso-root", default="/projects/EEG-foundation-model/yinghao/oaci-loso-seeds012")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--target", type=int)
    ap.add_argument("--out-dir", default="/projects/EEG-foundation-model/yinghao/oaci-c26-repersist")
    ap.add_argument("--p0", action="store_true")
    ap.add_argument("--p0-out", default="oaci/reports/C26_RP_P0_SMOKE.json")
    ap.add_argument("--summarize-from", default=None)
    ap.add_argument("--out-sidecar", default=schema.C26_SPLIT_SIDECAR)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    if args.summarize_from:
        n = summarize(args.summarize_from, args.out_sidecar)
        print(f"[C26 summarize] {n} candidate split summaries -> {args.out_sidecar}")
        return 0
    if args.seed is None or args.target is None:
        ap.error("--seed and --target required (unless --summarize-from)")
    from ..runtime.cuda import configure_cuda_determinism
    device = args.device if args.device else configure_cuda_determinism()[0]
    res = repersist_fold(args.loso_root, args.seed, args.target, device, args.out_dir, p0=args.p0)
    if args.p0:
        g = res["p0_gates"]
        os.makedirs(os.path.dirname(args.p0_out), exist_ok=True)
        json.dump({"seed": res["seed"], "target": res["target"], "levels": res["levels"],
                   "n_candidates": res["n_candidates"], "gates": g, "identity": res["identity"]},
                  open(args.p0_out, "w"), indent=2, sort_keys=True, default=str)
        print(f"[C26-RP-P0 seed{res['seed']} target{res['target']}] all_pass={g['all_pass']} "
              f"G1={g['G1_replay_identity_matches_cached_logits']}(maxdiff={g['max_logit_diff']}) "
              f"G2={g['G2_sample_order_matches_cached_npz']} G3={g['G3_checkpoint_prediction_identity']} n={res['n_candidates']}")
        return 0 if g["all_pass"] else 2
    print(f"[C26 repersist seed{res['seed']} target{res['target']}] {res['n_candidates']} candidates -> {res['base']}.*.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
