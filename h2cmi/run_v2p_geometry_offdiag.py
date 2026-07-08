"""Review-completion off-diagonal geometry stress.

Additive runner: reuses the frozen V2P source bundles and the W1.geometry
evaluation path, but applies stronger sensor-space linear perturbations than the
frozen W1 set. Outputs raw JSONL rows; no existing Wave0/W1 artifact is
overwritten.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from h2cmi.eval.harness import _embed
from h2cmi.eval.p0_eval import eval_unit_p0
from h2cmi.grid_io import append_row, require_clean_git, sha256_file, source_code_signature, stable_hash_int
from h2cmi.p0_source import ProvenanceError, get_source_p0
from h2cmi.run_v2 import PAIRS_B, build_cfg
from h2cmi.run_v2p_geometry import DIAG_OPS, K, _apply, _bacc, _coral_latent
from h2cmi.run_v2p_wave0 import _pair_units
from h2cmi.tta.class_conditional import ClassConditionalTTA


PERTS = ["rotation", "mixing", "strong_reref", "block_mixing"]


def _orthogonal(C: int, rng: np.random.Generator) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.normal(size=(C, C)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def _perturbation(kind: str, C: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "rotation":
        return _orthogonal(C, rng)
    if kind == "mixing":
        A = rng.normal(size=(C, C)).astype(np.float32)
        A /= max(np.linalg.norm(A, ord=2), 1e-6)
        return (np.eye(C, dtype=np.float32) + 0.35 * A).astype(np.float32)
    if kind == "strong_reref":
        P = np.eye(C, dtype=np.float32)
        ref = min(2, C)
        P -= np.ones((C, ref), dtype=np.float32) @ np.eye(C, dtype=np.float32)[:ref] / float(ref)
        return P
    if kind == "block_mixing":
        P = np.eye(C, dtype=np.float32)
        order = rng.permutation(C)
        for start in range(0, C, 4):
            idx = order[start:start + 4]
            if len(idx) < 2:
                continue
            Q = _orthogonal(len(idx), rng)
            P[np.ix_(idx, idx)] = Q
        return P
    raise ValueError(kind)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--seed0-root", default="results/h2cmi/v2_bundles")
    ap.add_argument("--new-root", default="results/h2cmi/p0_v2pw_bundles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", default="")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    only_subj = set(int(s) for s in args.subjects.split(",") if s) if args.subjects else None
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    commit = require_clean_git(
        allow_dirty=args.allow_dirty,
        ignore_prefixes=["results/h2cmi", args.new_root, args.seed0_root, "h2cmi/results/review_completion"],
    )
    code_sig = source_code_signature()
    seeds = [int(s) for s in args.seeds.split(",") if s]
    pairs = []
    for p in (args.pairs.split(",") if args.pairs else []):
        d, ss = p.split(":")
        a, b = ss.split(">")
        pairs.append((d, int(a), int(b)))
    pairs = pairs or PAIRS_B
    uni = np.full(K, 1.0 / K)

    done = set()
    if os.path.exists(args.out):
        import json as _json

        for line in open(args.out):
            if line.strip():
                r = _json.loads(line)
                if r.get("perturbation") == PERTS[-1]:
                    done.add((r.get("dataset"), int(r.get("subject")), int(r.get("seed"))))

    for ds, s_src, s_tgt, subj, ai, ei, Xa, ya, Xe, ye, ep, m_src in _pair_units(
        pairs, seeds, args, code_sig, commit, only_subj
    ):
        if only_subj is not None and int(subj) not in only_subj:
            continue
        C = Xa.shape[1]
        for seed in seeds:
            if (ds, int(subj), int(seed)) in done:
                continue
            cfg = build_cfg(ep.X.shape[1], args.epochs, args.device, seed=seed)
            tag = f"B:{ds}:s{int(subj)}:sess{s_src}"
            try:
                model, pooled_ref, R_src, pi_star, val = get_source_p0(
                    args.seed0_root,
                    args.new_root,
                    tag,
                    cfg,
                    code_sig,
                    K,
                    lambda ms=m_src: (ep.X[ms], ep.y[ms], ep.subject[ms]),
                )
            except ProvenanceError as pe:
                append_row(
                    args.out,
                    dict(
                        panel="W1GEOM_OFFDIAG",
                        pair=f"{ds}:{s_src}>{s_tgt}",
                        dataset=ds,
                        subject=int(subj),
                        seed=int(seed),
                        provenance_fail=str(pe),
                    ),
                )
                continue
            Us = _embed(model, ep.X[m_src], args.device)
            base = dict(
                panel="W1GEOM_OFFDIAG",
                commit=commit,
                code_sig=code_sig,
                pair=f"{ds}:{s_src}>{s_tgt}",
                dataset=ds,
                subject=int(subj),
                seed=int(seed),
                seed0_validated=bool(val),
                n_chans=int(C),
                n_adapt=int(len(ai)),
                n_eval=int(len(ei)),
            )
            for kind in PERTS:
                pseed = stable_hash_int("W1offdiag", kind, ds, int(C))
                P = _perturbation(kind, C, pseed)
                Xa_p, Xe_p = _apply(P, Xa), _apply(P, Xe)
                Ua_p, Ue_p = _embed(model, Xa_p, args.device), _embed(model, Xe_p, args.device)
                tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, K, args.device)
                ts = stable_hash_int("W1offdiag_eval", ds, int(subj), int(seed), kind)
                branches, _ = eval_unit_p0(
                    model, tta, pooled_ref, R_src, Xa_p, Xe_p, Ua_p, Ue_p, ye, args.device, K, ts
                )
                ba = {op: branches[op]["bacc"] for op in (["identity_uniform", "source_recolored_ea"] + DIAG_OPS)}
                ba["coral_latent"] = _bacc(_coral_latent(model, Us, Ua_p, Ue_p, uni), ye)
                ba["P_hash"] = int(stable_hash_int("W1offdiag_P", kind, ds, int(C)))
                append_row(args.out, dict(base, perturbation=kind, **{f"bacc_{k}": v for k, v in ba.items()}))
            print(f"[W1OFFDIAG {ds}:{s_src}>{s_tgt}] subj={subj} seed={seed} done", flush=True)

    if os.path.exists(args.out):
        print(f"[W1OFFDIAG] -> {args.out} sha={sha256_file(args.out)[:12]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
