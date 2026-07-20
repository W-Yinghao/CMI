"""C29 — extract the LINEAR classifier head parameters (b, per-class weight norms, pairwise class-weight angles)
per candidate from the frozen checkpoint state_dicts (staging trained pickles). CPU read of frozen parameters --
NO training, NO GPU, NO inference. Produces a small head sidecar; W.z is then recovered downstream as
(logit - b) from the persisted logits (no need to store the full 4x800 W)."""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re

import numpy as np

from . import schema


def extract_head_params(loso_root=None, out_sidecar=None) -> int:
    root = loso_root or schema.LOSO_ROOT
    out = []
    for staging in sorted(glob.glob(os.path.join(root, "seed-*", "target-*", "staging"))):
        m = re.search(r"seed-(\d+)[/\\]target-(\d+)", staging)
        seed, target = int(m.group(1)), int(m.group(2))
        meta = json.load(open(os.path.join(staging, "phase_a.json")))
        from ..runner.staged_fold import load_phase_a_fold
        fold = load_phase_a_fold(staging)
        tol = float(fold.execution_config.engine_template.numerical_tol)
        for tf in sorted(glob.glob(os.path.join(staging, "level-*-trained.pkl"))):
            level = int(re.search(r"level-(\d+)-trained", tf).group(1))
            t = pickle.load(open(tf, "rb"))
            from ..diagnostics.candidate_replay import candidate_records
            for origin, rec, feasible, is_erm in candidate_records(t["stage1"], t["trained"], tol):
                ms = rec.model_state
                W = np.asarray(ms[schema.HEAD_WEIGHT_KEY], dtype=np.float64)   # (C, D)
                b = np.asarray(ms[schema.HEAD_BIAS_KEY], dtype=np.float64)     # (C,)
                norms = np.linalg.norm(W, axis=1)
                ang = (W @ W.T) / (norms[:, None] * norms[None, :] + 1e-12)
                out.append({"seed": seed, "target": target, "level": level, "model_hash": rec.model_hash,
                            "is_erm": bool(is_erm), "bias": b.tolist(), "weight_norms": norms.tolist(),
                            "weight_angles": ang.tolist()})
    payload = {"config_hash": schema.LOCKED_C19_CONFIG_HASH, "n_candidates": len(out), "head_params": out,
               "note": "linear classifier head b + per-class weight norms + pairwise class-weight cosines; W.z "
                       "recovered downstream as (logit - b). CPU read of frozen params, no training/inference."}
    path = out_sidecar or schema.C29_HEAD_SIDECAR
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(payload, open(path, "w"), sort_keys=True, default=str)
    return len(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.rep_head_geometry.head_extractor")
    ap.add_argument("--loso-root", default=None)
    ap.add_argument("--out-sidecar", default=None)
    args = ap.parse_args(argv)
    n = extract_head_params(args.loso_root, args.out_sidecar)
    print(f"[C29 head extract] {n} candidate head params -> {args.out_sidecar or schema.C29_HEAD_SIDECAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
