"""C18-P0 — candidate-replay smoke / identity gate. Pick a small deterministic slice (one seed x target),
GPU re-infer the candidate checkpoints, and verify BEFORE any full run:
  (1) IDENTITY: selected ERM/OACI reproduce the stored artifact logits (argmax parity + max|dlogit|<tol);
  (2) S0-vs-C10: the recomputed no-mask source scalars reproduce the persisted C10 atlas within tolerance;
  (3) PERSISTENCE: the per-unit source logits + per-candidate Z-features round-trip with consistent ids.
Only if all three pass may the full C18-P replay run. Writes oaci/reports/C18_P0_SMOKE.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .replay_extract import _PERSIST_ROLES, extract_fold


def _roundtrip(out_dir, seed, target) -> dict:
    fdir = os.path.join(out_dir, f"seed-{seed}-target-{target:03d}")
    levels = sorted(d for d in os.listdir(fdir) if d.startswith("level-"))
    checks = []
    for ld in levels:
        p = os.path.join(fdir, ld)
        for role in _PERSIST_ROLES:
            u = np.load(os.path.join(p, f"units-{role}.npz"), allow_pickle=True)
            lg = np.load(os.path.join(p, f"logits-{role}.npy"))
            nu = len(u["y"])
            ok = (lg.ndim == 3 and lg.shape[1] == nu and lg.shape[2] >= 2
                  and len(u["domain_raw"]) == nu and len(u["group"]) == nu and len(u["sample_id"]) == nu
                  and set(np.unique(u["y"]).tolist()) <= set(range(lg.shape[2])))
            checks.append({"level": ld, "role": role, "n_cand": int(lg.shape[0]), "n_units": int(nu),
                           "logits_shape": list(lg.shape), "n_domains": int(len(set(u["domain_raw"].tolist()))),
                           "ok": bool(ok)})
        for dz in ("selection", "audit"):
            fp = os.path.join(p, f"featz-{dz}.npz")
            if os.path.exists(fp):
                z = np.load(fp, allow_pickle=True)
                Z, y = z["Z"], z["y"]
                ok = (Z.ndim == 3 and Z.shape[1] == len(y) == len(z["d"]) == len(z["group"]) == len(z["sample_id"])
                      and np.isfinite(Z).all())
                checks.append({"level": ld, "featz": dz, "n_cand": int(Z.shape[0]), "n_rows": int(Z.shape[1]),
                               "z_dim": int(Z.shape[2]), "ok": bool(ok)})
    return {"checks": checks, "all_ok": all(c["ok"] for c in checks)}


def run(loso_root, seed, target, out_dir, c10_dir, device, report_path) -> int:
    res = extract_fold(loso_root, seed, target, device, out_dir, require_identity=True, c10_dir=c10_dir)
    identity_ok = all(c["match"] for c in res["identity"])
    s0 = res["s0_vs_c10"]
    s0_ok = len(s0) > 0 and all(c["all_ok"] for c in s0)
    # worst per-key diff across candidates, for disclosure
    worst = {}
    for c in s0:
        for k, v in c["checks"].items():
            if v["diff"] is not None and v["diff"] > worst.get(k, -1.0):
                worst[k] = v["diff"]
    rt = _roundtrip(out_dir, seed, target)
    verdict = "PASS" if (identity_ok and s0_ok and rt["all_ok"]) else "FAIL"
    report = {"verdict": verdict, "seed": seed, "target": target,
              "identity": {"n": len(res["identity"]), "n_match": sum(c["match"] for c in res["identity"]),
                           "ok": identity_ok},
              "s0_vs_c10": {"n_candidates": len(s0), "n_all_ok": sum(c["all_ok"] for c in s0), "ok": s0_ok,
                            "worst_abs_diff_by_key": {k: round(v, 6) for k, v in sorted(worst.items())}},
              "persistence_roundtrip": rt, "note": "C18-P0 gate; full C18-P replay may run only on PASS."}
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    json.dump(report, open(report_path, "w"), indent=2, sort_keys=True)
    print(f"[C18-P0] verdict={verdict} identity={report['identity']['n_match']}/{report['identity']['n']} "
          f"s0={report['s0_vs_c10']['n_all_ok']}/{report['s0_vs_c10']['n_candidates']} "
          f"roundtrip={'ok' if rt['all_ok'] else 'FAIL'} worst={report['s0_vs_c10']['worst_abs_diff_by_key']}")
    return 0 if verdict == "PASS" else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.support_stress.smoke")
    ap.add_argument("--loso-root", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--c10-dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--report", default="oaci/reports/C18_P0_SMOKE.json")
    args = ap.parse_args(argv)
    from ..runtime.cuda import configure_cuda_determinism
    device = args.device if args.device else configure_cuda_determinism()[0]
    return run(args.loso_root, args.seed, args.target, args.out_dir, args.c10_dir, device, args.report)


if __name__ == "__main__":
    raise SystemExit(main())
