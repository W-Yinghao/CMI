#!/usr/bin/env python
"""E3 runner — K*_subj on beneficial vs legitimate-use worlds (Proposition 2).

Self-contained: builds the two deployment worlds from the ported spurious-task DGP, fits shared source heads,
and evaluates the EXACT squared-loss K* identity on each world. The identity gate (|Gain*-Gain_direct|<=tol on
BOTH worlds) MUST be green; a mismatch is a bug (fix before reporting).

  probe:  python scripts/run_e3_kstar.py --probe
  fleet:  python scripts/run_e3_kstar.py --seeds 0 1 2 3 4 --spur_strengths 2.0 3.0 4.0
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmi.eval.kstar_worlds import run_worlds                       # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--spur_strengths", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    ap.add_argument("--out_dir", default=str(REPO / "results" / "kstar"))
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.probe:
        out = run_worlds(spur_strength=3.0, seed=0)
        qc = {
            "PROBE": "spur_strength=3.0 seed=0",
            "identity_ok": out["identity_ok"],
            "beneficial": {k: round(out["beneficial"][k], 6) for k in
                           ("E_delta2", "E_r_delta", "K_star", "gain_star", "gain_direct", "identity_residual")},
            "legitimate": {k: round(out["legitimate"][k], 6) for k in
                           ("E_delta2", "E_r_delta", "K_star", "gain_star", "gain_direct", "identity_residual")},
            "worlds_separate": out["worlds_separate"],
        }
        print(json.dumps(qc, indent=2))
        if not out["identity_ok"]:
            raise SystemExit("IDENTITY GATE FAILED — bug, do not report worlds.")
        return

    rows = []
    for ss in args.spur_strengths:
        for seed in args.seeds:
            r = run_worlds(spur_strength=ss, seed=seed)
            rows.append(r)
            if not r["identity_ok"]:
                raise SystemExit(f"IDENTITY GATE FAILED at spur={ss} seed={seed}: {r}")
    agg = {
        "n_cells": len(rows),
        "identity_all_ok": all(r["identity_ok"] for r in rows),
        "beneficial_Kstar_mean": float(np.mean([r["beneficial"]["K_star"] for r in rows])),
        "legitimate_Kstar_mean": float(np.mean([r["legitimate"]["K_star"] for r in rows])),
        "frac_worlds_separate": float(np.mean([r["worlds_separate"] for r in rows])),
        "cells": rows,
    }
    (out_dir / "worlds.json").write_text(json.dumps(agg, indent=2))
    print(f"[e3] {len(rows)} cells; identity_all_ok={agg['identity_all_ok']}; "
          f"separate_frac={agg['frac_worlds_separate']:.2f} -> {out_dir/'worlds.json'}")


if __name__ == "__main__":
    main()
