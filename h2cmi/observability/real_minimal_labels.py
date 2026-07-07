"""Project A Step 13 — real minimal-label curves.

Reads audited run dirs that carry `per_trial_oracle_predictions` (Step-13 instrumented runs) and asks,
on REAL target predictions/labels: how many iid-sampled target labels are needed to recover the sign
of the full-target offline-TTA accuracy gain?

Boundary discipline (hard):
  * k = 0 is the R1 boundary: the gain sign is NOT identified (chance);
  * k > 0 is an R2 labeled slice under an iid SAMPLING contract — it estimates the labeled-slice gain
    and is compared to the (oracle) full-target sign; it is NOT full-target identification without
    that contract;
  * the per-trial oracle labels are used ONLY here (R2 slice + evaluation), never as an R0/R1 feature.

  python -m h2cmi.observability.real_minimal_labels --roots <dir> ... --ks 0 1 2 4 8 16 32 64 \
      --repeats 200 --out-json ... --out-md ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from .result_index import _load_json, write_json_lf, write_text_lf

_K_DEFAULT = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
_Z = 1.96
_R2_CLAIM = ("k>0 estimates a labeled-slice gain under an iid sampling contract and compares it to the "
             "oracle full-target sign; NOT full-target identification without that contract.")


def _load_runs(roots: List[str]) -> List[Dict[str, Any]]:
    runs = []
    for root in roots:
        for mp in sorted(Path(root).glob("*/run_manifest.json")):
            manifest = _load_json(mp) or {}
            if manifest.get("status") != "ok":
                continue
            raw = _load_json(mp.parent / "raw_results.json") or {}
            pt = raw.get("per_trial_oracle_predictions") or {}
            if pt.get("y_true") and pt.get("identity_pred") and pt.get("adapt_pred"):
                runs.append({"dataset": manifest.get("dataset"),
                             "target_subject": manifest.get("target_subject"),
                             "seed": manifest.get("seed"), "per_trial": pt})
    return runs


def _curve(roots, ks, repeats, seed=0, z=_Z) -> Dict[str, Any]:
    import numpy as np
    runs = _load_runs(roots)
    rng = np.random.default_rng(int(seed))
    # accumulate decisive/correct/abstain and CI widths per k across all runs x repeats
    agg = {k: {"decisive": 0, "correct": 0, "abstain": 0, "width_sum": 0.0, "total": 0} for k in ks}
    for r in runs:
        pt = r["per_trial"]
        y = np.asarray(pt["y_true"]); ip = np.asarray(pt["identity_pred"]); ap = np.asarray(pt["adapt_pred"])
        d = (ap == y).astype(float) - (ip == y).astype(float)      # paired per-trial accuracy gain
        n = len(y)
        full_harm = float(d.mean()) < 0                            # oracle full-target accuracy-harm sign
        for k in ks:
            a = agg[k]
            if k == 0:
                a["abstain"] += repeats; a["total"] += repeats     # R1: no slice, cannot call sign
                continue
            kk = min(k, n)
            for _ in range(repeats):
                ds = d[rng.choice(n, kk, replace=False)]
                gh = float(ds.mean())
                se = float(ds.std(ddof=1)) / (kk ** 0.5) if kk > 1 else 0.0
                a["width_sum"] += 2 * z * se; a["total"] += 1
                # decisive if the sign is called: a zero-variance non-zero slice is perfectly decisive
                decisive = (gh != 0.0) and (se == 0.0 or not (gh - z * se <= 0 <= gh + z * se))
                if decisive:
                    a["decisive"] += 1
                    if (gh < 0) == full_harm:
                        a["correct"] += 1
                else:
                    a["abstain"] += 1

    per_k = {}
    for k in ks:
        a = agg[k]; tot = max(1, a["total"])
        per_k[str(k)] = {
            "k": k,
            "harm_sign_accuracy": 0.5 if k == 0 else round(a["correct"] / tot, 4),
            "decisive_rate": round(a["decisive"] / tot, 4),
            "abstention_rate": round(a["abstain"] / tot, 4),
            "mean_ci_width": None if k == 0 else round(a["width_sum"] / tot, 4),
            "identified_status": "not_identified_R1" if k == 0 else "r2_labeled_slice_under_iid_sampling_contract",
            "claim_boundary": ("k=0 gain sign non-identifiable under R1" if k == 0 else _R2_CLAIM),
        }

    def _best(thr):
        return next((k for k in ks if k > 0 and per_k[str(k)]["harm_sign_accuracy"] >= thr), None)

    return {
        "project": "Project A", "step": "Step 13",
        "scope": "real minimal-label curves (R2 labeled slice); not SOTA",
        "n_runs": len(runs), "ks": ks, "repeats": repeats,
        "per_k": per_k,
        "best_k_for_0_8_accuracy": _best(0.8),
        "best_k_for_0_9_accuracy": _best(0.9),
        "k0_status": "not_identified_R1",
        "claim_boundary_ok": True,
        "oracle_labels_used_only_for_r2_slice_and_evaluation": True,
        "claim_boundary": ("k=0 = R1 non-identifiable; k>0 = R2 labeled slice under an iid sampling "
                           "contract compared to the oracle full-target sign; not full-target "
                           "identification. Oracle labels are used only here (R2/evaluation)."),
    }


def write_md(s: Dict[str, Any], path) -> str:
    lines = ["# Step 13 — real minimal-label curves", "",
             f"Scope: {s['scope']}. {s['claim_boundary']}", "",
             f"- runs with per-trial oracle predictions: **{s['n_runs']}** · repeats **{s['repeats']}**",
             f"- best k for harm-sign acc ≥ 0.8: **{s['best_k_for_0_8_accuracy']}** · ≥ 0.9: "
             f"**{s['best_k_for_0_9_accuracy']}**",
             f"- k=0 status: **{s['k0_status']}**", "",
             "| k | harm_sign_acc | decisive_rate | abstention | mean_ci_width | status |",
             "|---:|---:|---:|---:|---:|---|"]
    for k in s["ks"]:
        r = s["per_k"][str(k)]
        lines.append(f"| {r['k']} | {r['harm_sign_accuracy']} | {r['decisive_rate']} | "
                     f"{r['abstention_rate']} | {r['mean_ci_width']} | {r['identified_status']} |")
    lines += ["", "> " + _R2_CLAIM]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A Step 13 real minimal-label curves")
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=_K_DEFAULT)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)
    s = _curve(args.roots, args.ks, args.repeats, args.seed)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        write_json_lf(args.out_json, s)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_md(s, args.out_md)
    print(f"real_minimal_labels n_runs={s['n_runs']} best_k_0.8={s['best_k_for_0_8_accuracy']} "
          f"best_k_0.9={s['best_k_for_0_9_accuracy']} k0={s['k0_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
