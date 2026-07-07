"""Project A Step 13/14 — real minimal-label curves (coverage-decomposed).

Reads audited run dirs that carry `per_trial_oracle_predictions` and asks, on REAL target
predictions/labels: how many iid-sampled target labels are needed to recover the sign of the
full-target offline-TTA accuracy gain?

Step-14 metric decomposition (the earlier `harm_sign_accuracy` conflated accuracy with coverage):
  * decisive_rate (coverage)               = P(the k-slice CI excludes 0, so a sign is called);
  * unconditional_correct_rate             = P(decisive AND correct sign);
  * conditional_accuracy_given_decisive    = P(correct | decisive)  (null when never decisive);
  * abstention_rate                        = 1 - decisive_rate.

Boundary discipline (hard):
  * k = 0 is the R1 boundary: NO estimator is licensed -> accuracy is NULL (not 0.5), decisive 0;
  * k > 0 is an R2 labeled slice under an iid SAMPLING contract, compared to the oracle full-target
    sign; NOT full-target identification;
  * per-trial oracle labels are used ONLY here (R2 slice + evaluation), never as an R0/R1 feature.

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
    agg = {k: {"decisive": 0, "correct": 0, "abstain": 0, "width_sum": 0.0, "total": 0} for k in ks}
    n_harm = n_benefit = 0
    for r in runs:
        pt = r["per_trial"]
        y = np.asarray(pt["y_true"]); ip = np.asarray(pt["identity_pred"]); ap = np.asarray(pt["adapt_pred"])
        d = (ap == y).astype(float) - (ip == y).astype(float)      # paired per-trial accuracy gain
        n = len(y)
        full_harm = float(d.mean()) < 0                            # oracle full-target accuracy-harm sign
        n_harm += int(full_harm); n_benefit += int(not full_harm)
        for k in ks:
            a = agg[k]
            if k == 0:
                a["abstain"] += repeats; a["total"] += repeats     # R1: no slice, no estimator licensed
                continue
            kk = min(k, n)
            for _ in range(repeats):
                ds = d[rng.choice(n, kk, replace=False)]
                gh = float(ds.mean())
                se = float(ds.std(ddof=1)) / (kk ** 0.5) if kk > 1 else 0.0
                a["width_sum"] += 2 * z * se; a["total"] += 1
                decisive = (gh != 0.0) and (se == 0.0 or not (gh - z * se <= 0 <= gh + z * se))
                if decisive:
                    a["decisive"] += 1
                    if (gh < 0) == full_harm:
                        a["correct"] += 1
                else:
                    a["abstain"] += 1

    per_k = {}
    for k in ks:
        a = agg[k]; tot = max(1, a["total"]); dec = a["decisive"]
        if k == 0:
            per_k["0"] = {"k": 0, "decisive_rate": 0.0, "unconditional_correct_rate": None,
                          "conditional_accuracy_given_decisive": None, "abstention_rate": 1.0,
                          "coverage": 0.0, "precision_when_decisive": None, "mean_ci_width": None,
                          "identified_status": "not_identified_R1",
                          "claim_boundary": "k=0: no estimator licensed under R1 (non-identifiable)."}
            continue
        cond = round(a["correct"] / dec, 4) if dec else None
        per_k[str(k)] = {
            "k": k, "decisive_rate": round(dec / tot, 4),
            "unconditional_correct_rate": round(a["correct"] / tot, 4),
            "conditional_accuracy_given_decisive": cond,
            "abstention_rate": round(a["abstain"] / tot, 4),
            "coverage": round(dec / tot, 4), "precision_when_decisive": cond,
            "mean_ci_width": round(a["width_sum"] / tot, 4),
            "identified_status": "r2_labeled_slice_under_iid_sampling_contract",
            "claim_boundary": _R2_CLAIM,
        }

    def _best(field, thr):
        return next((k for k in ks if k > 0 and (per_k[str(k)][field] or 0) >= thr), None)

    n_runs = len(runs)
    return {
        "project": "Project A", "step": "Step 13",
        "scope": "real minimal-label curves (R2 labeled slice, coverage-decomposed); not SOTA",
        "n_runs": n_runs, "ks": ks, "repeats": repeats,
        "per_k": per_k,
        # coverage-limited unconditional vs precision-when-decisive conditional
        "best_k_for_0_8_unconditional": _best("unconditional_correct_rate", 0.8),
        "best_k_for_0_9_unconditional": _best("unconditional_correct_rate", 0.9),
        "best_k_for_0_8_conditional": _best("conditional_accuracy_given_decisive", 0.8),
        "best_k_for_0_9_conditional": _best("conditional_accuracy_given_decisive", 0.9),
        # DEPRECATED alias (do not use in the dashboard): it conflated accuracy and coverage
        "harm_sign_accuracy_deprecated_per_k": {str(k): per_k[str(k)]["unconditional_correct_rate"]
                                                for k in ks},
        "baselines": {
            "always_predict_harm_baseline": round(n_harm / n_runs, 4) if n_runs else None,
            "always_abstain_baseline": {"decisive_rate": 0.0, "unconditional_correct_rate": 0.0},
            "oracle_full_target_sign_distribution": {"harm": n_harm, "benefit_or_no_harm": n_benefit},
            "note": ("always_predict_harm is an ORACLE class-balance baseline for evaluating "
                     "retrospective sign predictors; it is NOT licensed under R1."),
        },
        "k0_status": "not_identified_R1",
        "claim_boundary_ok": True,
        "oracle_labels_used_only_for_r2_slice_and_evaluation": True,
        "claim_boundary": ("k=0 = R1 non-identifiable (no estimator, accuracy NULL); k>0 = R2 labeled "
                           "slice under an iid sampling contract vs the oracle full-target sign; not "
                           "full-target identification. Oracle labels used only here (R2/evaluation)."),
    }


def write_md(s: Dict[str, Any], path) -> str:
    lines = ["# Step 13/14 — real minimal-label curves (coverage-decomposed)", "",
             f"Scope: {s['scope']}. {s['claim_boundary']}", "",
             f"- runs with per-trial oracle predictions: **{s['n_runs']}** · repeats **{s['repeats']}**",
             f"- best k (unconditional ≥0.8): **{s['best_k_for_0_8_unconditional']}** · "
             f"(conditional ≥0.8): **{s['best_k_for_0_8_conditional']}**",
             f"- oracle sign distribution: **{s['baselines']['oracle_full_target_sign_distribution']}** · "
             f"always-predict-harm baseline **{s['baselines']['always_predict_harm_baseline']}** (oracle, not R1)",
             f"- k=0 status: **{s['k0_status']}** (no estimator; accuracy NULL, not 0.5)", "",
             "| k | coverage (decisive) | uncond correct | conditional acc (decisive) | abstention | ci_width |",
             "|---:|---:|---:|---:|---:|---:|"]
    for k in s["ks"]:
        r = s["per_k"][str(k)]
        lines.append(f"| {r['k']} | {r['decisive_rate']} | {r['unconditional_correct_rate']} | "
                     f"{r['conditional_accuracy_given_decisive']} | {r['abstention_rate']} | "
                     f"{r['mean_ci_width']} |")
    lines += ["", "> " + _R2_CLAIM,
              "> Coverage–confidence tradeoff: when the slice is DECISIVE it may be accurate "
              "(conditional accuracy), but COVERAGE (decisive rate) stays low; the burden is coverage."]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A real minimal-label curves (coverage-decomposed)")
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=_K_DEFAULT)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step-label", default="Step 13")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)
    s = _curve(args.roots, args.ks, args.repeats, args.seed)
    s["step"] = args.step_label
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        write_json_lf(args.out_json, s)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_md(s, args.out_md)
    print(f"real_minimal_labels[{s['step']}] n_runs={s['n_runs']} "
          f"best_k_0.8_uncond={s['best_k_for_0_8_unconditional']} "
          f"best_k_0.8_cond={s['best_k_for_0_8_conditional']} k0={s['k0_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
