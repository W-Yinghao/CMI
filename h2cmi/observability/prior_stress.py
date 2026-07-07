"""Project A Step 18 — deployment-prior stress on TTA gain (counterfactual; contract C14).

A prior-weighted target gain is a linear functional of the per-class recall deltas:

    gain(pi) = sum_c pi_c * recall_delta_c        (balanced-accuracy gain is the uniform-prior case)

Over the probability simplex the extreme prior-weighted gains are simply the extreme class deltas:

    gain_min_prior = min_c recall_delta_c ,   gain_max_prior = max_c recall_delta_c

so the sign of the gain is prior-dependent iff  min_delta < 0 < max_delta.  This is an EXACT,
non-neural result read straight off Step-18 harm_mechanisms. It answers: is offline-TTA harm robust to
ALL deployment priors, or only to the benchmark-uniform prior?

Boundary: these are COUNTERFACTUAL deployment-prior scenarios. The priors are DECLARED (contract C14),
not estimated from target data; this does NOT identify the actual target prior (that needs TU-1). The
class deltas themselves come from oracle labels, so the whole analysis is oracle/evaluation-only.

  python -m h2cmi.observability.prior_stress --harm-mechanisms step18_harm_mechanisms.json \
      --out-json step18_prior_stress.json --out-md step18_prior_stress.md
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from .result_index import _load_json, write_json_lf, write_text_lf

_EPS = 1e-9
_C14 = "C14"
_DOMINANT_MASS = 0.9                                            # a skewed deployment prior scenario


def prior_weighted_gain(prior: List[float], deltas: List[float]) -> float:
    """gain(pi) = <pi, class recall deltas>. Prior need not be normalised here; callers pass a simplex."""
    return float(sum(p * d for p, d in zip(prior, deltas)))


def _delta_vector(run) -> List[float]:
    order = run.get("transition_matrix_class_order") or sorted(int(c) for c in run["per_class"])
    return [run["per_class"][str(int(c))]["recall_delta"] for c in order]


def _scenarios(deltas: List[float]) -> Dict[str, float]:
    K = len(deltas)
    out = {"uniform": prior_weighted_gain([1.0 / K] * K, deltas)}
    rest = (1.0 - _DOMINANT_MASS) / (K - 1) if K > 1 else 0.0
    for c in range(K):
        prior = [_DOMINANT_MASS if i == c else rest for i in range(K)]
        out[f"one_class_{c}_mass_0_9"] = round(prior_weighted_gain(prior, deltas), 6)
    out["uniform"] = round(out["uniform"], 6)
    return out


def _per_run(run) -> Dict[str, Any]:
    deltas = _delta_vector(run)
    order = run.get("transition_matrix_class_order") or sorted(int(c) for c in run["per_class"])
    uniform = round(sum(deltas) / len(deltas), 6)
    min_d, max_d = min(deltas), max(deltas)
    prior_dependent = bool(min_d < -_EPS and max_d > _EPS)
    harmful_all = bool(max_d <= _EPS and min_d < -_EPS)
    beneficial_all = bool(min_d >= -_EPS and max_d > _EPS)
    return {
        "dataset": run.get("dataset"), "target_subject": run.get("target_subject"), "seed": run.get("seed"),
        "n_classes": len(deltas),
        "class_delta_vector": [round(d, 6) for d in deltas], "class_order": [int(c) for c in order],
        "uniform_gain": uniform, "min_prior_gain": round(min_d, 6), "max_prior_gain": round(max_d, 6),
        "prior_sign_width": round(max_d - min_d, 6),
        "prior_dependent_sign": prior_dependent,
        "harmful_under_all_priors": harmful_all,
        "beneficial_under_all_priors": beneficial_all,
        "uniform_harm_but_some_prior_benefit": bool(uniform < -_EPS and max_d > _EPS),
        "uniform_benefit_but_some_prior_harm": bool(uniform > _EPS and min_d < -_EPS),
        "worst_class": int(order[int(min(range(len(deltas)), key=lambda i: deltas[i]))]),
        "best_class": int(order[int(max(range(len(deltas)), key=lambda i: deltas[i]))]),
        "declared_prior_scenarios": _scenarios(deltas),
        "claim_boundary": ("counterfactual deployment-prior stress (declared priors); C14 required for "
                           "deployment-weighted claims; does NOT identify the actual target prior"),
    }


def build_summary(harm_mechanisms: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np
    runs = [_per_run(r) for r in (harm_mechanisms or {}).get("runs", []) if r.get("per_class")]
    n = len(runs)

    def frac(key):
        return round(sum(r[key] for r in runs) / n, 4) if n else None

    return {
        "project": "Project A", "step": "Step 18",
        "scope": "counterfactual deployment-prior stress on TTA gain (contract C14); not SOTA",
        "n_runs": n, "dominant_prior_mass": _DOMINANT_MASS,
        "fraction_prior_dependent_sign": frac("prior_dependent_sign"),
        "fraction_harmful_under_all_priors": frac("harmful_under_all_priors"),
        "fraction_beneficial_under_all_priors": frac("beneficial_under_all_priors"),
        "fraction_uniform_harm_but_some_prior_benefit": frac("uniform_harm_but_some_prior_benefit"),
        "fraction_uniform_benefit_but_some_prior_harm": frac("uniform_benefit_but_some_prior_harm"),
        "mean_prior_sign_width": round(float(np.mean([r["prior_sign_width"] for r in runs])), 6) if n else None,
        "prior_contract_required": _C14,
        "deployment_prior_identified_under_R1": False,          # C14 declares, TU-1 identifies — not here
        "deployment_prior_identified": False,
        "runs": runs,
        "claim_boundary_ok": True,
        "claim_boundary": ("prior-weighted gains are counterfactual evaluations under DECLARED priors "
                           "(contract C14); the actual target prior is NOT identified (that needs TU-1); "
                           "class deltas are oracle/evaluation-only. No SOTA."),
    }


def write_md(s: Dict[str, Any], path) -> str:
    lines = ["# Step 18 — deployment-prior stress on TTA gain", "",
             f"Scope: {s['scope']}. {s['claim_boundary']}", "",
             f"- runs: **{s['n_runs']}** · prior-dependent-sign **{s['fraction_prior_dependent_sign']}** · "
             f"harmful-under-all-priors **{s['fraction_harmful_under_all_priors']}** · "
             f"beneficial-under-all-priors **{s['fraction_beneficial_under_all_priors']}**",
             f"- uniform-harm-but-some-prior-benefit **{s['fraction_uniform_harm_but_some_prior_benefit']}** · "
             f"uniform-benefit-but-some-prior-harm **{s['fraction_uniform_benefit_but_some_prior_harm']}** · "
             f"mean prior-sign-width **{s['mean_prior_sign_width']}**",
             f"- prior contract required **{s['prior_contract_required']}** · deployment prior identified "
             f"under R1 **{s['deployment_prior_identified_under_R1']}**", "",
             "> " + s["claim_boundary"]]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A Step 18 deployment-prior stress")
    ap.add_argument("--harm-mechanisms", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)
    s = build_summary(_load_json(Path(args.harm_mechanisms)) or {})
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        write_json_lf(args.out_json, s)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_md(s, args.out_md)
    print(f"prior_stress n_runs={s['n_runs']} prior_dep={s['fraction_prior_dependent_sign']} "
          f"harmful_all={s['fraction_harmful_under_all_priors']} "
          f"benef_all={s['fraction_beneficial_under_all_priors']} "
          f"uni_harm_some_benefit={s['fraction_uniform_harm_but_some_prior_benefit']} "
          f"mean_width={s['mean_prior_sign_width']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
