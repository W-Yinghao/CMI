"""C86H Stage H2/H3 — held evaluation, endpoints, and two-level classification.

Unlike C86D (which replayed the historical C85U construction utility), C86H's held
evaluation is the target's OWN label-blind held split, opened only after every H1
selection freeze is verified. The candidate composite pipeline is the frozen C86D
production dispatcher (imported, not reimplemented).

Endpoints per (method, budget, cohort): target-first held standardized regret (primary),
exact target-tail CVaR, indicator-first near-optimal probability; per-cohort active-vs-P0
mean/tail qualification with the registered thresholds; FULL construction-view ceiling for
the Level-2 descriptor. Then the registered within-cohort max-T decides familywise
significance and ``analysis.classify`` emits the two-level output.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from ..c86d.policies import composite_select
from oaci.theory.c86_active_program import empirical_upper_cvar
from . import analysis as AN
from . import contract as K

_PROB_FLOOR = 1e-7


def contribs_from_probs(probs: np.ndarray, labels) -> dict:
    """Derive the plugin contribution fields (nll/correct/confidence/conf_bin) from
    candidate probabilities [n,81,2] and true labels [n] — the exact C86 field semantics."""
    probs = np.asarray(probs, dtype=np.float64)
    n, K_, _ = probs.shape
    y = np.asarray(labels).astype(int)
    p_true = probs[np.arange(n)[:, None], np.arange(K_)[None, :], y[:, None]]  # [n,81]
    nll = -np.log(np.clip(p_true, _PROB_FLOOR, 1.0))
    pred = probs.argmax(axis=2)
    correct = (pred == y[:, None]).astype(np.float64)
    confidence = probs.max(axis=2)
    conf_bin = np.clip((confidence * 15).astype(int), 0, 14)
    return {"nll": nll, "correct": correct, "confidence": confidence, "conf_bin": conf_bin}


def held_composite(probs: np.ndarray, labels) -> tuple:
    """Full held-eval composite over all 81 candidates. Returns (composite[81], std_regret[81])."""
    contribs = contribs_from_probs(probs, labels)
    n = len(labels)
    _, metrics = composite_select(list(np.asarray(labels).astype(int)), contribs,
                                  weights=None, full=True, n_pool=n)
    return metrics["composite"], metrics["std_regret_construction"]


def evaluate(freezes: Mapping, held_field: Mapping, target_cohort: Mapping,
             cohort_dataset: Mapping, epsilon: float = K.PRIMARY_EPSILON,
             blocker: bool = False) -> dict:
    """Compute C86H endpoints + two-level classification.

    freezes[(method, target, chain)] = {budget_str: {status, selected_by_context, ...}}
    held_field[(target, ctx)]        = {"probs": [n,81,2], "labels": [n]}
    target_cohort[target]            = cohort name
    cohort_dataset[cohort]           = registered dataset id used for the max-T seed

    Near-opt is C86D-geometry indicator-first: per replicate (chain) the 8-context selected
    gaps are averaged FIRST, then thresholded at epsilon; the per-target near-opt is the mean
    over chains. Formal-gate mean qualification excludes LOTO (a C86-A-only stability check).
    The Level-2 descriptor is computed from the C86D CROSSED margins (DESC_TAU/DESC_NEAROPT),
    not from the Level-1 gate booleans.
    """
    held_std, held_gap = {}, {}
    for (tgt, ctx), d in held_field.items():
        comp, stdr = held_composite(d["probs"], d["labels"])
        held_std[(tgt, ctx)] = stdr
        held_gap[(tgt, ctx)] = float(comp.max()) - comp     # raw composite-utility gap

    methods = list(K.METHOD_REGISTRY)
    contexts = sorted({ctx for (_t, ctx) in held_field})
    targets = sorted({t for t in target_cohort})
    cohorts = sorted({target_cohort[t] for t in targets})
    chains_by = {}
    for (m, t, c) in freezes:
        chains_by.setdefault((m, t), []).append(c)

    def per_target(method, budget, tgt):
        """Per-chain (mean-8ctx regret, indicator-first near-opt) then mean over chains;
        also per-context mean-over-chains regret for the 8-cell effects."""
        regrets, inds, ctx_regs = [], [], {ctx: [] for ctx in contexts}
        for chain in sorted(chains_by.get((method, tgt), [])):
            fr = freezes[(method, tgt, chain)].get(str(budget))
            if fr is None or fr.get("status") != "AVAILABLE":
                continue
            ctx_sr, ctx_gap = [], []
            for ctx in contexts:
                sel = int(fr["selected_by_context"][ctx])
                sr = float(held_std[(tgt, ctx)][sel])
                ctx_sr.append(sr)
                ctx_gap.append(float(held_gap[(tgt, ctx)][sel]))
                ctx_regs[ctx].append(sr)
            regrets.append(float(np.mean(ctx_sr)))
            inds.append(1.0 if float(np.mean(ctx_gap)) <= epsilon else 0.0)   # C86D geometry
        if not regrets:
            return None
        return {"regret": float(np.mean(regrets)), "near_opt": float(np.mean(inds)),
                "ctx_regret": {c: float(np.mean(v)) for c, v in ctx_regs.items() if v}}

    per = {}
    for method in methods:
        for budget in K.BUDGET_GRID:
            tr, tn, cr = {}, {}, {}
            for tgt in targets:
                pt = per_target(method, budget, tgt)
                if pt is None:
                    continue
                tr[tgt] = pt["regret"]; tn[tgt] = pt["near_opt"]
                for ctx, v in pt["ctx_regret"].items():
                    cr[(tgt, ctx)] = v
            per[(method, budget)] = {"target_regret": tr, "target_nearopt": tn, "ctx_regret": cr}

    active = list(K.ACTIVE_METHODS)
    finite = list(K.FINITE_BUDGETS)
    endpoints, per_cohort, full_ceiling, inference_detail = {}, {}, {}, {}

    def cohort_stat(method, budget, c_targets):
        tr = per[(method, budget)]["target_regret"]
        tn = per[(method, budget)]["target_nearopt"]
        reg = [tr[t] for t in c_targets if t in tr]
        near = [tn[t] for t in c_targets if t in tn]
        return {"mean": float(np.mean(reg)) if reg else None,
                "tail": float(empirical_upper_cvar(reg, K.PRIMARY_CVAR_ALPHA)) if reg else None,
                "near_opt": float(np.mean(near)) if near else None, "n": len(reg)}

    desc_material, desc_crossed = False, False

    for cohort in cohorts:
        c_targets = [t for t in targets if target_cohort[t] == cohort]
        ds = cohort_dataset[cohort]
        mean_q, tail_q, stab_q = {}, {}, {}

        full = cohort_stat("P0", "FULL", c_targets)          # method-invariant at FULL
        full_ceiling[cohort] = {"mean": full["mean"] if full["mean"] is not None else 1.0,
                                "tail": full["tail"] if full["tail"] is not None else 1.0,
                                "near_opt": full["near_opt"] if full["near_opt"] is not None else 0.0}

        for method in methods:
            for budget in K.BUDGET_GRID:
                st = cohort_stat(method, budget, c_targets)
                endpoints[f"{cohort}|{method}|{budget}"] = {
                    "mean_regret_std": st["mean"], "tail_regret_std": st["tail"],
                    "near_opt_prob": st["near_opt"], "n_targets": st["n"]}

        # common target set across P0 and all active x finite budgets (aligned max-T cluster)
        def present(m, b, t):
            return t in per[(m, b)]["target_regret"]
        common = [t for t in c_targets
                  if all(present("P0", b, t) for b in finite)
                  and all(present(am, b, t) for am in active for b in finite)]

        family_effects = {(am, b): np.array(
            [per[("P0", b)]["target_regret"][t] - per[(am, b)]["target_regret"][t]
             for t in common], dtype=np.float64) for am in active for b in finite}
        maxt = (AN.maxt_familywise(family_effects, ds)
                if len(common) >= 2 else {"significant": {}})

        for am in active:
            for b in finite:
                eff = family_effects[(am, b)]
                if len(eff) < 2:
                    mean_q[(am, b)] = tail_q[(am, b)] = stab_q[(am, b)] = False
                    continue
                cell_means = []
                for ctx in contexts:
                    p0c = per[("P0", b)]["ctx_regret"]; amc = per[(am, b)]["ctx_regret"]
                    ce = [p0c[(t, ctx)] - amc[(t, ctx)] for t in common
                          if (t, ctx) in p0c and (t, ctx) in amc]
                    cell_means.append(float(np.mean(ce)) if ce else 0.0)
                mean_q[(am, b)] = AN.mean_qualification(
                    eff, cell_means, maxt["significant"].get((am, b), False))["qualified"]
                stab_q[(am, b)] = AN.stability_qualification(eff)["qualified"]
                p0_vals = [per[("P0", b)]["target_regret"][t] for t in common]
                a_vals = [per[(am, b)]["target_regret"][t] for t in common]
                tq = AN.tail_qualification(p0_vals, a_vals)
                tail_q[(am, b)] = tq["qualified"]
                # complete inference detail frozen for independent re-derivation of the gate
                hyp = maxt.get("hypotheses", {}).get((am, b), {})
                inference_detail[f"{cohort}|{am}|{b}"] = {
                    "observed_t": hyp.get("observed"),
                    "adjusted_maxt_p": hyp.get("adjusted_p"),
                    "critical": maxt.get("critical"),
                    "maxt_seed": maxt.get("seed"), "sign_mode": maxt.get("sign_mode"),
                    "n_signs": maxt.get("n_signs"), "family": maxt.get("family"),
                    "family_sha256": maxt.get("family_sha256"),
                    "mean_effect": float(eff.mean()),
                    "favorable_fraction": AN.favorable_fraction(eff),
                    "worst_target": AN.worst_target(eff),
                    "cell_effects": [float(c) for c in cell_means],
                    "loto": AN.loto_preservation(eff),
                    "tail_effects": {str(a): float(v) for a, v in tq["tail_effects"].items()},
                    "mean_qualified": mean_q[(am, b)], "tail_qualified": tail_q[(am, b)],
                    "stability_qualified": stab_q[(am, b)], "n_targets": int(len(eff)),
                    # raw cluster so max-T (observed_t / adjusted_maxt_p) is recomputable from the
                    # committed artifact alone (the whole cohort family shares `common_targets`)
                    "effect_vector": [float(x) for x in eff],
                    "common_targets": [list(t) for t in common],
                }

        per_cohort[cohort] = {"mean": mean_q, "tail": tail_q, "stability": stab_q}

    # Level-2 descriptor 'any_material': POOLED cross-cohort mean improvement >= DESC_TAU
    # for some (active, budget) (matches the C86D pooled-mean TAU entry gate).
    for am in active:
        for b in finite:
            pool = [t for t in targets if t in per[("P0", b)]["target_regret"]
                    and t in per[(am, b)]["target_regret"]]
            if pool:
                eff = float(np.mean([per[("P0", b)]["target_regret"][t]
                                     - per[(am, b)]["target_regret"][t] for t in pool]))
                if eff >= K.DESC_TAU:
                    desc_material = True

    # robust = one (active,budget) that meets the C86D CROSSED margins in EVERY cohort
    for am in active:
        for b in finite:
            ok = True
            for cohort in cohorts:
                c_targets = [t for t in targets if target_cohort[t] == cohort]
                p0s = cohort_stat("P0", b, c_targets); ams = cohort_stat(am, b, c_targets)
                if None in (p0s["mean"], ams["mean"], p0s["tail"], ams["tail"],
                            p0s["near_opt"], ams["near_opt"]):
                    ok = False; break
                if not ((p0s["mean"] - ams["mean"]) >= K.DESC_TAU
                        and (p0s["tail"] - ams["tail"]) >= K.DESC_TAU
                        and (ams["near_opt"] - p0s["near_opt"]) >= K.DESC_NEAROPT_MARGIN):
                    ok = False; break
            if ok:
                desc_crossed = True

    active_gain = {"any_material": bool(desc_material),
                   "robust_same_method_cross_cohort": bool(desc_crossed)}
    classification = AN.classify(per_cohort, full_ceiling, active_gain, blocker=blocker)
    return {"endpoints": endpoints, "per_cohort_qualification": per_cohort,
            "full_ceiling": full_ceiling, "active_gain": active_gain,
            "classification": classification, "inference_detail": inference_detail,
            "materiality_margin": K.MATERIALITY_MARGIN, "maxt_draws": K.MAXT_DRAWS}
