"""Stage-B1b-2: compare four pre-registered routers over the deployable actions, using the A/B/C
signals from run_b1a_router_signals.py. NO learned gate -- an auditable nested conjunction:

  R0  raw evidence            argmax evidence gain, identity if max<=0      (the failed baseline)
  R1  null p-value only        eligible: conformal p < 0.05/n_actions        (A)
  R2  null + stability         R1 AND cross-subject reproducibility          (A+B)
  R3  null + stability + struct R2 AND class-structure certificates          (A+B+C)

If no action is eligible -> identity. Among eligible, rank by delta_snd, then disc<->gen
agreement, then smaller capacity (pooled < gen_oneshot < gen_iterative). Outcomes are reported
both OOF and full-refit. R3 must clear the frozen pass criteria before pooled-SPD / confirmatory.

  python -m h2cmi.analyze_b1a_routers --in results/h2cmi/b1a_router_signals_standard.jsonl \
      results/h2cmi/b1a_router_signals_hard.jsonl --out results/h2cmi/b1a_routers.report.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

IDENTITY = "identity"
ACTIONS = ("pooled_empirical_diag", "gen_oneshot_diag", "gen_iterative_diag")
CAPACITY = {"pooled_empirical_diag": 0, "gen_oneshot_diag": 1, "gen_iterative_diag": 2}
NULL_SCENARIOS = ("population_null", "matched_domain_null")
ROTATION = ("conditional_rotation", "cov_conditional_rotation")
COV_FAMILY = ("cov", "cov_prior", "cov_conditional_rotation")

PASS = dict(false_adapt_max=0.10, hardnull_harm_max=0.10, hardnull_dbacc_min=-0.005,
            disagreement_max=0.02, nonnull_harm_max=0.20, top1_min=0.50, regret_max=0.02,
            coverage_min=0.25)


def _fin(x):
    return x is not None and x == x


def _eligible(r, level) -> bool:
    n = r.get("n_actions", 3)
    if not (_fin(r.get("null_pvalue")) and r["null_pvalue"] < 0.05 / n):     # A
        return False
    if level == 1:
        return True
    stab = (_fin(r.get("transform_direction_cosine")) and r["transform_direction_cosine"] >= r.get("src_q05_cosine", -2)
            and (1.0 - r.get("crossfit_prediction_disagreement", 1.0)) >= r.get("src_q05_predstab", 2))
    if not stab:                                                            # B
        return False
    if level == 2:
        return True
    af = r.get("anchor_flip_rate")
    ok_anchor = (not _fin(af)) or af <= r.get("src_q95_anchor", -1) or r.get("anchor_n", 0) == 0
    struct = (ok_anchor and r.get("min_class_occupancy", 0) >= 0.05
              and _fin(r.get("delta_snd")) and r["delta_snd"] > r.get("src_q95_dsnd", 1e9))
    return bool(struct)                                                     # C


def _rank_key(r):
    return (-r.get("delta_snd", -1e9), -r.get("delta_disc_gen_agreement", -1e9), CAPACITY.get(r["action"], 9))


def route_unit(vm, router) -> str:
    if router == "R0":
        cand = {a: vm[a]["evidence_target"] for a in vm if _fin(vm[a].get("evidence_target"))}
        if not cand:
            return IDENTITY
        best = max(cand, key=cand.get)
        return best if cand[best] > 0 else IDENTITY
    level = {"R1": 1, "R2": 2, "R3": 3}[router]
    elig = [vm[a] for a in vm if _eligible(vm[a], level)]
    if not elig:
        return IDENTITY
    return min(elig, key=_rank_key)["action"]


def _outcome(vm, sel, metric):
    idk = "identity_bacc_uniform" if metric == "bacc_uniform" else "identity_grouped_oof_bacc"
    idv = next(iter(vm.values()))[idk]
    if sel == IDENTITY:
        return 0.0, idv
    d = (vm[sel][metric] - idv) if (_fin(vm[sel].get(metric)) and _fin(idv)) else float("nan")
    return d, idv


def analyze(rows) -> dict:
    units = defaultdict(dict)
    for r in rows:
        units[(r.get("difficulty", "standard"), r["scenario"], r["data_seed"], r["target_site"])][r["action"]] = r
    rep = {"pass_criteria": PASS, "by_difficulty": defaultdict(dict)}
    diffs = sorted({k[0] for k in units})
    for diff in diffs:
        du = {k: v for k, v in units.items() if k[0] == diff}
        for router in ("R0", "R1", "R2", "R3"):
            d_full, d_oof, adapt, top1, regret, sel_dis = [], [], [], [], [], []
            fa_null, harm_full_null, dbacc_full_null = [], [], []
            by_fam = defaultdict(list)
            for (df, scen, seed, site), vm in du.items():
                if IDENTITY in vm:                                          # safety: identity rows absent here
                    vm = {a: vm[a] for a in vm if a in ACTIONS}
                sel = route_unit(vm, router)
                df_full, _ = _outcome(vm, sel, "bacc_uniform")
                df_oof, _ = _outcome(vm, sel, "grouped_oof_bacc")
                d_full.append(df_full); d_oof.append(df_oof); adapt.append(sel != IDENTITY)
                # oracle (incl identity) by full bAcc
                cand = {IDENTITY: next(iter(vm.values()))["identity_bacc_uniform"]}
                cand.update({a: vm[a]["bacc_uniform"] for a in vm if _fin(vm[a].get("bacc_uniform"))})
                oa = max(cand, key=cand.get)
                top1.append(sel == oa); regret.append(cand[oa] - cand.get(sel, cand[IDENTITY]))
                if sel != IDENTITY:
                    sel_dis.append(vm[sel].get("crossfit_prediction_disagreement"))
                is_null = scen in NULL_SCENARIOS
                if is_null:
                    fa_null.append(sel != IDENTITY); harm_full_null.append(df_full < -1e-9); dbacc_full_null.append(df_full)
                else:
                    by_fam["all_shift"].append(df_full)
                    if scen in COV_FAMILY: by_fam["cov_family"].append(df_full)
                    if scen == "prior": by_fam["prior"].append(df_full)
                    if scen in ROTATION: by_fam["rotation"].append(df_full)
            nonnull_harm = float(np.mean([x < -1e-9 for x in by_fam["all_shift"]])) if by_fam["all_shift"] else float("nan")
            res = dict(
                n_units=len(du), adaptation_rate=float(np.mean(adapt)),
                false_adaptation_rate_null=float(np.mean(fa_null)) if fa_null else float("nan"),
                harm_rate_full=float(np.mean([x < -1e-9 for x in d_full if _fin(x)])),
                mean_dbacc_full_shift=float(np.nanmean(by_fam["all_shift"])) if by_fam["all_shift"] else float("nan"),
                mean_dbacc_full_cov=float(np.nanmean(by_fam["cov_family"])) if by_fam["cov_family"] else float("nan"),
                mean_dbacc_full_prior=float(np.nanmean(by_fam["prior"])) if by_fam["prior"] else float("nan"),
                mean_dbacc_full_rotation=float(np.nanmean(by_fam["rotation"])) if by_fam["rotation"] else float("nan"),
                top1_oracle_full=float(np.mean(top1)), mean_regret_full=float(np.nanmean(regret)),
                mean_selected_disagreement=float(np.nanmean(sel_dis)) if sel_dis else 0.0,
                coverage=float(np.mean(adapt)),
                hardnull_harm_full=float(np.mean(harm_full_null)) if harm_full_null else float("nan"),
                hardnull_mean_dbacc_full=float(np.nanmean(dbacc_full_null)) if dbacc_full_null else float("nan"),
                nonnull_harm_rate=nonnull_harm)
            if diff == "hard" and router == "R3":
                res["passes"] = _check_pass(res)
            rep["by_difficulty"][diff][router] = res
    rep["by_difficulty"] = {k: dict(v) for k, v in rep["by_difficulty"].items()}
    rep["R3_pass"] = _r3_pass(rep)
    return rep


def _check_pass(hard_res) -> dict:
    return dict(false_adapt=hard_res["false_adaptation_rate_null"] <= PASS["false_adapt_max"],
                hardnull_harm=hard_res["hardnull_harm_full"] <= PASS["hardnull_harm_max"],
                hardnull_dbacc=hard_res["hardnull_mean_dbacc_full"] >= PASS["hardnull_dbacc_min"],
                disagreement=hard_res["mean_selected_disagreement"] <= PASS["disagreement_max"])


def _r3_pass(rep) -> dict:
    out = {}
    std = rep["by_difficulty"].get("standard", {}).get("R3")
    hard = rep["by_difficulty"].get("hard", {}).get("R3")
    if std:
        out.update(std_false_adapt=std["false_adaptation_rate_null"] <= PASS["false_adapt_max"],
                   shift_utility=std["mean_dbacc_full_shift"] > 0,
                   nonnull_harm=std["nonnull_harm_rate"] <= PASS["nonnull_harm_max"],
                   top1=std["top1_oracle_full"] >= PASS["top1_min"],
                   regret=std["mean_regret_full"] <= PASS["regret_max"],
                   coverage=std["coverage"] >= PASS["coverage_min"])
    if hard:
        out.update(hard_false_adapt=hard["false_adaptation_rate_null"] <= PASS["false_adapt_max"],
                   hard_harm=hard["hardnull_harm_full"] <= PASS["hardnull_harm_max"],
                   hard_dbacc=hard["hardnull_mean_dbacc_full"] >= PASS["hardnull_dbacc_min"])
    out["ALL"] = bool(out) and all(out.values())
    return out


def _load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rep = analyze(_load(args.inp))
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=2)
    for diff, routers in rep["by_difficulty"].items():
        print(f"=== {diff} ===")
        for R, d in routers.items():
            print(f"  {R}: adapt={d['adaptation_rate']:.2f} false_adapt_null={d['false_adaptation_rate_null']:.2f} "
                  f"harm_full={d['harm_rate_full']:.2f} ΔbAcc_shift={d['mean_dbacc_full_shift']:+.3f} "
                  f"top1={d['top1_oracle_full']:.2f} regret={d['mean_regret_full']:.3f} cov={d['coverage']:.2f}")
    print(f"R3 pass: {rep['R3_pass']}")
    if args.out:
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
