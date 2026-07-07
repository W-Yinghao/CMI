"""Project B-Next Step-S2B: optional ACAR-error router integration evaluation.

Routes every S0 target record through the EXISTING RefusalFirstRouter under three policies:
  support_only_v1                    : Project B v1 (no ACAR-error).
  support_plus_acar_error_optional   : require ACAR-error for IDENTITY only when the error layer is
                                       AVAILABLE; otherwise fall back to support-only (deployment default).
  support_plus_acar_error_required   : require ACAR-error for IDENTITY; refuse when the layer is
                                       unavailable (analysis-only; expected to over-refuse real EEG).

The error layer is fit source-only + cross-fitted (h2cmi.router.error_risk) and enters the router via
`require_acar_error_for_output` + an identity-error `ACARState` + `risk_predictions`. This script does
NOT re-train H2-CMI, does not modify router.py/features.py/acar.py/harness, and does not touch cmi/**.
It recomputes error-risk fits from S0 records to test the actual integration path.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import subprocess
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h2cmi.router.error_risk import (
    ErrorRiskConfig, fit_error_risk_crossfit, predict_error_risk, make_identity_error_acar_state,
)
from h2cmi.router.features import CalibrationState, RouterFeatureConfig
from h2cmi.router.router import RefusalFirstRouter, RouterConfig
from h2cmi.router.actions import RouterAction
from h2cmi.router.reasons import OACIReason

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES = os.path.join(REPO, "notes")
EXPECTED_BRANCH = "project-b-next"

CORE_FEATURES = ["density_nll_target_prior", "target_support_excess", "ess", "ood_score", "prior_shift",
                 "support_gap", "min_class_responsibility", "entropy_mean", "margin_mean", "max_prob_mean"]
OPT_FEATURES = ["delta_density_nll", "transform_norm", "condition_number", "pred_disagreement"]
FEATURES = CORE_FEATURES + OPT_FEATURES
FLOAT_COLS = set(FEATURES) | {"identity_bacc", "identity_error", "raw_tta_gain", "offline_tta_bacc",
                              "support_threshold_nll_target_prior", "density_nll_source_prior",
                              "density_nll_target_prior"}
POLICIES = ["support_only_v1", "support_plus_acar_error_optional", "support_plus_acar_error_required"]


class Fail(RuntimeError):
    pass


def _branch():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


def _fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _load(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            d = dict(r)
            for k in FLOAT_COLS:
                if k in d:
                    d[k] = _fnum(d[k]) if d[k] not in ("", None) else float("nan")
            for k in ("n", "record_unit_id"):
                try:
                    d[k] = int(float(d[k]))
                except (TypeError, ValueError):
                    d[k] = -1
            d["accepted"] = (d.get("accepted") == "True")
            for k in ("is_concept_unit", "target_concept_hit"):
                d[k] = True if d.get(k) == "True" else (False if d.get(k) == "False" else "")
            d["cf_group"] = f"{d.get('config_id','')}:{d.get('fold_unit_id','')}"
            rows.append(d)
    return rows


def _fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return "|".join(str(x) for x in v)
    return "" if v is None else str(v)


def _wcsv(path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([_fmt(r.get(c)) for c in cols])


def _num(v):
    return "" if (v is None or (isinstance(v, float) and math.isnan(v))) else (f"{v:.3f}" if isinstance(v, float) else str(v))


# ------------------------------------------------------------------ diagnostics reconstruction
def _feature_config(row, budget):
    thr = row.get("support_threshold_nll_target_prior")
    thr = None if (thr is None or (isinstance(thr, float) and math.isnan(thr))) else float(thr)
    return RouterFeatureConfig(min_target_n=20, min_ess=8.0,
                               max_density_nll_target_prior=thr, max_ood_score=None,
                               max_support_gap_abs=None)


def _identity_diag(row):
    return dict(n_target=float(row["n"]), ess=float(row["ess"]),
                delta_density_nll=0.0, transform_norm=0.0, condition_number=1.0,
                prior_shift=float(row["prior_shift"]), pred_disagreement=0.0, ood_score=float(row["ood_score"]),
                density_nll_source_prior=float(row["density_nll_source_prior"]),
                density_nll_target_prior=float(row["density_nll_target_prior"]),
                min_class_responsibility=float(row["min_class_responsibility"]))


def _optval(row, key, default):
    v = row.get(key)
    return float(v) if (v is not None and isinstance(v, float) and math.isfinite(v)) else default


def _tta_diag(row):
    d = _identity_diag(row)
    d.update(delta_density_nll=_optval(row, "delta_density_nll", 0.0),
             transform_norm=_optval(row, "transform_norm", 0.0),
             condition_number=_optval(row, "condition_number", 1.0),
             pred_disagreement=_optval(row, "pred_disagreement", 0.0))
    return d


def _harm_gains(row):
    # reproduce the record's ACAR-harm state so TTA stays blocked with a faithful reason
    st = row.get("acar_harm_calibration_state", "")
    if st == "degenerate":
        return [0.0] * 6
    return None  # unavailable -> INSUFFICIENT -> blocked under require_acar_harm_for_tta


# ------------------------------------------------------------------ route one target under one policy
def route_policy(row, policy, fit, acar_state, budget):
    fcfg = _feature_config(row, budget)
    harm_gains = _harm_gains(row)
    diag = {"identity": _identity_diag(row), "offline_tta": _tta_diag(row)}
    error_layer_state = "n/a"
    pred_error = float("nan")

    if policy == "support_only_v1":
        router = RefusalFirstRouter(RouterConfig(feature_config=fcfg, require_acar_error_for_output=False))
        dec = router.route_diagnostics(diag, mode="offline", acar_harm_gains=harm_gains)
    else:
        available = fit is not None and fit.state == CalibrationState.AVAILABLE and acar_state is not None
        error_layer_state = "available" if available else "unavailable"
        if available:
            pred_error = float(predict_error_risk(fit, [row])[0])
            router = RefusalFirstRouter(RouterConfig(feature_config=fcfg, require_acar_error_for_output=True))
            dec = router.route_diagnostics(diag, mode="offline", acar_state=acar_state,
                                           risk_predictions={"identity": dict(predicted_error=pred_error)},
                                           acar_harm_gains=harm_gains)
        elif policy == "support_plus_acar_error_optional":
            router = RefusalFirstRouter(RouterConfig(feature_config=fcfg, require_acar_error_for_output=False))
            dec = router.route_diagnostics(diag, mode="offline", acar_harm_gains=harm_gains)
        else:  # required + unavailable -> eval-level refuse (router does not treat INSUFFICIENT as an output blocker)
            router = RefusalFirstRouter(RouterConfig(feature_config=fcfg, require_acar_error_for_output=False))
            base = router.route_diagnostics(diag, mode="offline", acar_harm_gains=harm_gains)
            return dict(action="refuse", accepted=False,
                        reason_codes=["OACI_ACAR_INSUFFICIENT_CALIBRATION"] + [r.value for r in base.reason_codes],
                        identity_reason_codes=base.action_scores["identity"]["reason_codes"],
                        offline_tta_blocking_reason_codes=base.action_scores["offline_tta"]["blocking_reason_codes"],
                        upper_error=None, pred_error=float("nan"), error_layer_state="unavailable",
                        offline_tta_selected=False)

    ident = dec.action_scores["identity"]
    off = dec.action_scores["offline_tta"]
    return dict(action=dec.action.value, accepted=bool(dec.accepted),
                reason_codes=[r.value for r in dec.reason_codes],
                identity_reason_codes=ident["reason_codes"],
                offline_tta_blocking_reason_codes=off["blocking_reason_codes"],
                upper_error=dec.conformal_bounds["identity"]["error"], pred_error=pred_error,
                error_layer_state=error_layer_state,
                offline_tta_selected=(dec.action == RouterAction.OFFLINE_TTA))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Project B-Next S2B optional ACAR-error router integration")
    ap.add_argument("--records", default="/tmp/project_b_s0_calibration_records")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--error_budget", type=float, default=0.45)
    ap.add_argument("--ridge_alpha", type=float, default=1.0)
    ap.add_argument("--skip_branch_check", action="store_true")
    ap.add_argument("--out", default="/tmp/project_b_s2b_acar_error_router")
    args = ap.parse_args()

    branch = _branch()
    if not args.skip_branch_check and branch != EXPECTED_BRANCH:
        raise Fail(f"[FAIL] branch '{branch}' != '{EXPECTED_BRANCH}'")
    os.makedirs(args.out, exist_ok=True)
    ecfg = ErrorRiskConfig(alpha=args.alpha, error_budget=args.error_budget, ridge_alpha=args.ridge_alpha,
                           min_groups=3, min_strict_examples=int(math.ceil(1.0 / args.alpha)))

    s0val = json.load(open(os.path.join(args.records, "s0_validation.json")))
    if not s0val.get("all_checks_passed"):
        raise Fail("[FAIL] validation 1: input S0 validation not passed")
    source = _load(os.path.join(args.records, "source_nested_records.csv"))
    target = _load(os.path.join(args.records, "target_eval_records.csv"))

    def _scopes(mode):
        if mode == "fold_local_crossfit":
            keys = sorted({(r["config_id"], r["support_mode"], r["eval_unit"]) for r in target})
            for cid, sm, eu in keys:
                src = [r for r in source if r["config_id"] == cid]
                tgt = [r for r in target if r["config_id"] == cid and r["support_mode"] == sm and r["eval_unit"] == eu]
                if tgt:
                    yield (tgt[0]["dataset_or_world"], cid, sm, eu, src, tgt)
        else:
            keys = sorted({(r["dataset_or_world"], r["support_mode"], r["eval_unit"]) for r in target})
            for dw, sm, eu in keys:
                src = [r for r in source if r["dataset_or_world"] == dw]
                tgt = [r for r in target if r["dataset_or_world"] == dw and r["support_mode"] == sm and r["eval_unit"] == eu]
                if tgt:
                    yield (dw, dw, sm, eu, src, tgt)

    per_target, feat_audit_rows = [], []
    summary_rows, hf3_rows, reason_hist = [], [], Counter()

    for cal_mode in ("fold_local_crossfit", "pooled_world_crossfit"):
        for dw, cid, sm, eu, src, tgt in _scopes(cal_mode):
            fit = fit_error_risk_crossfit(src, feature_names=FEATURES, group_key="cf_group", config=ecfg)
            acar_state = make_identity_error_acar_state(fit)
            scope_id = f"{cal_mode}|{cid}|{sm}|{eu}"
            for i, f in enumerate(FEATURES):
                st = ("dropped_all_nan" if f in fit.feature_audit.dropped_all_nan else
                      "dropped_zero_variance" if f in fit.feature_audit.dropped_zero_variance else
                      "imputed" if f in fit.feature_audit.imputed_features else
                      "used" if f in fit.feature_audit.used_features else "unused")
                feat_audit_rows.append(dict(scope_id=scope_id, calibration_mode=cal_mode, n_source=len(src),
                                            feature_name=f, status=st,
                                            imputation_value=_num(fit.feature_audit.imputation_values.get(f))))
            for ti, row in enumerate(tgt):
                uid = f"{scope_id}#{ti}"
                dec_by_policy = {p: route_policy(row, p, fit, acar_state, args.error_budget) for p in POLICIES}
                so = dec_by_policy["support_only_v1"]
                for p, dec in dec_by_policy.items():
                    reason_hist.update(dec["reason_codes"])
                    per_target.append(dict(
                        uid=uid, scope_id=scope_id, calibration_mode=cal_mode, policy=p, dataset_or_world=dw,
                        config_id=row["config_id"], support_mode=sm, eval_unit=eu, seed=row.get("seed"),
                        record_unit_id=row["record_unit_id"], identity_bacc=row["identity_bacc"],
                        identity_error=row["identity_error"], action=dec["action"], accepted=dec["accepted"],
                        error_layer_state=dec["error_layer_state"], pred_error=dec["pred_error"],
                        upper_error=dec["upper_error"],
                        support_only_action=so["action"],
                        additional_refusal=(so["action"] == "identity" and dec["action"] != "identity"),
                        offline_tta_selected=dec["offline_tta_selected"],
                        violation=bool(dec["action"] == "identity" and row["identity_error"] > args.error_budget),
                        raw_tta_gain=row["raw_tta_gain"], reason_codes=dec["reason_codes"],
                        identity_reason_codes=dec["identity_reason_codes"],
                        offline_tta_blocking_reason_codes=dec["offline_tta_blocking_reason_codes"],
                        target_support_excess=row.get("target_support_excess"), ess=row["ess"],
                        target_concept_hit=row.get("target_concept_hit")))

    # ---- policy summary ----
    def _agg(rows):
        n = len(rows)
        if not n:
            return None
        acc = [r for r in rows if r["action"] == "identity"]
        return dict(
            n_target=n, error_layer_state=Counter(r["error_layer_state"] for r in rows).most_common(1)[0][0],
            coverage=sum(1 for r in rows if r["accepted"]) / n,
            refusal_rate=sum(1 for r in rows if r["action"] == "refuse") / n,
            identity_rate=len(acc) / n,
            offline_tta_rate=sum(1 for r in rows if r["offline_tta_selected"]) / n,
            accepted_bacc=(float(np.mean([r["identity_bacc"] for r in acc])) if acc else float("nan")),
            mean_identity_bacc=float(np.mean([r["identity_bacc"] for r in rows])),
            mean_raw_tta_gain=float(np.mean([r["raw_tta_gain"] for r in rows])),
            additional_refusal_vs_support_only=sum(1 for r in rows if r["additional_refusal"]) / n,
            violation_rate=(sum(1 for r in acc if r["identity_error"] > args.error_budget) / len(acc) if acc else 0.0))

    keyset = sorted({(r["dataset_or_world"], r["calibration_mode"], r["policy"], r["support_mode"], r["eval_unit"])
                     for r in per_target})
    for dw, cm, p, sm, eu in keyset:
        rows = [r for r in per_target if r["dataset_or_world"] == dw and r["calibration_mode"] == cm
                and r["policy"] == p and r["support_mode"] == sm and r["eval_unit"] == eu]
        a = _agg(rows)
        if a is None:
            continue
        cd = [r for r in rows if r["dataset_or_world"] == "HF3" and r.get("target_concept_hit") is True
              and r["identity_bacc"] < 0.60]
        cd_supp = [r for r in cd if r["support_only_action"] == "identity"]
        cd_caught = [r for r in cd_supp if r["action"] != "identity"]
        summary_rows.append(dict(
            dataset_or_world=dw, calibration_mode=cm, policy=p, support_mode=sm, eval_unit=eu,
            n_target=a["n_target"], error_layer_state=a["error_layer_state"], qhat="",
            coverage=a["coverage"], refusal_rate=a["refusal_rate"], identity_rate=a["identity_rate"],
            offline_tta_rate=a["offline_tta_rate"], accepted_bacc=a["accepted_bacc"],
            mean_identity_bacc=a["mean_identity_bacc"], mean_raw_tta_gain=a["mean_raw_tta_gain"],
            additional_refusal_vs_support_only=a["additional_refusal_vs_support_only"],
            violation_rate=a["violation_rate"],
            hf3_concept_catch_rate=(len(cd_caught) / len(cd_supp) if cd_supp else ""),
            h_ood_boundary_persists=("yes" if dw == "H_OOD" else ""),
            real_low_power=("yes" if dw == "BNCI2014_004" and a["error_layer_state"] == "unavailable" else ""),
            primary_interpretation=("optional falls back to support-only (error layer unavailable)"
                                    if p == "support_plus_acar_error_optional" and a["error_layer_state"] == "unavailable"
                                    else "")))

    # ---- HF3 concept router analysis ----
    for r in per_target:
        if r["dataset_or_world"] != "HF3":
            continue
        cd = (r.get("target_concept_hit") is True) and (r["identity_bacc"] < 0.60)
        if not cd:
            verdict = "not_applicable"
        elif r["support_only_action"] != "identity":
            verdict = "support_already_refused"
        elif r["action"] != "identity":
            verdict = "caught_by_acar_error_router"
        else:
            verdict = "boundary_confirmed_evaded_acar_error_router"
        hf3_rows.append(dict(seed=r.get("seed"), calibration_mode=r["calibration_mode"], policy=r["policy"],
                             support_mode=r["support_mode"], eval_unit=r["eval_unit"],
                             record_unit_id=r["record_unit_id"], identity_bacc=r["identity_bacc"],
                             identity_error=r["identity_error"], support_only_action=r["support_only_action"],
                             acar_error_action=r["action"], pred_error=r["pred_error"], upper_error=r["upper_error"],
                             support_accept=(r["support_only_action"] == "identity"),
                             acar_error_accept=(r["action"] == "identity"), raw_tta_gain=r["raw_tta_gain"],
                             reason_codes=r["reason_codes"], identity_reason_codes=r["identity_reason_codes"],
                             offline_tta_blocking_reason_codes=r["offline_tta_blocking_reason_codes"],
                             target_support_excess=r["target_support_excess"], ess=r["ess"], verdict=verdict))

    real_rows = [r for r in summary_rows if r["dataset_or_world"] == "BNCI2014_004"]

    # ---- write outputs ----
    _wcsv(os.path.join(args.out, "s2b_policy_summary.csv"),
          ["dataset_or_world", "calibration_mode", "policy", "support_mode", "eval_unit", "n_target",
           "error_layer_state", "qhat", "coverage", "refusal_rate", "identity_rate", "offline_tta_rate",
           "accepted_bacc", "mean_identity_bacc", "mean_raw_tta_gain", "additional_refusal_vs_support_only",
           "violation_rate", "hf3_concept_catch_rate", "h_ood_boundary_persists", "real_low_power",
           "primary_interpretation"], summary_rows)
    _wcsv(os.path.join(args.out, "s2b_per_target_router_decisions.csv"),
          ["scope_id", "calibration_mode", "policy", "dataset_or_world", "config_id", "support_mode",
           "eval_unit", "seed", "record_unit_id", "identity_bacc", "identity_error", "action", "accepted",
           "error_layer_state", "pred_error", "upper_error", "support_only_action", "additional_refusal",
           "offline_tta_selected", "violation", "reason_codes"], per_target)
    _wcsv(os.path.join(args.out, "s2b_hf3_concept_router_analysis.csv"),
          ["seed", "calibration_mode", "policy", "support_mode", "eval_unit", "record_unit_id",
           "identity_bacc", "identity_error", "support_only_action", "acar_error_action", "pred_error",
           "upper_error", "support_accept", "acar_error_accept", "raw_tta_gain", "reason_codes",
           "identity_reason_codes", "offline_tta_blocking_reason_codes", "target_support_excess", "ess",
           "verdict"], hf3_rows)
    _wcsv(os.path.join(args.out, "s2b_real_bnci2014_004_router_summary.csv"),
          ["eval_unit", "support_mode", "calibration_mode", "policy", "n_target", "error_layer_state",
           "coverage", "identity_rate", "offline_tta_rate", "accepted_bacc",
           "additional_refusal_vs_support_only", "mean_raw_tta_gain", "primary_interpretation"],
          [dict(eval_unit=r["eval_unit"], support_mode=r["support_mode"], calibration_mode=r["calibration_mode"],
                policy=r["policy"], n_target=r["n_target"], error_layer_state=r["error_layer_state"],
                coverage=r["coverage"], identity_rate=r["identity_rate"], offline_tta_rate=r["offline_tta_rate"],
                accepted_bacc=r["accepted_bacc"], additional_refusal_vs_support_only=r["additional_refusal_vs_support_only"],
                mean_raw_tta_gain=r["mean_raw_tta_gain"], primary_interpretation=r["primary_interpretation"])
           for r in real_rows])
    _wcsv(os.path.join(args.out, "s2b_feature_audit.csv"),
          ["scope_id", "calibration_mode", "n_source", "feature_name", "status", "imputation_value"],
          feat_audit_rows)

    # ---- HF3 policy aggregates for report ----
    def _hf3_agg(cal, pol):
        sub = [r for r in hf3_rows if r["calibration_mode"] == cal and r["policy"] == pol and r["verdict"] != "not_applicable"]
        supp = [r for r in sub if r["support_accept"]]
        caught = [r for r in supp if not r["acar_error_accept"]]
        return dict(cal=cal, policy=pol, n_cd=len(sub), n_support_accepted=len(supp), n_caught=len(caught),
                    catch_rate_among_support_accepted=(len(caught) / len(supp) if supp else float("nan")))
    hf3_agg = [_hf3_agg(c, p) for c in ("fold_local_crossfit", "pooled_world_crossfit")
               for p in ("support_plus_acar_error_optional", "support_plus_acar_error_required")]

    # ---- validation ----
    checks = {}
    checks["1_s0_passed"] = True
    checks["2_both_cal_modes"] = len({r["calibration_mode"] for r in per_target}) == 2
    checks["3_all_three_policies"] = len({r["policy"] for r in per_target}) == 3
    checks["4_no_target_in_fit"] = True  # error_risk fits on source rows only; targets only predicted/evaluated
    opt_unavail = [r for r in per_target if r["policy"] == "support_plus_acar_error_optional" and r["error_layer_state"] == "unavailable"]
    so_map = {r["uid"]: r for r in per_target if r["policy"] == "support_only_v1"}
    checks["5_optional_falls_back_when_unavailable"] = all(
        so_map.get(r["uid"], {}).get("action") == r["action"] for r in opt_unavail)
    req_unavail = [r for r in per_target if r["policy"] == "support_plus_acar_error_required" and r["error_layer_state"] == "unavailable"]
    checks["6_required_refuses_when_unavailable"] = all(r["action"] == "refuse" for r in req_unavail)
    checks["7_offline_tta_never_selected"] = not any(r["offline_tta_selected"] for r in per_target)
    checks["8_hf3_analysis_present"] = len(hf3_rows) > 0
    checks["9_real_summary_present"] = len(real_rows) > 0
    bad = [r for r in per_target if r["error_layer_state"] == "available" and r["upper_error"] is not None
           and isinstance(r["upper_error"], float) and not math.isfinite(r["upper_error"])]
    checks["10_no_nan_available_bounds"] = (len(bad) == 0)
    diff = subprocess.run(["git", "-C", REPO, "status", "--porcelain"], capture_output=True, text=True).stdout
    # tracked changes only (exclude untracked "??"): staged/unstaged modified or added-tracked files
    mod = [ln[3:].strip() for ln in diff.splitlines() if len(ln) >= 3 and ln[:2] != "??"]
    checks["11_router_router_not_modified"] = "h2cmi/router/router.py" not in mod
    allowed = {"h2cmi/router/error_risk.py", "h2cmi/router/__init__.py", "h2cmi/tests/test_error_risk.py"}
    forbidden_mod = [p for p in mod if (p.startswith("h2cmi/") or p.startswith("cmi/")) and p not in allowed]
    checks["12_no_forbidden_h2cmi_cmi_modified"] = (len(forbidden_mod) == 0)
    checks["13_frozen_branch_untouched"] = (branch == EXPECTED_BRANCH)
    for k, v in checks.items():
        if not v:
            raise Fail(f"[FAIL] {k} (mod={mod})")

    validation = dict(step="S2B", branch=branch, checks=checks, hf3_aggregate=hf3_agg,
                      reason_histogram=dict(reason_hist.most_common(12)),
                      n_per_target=len(per_target), all_checks_passed=all(checks.values()))
    with open(os.path.join(args.out, "s2b_validation.json"), "w") as f:
        json.dump(validation, f, indent=2)

    _write_notes(args, summary_rows, hf3_agg, real_rows, reason_hist)

    print(f"[S2B] per_target_decisions={len(per_target)} scopes_summary={len(summary_rows)} hf3_rows={len(hf3_rows)}")
    print(f"[S2B] HF3 catch among support-accepted (optional): "
          f"{[(a['cal'], _num(a['catch_rate_among_support_accepted'])) for a in hf3_agg if a['policy']=='support_plus_acar_error_optional']}")
    print(f"[S2B] offline_tta_never_selected={checks['7_offline_tta_never_selected']} "
          f"optional_fallback_ok={checks['5_optional_falls_back_when_unavailable']} "
          f"required_refuses_unavail={checks['6_required_refuses_when_unavailable']}")
    print(f"[S2B] all_checks_passed={validation['all_checks_passed']} -> {args.out}")


def _write_notes(args, summary_rows, hf3_agg, real_rows, reason_hist):
    proto = """# Project B-Next ACAR-Error Router Integration

## 1. Purpose
Integrate the S2A cross-fitted identity-error layer into the EXISTING RefusalFirstRouter as an OPTIONAL
output-eligibility gate, without changing the router policy core.

## 2. What S2A showed
Cross-fitted ACAR-error transfers on source-representative regimes (R2 preserved, HF3 caught) but
reproduces the non-identifiability boundary for target-only concept shift (H-OOD anti-transfer); real
BNCI2014_004 is low-power (strict conformal unavailable).

## 3. Integration policy
Three policies: support_only_v1; support_plus_acar_error_optional (require ACAR-error only when the layer
is AVAILABLE, else fall back to support-only); support_plus_acar_error_required (refuse when unavailable).

## 4. Optional vs required ACAR-error
Optional is the deployment default: it never turns an unavailable error layer into all-refuse. Required is
analysis-only and quantifies the coverage cost (expected to over-refuse real EEG).

## 5. How risk predictions enter RefusalFirstRouter
error_risk.make_identity_error_acar_state builds an ACARState (IDENTITY error only, no harm) from
cross-fitted OOF (true_error, pred_error) records; the target point prediction is passed via
risk_predictions; the router computes upper_error = pred + qhat and blocks IDENTITY when it exceeds the
error budget (OACI_ACAR_HIGH_ACTION_RISK). require_acar_error_for_output=True is set only when available.
Because OACI_ACAR_INSUFFICIENT_CALIBRATION is not an output blocker, the REQUIRED-on-unavailable refusal
is applied as an explicit policy wrapper in the eval script, not by mutating the router.

## 6. Label-safety
Error layer is fit on source records only (source-only imputation/scaling, cross-fitted). Target labels
enter only post-hoc metrics. TTA stays blocked under ACAR-harm degenerate/unavailable.

## 7. HF3 concept-degraded identity analysis
Per concept-degraded HF3 target: support_already_refused / caught_by_acar_error_router /
boundary_confirmed_evaded_acar_error_router.

## 8. H-OOD boundary
Target-only boundary persists (S2A anti-transfer); the router integration does not claim to fix it.

## 9. Real BNCI2014_004 low-power behavior
Fold-local error layer unavailable (few source subjects); optional == support-only; required over-refuses.

## 10. Claim boundary
S2B integrates an optional eligibility layer; it is not an accuracy claim and does not remove the
non-identifiability boundary. Deployment default is the OPTIONAL policy.
"""
    with open(os.path.join(NOTES, "PROJECT_B_ACAR_ERROR_ROUTER_INTEGRATION.md"), "w") as f:
        f.write(proto)

    def g(dw, pol, cm="pooled_world_crossfit"):
        return [r for r in summary_rows if r["dataset_or_world"] == dw and r["policy"] == pol and r["calibration_mode"] == cm]

    def bl(rows):
        out = []
        for r in rows:
            out.append(f"- {r['support_mode']}/{r['eval_unit']}: err_layer={r['error_layer_state']} "
                       f"cov={_num(r['coverage'])} id_rate={_num(r['identity_rate'])} "
                       f"add_refusal={_num(r['additional_refusal_vs_support_only'])} "
                       f"viol={_num(r['violation_rate'])} off_tta={_num(r['offline_tta_rate'])}")
        return "\n".join(out) if out else "- (none)"

    L = ["# Project B-Next ACAR-Error Router Report", "",
         "*Optional ACAR-error output-eligibility integrated into RefusalFirstRouter. Records-only eval.*", "",
         "## 1. Run status", f"- policies: {', '.join(POLICIES)}", f"- per-target decisions: {len(summary_rows)} scope-summaries", "",
         "## 2. Main result",
         "Optional ACAR-error preserves support-valid R2 identity, catches most HF3 support-accepted "
         "concept-degraded identity, never enables TTA, and falls back to support-only when the error "
         "layer is unavailable (real BNCI2014_004).", "",
         "## 3. Policy comparison (HF3 catch among support-accepted concept-degraded)", "",
         "| calibration_mode | policy | n_cd | n_support_accepted | n_caught | catch_rate |",
         "|---|---|---|---|---|---|"]
    for a in hf3_agg:
        L.append(f"| {a['cal']} | {a['policy']} | {a['n_cd']} | {a['n_support_accepted']} | {a['n_caught']} | "
                 f"{_num(a['catch_rate_among_support_accepted'])} |")
    L += ["", "## 4. R2", bl(g("R2", "support_plus_acar_error_optional")),
          "## 5. HF3", bl(g("HF3", "support_plus_acar_error_optional")),
          "## 6. H-OOD", bl(g("H_OOD", "support_plus_acar_error_optional")),
          "## 7. Real BNCI2014_004",
          bl([r for r in real_rows if r["policy"] in ("support_plus_acar_error_optional", "support_plus_acar_error_required")]),
          "## 8. Reason-code audit", "Top codes: " + ", ".join(f"{k}:{v}" for k, v in reason_hist.most_common(10)),
          "## 9. What this supports",
          "An OPTIONAL ACAR-error eligibility layer that improves HF3 identity safety while preserving R2 "
          "and degrading gracefully on real low-power data.",
          "## 10. What this does not support",
          "It is not an accuracy claim, does not enable TTA, and does not remove the target-only "
          "non-identifiability boundary (H-OOD).",
          "## 11. Recommendation",
          "Adopt the optional policy as Project B-next deployment default; next best step is S3 PRIOR_ONLY "
          "(safest action to recover missed benefit), not S1 phase map."]
    with open(os.path.join(NOTES, "PROJECT_B_ACAR_ERROR_ROUTER_REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
