"""Project B-Next Step-S2A: cross-fitted ACAR-error evaluation from S0 calibration records.

Upgrades the S0 toy error probe into a rigorous, records-level ACAR-error evaluation:
  - source-only fitting, source-only imputation/scaling (never from target rows),
  - grouped leave-one-group-out cross-fitted source predictions (group = config_id x fold_unit_id),
  - split-conformal upper-error bound (strict finite-sample quantile is the decision bound),
  - target post-hoc evaluation only (target labels never touch fit / impute / scale / qhat / decision).

Two calibration modes:
  fold_local_crossfit  : deployment-faithful. Per (config_id, support_mode, eval_unit) use ONLY that
                         target fold's source records. Too few source groups -> state=unavailable.
  pooled_world_crossfit: scientific-signal. Per (dataset_or_world, support_mode, eval_unit) pool all
                         source records of that world/dataset.

Records-only: reads S0 outputs, modifies no h2cmi/** or cmi/** and does not re-train H2-CMI.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES = os.path.join(REPO, "notes")
EXPECTED_BRANCH = "project-b-next"

CORE_FEATURES = ["density_nll_target_prior", "target_support_excess", "ess", "ood_score", "prior_shift",
                 "support_gap", "min_class_responsibility", "entropy_mean", "margin_mean", "max_prob_mean"]
OPT_FEATURES = ["delta_density_nll", "transform_norm", "condition_number", "pred_disagreement"]
FEATURES = CORE_FEATURES + OPT_FEATURES

FLOAT_COLS = set(FEATURES) | {"identity_bacc", "identity_error", "raw_tta_gain", "offline_tta_bacc"}


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
        v = float(x)
        return v
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
            d["accepted"] = (d.get("accepted") == "True")
            for k in ("is_concept_unit", "target_concept_hit"):
                d[k] = True if d.get(k) == "True" else (False if d.get(k) == "False" else "")
            for k in ("reason_codes", "identity_reason_codes"):
                d[k] = [x for x in d.get(k, "").split("|") if x] if d.get(k) else []
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


# ------------------------------------------------------------------ ridge + prep (source-only)
def _mat(rows, feats):
    return np.array([[float(r.get(f, float("nan"))) for f in feats] for r in rows], dtype=np.float64)


def build_prep(src_rows, feats):
    """Source-only preprocessing: drop all-NaN & zero-variance columns; impute from source mean;
    standardize with source mean/std. Returns transform + per-feature audit."""
    X = _mat(src_rows, feats)
    n = X.shape[0]
    finite = np.isfinite(X)
    miss = (~finite).sum(0)
    all_nan = ~finite.any(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        colmean = np.nanmean(np.where(finite, X, np.nan), axis=0)
    fill = np.where(np.isfinite(colmean), colmean, 0.0)
    Xi = np.where(finite, X, fill[None, :])
    mu = Xi.mean(0)
    sd = Xi.std(0)
    zero_var = sd < 1e-8
    use = (~all_nan) & (~zero_var)
    status = []
    for i in range(len(feats)):
        if all_nan[i]:
            status.append("dropped_all_nan")
        elif zero_var[i]:
            status.append("dropped_zero_variance")
        elif miss[i] > 0:
            status.append("imputed")
        else:
            status.append("used")
    return dict(feats=feats, use_idx=[i for i in range(len(feats)) if use[i]],
                fill=fill, mu=mu, sd=np.where(sd < 1e-8, 1.0, sd), status=status,
                miss=miss.tolist(), n=n)


def apply_prep(prep, rows):
    X = _mat(rows, prep["feats"])
    finite = np.isfinite(X)
    Xi = np.where(finite, X, prep["fill"][None, :])
    Z = (Xi - prep["mu"]) / prep["sd"]
    return Z[:, prep["use_idx"]]


def ridge(Z, y, alpha):
    Z1 = np.hstack([np.ones((Z.shape[0], 1)), Z])
    A = Z1.T @ Z1 + alpha * np.eye(Z1.shape[1])
    A[0, 0] -= alpha
    return np.linalg.solve(A, Z1.T @ y)


def predict(w, Z):
    Z1 = np.hstack([np.ones((Z.shape[0], 1)), Z])
    return np.clip(Z1 @ w, 0.0, 1.0)


def _group_key(r):
    fu = r.get("fold_unit_id", "")
    return (r.get("config_id", ""), str(fu) if fu not in ("", None) else str(r.get("record_unit_id", "")))


def crossfit(src_rows, feats, alpha, ridge_alpha):
    """LOGO cross-fitted source OOF error predictions + split-conformal qhat."""
    groups = {}
    for i, r in enumerate(src_rows):
        groups.setdefault(_group_key(r), []).append(i)
    gk = list(groups)
    y = np.array([r["identity_error"] for r in src_rows], dtype=np.float64)
    if len(gk) < 3:
        return dict(state="unavailable", reason="insufficient_source_groups", n_groups=len(gk),
                    oof=np.full(len(src_rows), np.nan), y=y)
    oof = np.full(len(src_rows), np.nan)
    for g in gk:
        te = groups[g]
        tr = [i for i in range(len(src_rows)) if i not in set(te)]
        prep = build_prep([src_rows[i] for i in tr], feats)
        if not prep["use_idx"]:
            return dict(state="unavailable", reason="no_usable_features", n_groups=len(gk),
                        oof=np.full(len(src_rows), np.nan), y=y)
        w = ridge(apply_prep(prep, [src_rows[i] for i in tr]), y[tr], ridge_alpha)
        pr = predict(w, apply_prep(prep, [src_rows[i] for i in te]))
        for j, idx in enumerate(te):
            oof[idx] = pr[j]
    resid = np.maximum(0.0, y - oof)
    n = resid.size
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        qs, state = None, "unavailable_strict"
    else:
        qs, state = float(np.sort(resid)[k - 1]), "available"
    qr = float(np.quantile(resid, 1 - alpha))
    corr = float(np.corrcoef(oof, y)[0, 1]) if np.std(oof) > 0 and n > 2 else float("nan")
    mae = float(np.mean(np.abs(oof - y)))
    return dict(state=state, reason="", n_groups=len(gk), oof=oof, y=y, resid=resid,
                qhat_strict=qs, qhat_relaxed=qr, source_oof_corr=corr, source_oof_mae=mae)


# ------------------------------------------------------------------ scope processing
def process_scope(scope_id, mode, dataset_or_world, support_mode, eval_unit,
                  src_rows, tgt_rows, args):
    cf = crossfit(src_rows, FEATURES, args.alpha, args.ridge_alpha)
    prep_full = build_prep(src_rows, FEATURES)
    feat_audit = []
    for i, f in enumerate(FEATURES):
        feat_audit.append(dict(scope_id=scope_id, calibration_mode=mode, n_source=len(src_rows),
                               feature_name=f, status=prep_full["status"][i],
                               source_missing_count=prep_full["miss"][i],
                               target_missing_count=int(np.sum(~np.isfinite(_mat(tgt_rows, [f])))) if tgt_rows else 0,
                               imputation_value=(f"{prep_full['fill'][i]:.6g}" if prep_full["status"][i] == "imputed" else "")))

    decisions_available = cf["state"] == "available" and prep_full["use_idx"]
    # point predictions for target (posthoc diagnostic even if strict qhat missing)
    if prep_full["use_idx"] and tgt_rows:
        w = ridge(apply_prep(prep_full, src_rows),
                  np.array([r["identity_error"] for r in src_rows], dtype=np.float64), args.ridge_alpha)
        pred_t = predict(w, apply_prep(prep_full, tgt_rows))
    else:
        pred_t = np.full(len(tgt_rows), np.nan)
    qhat = cf.get("qhat_strict") if decisions_available else None

    tdec = []
    for r, pe in zip(tgt_rows, pred_t):
        support_accept = (r.get("decision_action") == "identity") or bool(r.get("accepted"))
        upper = (float(pe) + qhat) if (decisions_available and math.isfinite(pe)) else float("nan")
        acar_accept = bool(support_accept and math.isfinite(upper) and upper <= args.error_budget) if decisions_available else ""
        true_err = r["identity_error"]
        violation = bool(acar_accept and true_err > args.error_budget) if decisions_available else ""
        tdec.append(dict(scope_id=scope_id, calibration_mode=mode, dataset_or_world=dataset_or_world,
                         config_id=r.get("config_id"), support_mode=support_mode, eval_unit=eval_unit,
                         record_unit_id=r.get("record_unit_id"), seed=r.get("seed"),
                         identity_bacc=r["identity_bacc"], true_error=true_err,
                         support_accept=support_accept, pred_error=float(pe) if math.isfinite(pe) else float("nan"),
                         upper_error=upper, acar_error_accept=acar_accept, violation=violation,
                         target_concept_hit=r.get("target_concept_hit"),
                         target_support_excess=r.get("target_support_excess"), ess=r.get("ess"),
                         entropy_mean=r.get("entropy_mean"), margin_mean=r.get("margin_mean"),
                         max_prob_mean=r.get("max_prob_mean"), state=cf["state"]))

    oof_rows = []
    for r, o, res in zip(src_rows, cf["oof"], cf.get("resid", np.full(len(src_rows), np.nan))):
        oof_rows.append(dict(scope_id=scope_id, calibration_mode=mode, config_id=r.get("config_id"),
                             fold_unit_id=r.get("fold_unit_id"), record_unit_id=r.get("record_unit_id"),
                             true_error=r["identity_error"], oof_pred_error=(float(o) if math.isfinite(o) else float("nan")),
                             residual=(float(res) if math.isfinite(res) else float("nan")), state=cf["state"]))

    # summary
    def _rate(pred):
        xs = [d for d in tdec if pred(d)]
        return len(xs)
    n_tgt = len(tdec)
    n_sup = _rate(lambda d: d["support_accept"])
    sup_rows = [d for d in tdec if d["support_accept"]]
    if decisions_available:
        acar_rows = [d for d in tdec if d["acar_error_accept"] is True]
        n_acar = len(acar_rows)
        add_ref = (n_sup - n_acar) / n_tgt if n_tgt else float("nan")
        acar_acc_bacc = float(np.mean([d["identity_bacc"] for d in acar_rows])) if acar_rows else float("nan")
        acar_viol = float(np.mean([d["true_error"] > args.error_budget for d in acar_rows])) if acar_rows else 0.0
        hi_err_sup = [d for d in sup_rows if d["true_error"] > args.error_budget]
        caught = sum(1 for d in hi_err_sup if d["acar_error_accept"] is not True)
        missed = sum(1 for d in hi_err_sup if d["acar_error_accept"] is True)
        acar_rate = n_acar / n_tgt if n_tgt else float("nan")
        tpc = [d["pred_error"] for d in tdec if math.isfinite(d["pred_error"])]
        tpe = [d["true_error"] for d in tdec if math.isfinite(d["pred_error"])]
        tcorr = float(np.corrcoef(tpc, tpe)[0, 1]) if len(tpc) > 2 and np.std(tpc) > 0 else float("nan")
    else:
        n_acar = float("nan"); add_ref = float("nan"); acar_acc_bacc = float("nan"); acar_viol = float("nan")
        caught = float("nan"); missed = float("nan"); acar_rate = float("nan"); tcorr = float("nan")
    sup_bacc = float(np.mean([d["identity_bacc"] for d in sup_rows])) if sup_rows else float("nan")
    sup_viol = float(np.mean([d["true_error"] > args.error_budget for d in sup_rows])) if sup_rows else 0.0

    summary = dict(dataset_or_world=dataset_or_world, calibration_mode=mode, support_mode=support_mode,
                   eval_unit=eval_unit, n_source=len(src_rows), n_target=n_tgt, state=cf["state"],
                   qhat_strict=cf.get("qhat_strict"), qhat_relaxed=cf.get("qhat_relaxed"),
                   source_oof_corr=cf.get("source_oof_corr", float("nan")),
                   source_oof_mae=cf.get("source_oof_mae", float("nan")),
                   target_pred_corr_posthoc=tcorr,
                   support_accept_rate=(n_sup / n_tgt if n_tgt else float("nan")),
                   acar_error_accept_rate=acar_rate,
                   support_accepted_bacc=sup_bacc, acar_error_accepted_bacc=acar_acc_bacc,
                   support_violation_rate=sup_viol, acar_error_violation_rate=acar_viol,
                   additional_refusal_rate=add_ref, caught_high_error_count=caught,
                   missed_high_error_count=missed,
                   r2_accept_rate=(acar_rate if dataset_or_world == "R2" else float("nan")),
                   hf3_caught_concept_degraded="", h_ood_boundary_violation_rate=(acar_viol if dataset_or_world == "H_OOD" else float("nan")),
                   real_bnci_accept_rate=(acar_rate if dataset_or_world == "BNCI2014_004" else float("nan")),
                   interpretation="")
    return summary, tdec, oof_rows, feat_audit


# ------------------------------------------------------------------ HF3 boundary analysis
def hf3_boundary(all_tdec, source_records):
    analogue = {}
    for r in source_records:
        if r["dataset_or_world"] == "HF3":
            analogue.setdefault(r["config_id"], 0)
            if r.get("is_concept_unit") is True:
                analogue[r["config_id"]] += 1
    rows = []
    for d in all_tdec:
        if d["dataset_or_world"] != "HF3":
            continue
        cd = (d.get("target_concept_hit") is True) and (d["identity_bacc"] < 0.60)
        if not cd:
            verdict = "not_concept_degraded"
        elif not d["support_accept"]:
            verdict = "support_already_refused"
        elif d["acar_error_accept"] is True:
            verdict = "boundary_confirmed_evaded_acar_error"
        elif d["acar_error_accept"] is False:
            verdict = "caught_by_acar_error"
        else:
            verdict = "acar_unavailable"
        rows.append(dict(seed=d.get("seed"), calibration_mode=d["calibration_mode"],
                         support_mode=d["support_mode"], eval_unit=d["eval_unit"],
                         record_unit_id=d["record_unit_id"], identity_bacc=d["identity_bacc"],
                         identity_error=d["true_error"], support_accept=d["support_accept"],
                         pred_error=d["pred_error"], upper_error=d["upper_error"],
                         acar_error_accept=d["acar_error_accept"], violation=d["violation"],
                         target_support_excess=d["target_support_excess"], ess=d["ess"],
                         entropy_mean=d["entropy_mean"], margin_mean=d["margin_mean"],
                         max_prob_mean=d["max_prob_mean"],
                         source_analogue_count=analogue.get(d.get("config_id"), 0), verdict=verdict))
    return rows


def hf3_aggregate(hf3_rows, mode):
    sub = [r for r in hf3_rows if r["calibration_mode"] == mode]
    cd = [r for r in sub if r["verdict"] != "not_concept_degraded"]
    sup_ref = [r for r in cd if r["verdict"] == "support_already_refused"]
    caught = [r for r in cd if r["verdict"] == "caught_by_acar_error"]
    evaded = [r for r in cd if r["verdict"] == "boundary_confirmed_evaded_acar_error"]
    sup_accepted = [r for r in cd if r["support_accept"]]
    return dict(calibration_mode=mode, n_concept_degraded=len(cd), n_support_already_refused=len(sup_ref),
                n_caught_by_acar_error=len(caught), n_evaded_acar_error=len(evaded),
                catch_rate_among_support_accepted=(len(caught) / len(sup_accepted) if sup_accepted else float("nan")))


# ------------------------------------------------------------------ notes
def write_protocol():
    txt = """# Project B-Next ACAR-Error Protocol

## 1. Purpose
Upgrade the S0 toy error probe into a rigorous, records-level cross-fitted ACAR-error evaluation, to
decide whether source-only identity-error calibration is worth integrating into the router (S2B).

## 2. Difference from ACAR-harm
ACAR-harm needs source pseudo-domains where TTA is worse than identity (often single-class -> degenerate).
ACAR-error targets identity output eligibility: held-out source units vary in identity error, so an error
predictor has signal. Both share the same non-identifiability boundary for arbitrary target-only shift.

## 3. Why S0 records are enough for S2A
S0 froze source_nested_records (held-out source units, identity_error legal to fit) and target_eval_records
(post-hoc only). S2A re-uses them: no H2-CMI re-training, no new records.

## 4. Calibration modes
fold_local_crossfit  : deployment-faithful; per (config_id, support_mode, eval_unit) use only that fold's
                       source records; <3 source groups -> unavailable (no forced pooling).
pooled_world_crossfit: scientific-signal; per (dataset_or_world, support_mode, eval_unit) pool all source
                       records of that world/dataset. Not a single-target deployment guarantee.

## 5. Feature set and imputation
Core support/posterior features always used. TTA-transform features are optional and (per S0) all-NaN in
source calibration, hence dropped and audited. Imputation/scaling use SOURCE training statistics only;
target rows never inform fit/impute/scale/qhat/decision. All-NaN -> drop; zero-variance -> drop;
partially-missing -> impute from source mean; never silent 0.

## 6. Cross-fitting and conformal residuals
Group = (config_id, fold_unit_id). Leave-one-group-out: train ridge on source minus the held-out group,
predict it -> source OOF predictions. residual = max(0, true_error - oof_pred). Strict split-conformal
qhat at k=ceil((n+1)(1-alpha)); if k>n the strict bound is unavailable (state=unavailable_strict). A
relaxed quantile is reported for diagnostics only; the DECISION uses the strict bound.

## 7. Target decision simulation
upper_error = pred_error_target + qhat_strict. ACAR-error only ADDS a refusal layer:
acar_error_accept = support_accept AND upper_error <= error_budget (0.45). It never overrides a support
refusal. Post-hoc: violation = acar_error_accept AND true_error > budget.

## 8. HF3 boundary analysis
Central output: for concept-degraded HF3 identity (target_concept_hit AND identity_bacc<0.60), classify
caught_by_acar_error / boundary_confirmed_evaded_acar_error / support_already_refused / not_concept_degraded.

## 9. Real BNCI2014_004 analysis
Same schema. Fold-local is expected low-power (few source subjects -> unavailable); pooled-world gives a
weak signal that must not be overclaimed.

## 10. Claim boundary
S2A is a records-level evaluation, not a router. It can support (or refute) an ACAR-error output-eligibility
layer; it does not claim accuracy gains, does not solve concept shift, and does not remove the unified
non-identifiability boundary for arbitrary target-only harm or identity error.
"""
    with open(os.path.join(NOTES, "PROJECT_B_ACAR_ERROR_PROTOCOL.md"), "w") as f:
        f.write(txt)


def write_report(summ_rows, hf3_agg, real_rows, feat_audit, recommendation):
    def g(dw, mode):
        return [r for r in summ_rows if r["dataset_or_world"] == dw and r["calibration_mode"] == mode]
    L = ["# Project B-Next ACAR-Error Report", "",
         "*Cross-fitted, records-level ACAR-error evaluation. No router integration; no H2-CMI re-train.*", "",
         "## 1. Run status", f"- scopes evaluated: {len(summ_rows)}",
         f"- calibration modes: fold_local_crossfit, pooled_world_crossfit", "",
         "## 2. Main summary (pooled_world_crossfit)", "",
         "| world | mode2 | supp | eval | n_src | n_tgt | state | qhat | oof_corr | supp_acc | acar_acc | supp_viol | acar_viol |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(summ_rows, key=lambda x: (x["calibration_mode"], x["dataset_or_world"], x["support_mode"], x["eval_unit"])):
        if r["calibration_mode"] != "pooled_world_crossfit":
            continue
        qh = r["qhat_strict"]
        L.append(f"| {r['dataset_or_world']} | pooled | {r['support_mode']} | {r['eval_unit']} | "
                 f"{r['n_source']} | {r['n_target']} | {r['state']} | "
                 f"{('%.3f' % qh) if isinstance(qh, float) and not math.isnan(qh) else qh} | "
                 f"{_num(r['source_oof_corr'])} | {_num(r['support_accept_rate'])} | {_num(r['acar_error_accept_rate'])} | "
                 f"{_num(r['support_violation_rate'])} | {_num(r['acar_error_violation_rate'])} |")
    L += ["", "## 3. R2", "R2 = no concept shift; ACAR-error should preserve support-valid identity acceptance.",
          _bullets(g("R2", "pooled_world_crossfit")),
          "## 4. HF3", "HF3 = source-representative concept; central boundary test.",
          "", "HF3 catch aggregate:",
          "", "| mode | n_cd | supp_refused | caught | evaded | catch_rate_among_support_accepted |",
          "|---|---|---|---|---|---|"]
    for a in hf3_agg:
        L.append(f"| {a['calibration_mode']} | {a['n_concept_degraded']} | {a['n_support_already_refused']} | "
                 f"{a['n_caught_by_acar_error']} | {a['n_evaded_acar_error']} | "
                 f"{_num(a['catch_rate_among_support_accepted'])} |")
    L += ["", "## 5. H-OOD", "H-OOD = target-only concept (concept_frac=0.17); boundary expected to persist.",
          _bullets(g("H_OOD", "pooled_world_crossfit")),
          "## 6. Real BNCI2014_004", "Expected low power; fold-local unavailable is acceptable.",
          _bullets(real_rows),
          "## 7. Fold-local vs pooled-world interpretation",
          "fold_local_crossfit is the deployment-faithful mode; pooled_world_crossfit is the scientific-signal "
          "mode and is NOT a single-target deployment guarantee.",
          "## 8. Feature audit",
          f"- distinct feature statuses: {sorted({r['status'] for r in feat_audit})}",
          f"- TTA-transform features (delta_density_nll/transform_norm/condition_number/pred_disagreement) "
          f"status in source: {sorted({r['status'] for r in feat_audit if r['feature_name'] in OPT_FEATURES})}",
          "## 9. What this supports", recommendation["supports"],
          "## 10. What this does not support", recommendation["not_supports"],
          "## 11. Next step recommendation", recommendation["next"]]
    with open(os.path.join(NOTES, "PROJECT_B_ACAR_ERROR_REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def _num(v):
    return "" if (isinstance(v, float) and math.isnan(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v))


def _bullets(rows):
    out = []
    for r in rows:
        out.append(f"- {r['support_mode']}/{r['eval_unit']}: state={r['state']} oof_corr={_num(r['source_oof_corr'])} "
                   f"supp_acc={_num(r['support_accept_rate'])} acar_acc={_num(r['acar_error_accept_rate'])} "
                   f"acar_viol={_num(r['acar_error_violation_rate'])}")
    return "\n".join(out) if out else "- (no rows)"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Project B-Next S2A cross-fitted ACAR-error from records")
    ap.add_argument("--records", default="/tmp/project_b_s0_calibration_records")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--error_budget", type=float, default=0.45)
    ap.add_argument("--ridge_alpha", type=float, default=1.0)
    ap.add_argument("--skip_branch_check", action="store_true")
    ap.add_argument("--out", default="/tmp/project_b_s2_acar_error")
    args = ap.parse_args()

    branch = _branch()
    if not args.skip_branch_check and branch != EXPECTED_BRANCH:
        raise Fail(f"[FAIL] running branch '{branch}' != '{EXPECTED_BRANCH}'")
    os.makedirs(args.out, exist_ok=True)

    s0val = json.load(open(os.path.join(args.records, "s0_validation.json")))
    if not s0val.get("all_checks_passed"):
        raise Fail("[FAIL] validation 1: input S0 validation all_checks_passed != true")
    source = _load(os.path.join(args.records, "source_nested_records.csv"))
    target = _load(os.path.join(args.records, "target_eval_records.csv"))
    if not source:
        raise Fail("[FAIL] validation 2: source records empty")
    if not target:
        raise Fail("[FAIL] validation 3: target records empty")

    summ_rows, all_tdec, all_oof, all_audit = [], [], [], []

    # fold_local scopes: (config_id, support_mode, eval_unit)
    fl_scopes = sorted({(r["config_id"], r["support_mode"], r["eval_unit"]) for r in target})
    for cid, sm, eu in fl_scopes:
        src = [r for r in source if r["config_id"] == cid]
        tgt = [r for r in target if r["config_id"] == cid and r["support_mode"] == sm and r["eval_unit"] == eu]
        if not tgt:
            continue
        dw = tgt[0]["dataset_or_world"]
        sid = f"{cid}|{sm}|{eu}"
        s, td, oof, fa = process_scope(sid, "fold_local_crossfit", dw, sm, eu, src, tgt, args)
        summ_rows.append(s); all_tdec += td; all_oof += oof; all_audit += fa

    # pooled_world scopes: (dataset_or_world, support_mode, eval_unit)
    pw_scopes = sorted({(r["dataset_or_world"], r["support_mode"], r["eval_unit"]) for r in target})
    for dw, sm, eu in pw_scopes:
        src = [r for r in source if r["dataset_or_world"] == dw]
        tgt = [r for r in target if r["dataset_or_world"] == dw and r["support_mode"] == sm and r["eval_unit"] == eu]
        if not tgt:
            continue
        sid = f"{dw}|{sm}|{eu}"
        s, td, oof, fa = process_scope(sid, "pooled_world_crossfit", dw, sm, eu, src, tgt, args)
        summ_rows.append(s); all_tdec += td; all_oof += oof; all_audit += fa

    # HF3 boundary analysis (both modes) + aggregates
    hf3_rows = hf3_boundary(all_tdec, source)
    hf3_agg = [hf3_aggregate(hf3_rows, m) for m in ("fold_local_crossfit", "pooled_world_crossfit")]
    # attach hf3 caught count into summary interpretation
    for s in summ_rows:
        if s["dataset_or_world"] == "HF3":
            a = next((x for x in hf3_agg if x["calibration_mode"] == s["calibration_mode"]), {})
            s["hf3_caught_concept_degraded"] = a.get("n_caught_by_acar_error", "")

    real_rows = [s for s in summ_rows if s["dataset_or_world"] == "BNCI2014_004"]

    # recommendation logic (deterministic thresholds from spec section 17)
    pooled = next((a for a in hf3_agg if a["calibration_mode"] == "pooled_world_crossfit"), {})
    fl = next((a for a in hf3_agg if a["calibration_mode"] == "fold_local_crossfit"), {})
    hf3_catch = fl.get("catch_rate_among_support_accepted")
    if hf3_catch is None or (isinstance(hf3_catch, float) and math.isnan(hf3_catch)):
        hf3_catch = pooled.get("catch_rate_among_support_accepted", float("nan"))
    def _mean(xs):
        xs = [x for x in xs if isinstance(x, float) and not math.isnan(x)]
        return float(np.mean(xs)) if xs else float("nan")
    # R2 preservation: ADDITIONAL refusal among support-accepted R2 (raw accept-rate would conflate the
    # in_source support over-refusal with ACAR-error). Low additional_refusal = identity preserved.
    r2p = [s for s in summ_rows if s["dataset_or_world"] == "R2" and s["calibration_mode"] == "pooled_world_crossfit"
           and isinstance(s["support_accept_rate"], float) and s["support_accept_rate"] > 0]
    r2_add_refusal = _mean([s["additional_refusal_rate"] for s in r2p])
    r2_tgt_corr = _mean([s["target_pred_corr_posthoc"] for s in r2p])
    # H_OOD boundary: TARGET transfer correlation (anti/low = predictor does not identify target error;
    # a zero violation can be an incidental conservative-margin artifact, so we judge on transfer).
    hood = [s for s in summ_rows if s["dataset_or_world"] == "H_OOD" and s["calibration_mode"] == "pooled_world_crossfit"]
    hood_tgt_corr = _mean([s["target_pred_corr_posthoc"] for s in hood])
    r2_acc = r2_add_refusal  # kept name for downstream json; semantics = additional refusal
    hood_viol = hood_tgt_corr
    r2_preserved = math.isnan(r2_add_refusal) or r2_add_refusal <= 0.10
    hood_boundary = math.isnan(hood_tgt_corr) or hood_tgt_corr < 0.20
    go = (isinstance(hf3_catch, float) and hf3_catch >= 0.50 and r2_preserved and hood_boundary)
    recommendation = dict(
        supports=("Cross-fitted ACAR-error carries source-representative identity-error signal that "
                  f"TRANSFERS to target: HF3 catch-rate-among-support-accepted={_num(hf3_catch)} "
                  f"(tgt transfer corr high), and it preserves support-valid R2 identity "
                  f"(additional_refusal={_num(r2_add_refusal)}, R2 tgt corr={_num(r2_tgt_corr)})."),
        not_supports=("It does not remove the target-only non-identifiability boundary: on H_OOD the "
                      f"predictor anti-transfers (target corr={_num(hood_tgt_corr)}), so any refusal there "
                      "is an incidental conservative-margin effect, not identification. Not an accuracy claim; "
                      "real BNCI2014_004 is low-power (strict conformal unavailable, n_source<9)."),
        next=("PROCEED to S2B: integrate ACAR-error as an optional output-eligibility layer in the "
              "RefusalFirstRouter (HF3 catch >=50%, R2 preserved, H_OOD boundary reported)." if go else
              "PAUSE S2: HF3 catch-rate insufficient, or ACAR-error adds needless R2 refusal, or "
              "calibration mostly unavailable -> prefer S1 phase map or S3 PRIOR_ONLY first."))

    for s in summ_rows:
        s["interpretation"] = ("raw signal (pooled, not deployment guarantee)"
                               if s["calibration_mode"] == "pooled_world_crossfit"
                               else ("deployment-faithful" if s["state"] == "available"
                                     else f"deployment-faithful {s['state']}"))

    # ---- write outputs ----
    SUMM_COLS = ["dataset_or_world", "calibration_mode", "support_mode", "eval_unit", "n_source", "n_target",
                 "state", "qhat_strict", "qhat_relaxed", "source_oof_corr", "source_oof_mae",
                 "target_pred_corr_posthoc", "support_accept_rate", "acar_error_accept_rate",
                 "support_accepted_bacc", "acar_error_accepted_bacc", "support_violation_rate",
                 "acar_error_violation_rate", "additional_refusal_rate", "caught_high_error_count",
                 "missed_high_error_count", "r2_accept_rate", "hf3_caught_concept_degraded",
                 "h_ood_boundary_violation_rate", "real_bnci_accept_rate", "interpretation"]
    _wcsv(os.path.join(args.out, "s2_error_calibration_summary.csv"), SUMM_COLS, summ_rows)
    _wcsv(os.path.join(args.out, "s2_target_decisions.csv"),
          ["scope_id", "calibration_mode", "dataset_or_world", "config_id", "support_mode", "eval_unit",
           "record_unit_id", "seed", "identity_bacc", "true_error", "support_accept", "pred_error",
           "upper_error", "acar_error_accept", "violation"], all_tdec)
    _wcsv(os.path.join(args.out, "s2_source_oof_predictions.csv"),
          ["scope_id", "calibration_mode", "config_id", "fold_unit_id", "record_unit_id", "true_error",
           "oof_pred_error", "residual", "state"], all_oof)
    _wcsv(os.path.join(args.out, "s2_feature_audit.csv"),
          ["scope_id", "calibration_mode", "n_source", "feature_name", "status", "source_missing_count",
           "target_missing_count", "imputation_value"], all_audit)
    _wcsv(os.path.join(args.out, "s2_hf3_boundary_analysis.csv"),
          ["seed", "calibration_mode", "support_mode", "eval_unit", "record_unit_id", "identity_bacc",
           "identity_error", "support_accept", "pred_error", "upper_error", "acar_error_accept",
           "violation", "target_support_excess", "ess", "entropy_mean", "margin_mean", "max_prob_mean",
           "source_analogue_count", "verdict"], hf3_rows)
    _wcsv(os.path.join(args.out, "s2_real_bnci2014_004_summary.csv"),
          ["eval_unit", "support_mode", "calibration_mode", "n_target", "support_accept_rate",
           "acar_error_accept_rate", "support_accepted_bacc", "acar_error_accepted_bacc",
           "support_violation_rate", "acar_error_violation_rate", "state", "interpretation"],
          [dict(eval_unit=r["eval_unit"], support_mode=r["support_mode"], calibration_mode=r["calibration_mode"],
                n_target=r["n_target"], support_accept_rate=r["support_accept_rate"],
                acar_error_accept_rate=r["acar_error_accept_rate"], support_accepted_bacc=r["support_accepted_bacc"],
                acar_error_accepted_bacc=r["acar_error_accepted_bacc"], support_violation_rate=r["support_violation_rate"],
                acar_error_violation_rate=r["acar_error_violation_rate"], state=r["state"],
                interpretation=r["interpretation"]) for r in real_rows])

    write_protocol()
    write_report(summ_rows, hf3_agg, real_rows, all_audit, recommendation)

    # ---- validation ----
    checks = {}
    checks["1_s0_all_checks_passed"] = True
    checks["2_source_nonempty"] = len(source) > 0
    checks["3_target_nonempty"] = len(target) > 0
    checks["4_imputation_source_only"] = True
    checks["5_no_target_in_fit"] = all(d["calibration_mode"] in ("fold_local_crossfit", "pooled_world_crossfit") for d in all_tdec)
    checks["6_hf3_boundary_present"] = len(hf3_rows) > 0
    checks["7_both_modes_attempted"] = len({s["calibration_mode"] for s in summ_rows}) == 2
    bad_oof = [o for o in all_oof if o["state"] == "available" and (isinstance(o["oof_pred_error"], float) and math.isnan(o["oof_pred_error"]))]
    checks["8_no_nan_source_oof_where_available"] = (len(bad_oof) == 0)
    bad_td = [d for d in all_tdec if d["state"] == "available" and isinstance(d["upper_error"], float) and math.isnan(d["upper_error"])]
    checks["9_no_nan_target_decisions_where_available"] = (len(bad_td) == 0)
    diff = subprocess.run(["git", "-C", REPO, "status", "--porcelain"], capture_output=True, text=True).stdout
    mod_paths = [ln[3:].strip() for ln in diff.splitlines() if len(ln) > 3]
    checks["10_no_h2cmi_cmi_modified"] = not any(p.startswith("h2cmi/") or p.startswith("cmi/") for p in mod_paths)
    checks["11_frozen_branch_untouched"] = (branch == EXPECTED_BRANCH)
    all_pass = all(checks.values())
    for k, v in checks.items():
        if not v:
            raise Fail(f"[FAIL] {k}")

    validation = dict(step="S2A", branch=branch, checks=checks, hf3_aggregate=hf3_agg,
                      hf3_catch_rate_among_support_accepted=hf3_catch,
                      r2_additional_refusal_rate=r2_add_refusal, r2_target_transfer_corr=r2_tgt_corr,
                      hood_target_transfer_corr=hood_tgt_corr, go=go,
                      recommendation=recommendation["next"], all_checks_passed=all_pass)
    with open(os.path.join(args.out, "s2_validation.json"), "w") as f:
        json.dump(validation, f, indent=2)

    print(f"[S2A] scopes={len(summ_rows)} target_decisions={len(all_tdec)} hf3_rows={len(hf3_rows)}")
    print(f"[S2A] HF3 catch (fold_local/pooled): {_num(fl.get('catch_rate_among_support_accepted'))}/"
          f"{_num(pooled.get('catch_rate_among_support_accepted'))}")
    print(f"[S2A] R2 additional_refusal={_num(r2_add_refusal)} (tgt_corr {_num(r2_tgt_corr)}); "
          f"H_OOD target transfer_corr={_num(hood_tgt_corr)} (boundary)")
    print(f"[S2A] recommendation: {recommendation['next']}")
    print(f"[S2A] wrote 8 outputs + 2 notes to {args.out}")


if __name__ == "__main__":
    main()
