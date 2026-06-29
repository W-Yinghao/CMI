"""ACAR v4 (CURB) — Phase-1 EXPLORATORY DEV orchestration.

NON-BINDING / POST-V3 DEV_STOP / DEV-ONLY / NO EXTERNAL ARM / NO LOCKBOX.

This is NOT v3's binding DEV gate runner. It is the V4 Phase-1 *exploratory* runner: it wires the three hardened
primitives — Direction C frontiers (`frontiers.py`), Direction A finite-grid risk control (`risk_control.py`), and
Direction B deployed-risk objects (`hierarchy.py`) — into per-disease + disease-macro exploratory reports with a
strict-but-NON-BINDING G0–G6 candidate gate. It reads no real cohort here (synthetic `V4OOFRecord` fixtures only); the
real run later derives `V4OOFRecord`s from v3's single-execution cache. It NEVER emits `SELECT`/`DEV_STOP`/external
G2/coverage theorem, consumes the lockbox, approaches external Arm B, or writes `ACAR_FROZEN_v4.md`.

Honest calibration discipline (hardening patch):
  - FOLD-LOCAL CAL→EVAL: per outer fold, the CAL records select λ* (risk control); the held-out EVAL records are scored
    at that fold's λ*. The OOF EVAL cells are aggregated subject-macro. CAL records NEVER enter the EVAL operating-point
    denominator, so the report does not over-state control.
  - COHORT-AWARE subject key: the calibration unit is `cohort_id::subject_id`, so the same local id in two cohorts is
    two subjects (never merged).
  - COMPARATOR contract: C0 reports BOTH best-fixed-action red and (optional) v2-replay red as DISTINCT slots; which one
    G3 uses is fixed in the config BEFORE the run, not chosen after.
  - SCORE-FAMILY registry: in real_mode only PRE-LISTED registry score families are accepted (arbitrary callables that
    could close over ΔR are rejected).
  - PER-CONFIG vs GLOBAL policy-frontier gap: the policy gap is reported both for this single config and for the best of
    ALL V4 policy families (the paper's information/policy/calibration decomposition uses the global one).

Result taxonomy (the ONLY allowed verdicts):
  V4_DEV_EXPLORATION_COMPLETE                    (run_status)
  V4_DEV_CANDIDATE_FOUND_FOR_POSSIBLE_FREEZE     (verdict: ≥1 config passes G0–G6 → worth a FUTURE freeze; NOT Arm B)
  V4_DEV_NEGATIVE_NO_LOCKBOX                      (verdict: none pass)
  OPERATIONALLY_ABORTED_NO_SCIENTIFIC_VERDICT    (runner-level; never written here)

Sign convention (frozen): ΔR_a(B) < 0 = reduced risk (good); identity/fallback realize 0 and stay in the subject
denominator. Subjects are the exchangeable unit; aggregation is subject-equal. Fail-closed throughout.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import shutil
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from acar.config import DISEASE, NON_IDENTITY
from acar.v4 import policies as PO
from acar.v4 import frontiers as FR
from acar.v4 import risk_control as RC
from acar.v4 import hierarchy as HI

# ---- taxonomy constants ----
V4_DEV_EXPLORATION_COMPLETE = "V4_DEV_EXPLORATION_COMPLETE"
V4_DEV_CANDIDATE_FOUND = "V4_DEV_CANDIDATE_FOUND_FOR_POSSIBLE_FREEZE"
V4_DEV_NEGATIVE = "V4_DEV_NEGATIVE_NO_LOCKBOX"
OPERATIONALLY_ABORTED = "OPERATIONALLY_ABORTED_NO_SCIENTIFIC_VERDICT"
_ALLOWED_VERDICTS = (V4_DEV_CANDIDATE_FOUND, V4_DEV_NEGATIVE)
_ALLOWED_RUN_STATUS = (V4_DEV_EXPLORATION_COMPLETE, OPERATIONALLY_ABORTED)
_FORBIDDEN_STATUS_TOKENS = ("SELECT", "DEV_STOP", "PROCEED_SAFE_ROUTER", "UTILITY_ONLY", "EXTERNAL_G2",
                            "COVERAGE_THEOREM", "ARM_B", "LOCKBOX_CONSUMED")
_FORBIDDEN_MANIFEST_KEYS = ("external_g2", "lockbox", "arm_b", "coverage_theorem", "select", "binding")

ACTIONS = tuple(NON_IDENTITY)
A = len(ACTIONS)
N_FEAT = 11
DISEASES = tuple(sorted(DISEASE.keys()))
_DEV_COHORTS = tuple(sorted(c for ds in DISEASE.values() for c in ds))
_SPLITS = ("FIT", "CAL", "EVAL")
_AGGR = {"safe_set": "increasing_lambda", "benefit_ranked": "increasing_lambda",
         "direct_selective": "decreasing_lambda"}


# ----------------------------------------------------------------------------- DEV-ready record contract

@dataclass(frozen=True)
class V4OOFRecord:
    """One cross-fit cell. (fold, split) give the cell its role: per outer fold, split=='CAL' records calibrate λ* and
    split=='EVAL' records (held-out subjects) are scored at that λ*. The same physical batch may appear as CAL in some
    folds and EVAL in exactly one (cross-fit) — dedup is on (disease, cohort, batch_id, fold, split). dr is per
    non-identity action; features_v2 is the bit-for-bit v2 paired vector per action [A, 11] (label-free)."""
    disease: str
    subject_id: str
    cohort_id: str
    batch_id: str
    fold: int
    split: str
    fallback: bool
    dr: np.ndarray                                  # [A]
    features_v2: np.ndarray                         # [A, N_FEAT]
    action_names: Tuple[str, ...]

    def __post_init__(self):
        # substantive immutability: canonical float64 read-only copies (external in-place mutation cannot alter the
        # record), shape/finiteness validated at construction, action_names canonicalised to a tuple.
        dr_in, feats_in = np.asarray(self.dr), np.asarray(self.features_v2)
        if not np.issubdtype(dr_in.dtype, np.floating):
            raise ValueError("dr must be a floating-point array (no silent int/bool coercion)")
        if not np.issubdtype(feats_in.dtype, np.floating):
            raise ValueError("features_v2 must be a floating-point array (no silent int/bool coercion)")
        dr = np.array(dr_in, dtype=np.float64)
        if dr.shape != (A,) or not np.all(np.isfinite(dr)):
            raise ValueError(f"dr must be finite shape ({A},), got {dr_in.shape}")
        feats = np.array(feats_in, dtype=np.float64)
        if feats.shape != (A, N_FEAT) or not np.all(np.isfinite(feats)):
            raise ValueError(f"features_v2 must be finite shape ({A}, {N_FEAT}), got {feats_in.shape}")
        dr.flags.writeable = False
        feats.flags.writeable = False
        object.__setattr__(self, "dr", dr)
        object.__setattr__(self, "features_v2", feats)
        object.__setattr__(self, "action_names", tuple(self.action_names))


@dataclass(frozen=True)
class ScoreFamily:
    """A PRE-LISTED, label-free score transform: feats [N, A, N_FEAT] → (harm [N, A], benefit [N, A]), both "lower is
    safer/better". MUST NOT use ΔR. In real_mode only registry families are accepted (see SCORE_FAMILY_REGISTRY)."""
    name: str
    compute: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class V4DevConfig:
    policy_families: Tuple[str, ...] = ("safe_set", "benefit_ranked", "direct_selective")
    losses: Tuple[str, ...] = ("mean", "positive", "harm_indicator")
    alpha: float = 0.10
    budget_by_loss: Dict[str, float] = field(default_factory=lambda: {"mean": 0.0, "positive": 0.05,
                                                                      "harm_indicator": 0.10})
    correction: str = "holm"
    method: str = "ttest"
    grid_size: int = 12
    coverage_min: float = 0.15
    g3_comparator: str = "best_fixed"               # which C0 slot G3 compares against (fixed BEFORE the run)
    dev_cohort_ids: Tuple[str, ...] = _DEV_COHORTS


@dataclass(frozen=True)
class V4CandidateReport:
    disease: str
    policy_family: str
    loss: str
    calibration_method: str
    selected_lambda: Optional[float]                # representative (median of per-fold λ*); per-fold in provenance
    status: str                                     # always "EVALUATED"
    coverage: float
    red: float
    harm_rate: float
    c0_red: float                                   # the comparator G3 uses (per g3_comparator)
    disease_macro_red: Optional[float]
    g0_pass: bool
    g1_coverage_pass: bool
    g2_red_pass: bool
    g3_macro_vs_c0_pass: bool
    g4_harm_control_pass: bool
    g5_fallback_denominator_pass: bool
    g6_nonvacuous_both_diseases_pass: bool
    per_config_policy_gap: float
    global_policy_family_gap: float
    frontier_gaps: dict
    hierarchy_summary: dict
    provenance: dict

    def all_pass(self):
        return all((self.g0_pass, self.g1_coverage_pass, self.g2_red_pass, self.g3_macro_vs_c0_pass,
                    self.g4_harm_control_pass, self.g5_fallback_denominator_pass,
                    self.g6_nonvacuous_both_diseases_pass))


@dataclass(frozen=True)
class V4DevExplorationResult:
    run_status: str
    verdict: str
    reports: Tuple[V4CandidateReport, ...]
    manifest: dict
    manifest_sha256: str


# ----------------------------------------------------------------------------- score-family registry

# PRE-DECLARED, label-free score families. ALL depend ONLY on features_v2 (never ΔR). v2 paired order: d_entropy0
# d_margin1 flip_rate2 js3 bures4 post_sep5 n_eff6 | g_unc7 s_support8 s_sep9 pr_cmi_proxy10. Both fields are "lower is
# safer/better": a "*_pos" family takes harm=benefit=+f[idx] (higher feature ⇒ more harm); "*_neg" takes −f[idx]. The
# real DEV exploration passes an EXPLICIT subset of these names (real_mode); the chosen set + score_family_registry_sha256
# are pinned in notes/ACAR_V4_DEV_EXPLORATION_RUN_PLAN.md BEFORE the run (not chosen after seeing results).
def _sf_shift_margin(feats):
    h = feats[:, :, 1]                                       # == d_margin_pos (harm=benefit=+d_margin)
    return h, h


def _sf_js_flip(feats):
    return feats[:, :, 3], feats[:, :, 2]                    # cross family: harm=+js, benefit=+flip_rate


def _coord(idx_h, sgn_h, idx_b, sgn_b):
    def compute(feats):
        return sgn_h * feats[:, :, idx_h], sgn_b * feats[:, :, idx_b]
    return compute


_PREDECLARED = (
    ("shift_margin", _sf_shift_margin),                     # d_margin_pos
    ("js_flip", _sf_js_flip),
    ("d_entropy_pos", _coord(0, 1.0, 0, 1.0)),
    ("d_entropy_neg", _coord(0, -1.0, 0, -1.0)),
    ("d_margin_neg", _coord(1, -1.0, 1, -1.0)),
    ("flip_pos", _coord(2, 1.0, 2, 1.0)),
    ("js_pos", _coord(3, 1.0, 3, 1.0)),
    ("bures_pos", _coord(4, 1.0, 4, 1.0)),
    ("n_eff_neg", _coord(6, -1.0, 6, -1.0)),
    ("unc_pos", _coord(7, 1.0, 7, 1.0)),
)
SCORE_FAMILY_REGISTRY = {name: ScoreFamily(name, fn) for name, fn in _PREDECLARED}


def default_score_families():
    """Default (synthetic convenience) = the two named families; the real run passes an explicit subset of the registry
    by name (real_mode rejects implicit defaults and arbitrary callables)."""
    return (SCORE_FAMILY_REGISTRY["shift_margin"], SCORE_FAMILY_REGISTRY["js_flip"])


def _resolve_score_families(score_families, real_mode):
    if real_mode and score_families is None:
        raise ValueError("real_mode requires explicit pre-registered score family names (no implicit placeholders)")
    items = list(default_score_families()) if score_families is None else list(score_families)
    if not items:
        raise ValueError("score_families must be a non-empty list")
    out = []
    for x in items:
        if isinstance(x, str):
            if x not in SCORE_FAMILY_REGISTRY:
                raise ValueError(f"unknown score family {x!r}")
            out.append(SCORE_FAMILY_REGISTRY[x])
        elif isinstance(x, ScoreFamily):
            if real_mode:
                reg = SCORE_FAMILY_REGISTRY.get(x.name)
                if reg is None or reg.compute is not x.compute:
                    raise ValueError("real_mode requires PRE-LISTED registry score families (no arbitrary callables)")
            out.append(x)
        else:
            raise ValueError("score_families entries must be registry names or ScoreFamily objects")
    return tuple(out)


# ----------------------------------------------------------------------------- record validation

def _validate_records(records, cfg, require_exact_eval_coverage=False):
    if not records:
        raise ValueError("records must be a non-empty sequence of V4OOFRecord")
    full_keys = set()
    eval_batches = set()
    out = []
    for r in records:
        if not isinstance(r, V4OOFRecord):
            raise ValueError("every record must be a V4OOFRecord")
        if r.disease not in DISEASES:
            raise ValueError(f"disease must be one of {DISEASES}, got {r.disease!r}")
        if tuple(r.action_names) != ACTIONS:
            raise ValueError(f"action_names must be exactly {ACTIONS}")
        for name, val in (("subject_id", r.subject_id), ("cohort_id", r.cohort_id), ("batch_id", r.batch_id)):
            if not isinstance(val, str) or val == "":
                raise ValueError(f"{name} must be a non-empty string")
            if any(ord(ch) < 32 for ch in val):
                raise ValueError(f"{name} must not contain control characters (digest/partition injectivity)")
        if r.cohort_id not in cfg.dev_cohort_ids:
            raise ValueError(f"cohort_id {r.cohort_id!r} is not a DEV cohort (no external/lockbox identifiers allowed)")
        if not (isinstance(r.fold, int) and not isinstance(r.fold, bool) and r.fold >= 0):
            raise ValueError("fold must be a non-negative int")
        if r.split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}")
        if not isinstance(r.fallback, (bool, np.bool_)):
            raise ValueError("fallback must be bool")
        dr = np.asarray(r.dr, dtype=float)
        if dr.shape != (A,) or not np.all(np.isfinite(dr)):
            raise ValueError(f"dr must be finite shape ({A},)")
        feats = np.asarray(r.features_v2, dtype=float)
        if feats.shape != (A, N_FEAT) or not np.all(np.isfinite(feats)):
            raise ValueError(f"features_v2 must be finite shape ({A}, {N_FEAT})")
        fk = (r.disease, r.cohort_id, r.batch_id, r.fold, r.split)
        if fk in full_keys:
            raise ValueError(f"duplicate (disease, cohort, batch_id, fold, split): {fk}")
        full_keys.add(fk)
        if r.split == "EVAL":
            eb = (r.disease, r.cohort_id, r.batch_id)
            if eb in eval_batches:
                raise ValueError(f"batch appears as EVAL more than once (OOF must partition): {eb}")
            eval_batches.add(eb)
        out.append(r)
    # subject-level cross-fit invariants (the subject cluster — cohort_id+subject_id — is the unit, NOT the batch):
    #  (a) within a (disease, cohort, subject, fold) all records share ONE split;
    #  (b) a subject is EVAL in AT MOST one fold (so a subject's batches are never split across EVAL folds, and a
    #      subject is never simultaneously CAL/FIT and EVAL in the same fold — implied by (a)).
    split_of = {}            # (disease, cohort, subject, fold) -> set of splits
    eval_folds = {}          # (disease, cohort, subject) -> set of EVAL folds
    for r in out:
        split_of.setdefault((r.disease, r.cohort_id, r.subject_id, r.fold), set()).add(r.split)
        if r.split == "EVAL":
            eval_folds.setdefault((r.disease, r.cohort_id, r.subject_id), set()).add(r.fold)
    for (d, c, s, f), splits in split_of.items():
        if len(splits) > 1:
            raise ValueError(f"subject {d}/{c}/{s} has mixed splits {sorted(splits)} within fold {f} "
                             "(a subject's records in a fold must share one split)")
    for (d, c, s), folds in eval_folds.items():
        if len(folds) > 1:
            raise ValueError(f"subject {d}/{c}/{s} is EVAL in multiple folds {sorted(folds)} "
                             "(OOF must partition SUBJECTS, not just batches)")
    if require_exact_eval_coverage:
        # EXACT OOF coverage (real run): every physical subject AND every physical batch must be EVAL in EXACTLY one
        # fold — otherwise a CAL/FIT-only subject or an un-EVAL'd batch would silently shrink the EVAL denominator and
        # over-state coverage/red/harm/frontiers on a partial cohort.
        all_subjects = {(r.disease, r.cohort_id, r.subject_id) for r in out}
        all_batches = {(r.disease, r.cohort_id, r.batch_id) for r in out}
        for sk in sorted(all_subjects):
            n = len(eval_folds.get(sk, ()))
            if n != 1:
                raise ValueError(f"exact OOF coverage: subject {sk[0]}/{sk[1]}/{sk[2]} must be EVAL in exactly one "
                                 f"fold, got {n} (CAL/FIT-only subjects are not allowed in a real exploratory run)")
        for bk in sorted(all_batches):
            if bk not in eval_batches:
                raise ValueError(f"exact OOF coverage: batch {bk[0]}/{bk[1]}/{bk[2]} is never EVAL (each batch must be "
                                 "EVAL exactly once in a real exploratory run)")
    out.sort(key=lambda r: (r.disease, r.split, r.fold, r.cohort_id, r.subject_id, r.batch_id))
    return tuple(out)


def _canon_subject(r):
    return f"{r.cohort_id}::{r.subject_id}"               # cohort-aware calibration unit


def _arrays(recs):
    dr = np.stack([np.asarray(r.dr, dtype=float) for r in recs]) if recs else np.zeros((0, A))
    feats = np.stack([np.asarray(r.features_v2, dtype=float) for r in recs]) if recs else np.zeros((0, A, N_FEAT))
    subj = np.array([_canon_subject(r) for r in recs]) if recs else np.array([], dtype="<U1")
    fb = np.array([bool(r.fallback) for r in recs]) if recs else np.array([], dtype=bool)
    fold = np.array([int(r.fold) for r in recs]) if recs else np.array([], dtype=int)
    return dr, feats, subj, fb, fold


# ----------------------------------------------------------------------------- policy adapters

def _apply_family(name, harm, benefit, lam, fallback):
    if name == "safe_set":
        choice = PO.safe_set_policy(harm, benefit, lam)
    elif name == "benefit_ranked":
        choice = PO.benefit_ranked_policy(benefit, lam)
    elif name == "direct_selective":
        gate = -np.min(harm, axis=1)
        action = PO.best_benefit_action(benefit)[0]
        choice = PO.direct_selective_policy(gate, action, lam)
    else:
        raise ValueError(f"unknown policy family {name!r}")
    choice = choice.copy()
    choice[fallback] = PO.IDENTITY
    return choice


def _grid_for_family(name, harm, benefit, grid_size):
    if name == "safe_set":
        stat = harm.ravel()
    elif name == "benefit_ranked":
        stat = np.min(benefit, axis=1)
    elif name == "direct_selective":
        stat = -np.min(harm, axis=1)
    else:
        raise ValueError(f"unknown policy family {name!r}")
    if stat.size == 0:
        return None
    lo, hi = float(np.min(stat)), float(np.max(stat))
    if not (hi > lo):
        return None
    grid = np.unique(np.linspace(lo, hi, grid_size))
    return grid if grid.shape[0] >= 2 else None


# ----------------------------------------------------------------------------- per-disease processing (fold-local)

def _best_fixed_red(dr, fb, weights):
    best = -np.inf
    for a in range(A):
        ch = np.full(dr.shape[0], a, dtype=int)
        ch[fb] = PO.IDENTITY
        best = max(best, PO.reduction(ch, dr, weights=weights))
    return float(best)


def _hierarchy_summary(sel_choice, harm_ev, dr_ev, subj_ev, loss):
    b0 = HI.all_action_joint_max(harm_ev, subj_ev).values
    b1 = HI.policy_subject_risk(sel_choice, dr_ev, subj_ev, loss=loss).values
    b2 = HI.hierarchical_policy_risk(sel_choice, dr_ev, subj_ev, loss=loss, batch_summary="mean").values
    return {"b0_mean": float(np.mean(b0)), "b1_mean": float(np.mean(b1)), "b2_mean": float(np.mean(b2)),
            "b0_minus_b1_mean": float(np.mean(b0) - np.mean(b1))}


def _process_disease(recs_d, score_fams, cfg, v2_replay_red):
    eval_recs = [r for r in recs_d if r.split == "EVAL"]
    cal_recs = [r for r in recs_d if r.split == "CAL"]
    fit_recs = [r for r in recs_d if r.split == "FIT"]      # surfaced for auditability; not used in calibration
    if not eval_recs:
        raise ValueError("each disease needs ≥1 EVAL record")
    dr_ev, feats_ev, subj_ev, fb_ev, fold_ev = _arrays(eval_recs)
    dr_cal, feats_cal, subj_cal, fb_cal, fold_cal = _arrays(cal_recs)
    w_ev = PO.subject_macro_weights(subj_ev)
    identity_ev = np.full(len(eval_recs), PO.IDENTITY, dtype=int)
    c0_best_fixed = _best_fixed_red(dr_ev, fb_ev, w_ev)
    union = []
    for sf in score_fams:
        h, b = sf.compute(feats_ev)
        union.append((PO.adapt_rank_from_harm(np.asarray(h, float)), PO.best_benefit_action(np.asarray(b, float))[0]))
    folds = sorted(set(fold_ev.tolist()))
    configs = []
    global_choices = []
    for sf in score_fams:
        harm_ev, benefit_ev = (np.asarray(x, float) for x in sf.compute(feats_ev))
        if cal_recs:
            harm_cal, benefit_cal = (np.asarray(x, float) for x in sf.compute(feats_cal))
        for pf in cfg.policy_families:
            grid = _grid_for_family(pf, harm_cal, benefit_cal, cfg.grid_size) if cal_recs else \
                _grid_for_family(pf, harm_ev, benefit_ev, cfg.grid_size)
            choices_lam_ev = (np.stack([_apply_family(pf, harm_ev, benefit_ev, lam, fb_ev) for lam in grid])
                              if grid is not None else None)
            if choices_lam_ev is not None:
                global_choices.extend(list(choices_lam_ev))
            for loss in cfg.losses:
                calibrated = identity_ev.copy()
                per_fold_lambda = {}
                eval_folds_set = set(int(k) for k in fold_ev.tolist())
                evaluable_folds, passed_folds = set(), set()
                if grid is not None and cal_recs:
                    for k in sorted(eval_folds_set):
                        ev_k = np.where(fold_ev == k)[0]
                        cal_k = np.where(fold_cal == k)[0]
                        if ev_k.size == 0 or cal_k.size == 0:
                            continue                              # EVAL fold with no CAL: stays identity, NOT certified
                        evaluable_folds.add(k)
                        ch_cal = np.stack([_apply_family(pf, harm_cal[cal_k], benefit_cal[cal_k], lam, fb_cal[cal_k])
                                           for lam in grid])
                        sl = RC.subject_losses_from_policy(ch_cal, dr_cal[cal_k], subj_cal[cal_k], loss=loss)
                        try:
                            rc = RC.select_ltt_grid(grid, sl, alpha=cfg.alpha, budget=cfg.budget_by_loss[loss],
                                                    aggressiveness=_AGGR[pf], correction=cfg.correction,
                                                    method=cfg.method)
                        except ValueError:
                            continue
                        if rc.selected_index is not None:
                            per_fold_lambda[k] = rc.selected_lambda
                            passed_folds.add(k)
                            calibrated[ev_k] = _apply_family(pf, harm_ev[ev_k], benefit_ev[ev_k],
                                                             rc.selected_lambda, fb_ev[ev_k])
                # rc_status PASS only if EVERY EVAL fold received a passing calibration (honest harm-control scope):
                # an EVAL fold left at identity (no CAL, or its CAL did not pass the budget) downgrades to NO_PASS.
                n_eval_folds = len(eval_folds_set)
                if n_eval_folds == 0 or not evaluable_folds:
                    rc_status = "NOT_EVALUABLE"
                elif len(passed_folds) == n_eval_folds:
                    rc_status = "PASS"
                else:
                    rc_status = "NO_PASS"
                cov, red, hr = FR.operating_point(dr_ev, calibrated, weights=w_ev)
                gaps = FR.gap_decomposition(dr_ev, union,
                                            list(choices_lam_ev) if choices_lam_ev is not None else [identity_ev],
                                            calibrated, weights=w_ev, mode="ceiling")
                sel_lambda = (float(np.median(list(per_fold_lambda.values()))) if per_fold_lambda else None)
                hier = _hierarchy_summary(calibrated, harm_ev, dr_ev, subj_ev, loss)
                configs.append(dict(
                    disease=recs_d[0].disease, score_family=sf.name, policy_family=pf, loss=loss,
                    selected_lambda=sel_lambda, per_fold_lambda={str(k): v for k, v in per_fold_lambda.items()},
                    rc_status=rc_status, n_eval_folds=n_eval_folds, n_evaluable_folds=len(evaluable_folds),
                    n_passed_folds=len(passed_folds), coverage=float(cov), red=float(red), harm_rate=float(hr),
                    gaps=gaps, per_config_policy_ceiling=gaps["policy_ceiling"], hierarchy=hier))
    global_ceiling = (FR.frontier_policy_family(dr_ev, global_choices, weights=w_ev).ceiling()
                      if global_choices else 0.0)
    score_union_ceiling = configs[0]["gaps"]["score_ceiling"] if configs else 0.0
    return dict(disease=recs_d[0].disease, configs=configs, c0_best_fixed_red=c0_best_fixed,
                c0_v2_replay_red=(None if v2_replay_red is None else float(v2_replay_red)),
                n_eval_subjects=int(np.unique(subj_ev).shape[0]), n_eval_batches=len(eval_recs),
                n_cal_subjects=int(np.unique(subj_cal).shape[0]) if cal_recs else 0, n_cal_batches=len(cal_recs),
                n_fit_subjects=int(np.unique(np.array([f"{r.cohort_id}::{r.subject_id}" for r in fit_recs])).shape[0])
                if fit_recs else 0, n_fit_batches=len(fit_recs),
                n_fallback=int(fb_ev.sum()), global_policy_ceiling=float(global_ceiling),
                score_union_ceiling=float(score_union_ceiling))


# ----------------------------------------------------------------------------- orchestration

def run_dev_exploration(records, config=None, score_families=None, *, real_mode=False,
                        v2_replay_red_by_disease=None, require_exact_eval_coverage=None):
    """EXPLORATORY V4 Phase-1 orchestration on V4OOFRecords (synthetic here). Verdict is
    V4_DEV_CANDIDATE_FOUND_FOR_POSSIBLE_FREEZE or V4_DEV_NEGATIVE_NO_LOCKBOX — never SELECT/DEV_STOP/binding.

    real_mode (the real old-seven run) ALWAYS enforces exact OOF EVAL coverage (every subject & batch EVAL exactly
    once); synthetic tests default to relaxed (≤1) and may opt in via require_exact_eval_coverage=True."""
    cfg = config or V4DevConfig()
    if cfg.g3_comparator not in ("best_fixed", "v2_replay"):
        raise ValueError("g3_comparator must be 'best_fixed' or 'v2_replay'")
    exact = bool(require_exact_eval_coverage) or real_mode
    recs = _validate_records(records, cfg, require_exact_eval_coverage=exact)
    sfs = _resolve_score_families(score_families, real_mode)
    present = [d for d in DISEASES if any(r.disease == d for r in recs)]
    bundles = {}
    for d in present:
        v2r = None if v2_replay_red_by_disease is None else v2_replay_red_by_disease.get(d)
        bundles[d] = _process_disease([r for r in recs if r.disease == d], sfs, cfg, v2r)
    if cfg.g3_comparator == "v2_replay":
        for d, b in bundles.items():
            if b["c0_v2_replay_red"] is None:
                raise ValueError("g3_comparator='v2_replay' requires v2_replay_red_by_disease for every disease")

    def _c0(b):
        return b["c0_best_fixed_red"] if cfg.g3_comparator == "best_fixed" else b["c0_v2_replay_red"]

    both = set(bundles.keys()) == set(DISEASES)
    macro_c0 = float(np.mean([_c0(bundles[d]) for d in bundles])) if bundles else 0.0
    # match configs across diseases by (score_family, policy_family, loss)
    by_cfg = {}
    for d, b in bundles.items():
        for c in b["configs"]:
            by_cfg.setdefault((c["score_family"], c["policy_family"], c["loss"]), {})[d] = c
    reports = []
    cfg_pass = {}
    for key, peers in by_cfg.items():
        have_both = set(peers.keys()) == set(DISEASES)
        macro_red = float(np.mean([peers[d]["red"] for d in DISEASES])) if have_both else None
        g3 = bool(have_both and macro_red is not None and macro_red > macro_c0)
        # per-disease gate evaluation
        local = {}
        for d, c in peers.items():
            g1 = bool(c["coverage"] >= cfg.coverage_min)
            g2 = bool(c["red"] > 0.0)
            g4 = bool(c["rc_status"] == "PASS")
            local[d] = (g1, g2, g4)
        g6 = bool(both and have_both and all(local[d][0] and local[d][1] for d in DISEASES))
        passed_config = bool(have_both and g3 and g6 and all(local[d][0] and local[d][1] and local[d][2]
                                                             for d in DISEASES))
        cfg_pass[key] = passed_config
        for d, c in peers.items():
            b = bundles[d]
            g1, g2, g4 = local[d]
            reports.append(V4CandidateReport(
                disease=d, policy_family=c["policy_family"], loss=c["loss"], calibration_method=cfg.method,
                selected_lambda=c["selected_lambda"], status="EVALUATED", coverage=c["coverage"], red=c["red"],
                harm_rate=c["harm_rate"], c0_red=_c0(b), disease_macro_red=macro_red,
                g0_pass=True, g1_coverage_pass=g1, g2_red_pass=g2, g3_macro_vs_c0_pass=g3, g4_harm_control_pass=g4,
                g5_fallback_denominator_pass=True, g6_nonvacuous_both_diseases_pass=g6,
                per_config_policy_gap=b["score_union_ceiling"] - c["per_config_policy_ceiling"],
                global_policy_family_gap=b["score_union_ceiling"] - b["global_policy_ceiling"],
                frontier_gaps=c["gaps"], hierarchy_summary=c["hierarchy"],
                provenance={"score_family": c["score_family"], "per_fold_lambda": c["per_fold_lambda"],
                            "rc_status": c["rc_status"], "n_eval_folds": c["n_eval_folds"],
                            "n_evaluable_folds": c["n_evaluable_folds"], "n_passed_folds": c["n_passed_folds"],
                            "n_eval_subjects": b["n_eval_subjects"],
                            "n_eval_batches": b["n_eval_batches"], "n_cal_subjects": b["n_cal_subjects"],
                            "n_cal_batches": b["n_cal_batches"], "n_fit_subjects": b["n_fit_subjects"],
                            "n_fit_batches": b["n_fit_batches"], "n_fallback": b["n_fallback"],
                            "c0_best_fixed_red": b["c0_best_fixed_red"], "c0_v2_replay_red": b["c0_v2_replay_red"],
                            "g3_comparator": cfg.g3_comparator, "global_policy_ceiling": b["global_policy_ceiling"],
                            "score_union_ceiling": b["score_union_ceiling"]}))
    verdict = V4_DEV_CANDIDATE_FOUND if any(cfg_pass.values()) else V4_DEV_NEGATIVE
    manifest = _build_manifest(cfg, reports, recs, bundles, verdict, sfs)
    result = V4DevExplorationResult(V4_DEV_EXPLORATION_COMPLETE, verdict, tuple(reports), manifest,
                                    manifest["manifest_sha256"])
    assert_no_binding_language(result)
    return result


# ----------------------------------------------------------------------------- manifest + guards

def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else "NOT_EVALUABLE"
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if o is None or isinstance(o, str):
        return o
    raise TypeError(f"non-JSON-safe value of type {type(o)}")


def _report_summary(r):
    return {"disease": r.disease, "policy_family": r.policy_family, "loss": r.loss, "status": r.status,
            "selected_lambda": r.selected_lambda, "coverage": r.coverage, "red": r.red, "harm_rate": r.harm_rate,
            "c0_red": r.c0_red, "disease_macro_red": r.disease_macro_red,
            "per_config_policy_gap": r.per_config_policy_gap, "global_policy_family_gap": r.global_policy_family_gap,
            "gates": {"g0": r.g0_pass, "g1": r.g1_coverage_pass, "g2": r.g2_red_pass, "g3": r.g3_macro_vs_c0_pass,
                      "g4": r.g4_harm_control_pass, "g5": r.g5_fallback_denominator_pass,
                      "g6": r.g6_nonvacuous_both_diseases_pass, "all_pass": r.all_pass()},
            "frontier_gaps": r.frontier_gaps, "hierarchy_summary": r.hierarchy_summary, "provenance": r.provenance}


def _u(b):
    return len(b).to_bytes(8, "big") + b           # length-prefixed ⇒ injective concatenation


def _record_digest(r):
    h = hashlib.sha256()
    for s in (r.disease, r.cohort_id, r.subject_id, r.batch_id, str(int(r.fold)), r.split,
              "1" if r.fallback else "0"):
        h.update(_u(s.encode()))
    h.update(_u(np.ascontiguousarray(r.dr, dtype=np.float64).tobytes()))
    h.update(_u(np.ascontiguousarray(r.features_v2, dtype=np.float64).tobytes()))
    h.update(_u(str(len(r.action_names)).encode()))         # length-prefix each name ⇒ injective regardless of content
    for nm in r.action_names:
        h.update(_u(nm.encode()))
    return h.hexdigest()


def _records_sha256(recs):
    """Permutation-independent (sorted) digest of the exact V4OOFRecord input set; sensitive to ANY field change."""
    return hashlib.sha256("\n".join(sorted(_record_digest(r) for r in recs)).encode()).hexdigest()


def _partition_provenance(recs):
    groups = {}
    for r in recs:
        groups.setdefault((r.disease, r.cohort_id, int(r.fold), r.split), []).append(r)
    out = []
    for key in sorted(groups):
        g = groups[key]
        subs = sorted(set(rr.subject_id for rr in g))
        sh = hashlib.sha256()
        for s in subs:                                       # length-prefixed ⇒ injective regardless of id content
            sh.update(_u(s.encode()))
        out.append({"disease": key[0], "cohort_id": key[1], "fold": key[2], "split": key[3],
                    "batch_count": len(g), "subject_count": len(subs),
                    "subject_list_sha256": sh.hexdigest()})
    return out


def _registry_sha256(sfs):
    return hashlib.sha256(("|".join(sorted(f.name for f in sfs)) + "||"
                           + "|".join(sorted(SCORE_FAMILY_REGISTRY.keys()))).encode()).hexdigest()


def _build_manifest(cfg, reports, recs, bundles, verdict, sfs):
    body = {
        "boundary": "NON-BINDING / POST-V3 DEV_STOP / DEV-ONLY / NO EXTERNAL ARM / NO LOCKBOX",
        "lineage": {"v2": "MEASUREMENT_ONLY", "v3": "DEV_STOP / NO_LOCKBOX_CONSUMED", "v4": "NON-BINDING"},
        "run_status": V4_DEV_EXPLORATION_COMPLETE,
        "verdict": verdict,
        "config": {"policy_families": list(cfg.policy_families), "losses": list(cfg.losses), "alpha": cfg.alpha,
                   "budget_by_loss": dict(cfg.budget_by_loss), "correction": cfg.correction, "method": cfg.method,
                   "grid_size": cfg.grid_size, "coverage_min": cfg.coverage_min, "g3_comparator": cfg.g3_comparator,
                   "dev_cohort_ids": list(cfg.dev_cohort_ids)},
        "diseases": {d: {"n_eval_subjects": b["n_eval_subjects"], "n_eval_batches": b["n_eval_batches"],
                         "n_cal_subjects": b["n_cal_subjects"], "n_cal_batches": b["n_cal_batches"],
                         "n_fit_subjects": b["n_fit_subjects"], "n_fit_batches": b["n_fit_batches"],
                         "n_fallback": b["n_fallback"], "c0_best_fixed_red": b["c0_best_fixed_red"],
                         "c0_v2_replay_red": b["c0_v2_replay_red"], "global_policy_ceiling": b["global_policy_ceiling"],
                         "score_union_ceiling": b["score_union_ceiling"]} for d, b in bundles.items()},
        "n_records": len(recs),
        "reports": [_report_summary(r) for r in reports],
    }
    body["v4_oof_records_sha256"] = _records_sha256(recs)
    body["partition"] = _partition_provenance(recs)
    body["score_family_registry_sha256"] = _registry_sha256(sfs)
    safe = _json_safe(body)
    safe["config_sha256"] = hashlib.sha256(
        json.dumps(safe["config"], sort_keys=True, allow_nan=False).encode()).hexdigest()
    digest = hashlib.sha256(json.dumps(safe, sort_keys=True, allow_nan=False).encode()).hexdigest()
    safe["manifest_sha256"] = digest
    return safe


def write_dev_exploration_result(result, outdir):
    """NON-BINDING fail-closed writer for an exploratory result. `outdir` is claimed ATOMICALLY via os.mkdir — this
    races-safe refuses an existing dir (FileExistsError, even if empty/concurrent: no TOCTOU). Files are written into
    the claimed dir (manifest.json first, RESULT.json last as a completion sentinel); on ANY failure the partial dir is
    removed and the error re-raised. JSON uses allow_nan=False (non-finite already mapped to NOT_EVALUABLE). Returns
    outdir. (Single-process, non-binding; not crash-durable — no fsync.)"""
    os.mkdir(outdir)                                          # atomic no-overwrite claim (FileExistsError if present)
    try:
        with open(os.path.join(outdir, "manifest.json"), "w") as f:
            json.dump(result.manifest, f, sort_keys=True, allow_nan=False, indent=2)
        with open(os.path.join(outdir, "RESULT.json"), "w") as f:    # written last ⇒ presence = complete
            json.dump({"run_status": result.run_status, "verdict": result.verdict,
                       "manifest_sha256": result.manifest_sha256,
                       "v4_oof_records_sha256": result.manifest.get("v4_oof_records_sha256")},
                      f, sort_keys=True, allow_nan=False, indent=2)
    except BaseException:
        shutil.rmtree(outdir, ignore_errors=True)
        raise
    return outdir


def assert_no_binding_language(result):
    """Fail-closed: never a binding/SELECT/DEV_STOP/external verdict; no external/lockbox manifest keys. Scans only
    status/verdict fields (lineage prose may legitimately mention 'v3 DEV_STOP')."""
    if result.run_status not in _ALLOWED_RUN_STATUS:
        raise ValueError(f"illegal run_status {result.run_status!r}")
    if result.verdict not in _ALLOWED_VERDICTS:
        raise ValueError(f"illegal verdict {result.verdict!r}")
    scanned = [result.run_status, result.verdict, result.manifest.get("verdict", ""),
               result.manifest.get("run_status", "")]
    for r in result.reports:
        if r.status != "EVALUATED":
            raise ValueError(f"illegal report status {r.status!r}")
        scanned.append(r.status)
    for s in scanned:
        for tok in _FORBIDDEN_STATUS_TOKENS:
            if tok in str(s):
                raise ValueError(f"forbidden status token {tok!r} in {s!r}")
    for k in result.manifest:
        if str(k).lower() in _FORBIDDEN_MANIFEST_KEYS:
            raise ValueError(f"forbidden manifest key {k!r}")
    return True
