"""ACAR v4 (CURB) — Phase-1 EXPLORATORY DEV orchestration.

NON-BINDING / POST-V3 DEV_STOP / DEV-ONLY / NO EXTERNAL ARM / NO LOCKBOX.

This is NOT v3's binding DEV gate runner. It is the V4 Phase-1 *exploratory* runner: it wires the three hardened
primitives — Direction C frontiers (`frontiers.py`), Direction A finite-grid risk control (`risk_control.py`), and
Direction B deployed-risk objects (`hierarchy.py`) — into per-disease + disease-macro exploratory reports with a
strict-but-NON-BINDING G0–G6 candidate gate. It reads no real cohort here (synthetic `V4OOFRecord` fixtures only); the
real run later derives `V4OOFRecord`s from v3's single-execution cache. It NEVER:
  - emits `SELECT`, `DEV_STOP`, `PROCEED_SAFE_ROUTER`, `UTILITY_ONLY`, an external G2, or a coverage theorem,
  - consumes the lockbox or approaches external Arm B,
  - writes `ACAR_FROZEN_v4.md` or selects a deployable router.

Result taxonomy (the ONLY allowed verdicts):
  V4_DEV_EXPLORATION_COMPLETE                    (run_status: orchestration finished cleanly)
  V4_DEV_CANDIDATE_FOUND_FOR_POSSIBLE_FREEZE     (verdict: ≥1 config passed G0–G6 → worth a FUTURE v4 freeze; NOT Arm B)
  V4_DEV_NEGATIVE_NO_LOCKBOX                      (verdict: no config passed)
  OPERATIONALLY_ABORTED_NO_SCIENTIFIC_VERDICT    (runner-level: process died before a verdict; never written here)

Sign convention (frozen): ΔR_a(B) < 0 = reduced risk (good); identity/fallback realize 0 and stay in the subject
denominator. Subjects are the exchangeable unit; aggregation is subject-equal. Fail-closed throughout.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import math
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
# forbidden as a STATUS/VERDICT (lineage prose may still mention "v3 DEV_STOP" — scanned fields only)
_FORBIDDEN_STATUS_TOKENS = ("SELECT", "DEV_STOP", "PROCEED_SAFE_ROUTER", "UTILITY_ONLY", "EXTERNAL_G2",
                            "COVERAGE_THEOREM", "ARM_B", "LOCKBOX_CONSUMED")
_FORBIDDEN_MANIFEST_KEYS = ("external_g2", "lockbox", "arm_b", "coverage_theorem", "select", "binding")

ACTIONS = tuple(NON_IDENTITY)                       # ("matched_coral", "spdim", "t3a")
A = len(ACTIONS)
N_FEAT = 11                                         # bit-for-bit v2 feature_vector width (7 paired + 4 context)
DISEASES = tuple(sorted(DISEASE.keys()))           # ("PD", "SCZ")
_DEV_COHORTS = tuple(sorted(c for ds in DISEASE.values() for c in ds))
_SPLITS = ("FIT", "CAL", "EVAL")


# ----------------------------------------------------------------------------- DEV-ready record contract

@dataclass(frozen=True)
class V4OOFRecord:
    """One out-of-fold EVAL cell. dr is per non-identity action; features_v2 is the bit-for-bit v2 paired vector per
    action ([A, 11]) — a single 11-vector per batch could not produce per-action harm/benefit scores. Synthetic
    fixtures build these directly; the real run derives them from v3's single-execution cache (label-free at deployment;
    dr from the separate labeled-risk path)."""
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


@dataclass(frozen=True)
class ScoreFamily:
    """A PRE-LISTED, label-free score transform: feats [N, A, N_FEAT] → (harm [N, A], benefit [N, A]), both "lower is
    safer/better". MUST NOT use ΔR (true ΔR is for evaluation/oracle only)."""
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
    dev_cohort_ids: Tuple[str, ...] = _DEV_COHORTS


@dataclass(frozen=True)
class V4CandidateReport:
    disease: str
    policy_family: str
    loss: str
    calibration_method: str
    selected_lambda: Optional[float]
    status: str                                     # always "EVALUATED" (never SELECT / DEV_STOP)
    coverage: float
    red: float
    harm_rate: float
    c0_red: float
    disease_macro_red: Optional[float]
    g0_pass: bool
    g1_coverage_pass: bool
    g2_red_pass: bool
    g3_macro_vs_c0_pass: bool
    g4_harm_control_pass: bool
    g5_fallback_denominator_pass: bool
    g6_nonvacuous_both_diseases_pass: bool
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


# ----------------------------------------------------------------------------- default (placeholder) score families

def default_score_families():
    """Coordinate-based placeholders (NOT scientific) so the module is self-contained — the REAL predeclared label-free
    transforms are fixed in ACAR_FROZEN_v4.md before any freeze. v2 paired order: d_entropy0 d_margin1 flip_rate2 js3
    bures4 post_sep5 n_eff6 | g_unc7 s_support8 s_sep9 pr_cmi_proxy10."""
    def _margin(feats):
        harm = feats[:, :, 1]                        # d_margin as a harm proxy (higher shift ⇒ more harm)
        return harm, harm                            # benefit center = same proxy (placeholder)

    def _js_flip(feats):
        return feats[:, :, 3], feats[:, :, 2]        # js as harm, flip_rate as benefit center (placeholder)

    return (ScoreFamily("shift_margin", _margin), ScoreFamily("js_flip", _js_flip))


# ----------------------------------------------------------------------------- record validation

def _validate_records(records, cfg):
    if not records:
        raise ValueError("records must be a non-empty sequence of V4OOFRecord")
    seen = set()
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
        key = (r.disease, r.cohort_id, r.batch_id)
        if key in seen:
            raise ValueError(f"duplicate (disease, cohort, batch_id): {key}")
        seen.add(key)
        out.append(r)
    # canonical order ⇒ permutation-independent downstream + manifest digest
    out.sort(key=lambda r: (r.disease, r.subject_id, r.cohort_id, r.batch_id, r.fold))
    return tuple(out)


# ----------------------------------------------------------------------------- per-disease block

@dataclass(frozen=True)
class _Block:
    disease: str
    dr: np.ndarray                     # [N, A]
    feats: np.ndarray                  # [N, A, N_FEAT]
    subject_ids: np.ndarray            # [N] str
    weights: np.ndarray               # [N] subject-macro
    fallback: np.ndarray              # [N] bool
    c0_red: float
    n_subjects: int


def _build_block(recs):
    dr = np.stack([np.asarray(r.dr, dtype=float) for r in recs])
    feats = np.stack([np.asarray(r.features_v2, dtype=float) for r in recs])
    subj = np.array([r.subject_id for r in recs])
    fb = np.array([bool(r.fallback) for r in recs])
    w = PO.subject_macro_weights(subj)
    # C0 comparator = best fixed non-identity action red (the v3 best_fixed analog; real run substitutes v2 replay)
    c0 = -np.inf
    for a in range(A):
        choice = np.full(dr.shape[0], a, dtype=int)
        choice[fb] = PO.IDENTITY                                # fallback always identity
        c0 = max(c0, PO.reduction(choice, dr, weights=w))
    return _Block(recs[0].disease, dr, feats, subj, w, fb, float(c0), int(np.unique(subj).shape[0]))


# ----------------------------------------------------------------------------- policy family adapters

def _apply_family(name, harm, benefit, lam, fallback):
    if name == "safe_set":
        choice = PO.safe_set_policy(harm, benefit, lam)
    elif name == "benefit_ranked":
        choice = PO.benefit_ranked_policy(benefit, lam)                       # lam = τ
    elif name == "direct_selective":
        gate = -np.min(harm, axis=1)                                         # higher ⇒ safer ⇒ adapt first
        action = PO.best_benefit_action(benefit)[0]
        choice = PO.direct_selective_policy(gate, action, lam)               # lam = gate threshold τ
    else:
        raise ValueError(f"unknown policy family {name!r}")
    choice = choice.copy()
    choice[fallback] = PO.IDENTITY                                           # fallback retained as identity
    return choice


def _grid_for_family(name, harm, benefit, grid_size):
    """Return (sorted-unique λ grid, aggressiveness) for the family, or (None, None) if a grid cannot be formed."""
    if name == "safe_set":
        stat = harm.ravel(); aggr = "increasing_lambda"                      # larger λ ⇒ larger safe set ⇒ more cover
    elif name == "benefit_ranked":
        stat = np.min(benefit, axis=1); aggr = "increasing_lambda"          # larger τ ⇒ more adapt
    elif name == "direct_selective":
        stat = -np.min(harm, axis=1); aggr = "decreasing_lambda"            # smaller τ ⇒ more adapt ⇒ more aggressive
    else:
        raise ValueError(f"unknown policy family {name!r}")
    lo, hi = float(np.min(stat)), float(np.max(stat))
    if not (hi > lo):
        return None, None
    grid = np.unique(np.linspace(lo, hi, grid_size))
    if grid.shape[0] < 2:
        return None, None
    return grid, aggr


# ----------------------------------------------------------------------------- one config evaluation

def _hierarchy_summary(sel_choice, harm, blk, loss):
    b0 = HI.all_action_joint_max(harm, blk.subject_ids).values
    if sel_choice is None:
        sel_choice = np.full(blk.dr.shape[0], PO.IDENTITY, dtype=int)
    b1 = HI.policy_subject_risk(sel_choice, blk.dr, blk.subject_ids, loss=loss).values
    b2 = HI.hierarchical_policy_risk(sel_choice, blk.dr, blk.subject_ids, loss=loss, batch_summary="mean").values
    return {"b0_mean": float(np.mean(b0)), "b1_mean": float(np.mean(b1)), "b2_mean": float(np.mean(b2)),
            "b0_minus_b1_mean": float(np.mean(b0) - np.mean(b1))}


def _eval_config(blk, sf, pf, loss, union_candidates, cfg):
    harm, benefit = sf.compute(blk.feats)
    harm = np.asarray(harm, dtype=float); benefit = np.asarray(benefit, dtype=float)
    if harm.shape != (blk.dr.shape[0], A) or benefit.shape != (blk.dr.shape[0], A):
        raise ValueError("score family must return harm/benefit of shape [N, A]")
    if not (np.all(np.isfinite(harm)) and np.all(np.isfinite(benefit))):
        raise ValueError("score family produced non-finite harm/benefit")
    grid, aggr = _grid_for_family(pf, harm, benefit, cfg.grid_size)
    identity = np.full(blk.dr.shape[0], PO.IDENTITY, dtype=int)
    selected_lambda = None
    rc_status = "NOT_EVALUABLE"
    cov, red, hr = 0.0, 0.0, float("nan")
    sel_choice = None
    choices_by_lambda = None
    if grid is not None:
        choices_by_lambda = np.stack([_apply_family(pf, harm, benefit, lam, blk.fallback) for lam in grid])
        subj_losses = RC.subject_losses_from_policy(choices_by_lambda, blk.dr, blk.subject_ids, loss=loss)
        rc = RC.select_ltt_grid(grid, subj_losses, alpha=cfg.alpha, budget=cfg.budget_by_loss[loss],
                                aggressiveness=aggr, correction=cfg.correction, method=cfg.method)
        rc_status = rc.status
        if rc.selected_index is not None:
            sel_choice = choices_by_lambda[rc.selected_index]
            selected_lambda = rc.selected_lambda
            cov, red, hr = FR.operating_point(blk.dr, sel_choice, weights=blk.weights)
    # Direction C frontier gaps (ceiling mode; score-oracle = union over ALL score families for this disease)
    policy_choices = list(choices_by_lambda) if choices_by_lambda is not None else [identity]
    gaps = FR.gap_decomposition(blk.dr, union_candidates, policy_choices,
                                sel_choice if sel_choice is not None else identity,
                                weights=blk.weights, mode="ceiling")
    hier = _hierarchy_summary(sel_choice, harm, blk, loss)
    g1 = bool(cov >= cfg.coverage_min)
    g2 = bool(red > 0.0)
    g4 = bool(rc_status == "PASS")
    prov = {"selected_lambda": selected_lambda, "rc_status": rc_status, "n_subjects": blk.n_subjects,
            "n_batches": int(blk.dr.shape[0]), "n_fallback": int(blk.fallback.sum()), "c0_red": blk.c0_red,
            "score_family": sf.name, "aggressiveness": aggr}
    return dict(disease=blk.disease, score_family=sf.name, policy_family=pf, loss=loss, selected_lambda=selected_lambda,
                coverage=float(cov), red=float(red), harm_rate=float(hr), c0_red=blk.c0_red, rc_status=rc_status,
                g1=g1, g2=g2, g4=g4, frontier_gaps=gaps, hierarchy_summary=hier, provenance=prov)


# ----------------------------------------------------------------------------- orchestration

def run_dev_exploration(records, config=None, score_families=None):
    """EXPLORATORY V4 Phase-1 orchestration on V4OOFRecords (synthetic here). Returns a V4DevExplorationResult whose
    verdict is V4_DEV_CANDIDATE_FOUND_FOR_POSSIBLE_FREEZE or V4_DEV_NEGATIVE_NO_LOCKBOX — never SELECT/DEV_STOP/binding."""
    cfg = config or V4DevConfig()
    recs = _validate_records(records, cfg)
    if score_families is None:
        score_families = default_score_families()
    if not score_families:
        raise ValueError("score_families must be a non-empty list")
    blocks = {d: _build_block([r for r in recs if r.disease == d]) for d in DISEASES if any(r.disease == d for r in recs)}
    # union score-oracle candidates per disease (information ceiling of the listed observables)
    union = {}
    for d, blk in blocks.items():
        cands = []
        for sf in score_families:
            harm, benefit = sf.compute(blk.feats)
            cands.append((PO.adapt_rank_from_harm(np.asarray(harm, float)),
                          PO.best_benefit_action(np.asarray(benefit, float))[0]))
        union[d] = cands
    # per-config evaluation
    raw = []
    for d, blk in blocks.items():
        for sf in score_families:
            for pf in cfg.policy_families:
                for loss in cfg.losses:
                    raw.append(_eval_config(blk, sf, pf, loss, union[d], cfg))
    # disease-macro: match configs across diseases by (score_family, policy_family, loss)
    by_cfg = {}
    for c in raw:
        by_cfg.setdefault((c["score_family"], c["policy_family"], c["loss"]), {})[c["disease"]] = c
    macro_c0 = float(np.mean([blocks[d].c0_red for d in blocks])) if blocks else 0.0
    both = set(blocks.keys()) == set(DISEASES)
    reports = []
    for c in raw:
        peers = by_cfg[(c["score_family"], c["policy_family"], c["loss"])]
        have_both = set(peers.keys()) == set(DISEASES)
        macro_red = float(np.mean([peers[d]["red"] for d in DISEASES])) if have_both else None
        g3 = bool(have_both and macro_red is not None and macro_red > macro_c0)
        g6 = bool(both and have_both and all(peers[d]["g1"] and peers[d]["g2"] for d in DISEASES))
        reports.append(V4CandidateReport(
            disease=c["disease"], policy_family=c["policy_family"], loss=c["loss"], calibration_method=cfg.method,
            selected_lambda=c["selected_lambda"], status="EVALUATED", coverage=c["coverage"], red=c["red"],
            harm_rate=c["harm_rate"], c0_red=c["c0_red"], disease_macro_red=macro_red,
            g0_pass=True, g1_coverage_pass=c["g1"], g2_red_pass=c["g2"], g3_macro_vs_c0_pass=g3,
            g4_harm_control_pass=c["g4"], g5_fallback_denominator_pass=True, g6_nonvacuous_both_diseases_pass=g6,
            frontier_gaps=c["frontier_gaps"], hierarchy_summary=c["hierarchy_summary"], provenance=c["provenance"]))
    # a CONFIG passes if BOTH disease reports all_pass
    cfg_pass = {}
    for key, peers in by_cfg.items():
        peer_reports = [r for r in reports if (r.policy_family, r.loss) == (key[1], key[2])
                        and r.provenance.get("score_family") == key[0]]
        cfg_pass[key] = (set(peers.keys()) == set(DISEASES) and len(peer_reports) == len(DISEASES)
                         and all(r.all_pass() for r in peer_reports))
    verdict = V4_DEV_CANDIDATE_FOUND if any(cfg_pass.values()) else V4_DEV_NEGATIVE
    manifest = _build_manifest(cfg, reports, recs, blocks, verdict)
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
            "gates": {"g0": r.g0_pass, "g1": r.g1_coverage_pass, "g2": r.g2_red_pass, "g3": r.g3_macro_vs_c0_pass,
                      "g4": r.g4_harm_control_pass, "g5": r.g5_fallback_denominator_pass,
                      "g6": r.g6_nonvacuous_both_diseases_pass, "all_pass": r.all_pass()},
            "frontier_gaps": r.frontier_gaps, "hierarchy_summary": r.hierarchy_summary, "provenance": r.provenance}


def _build_manifest(cfg, reports, recs, blocks, verdict):
    body = {
        "boundary": "NON-BINDING / POST-V3 DEV_STOP / DEV-ONLY / NO EXTERNAL ARM / NO LOCKBOX",
        "lineage": {"v2": "MEASUREMENT_ONLY", "v3": "DEV_STOP / NO_LOCKBOX_CONSUMED", "v4": "NON-BINDING"},
        "run_status": V4_DEV_EXPLORATION_COMPLETE,
        "verdict": verdict,
        "config": {"policy_families": list(cfg.policy_families), "losses": list(cfg.losses), "alpha": cfg.alpha,
                   "budget_by_loss": dict(cfg.budget_by_loss), "correction": cfg.correction, "method": cfg.method,
                   "grid_size": cfg.grid_size, "coverage_min": cfg.coverage_min,
                   "dev_cohort_ids": list(cfg.dev_cohort_ids)},
        "diseases": {d: {"n_subjects": blocks[d].n_subjects, "n_batches": int(blocks[d].dr.shape[0]),
                         "n_fallback": int(blocks[d].fallback.sum()), "c0_red": blocks[d].c0_red} for d in blocks},
        "n_records": len(recs),
        "reports": [_report_summary(r) for r in reports],
    }
    safe = _json_safe(body)
    digest = hashlib.sha256(json.dumps(safe, sort_keys=True, allow_nan=False).encode()).hexdigest()
    safe["manifest_sha256"] = digest
    return safe


def assert_no_binding_language(result):
    """Fail-closed: the run never emits a binding/SELECT/DEV_STOP/external verdict, and the manifest carries no
    external/lockbox keys. Scans only status/verdict fields (lineage prose may legitimately mention 'v3 DEV_STOP')."""
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
