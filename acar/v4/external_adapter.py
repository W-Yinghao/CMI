"""ACAR v4 — EXTERNAL Arm-B adapter: derive + evaluate the FROZEN candidate on a held-out site (single CAL/EVAL split).

NON-BINDING UNTIL TAGGED `acar-v4-protocol`. Reuses (imports, never modifies) the v3 substrate for ΔR + the bit-for-bit
v2 features (real path only), and the V4 primitives for the frozen candidate. The pure endpoint core (site-local split,
λ* selection, EVAL metrics, v2-replay comparator, taxonomy) is synthetic-tested in base env; the v3-coupled
`build_stratum_from_dump` runs only at the (gated) external read in eeg2025.

Frozen candidate (ACAR_FROZEN_v4.md §2): score_family = shift_margin (harm=benefit=+features_v2[:,:,1]); policy =
benefit_ranked (π=identity if min_a benefit>τ else argmin_a benefit; fallback→identity); loss = harm_indicator;
finite-grid LTT (ttest, holm, alpha 0.10, budget 0.10, grid_size 12, increasing-λ, most-aggressive passing λ).

Endpoint (§3c, criterion A) per (site,disease) stratum — CONFIRMED requires ALL of:
  CAL LTT certifies λ* (precond) · EVAL L_harm_all ≤ 0.10 (safety) · EVAL red > 0 AND red > v2_replay (utility) ·
  coverage ≥ 0.15. harm_among_adapted is DESCRIPTIVE (not a gate). NOT_EVALUABLE = data/split insufficiency.

Leakage firewall (§5): external diagnosis labels enter ONLY (a) CAL λ* selection and (b) EVAL endpoint scoring — never
f_0/source-state fitting, adapter execution, the label-free features, or the action choice before scoring; EVAL labels
never affect λ*; CAL labels affect λ* but never the score_family outputs.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from acar.v4.develop import ACTIONS, A, N_FEAT
from acar.v4 import policies as PO
from acar.v4 import risk_control as RC
from acar.regressor import ActionRegressor

# ---- frozen candidate constants ----
SCORE_COORD = 1                 # shift_margin: harm = benefit = +features_v2[:, :, 1] (= +d_margin)
LOSS = "harm_indicator"
BUDGET = 0.10
ALPHA = 0.10
GRID_SIZE = 12
DELTA = 0.0
CAL_FRAC = 0.40
MIN_CAL = 20
MIN_EVAL = 20
SPLIT_SEED = 0
C0_SEED = 1                     # secondary split of CAL into C0_FIT/C0_CAL for the v2 replay
C0_FIT_FRAC = 0.70
CLASSES = ("patient", "hc")
_STATUS = ("V4_EXTERNAL_CONFIRMED", "V4_EXTERNAL_NEGATIVE", "NOT_EVALUABLE")


@dataclass(frozen=True)
class StratumResult:
    site: str
    disease: str
    status: str                 # one of _STATUS
    reason: str
    n_cal_subjects: int
    n_eval_subjects: int
    selected_lambda: Optional[float]
    coverage: float
    red: float
    L_harm_all_eval: float      # the LTT-controlled object, checked ≤ budget on EVAL (criterion A)
    harm_among_adapted: float   # DESCRIPTIVE only
    v2_replay_red: float
    v2_replay_status: str


@dataclass(frozen=True)
class ExternalResult:
    run_status: str             # V4_EXTERNAL_COMPLETE | OPERATIONALLY_ABORTED_NO_SCIENTIFIC_VERDICT
    verdict: str                # V4_EXTERNAL_CONFIRMED | V4_EXTERNAL_NEGATIVE
    per_disease: dict           # {disease: {confirmed: bool, single_site: bool, strata: [...]}}
    strata: Tuple[StratumResult, ...]


# ----------------------------------------------------------------------------- deterministic hashing + splits

def _u01(key, salt):
    h = hashlib.sha256((str(salt) + "|" + str(key)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def site_local_split(subject_to_class, *, seed=SPLIT_SEED, cal_frac=CAL_FRAC, min_cal=MIN_CAL, min_eval=MIN_EVAL):
    """Subject-disjoint single CAL/EVAL split by subject hash. Returns (cal:set, eval:set, status, reason). Status
    NOT_EVALUABLE if either side < min or a class (patient/hc) is absent from either side (fail-closed)."""
    subs = list(subject_to_class)
    if not subs:
        return set(), set(), "NOT_EVALUABLE", "no subjects"
    for cc, cl in subject_to_class.items():
        if cl not in CLASSES:
            raise ValueError(f"subject {cc} class {cl!r} not in {CLASSES}")
    ordered = sorted(subs, key=lambda cc: _u01(cc, f"extsplit/{seed}"))
    n_cal = int(math.floor(cal_frac * len(ordered)))
    cal, ev = set(ordered[:n_cal]), set(ordered[n_cal:])
    if len(cal) < min_cal or len(ev) < min_eval:
        return cal, ev, "NOT_EVALUABLE", f"too few subjects (CAL {len(cal)} < {min_cal} or EVAL {len(ev)} < {min_eval})"
    for name, grp in (("CAL", cal), ("EVAL", ev)):
        present = {subject_to_class[cc] for cc in grp}
        if not (set(CLASSES) <= present):
            return cal, ev, "NOT_EVALUABLE", f"{name} missing a class (have {sorted(present)})"
    return cal, ev, "OK", ""


def _secondary_split(cal_subjects, *, seed=C0_SEED, fit_frac=C0_FIT_FRAC):
    ordered = sorted(cal_subjects, key=lambda cc: _u01(cc, f"c0split/{seed}"))
    n_fit = int(math.floor(fit_frac * len(ordered)))
    return set(ordered[:n_fit]), set(ordered[n_fit:])


# ----------------------------------------------------------------------------- array builders + scores

def _arrays_for(stratum, subjects):
    dr, feats, subj, fb = [], [], [], []
    for cc in sorted(subjects):
        cell = stratum["subjects"][cc]
        for (_bid, d, f) in cell["eligible"]:
            dr.append(np.asarray(d, float)); feats.append(np.asarray(f, float)); subj.append(cc); fb.append(False)
        for _bid in cell["fallback"]:
            dr.append(np.zeros(A)); feats.append(np.zeros((A, N_FEAT))); subj.append(cc); fb.append(True)
    if not dr:
        return np.zeros((0, A)), np.zeros((0, A, N_FEAT)), np.array([], dtype="<U1"), np.array([], dtype=bool)
    return np.stack(dr), np.stack(feats), np.array(subj), np.array(fb)


def _shift_margin(feats):
    h = feats[:, :, SCORE_COORD]
    return h, h                  # harm = benefit = +d_margin


def _benefit_ranked_choice(benefit, tau, fb):
    ch = PO.benefit_ranked_policy(benefit, tau).copy()
    ch[fb] = PO.IDENTITY
    return ch


def _one_sided_q(scores, alpha):
    s = sorted(float(x) for x in scores)
    m = len(s)
    if m == 0:
        return math.inf
    k = math.ceil((m + 1) * (1 - alpha))
    return math.inf if k > m else s[k - 1]


# ----------------------------------------------------------------------------- v2 replay (external, apples-to-apples)

def _v2_replay_red(stratum, c0fit, c0cal, eval_subjects, *, alpha=ALPHA, delta=DELTA):
    """v2 recipe (ActionRegressor seed 0 on C0_FIT; one-sided q on C0_CAL per-subject smax; route on EVAL), subject-macro
    red on the SAME EVAL subjects as V4 (apples-to-apples). NOT_EVALUABLE if C0_FIT or C0_CAL has no eligible batch."""
    Xy = {a: ([], []) for a in range(A)}
    for cc in c0fit:
        for (_bid, d, f) in stratum["subjects"][cc]["eligible"]:
            for a in range(A):
                Xy[a][0].append(np.asarray(f, float)[a]); Xy[a][1].append(float(np.asarray(d, float)[a]))
    if not Xy[0][0]:
        return 0.0, "NOT_EVALUABLE"
    regs = {a: ActionRegressor(seed=0).fit(np.array(Xy[a][0]), np.array(Xy[a][1])) for a in range(A)}
    cal_scores = []
    for cc in c0cal:
        elig = stratum["subjects"][cc]["eligible"]
        if not elig:
            continue
        smax = -math.inf
        for (_bid, d, f) in elig:
            d = np.asarray(d, float); f = np.asarray(f, float)
            for a in range(A):
                smax = max(smax, d[a] - float(regs[a].predict([f[a]])[0]))
        cal_scores.append(smax)
    if not cal_scores:
        return 0.0, "NOT_EVALUABLE"
    q = _one_sided_q(cal_scores, alpha)
    per_subj = []
    for cc in sorted(eval_subjects):
        cell = stratum["subjects"][cc]; vals = []
        for (_bid, d, f) in cell["eligible"]:
            d = np.asarray(d, float); f = np.asarray(f, float)
            ghat = [float(regs[a].predict([f[a]])[0]) for a in range(A)]
            U = [ghat[a] + q for a in range(A)]
            elig_acts = [a for a in range(A) if U[a] < -delta]
            vals.append(d[min(elig_acts, key=lambda a: U[a])] if elig_acts else 0.0)
        vals += [0.0] * len(cell["fallback"])                 # fallback identity
        if vals:
            per_subj.append(float(np.mean(vals)))
    red = -float(np.mean(per_subj)) if per_subj else 0.0
    return red, "OK"


# ----------------------------------------------------------------------------- stratum endpoint (criterion A)

def evaluate_stratum(stratum, *, alpha=ALPHA, budget=BUDGET, delta=DELTA):
    """Evaluate the FROZEN candidate on one (site,disease) stratum; return a StratumResult."""
    site = stratum["site"]; disease = stratum["disease"]
    subject_to_class = {cc: c["class"] for cc, c in stratum["subjects"].items()}

    def _na(reason, ncal=0, nev=0, v2=0.0, v2s="NOT_RUN"):
        return StratumResult(site, disease, "NOT_EVALUABLE", reason, ncal, nev, None, 0.0, 0.0,
                             float("nan"), float("nan"), v2, v2s)

    cal, ev, sstat, sreason = site_local_split(subject_to_class)
    if sstat != "OK":
        return _na(sreason, len(cal), len(ev))
    dr_cal, feats_cal, subj_cal, fb_cal = _arrays_for(stratum, cal)
    dr_ev, feats_ev, subj_ev, fb_ev = _arrays_for(stratum, ev)
    if dr_cal.shape[0] == 0 or dr_ev.shape[0] == 0:
        return _na("empty CAL or EVAL batch set", len(cal), len(ev))
    _, benefit_cal = _shift_margin(feats_cal)
    _, benefit_ev = _shift_margin(feats_ev)
    # frozen λ grid: min_a benefit over CAL batches; 12-pt linspace; unique
    stat = np.min(benefit_cal, axis=1)
    lo, hi = float(np.min(stat)), float(np.max(stat))
    if not (hi > lo):
        return _na("degenerate λ grid (CAL min-benefit constant)", len(cal), len(ev))
    grid = np.unique(np.linspace(lo, hi, GRID_SIZE))
    if grid.shape[0] < 2:
        return _na("λ grid < 2 unique points", len(cal), len(ev))
    # CAL LTT on L_harm_all (harm_indicator)
    choices_cal = np.stack([_benefit_ranked_choice(benefit_cal, t, fb_cal) for t in grid])
    sl = RC.subject_losses_from_policy(choices_cal, dr_cal, subj_cal, loss=LOSS)
    rc = RC.select_ltt_grid(grid, sl, alpha=alpha, budget=budget, aggressiveness="increasing_lambda",
                            correction="holm", method="ttest")
    if rc.status == "NOT_EVALUABLE":
        return _na("CAL LTT NOT_EVALUABLE (<2 CAL subjects)", len(cal), len(ev))
    # v2 replay (computed regardless; needed for the utility comparison)
    c0fit, c0cal = _secondary_split(cal)
    v2_red, v2_status = _v2_replay_red(stratum, c0fit, c0cal, ev)
    if v2_status == "NOT_EVALUABLE":
        return _na("v2_replay NOT_EVALUABLE (C0_FIT/C0_CAL too small)", len(cal), len(ev), v2_red, v2_status)
    if rc.selected_index is None:                              # CAL did not certify any λ
        return StratumResult(site, disease, "V4_EXTERNAL_NEGATIVE", "CAL LTT did not certify a λ (NO_PASS)",
                             len(cal), len(ev), None, 0.0, 0.0, float("nan"), float("nan"), v2_red, v2_status)
    lam = float(rc.selected_lambda)
    choice_ev = _benefit_ranked_choice(benefit_ev, lam, fb_ev)
    w_ev = PO.subject_macro_weights(subj_ev)
    coverage = PO.coverage(choice_ev, weights=w_ev)
    red = PO.reduction(choice_ev, dr_ev, weights=w_ev)
    harm_adapt = PO.harm_rate(choice_ev, dr_ev, weights=w_ev)
    L_harm_all_eval = float(RC.subject_losses_from_policy(np.stack([choice_ev]), dr_ev, subj_ev, loss=LOSS).mean())
    ok = (L_harm_all_eval <= budget) and (red > 0.0) and (red > v2_red) and (coverage >= 0.15)
    status = "V4_EXTERNAL_CONFIRMED" if ok else "V4_EXTERNAL_NEGATIVE"
    reason = "criterion A met" if ok else (
        f"failed: " + ", ".join(x for x, c in (("L_harm_all>budget", L_harm_all_eval > budget),
                                               ("red<=0", red <= 0.0), ("red<=v2_replay", red <= v2_red),
                                               ("coverage<0.15", coverage < 0.15)) if c))
    return StratumResult(site, disease, status, reason, len(cal), len(ev), lam, float(coverage), float(red),
                         L_harm_all_eval, float(harm_adapt) if harm_adapt == harm_adapt else float("nan"),
                         float(v2_red), v2_status)


# ----------------------------------------------------------------------------- multi-site taxonomy (§3d deterministic)

def external_taxonomy(strata_results):
    """Per-disease confirmation is DETERMINISTIC (no cross-stratum p-adjustment): a disease is confirmed iff ≥1 evaluable
    stratum is CONFIRMED and no evaluable stratum is NEGATIVE. Overall CONFIRMED iff BOTH diseases confirmed. NOT_EVALUABLE
    strata are listed, never silently dropped; a single evaluable stratum is flagged single_site."""
    per_disease = {}
    for d in ("PD", "SCZ"):
        ds = [r for r in strata_results if r.disease == d]
        evaluable = [r for r in ds if r.status != "NOT_EVALUABLE"]
        confirmed_n = sum(r.status == "V4_EXTERNAL_CONFIRMED" for r in evaluable)
        negative_n = sum(r.status == "V4_EXTERNAL_NEGATIVE" for r in evaluable)
        confirmed = bool(confirmed_n >= 1 and negative_n == 0)
        per_disease[d] = {"confirmed": confirmed, "single_site": len(evaluable) == 1,
                          "n_evaluable": len(evaluable), "n_confirmed": confirmed_n, "n_negative": negative_n,
                          "n_not_evaluable": len(ds) - len(evaluable),
                          "strata": [{"site": r.site, "status": r.status, "reason": r.reason} for r in ds]}
    both = bool(per_disease["PD"]["confirmed"] and per_disease["SCZ"]["confirmed"])
    verdict = "V4_EXTERNAL_CONFIRMED" if both else "V4_EXTERNAL_NEGATIVE"
    return ExternalResult("V4_EXTERNAL_COMPLETE", verdict, per_disease, tuple(strata_results))


# ----------------------------------------------------------------------------- gated real path (eeg2025; external read)

def build_stratum_from_dump(site_id, disease, dump_path, env=None):
    """REAL path (gated): build a stratum dict from a held-out site's erm_0 feature dump via the v3 substrate. Imports v3
    (torch/eeg2025) lazily. The per-subject class (patient/hc) is taken from the dump labels (diagnosis target). Not
    exercised by the synthetic guards; runs only at the authorized external read."""
    from acar.v3 import develop as V3D
    from acar.v3.loader import build_cohort_input
    from acar.v3.data import deployment_batch_digest, canon_subject
    ci = build_cohort_input(dump_path, disease=disease, dataset_id=site_id, env=env)
    reg = V3D._as_registry(ci.source_artifact, disease)
    idx = V3D._subject_batches(list(ci.batches))
    cache = V3D.disease_exec_cache(reg, list(ci.batches), ci.labels)
    subjects = {}
    for cc, slot in idx.items():
        key = slot["key"]
        elig = []
        cls = None
        for b in slot["eligible"]:
            c = cache[deployment_batch_digest(b)]
            dr = np.array([float(c["dr"][a]) for a in ACTIONS], float)
            feats = np.stack([np.asarray(c["c0feat"][a], float) for a in ACTIONS])
            elig.append((deployment_batch_digest(b), dr, feats))
            cls = _subject_class_from_labels(b, ci.labels) if cls is None else cls
        fb = [deployment_batch_digest(b) for b in slot["fallback"]]
        if cls is None and slot["fallback"]:
            cls = _subject_class_from_labels(slot["fallback"][0], ci.labels)
        subjects[cc] = {"class": cls, "eligible": elig, "fallback": fb,
                        "subject_id": key.subject_id, "cohort_id": key.dataset_id}
    return {"site": site_id, "disease": disease, "subjects": subjects}


def _subject_class_from_labels(batch, labels):
    """Map a batch's (constant-per-subject) diagnosis label to 'patient'/'hc'. y==1 -> patient (PD/SCZ), y==0 -> hc."""
    ys = [labels[wk] for wk in batch.window_keys if wk in labels]
    if not ys:
        return None
    return "patient" if int(round(float(np.mean(ys)))) == 1 else "hc"
