"""C16-A — post-hoc TARGET-ORACLE CEILING over OACI's own trajectory. C10 showed that no source-only selector
AND the source-audit oracle can rescue OACI (case C). This asks a strictly deeper question:

    Does the trajectory contain ANY target-good checkpoint at all?

A NON-DEPLOYABLE target oracle reads target_audit post hoc to pick the best candidate per fold-level, ONLY to
determine existence. It is a diagnostic ceiling, never a selection method. Comparing the target-oracle K2 to
the source-audit-oracle K2 splits C10's case C:

  C1  target oracle rescues K2, source oracle does not  -> source-side OBSERVABILITY / selector unidentifiability
  C2  target oracle ALSO cannot rescue K2               -> TRAJECTORY failure (OACI never enters a target-good state)
  C3  target oracle rescues bAcc but not NLL/ECE        -> CALIBRATION / confidence, not discrimination
  C4  target oracle rescues only some subjects/classes  -> target HETEROGENEITY, not global method failure

Reads only the committed C10 per-fold replay candidate tables (no retraining).
"""
from __future__ import annotations

import glob
import json
import os

from ..decision.k2_decision import k2_decision

_MARGIN = {"worst_domain_bacc": 0.0, "worst_domain_nll": 0.0}


def load_replay(replay_dir) -> list:
    folds = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(replay_dir, "seed-*-target-*.json")))]
    if not folds:
        raise ValueError(f"no C10 replay fold JSONs in {replay_dir}")
    return folds


def _erm(cands):
    return next(c for c in cands if c.get("is_erm"))


def _pick(cands, key, *, maximize, gate=None):
    """Deterministic argmax/argmin over eligible candidates (tie-break by model_hash)."""
    elig = [c for c in cands if (c.get("is_erm") or c.get("feasible")) and c.get(key) is not None
            and (gate is None or gate(c))]
    if not elig:
        return None
    elig.sort(key=lambda c: ((-c[key] if maximize else c[key]), c["model_hash"]))
    return elig[0]


# ---- diagnostic ceiling selectors (target oracle is NON-DEPLOYABLE) ----
def _sel_erm(cands, erm):
    return erm

def _sel_source_audit_oracle(cands, erm):                 # reproduces C10 S5
    return _pick(cands, "source_audit_worst_bacc", maximize=True) or erm

def _sel_target_oracle_bacc(cands, erm):                  # NON-DEPLOYABLE: best target worst-domain bAcc
    return _pick(cands, "target_worst_bacc", maximize=True) or erm

def _sel_target_oracle_joint(cands, erm):                 # NON-DEPLOYABLE: bAcc>=ERM, then min target NLL
    b = _pick(cands, "target_worst_nll", maximize=False,
              gate=lambda c: c.get("target_worst_bacc") is not None and erm.get("target_worst_bacc") is not None
              and c["target_worst_bacc"] >= erm["target_worst_bacc"])
    return b or erm

_SELECTORS = {"ERM": _sel_erm, "source_audit_oracle": _sel_source_audit_oracle,
              "target_oracle_bacc": _sel_target_oracle_bacc, "target_oracle_joint": _sel_target_oracle_joint}
_TARGET_ORACLES = {"target_oracle_bacc", "target_oracle_joint"}


def run_target_oracle(folds) -> dict:
    seeds = sorted({f["seed"] for f in folds})
    targets = sorted({f["target"] for f in folds})
    levels = sorted({int(L) for f in folds for L in f["levels"]})
    idx = {(f["seed"], f["target"], int(L)): lv for f in folds for L, lv in f["levels"].items()}

    # per (selector, seed, target, level) -> chosen target metrics
    choices = {name: {} for name in _SELECTORS}
    for (s, t, L), lv in idx.items():
        cands = lv["candidates"]; erm = _erm(cands)
        for name, fn in _SELECTORS.items():
            ch = fn(cands, erm)
            choices[name][(s, t, L)] = {"target_worst_bacc": ch.get("target_worst_bacc"),
                                        "target_worst_nll": ch.get("target_worst_nll"),
                                        "target_worst_ece": ch.get("target_worst_ece"),
                                        "epoch": ch.get("epoch"), "is_erm": ch.get("is_erm"),
                                        "model_hash": ch.get("model_hash")}

    def _erm_worst(s, L):
        bb = [choices["ERM"][(s, t, L)]["target_worst_bacc"] for t in targets]
        nn = [choices["ERM"][(s, t, L)]["target_worst_nll"] for t in targets]
        bb = [x for x in bb if x is not None]; nn = [x for x in nn if x is not None]
        return (min(bb) if bb else None, max(nn) if nn else None)

    # worst-held-out-target K2 per selector (same estimand as C8/C10)
    per_selector = {}
    for name in _SELECTORS:
        units, per_unit, rows = [], [], []
        for s in seeds:
            for L in levels:
                sb = [choices[name][(s, t, L)]["target_worst_bacc"] for t in targets]
                sn = [choices[name][(s, t, L)]["target_worst_nll"] for t in targets]
                sb = [x for x in sb if x is not None]; sn = [x for x in sn if x is not None]
                wb, wn = (min(sb) if sb else None), (max(sn) if sn else None)
                eb, en = _erm_worst(s, L)
                db = None if (wb is None or eb is None) else wb - eb
                dn = None if (wn is None or en is None) else wn - en
                units.append({"seed": s, "level": L, "deltas": {"worst_domain_bacc": db, "worst_domain_nll": dn}})
                per_unit.append({"seed": s, "level": L, "worst_bacc": wb, "worst_nll": wn,
                                 "delta_worst_bacc": db, "delta_worst_nll": dn})
        for s in seeds:
            for t in targets:
                for L in levels:
                    c = choices[name][(s, t, L)]; e = choices["ERM"][(s, t, L)]
                    rows.append({"seed": s, "target": t, "level": L, "chosen_epoch": c["epoch"],
                                 "chosen_is_erm": c["is_erm"],
                                 "target_bacc_delta": (None if (c["target_worst_bacc"] is None or e["target_worst_bacc"] is None)
                                                       else c["target_worst_bacc"] - e["target_worst_bacc"]),
                                 "target_nll_delta": (None if (c["target_worst_nll"] is None or e["target_worst_nll"] is None)
                                                      else c["target_worst_nll"] - e["target_worst_nll"]),
                                 "target_ece_delta": (None if (c["target_worst_ece"] is None or e["target_worst_ece"] is None)
                                                      else c["target_worst_ece"] - e["target_worst_ece"])})
        k2 = k2_decision(units, endpoints=["worst_domain_bacc", "worst_domain_nll"], min_seeds=3,
                         level_policy="both_levels", margins=_MARGIN)
        per_selector[name] = {"k2_status": k2["k2_status"], "reproduced_endpoints": k2.get("reproduced_endpoints"),
                              "per_endpoint": k2.get("per_endpoint"), "per_unit": per_unit, "per_fold": rows,
                              "is_target_oracle": name in _TARGET_ORACLES,
                              "non_deployable": name in _TARGET_ORACLES}

    # ---- taxonomy ----
    src = per_selector["source_audit_oracle"]["k2_status"]
    tob = per_selector["target_oracle_bacc"]["k2_status"]
    toj = per_selector["target_oracle_joint"]["k2_status"]
    src_rescues = src == "reproducible_gain"
    tgt_rescues_bacc = "worst_domain_bacc" in (per_selector["target_oracle_bacc"].get("reproduced_endpoints") or [])
    tgt_rescues_joint = toj == "reproducible_gain"
    # per-endpoint: does the target oracle at least reproduce bAcc even if joint stops?
    if not tgt_rescues_bacc and not tgt_rescues_joint and not src_rescues:
        case = "C2_trajectory_failure"
    elif tgt_rescues_joint and not src_rescues:
        case = "C1_source_observability_failure"
    elif tgt_rescues_bacc and not tgt_rescues_joint:
        case = "C3_calibration_not_discrimination"
    elif src_rescues:
        case = "control_partially_supported_review"
    else:
        case = "C4_or_mixed_review"
    return {"seeds": seeds, "targets": targets, "levels": levels, "selectors": per_selector,
            "source_audit_oracle_k2": src, "target_oracle_bacc_k2": tob, "target_oracle_joint_k2": toj,
            "target_oracle_rescues_bacc": tgt_rescues_bacc, "target_oracle_rescues_joint": tgt_rescues_joint,
            "source_oracle_rescues": src_rescues, "case_label": case,
            "interpretation": {
                "C1_source_observability_failure": "trajectory HAS target-good checkpoints; source signal cannot observe them",
                "C2_trajectory_failure": "even a target oracle cannot rescue -> OACI never enters a target-good state",
                "C3_calibration_not_discrimination": "target oracle recovers accuracy but not calibration",
                "C4_or_mixed_review": "heterogeneous / needs per-subject inspection",
                "control_partially_supported_review": "source-audit oracle itself rescued K2 — inspect (unexpected for OACI)",
            }.get(case, "review")}
