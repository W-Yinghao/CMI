"""C17-A — checkpoint-level source-only signal atlas over OACI trajectories, joined with C16 target-oracle
labels (diagnostic-only). Every source column is prefixed `src__`; every target/diagnostic column `tgt__` and
carries diagnostic_only_non_deployable=True. Reads the committed C10 replay per-fold candidate tables.

Per-candidate confidence-geometry (entropy/margin/logit-norm/conf-on-wrong) is NOT committed (needs logits;
only selected checkpoints have them), so the tested source observable family here is the worst-domain source
endpoints + leakage + risk/training signals. That scope is stated in the report.
"""
from __future__ import annotations

import glob
import json
import os

# source-only observables available per candidate (the tested measurement family)
SOURCE_SIGNALS = ("source_guard_worst_bacc", "source_guard_worst_nll", "source_guard_worst_ece",
                  "source_audit_worst_bacc", "source_audit_worst_nll", "source_audit_worst_ece",
                  "selection_leakage_point", "audit_leakage_point", "R_src", "balanced_err",
                  "train_surrogate", "epoch")
# which axis each source signal lives on (for C17-D)
SIGNAL_AXIS = {"source_guard_worst_bacc": "accuracy", "source_audit_worst_bacc": "accuracy",
               "source_guard_worst_nll": "calibration", "source_audit_worst_nll": "calibration",
               "source_guard_worst_ece": "calibration", "source_audit_worst_ece": "calibration",
               "selection_leakage_point": "leakage", "audit_leakage_point": "leakage",
               "R_src": "risk", "balanced_err": "accuracy", "train_surrogate": "objective", "epoch": "meta"}
_MARGIN = 1e-9


def load_replay(replay_dir) -> list:
    folds = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(replay_dir, "seed-*-target-*.json")))]
    if not folds:
        raise ValueError(f"no C10 replay fold JSONs in {replay_dir}")
    return folds


def build_atlas(folds) -> list:
    """One row per feasible OACI candidate: src__ signals + tgt__ diagnostic labels/deltas. ERM is the
    per-fold-level reference (excluded from the candidate rows)."""
    rows = []
    for f in folds:
        for L, lv in f["levels"].items():
            cands = lv["candidates"]
            erm = next(c for c in cands if c.get("is_erm"))
            oaci = [c for c in cands if not c.get("is_erm") and c.get("feasible")]
            ranked = sorted(oaci, key=lambda c: (-(c["target_worst_bacc"] if c["target_worst_bacc"] is not None else -1e9),
                                                 c["model_hash"]))
            rank = {c["model_hash"]: i + 1 for i, c in enumerate(ranked)}
            for c in oaci:
                r = {"seed": f["seed"], "target": f["target"], "level": int(L), "model_hash": c["model_hash"],
                     "diagnostic_only_non_deployable": True}
                for s in SOURCE_SIGNALS:
                    r["src__" + s] = c.get(s)
                dbacc = (None if (c["target_worst_bacc"] is None or erm["target_worst_bacc"] is None)
                         else c["target_worst_bacc"] - erm["target_worst_bacc"])
                dnll = (None if (c["target_worst_nll"] is None or erm["target_worst_nll"] is None)
                        else c["target_worst_nll"] - erm["target_worst_nll"])
                r["tgt__target_bacc_delta"] = dbacc
                r["tgt__target_nll_delta"] = dnll
                r["tgt__target_bacc_good"] = bool(dbacc is not None and dbacc > _MARGIN)
                r["tgt__target_joint_good"] = bool(dbacc is not None and dbacc > _MARGIN and dnll is not None and dnll < -_MARGIN)
                r["tgt__target_oracle_rank"] = rank[c["model_hash"]]
                r["tgt__n_candidates"] = len(oaci)
                rows.append(r)
    return rows


def source_columns(rows) -> list:
    return [f"src__{s}" for s in SOURCE_SIGNALS]


def target_columns(rows) -> list:
    return [k for k in (rows[0].keys() if rows else []) if k.startswith("tgt__")]
