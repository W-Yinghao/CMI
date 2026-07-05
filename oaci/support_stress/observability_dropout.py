"""C18-D — observability-dropout SECONDARY appendix (NOT the headline). This is the no-GPU proxy: it does
NOT recompute source signals under a masked source distribution (that is C18-P). Instead it models support
degradation as ESTIMABILITY-DRIVEN dropout of the source columns that WOULD become non-estimable (leakage
first, then held-out-audit endpoints), sets them to NaN so the finite-filter drops them, and re-runs the
C17 univariate + LOTO probe on the surviving observability set.

It answers only: "if support degradation simply makes certain source observables non-estimable, does the
LOTO signal disappear?" — NOT "what happens when the source distribution is actually masked?". It exists to
be COMPARED against the genuine C18-P replay, and must never be reported as the primary H1/H2 result.
"""
from __future__ import annotations

from ..identifiability.multivariate_probe import multivariate_probe
from ..identifiability.univariate import univariate_identifiability
from . import schema

LABEL = "secondary_observability_proxy_not_source_distribution_recompute"

# monotone estimability-dropout schedule: which source columns become non-estimable as support degrades.
# Leakage needs comparable classes -> drops first; audit endpoints need held-out audit cells -> drop next.
_DROP_SCHEDULE = {
    "S0_full_support": (),
    "S1_label_marginal_skew": (),
    "S2_rare_cells": ("selection_leakage_point",),
    "S3_nonestimable_cells": ("selection_leakage_point", "audit_leakage_point"),
    "S4_missing_cells": ("selection_leakage_point", "audit_leakage_point", "source_audit_worst_bacc",
                         "source_audit_worst_nll", "source_audit_worst_ece"),
    "S5_block_class_by_domain": ("selection_leakage_point", "audit_leakage_point"),
    "S6_boundary_aligned_mask": ("selection_leakage_point", "audit_leakage_point", "source_audit_worst_bacc"),
    "S7_random_matched_mask": ("selection_leakage_point", "audit_leakage_point", "source_audit_worst_bacc"),
}


def _drop_columns(rows, cols):
    dropped = set("src__" + c for c in cols)
    out = []
    for r in rows:
        rr = dict(r)
        for c in dropped:
            rr[c] = float("nan")            # finite-filter downstream drops NaN -> column effectively removed
        out.append(rr)
    return out


def observability_dropout_stress(atlas_rows, *, n_perm=None) -> dict:
    """Per regime, drop the WOULD-BE-non-estimable columns and re-run univariate + LOTO on the survivors."""
    per_regime = {}
    for regime in schema.REGIME_ORDER:
        cols = _DROP_SCHEDULE[regime]
        rows = _drop_columns(atlas_rows, cols) if cols else atlas_rows
        u = univariate_identifiability(rows)
        m = multivariate_probe(rows) if n_perm is None else multivariate_probe(rows, n_perm=n_perm)
        per_regime[regime] = {"dropped_columns": list(cols), "n_dropped": len(cols),
                              "univariate_verdict": u["univariate_verdict"],
                              "loto_auc": m["loto_auc"], "permutation_p": m["permutation_p"],
                              "beats_permutation": m["beats_permutation"]}
    return {"label": LABEL, "is_primary": False, "per_regime": per_regime,
            "note": ("SECONDARY proxy: drops columns that would become non-estimable under support "
                     "degradation; does NOT recompute surviving columns under the masked source "
                     "distribution. Compare against the genuine C18-P replay, never substitute for it.")}
