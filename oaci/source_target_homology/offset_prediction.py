"""C28 Q3 — can the SOURCE factor predict the TARGET offset? The decisive test: a strict source-only class-
conditioned confidence gauge, LOTO, permutation-baselined, no feature selection. If it fails (as C23), source-
unobservability is confirmed at the logit-factor level; if it recovers, the C23 source registry missed this
factor (H6). The target-carrier gauge (C27's +0.524) is reported as the reference ceiling."""
from __future__ import annotations

from . import artifact_loader, factor_registry, schema


def offset_prediction(cands, score_rows, mode, raw, oracle) -> dict:
    out = {}
    # target-carrier reference (C27's class-conditioned confidence recovery)
    out["target_carrier_reference"] = artifact_loader.recover(cands, score_rows, mode, raw, oracle, artifact_loader.tgt_carrier())
    for role in schema.SOURCE_ROLES:
        out[f"source_carrier__{role}"] = artifact_loader.recover(
            cands, score_rows, mode, raw, oracle, artifact_loader.src_carrier(role))
        out[f"source_occupancy__{role}"] = artifact_loader.recover(
            cands, score_rows, mode, raw, oracle, artifact_loader.src_family(role, "occupancy"))
        out[f"source_global_confidence__{role}"] = artifact_loader.recover(
            cands, score_rows, mode, raw, oracle, artifact_loader.src_family(role, "global_confidence"))
    src_keys = [f"source_carrier__{role}" for role in schema.SOURCE_ROLES]
    source_predicts = any(out[k]["gap_closed"] is not None and out[k]["gap_closed"] >= schema.SOURCE_PREDICTS_OFFSET_GAP
                          and out[k]["survives_permutation"] for k in src_keys)
    best = max(src_keys, key=lambda k: (out[k]["gap_closed"] if out[k]["gap_closed"] is not None else -9))
    return {"per_gauge": out, "source_predicts_offset": bool(source_predicts), "best_source_gauge": best,
            "best_source_gap": out[best]["gap_closed"], "target_carrier_gap": out["target_carrier_reference"]["gap_closed"],
            "note": ("a strict source-only class-conditioned confidence gauge RECOVERS the target offset (LOTO, "
                     "survives permutation) -> the C23 source registry missed this factor" if source_predicts else
                     "the source-only class-conditioned confidence gauge does NOT recover the target offset -> "
                     "source-unobservability confirmed at the logit-factor level")}
