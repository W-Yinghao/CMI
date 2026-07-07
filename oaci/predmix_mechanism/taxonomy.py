"""C26 — deterministic predicted-class-mix taxonomy P1-P7. Stage-1 (read-only Q2/Q3/Q4) resolves among the
signal-nature cases P2 (class-specific) / P3 (symmetric collapse) / P4 (identity fingerprint) / P5 (confidence
interaction). P1 (decision-occupancy), P6 (sample noise) and P7 (label boundary) require the Q1 split-stability
+ Q5 label diagnostics from the re-persistence re-inference; until then they are UNRESOLVED and C26 is NOT
finalized."""
from __future__ import annotations

from . import schema


def gauge_taxonomy(signed_sym, rotation, identity, interaction, split, label) -> dict:
    fingerprint = bool(identity.get("identity_fingerprint_dominant"))
    # signed-vs-symmetric is the decisive P2/P3 test (does the class DIRECTION matter, or only concentration?);
    # the per-target scramble is reported as supporting evidence but not gated (ridge symmetry + rank-AUC make it
    # a weak instrument for a strong signal).
    signed_specific = bool(signed_sym.get("signed_specific"))
    symmetric = bool(signed_sym.get("symmetric_carries") and not signed_sym.get("signed_specific"))
    scramble_supports = bool(rotation.get("class_index_alignment_matters"))
    scaffold = bool(interaction.get("predmix_needs_confidence_scaffold") or interaction.get("interaction_dominant"))

    established = []
    if fingerprint:
        established.append(schema.P4)
    if signed_specific:
        established.append(schema.P2)
    if symmetric:
        established.append(schema.P3)
    if scaffold:
        established.append(schema.P5)

    # Stage-2 resolution
    split_ok = split.get("status") == schema.STATUS_OK
    label_ok = label.get("status") == schema.STATUS_OK
    final = bool(split_ok and label_ok)
    unresolved = [] if final else [schema.P1, schema.P6, schema.P7]
    if split_ok:
        if split.get("split_stable"):
            if label_ok and label.get("tracks_target_error_geometry"):
                established.append(schema.P1)
            if label_ok and label.get("tracks_target_error_geometry"):
                established.append(schema.P7)
        else:
            established.append(schema.P6)

    primary = (schema.P4 if fingerprint else
               (schema.P2 if signed_specific else
                (schema.P3 if symmetric else (schema.P5 if scaffold else None))))
    if final and split.get("split_stable") and label.get("tracks_target_error_geometry") and not fingerprint:
        primary = schema.P1                                  # decision-occupancy is the finalized headline
    interp = {
        schema.P1: "predicted-class mix is a split-stable target DECISION-OCCUPANCY signal that tracks the frozen model's target class-error geometry.",
        schema.P2: "the SIGNED class vector carries the signal (which class is over-predicted matters); symmetric concentration alone does not -> class-index-specific decision occupancy.",
        schema.P3: "only symmetric concentration (entropy/max-mass/distance-to-uniform) matters -> collapse/confidence-concentration signal, class-index-invariant.",
        schema.P4: "predicted-class mix IN ISOLATION is a target-identity fingerprint (NN same-target rate high; its +0.003 standalone recovery FAILS the permutation control). It carries no standalone target-marginal offset signal -- C25's Shapley credit was SYNERGY allocation, not a main effect. (The FULL R3 interaction is still permutation-robust per C24; it is the isolated family that is a fingerprint.)",
        schema.P5: "the R3 recovery is a SYNERGISTIC interaction between predicted-class mix and confidence/margin geometry (both families ~0/negative alone; synergy positive) -- predicted-class mix contributes ONLY through the scaffold, not as a standalone signal.",
        schema.P6: "predicted-class mix is NOT split-stable -> finite-sample artifact.",
        schema.P7: "labels show the mix corresponds to target error geometry (diagnostic-only, non-deployable).",
        None: "no read-only case resolved.",
    }[primary]
    return {"primary_case": primary, "established": established, "final": final,
            "unresolved_pending_reinference": unresolved,
            "signed_specific": signed_specific, "symmetric_carries": symmetric, "scramble_supports_signed": scramble_supports,
            "identity_fingerprint_dominant": fingerprint, "predmix_needs_scaffold": scaffold,
            "split_status": split.get("status"), "label_status": label.get("status"),
            "interpretation": interp, "diagnostic_only_non_deployable": True,
            "identity_entanglement_disclosed": True}
