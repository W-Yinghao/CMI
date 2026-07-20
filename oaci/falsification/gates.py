"""C14 battery gates G0–G3 (integrity, selection optimism, held-out leakage/K1, endpoint transfer/K2). G4
(oracle) and G5 (anti-transfer) live in oracle.py / antitransfer.py. Each gate is a pure function of the
loaded evidence dicts and returns a schema.gate(...) record."""
from __future__ import annotations

from .schema import (G0, G1, G2, G3, INTEGRITY_OK, INVALID_EVIDENCE, K1_DETECTED, K1_STOP, K1_WEAK, K2_GAIN,
                     K2_MIXED, K2_STOP, NO_SELECTION_OPTIMISM, SELECTION_OPTIMISM_PRESENT, gate)


def g0_integrity(c8, c10, c12) -> dict:
    """Deep-verified artifacts, target_fit_ids empty, byte/numeric replay identity, and the SRC cells present.
    Failure => the downstream gates run on untrustworthy evidence."""
    idn = c10["part2_selector_replay"]["identity"]
    ok = (bool(c8.get("all_deep_verified")) and bool(c8.get("all_target_fit_empty"))
          and int(c8.get("n_folds", 0)) == 27 and bool(idn.get("all_pass"))
          and int(idn.get("total_argmax_flips", 1)) == 0 and len(c12.get("cells", [])) >= 1)
    return gate(G0, INTEGRITY_OK if ok else INVALID_EVIDENCE,
                c8_deep_verified=bool(c8.get("all_deep_verified")), c8_target_fit_empty=bool(c8.get("all_target_fit_empty")),
                c8_n_folds=c8.get("n_folds"), replay_identity_all_pass=bool(idn.get("all_pass")),
                replay_argmax_flips=idn.get("total_argmax_flips"), replay_max_logit_diff=idn.get("max_logit_diff"),
                c12_cells=len(c12.get("cells", [])))


def g1_selection_optimism(c10) -> dict:
    """Does the selection-time leakage improvement survive at the held-out audit split?"""
    o = c10["part1_transfer"]["selection_to_audit_optimism"]
    d_sel = o["delta_selection_leakage"]["mean"]
    d_aud = o["delta_audit_leakage"]["mean"]
    corr = o["corr_selection_vs_audit_delta"]["pearson"].get("r")
    n = o["n_fold_levels"]
    ratio = (d_aud / d_sel) if (d_sel not in (None, 0)) else None          # ~1 transfers, ~0 optimism
    optimism_gap = (abs(d_sel) - abs(d_aud)) if (d_sel is not None and d_aud is not None) else None
    frac_sign_preserved = (o["n_audit_reduced"] / n) if n else None
    present = (corr is not None and abs(corr) < 0.2 and d_sel is not None and d_sel < -0.05
               and (d_aud is None or abs(d_aud) < abs(d_sel) * 0.25))
    return gate(G1, SELECTION_OPTIMISM_PRESENT if present else NO_SELECTION_OPTIMISM,
                delta_selection_leakage_mean=d_sel, delta_audit_leakage_mean=d_aud,
                selection_to_audit_transfer_ratio=ratio, optimism_gap=optimism_gap,
                fraction_sign_preserved=frac_sign_preserved, corr_selection_vs_audit=corr,
                n_selection_reduced=o["n_selection_reduced"], n_audit_reduced=o["n_audit_reduced"], n_fold_levels=n)


def g2_heldout_leakage(c8) -> dict:
    """K1: did held-out audit leakage reduction survive multiplicity? Weak nominal signal is NOT a success."""
    ko = c8["k1_overall"]; m = ko["multiplicity"]
    n_nom = ko["n_leakage_reduction_detected"]; n_bh = m["n_bh_survive"]; n_bonf = m["n_bonferroni_survive"]
    sweep = ko["k1_sweep_status"]
    weak = (n_nom > 0 and n_bh == 0 and sweep == K1_STOP)
    status = K1_WEAK if weak else (K1_DETECTED if n_bh >= 1 else K1_STOP)
    return gate(G2, status, sweep_status=sweep, n_nominal_detected=n_nom, n_tests=ko["n_tests"],
                n_bh_survivors=n_bh, n_bonferroni_survivors=n_bonf, weak_nominal_signal=weak,
                observed_delta_mean=ko.get("observed_delta_mean"))


def g3_endpoint_transfer(c8) -> dict:
    """K2: did any nominal leakage reduction convert to reproducible worst-domain endpoint gain?"""
    k2 = c8["k2"]; st = k2["k2_status"]
    status = {"reproducible_gain": K2_GAIN, "stop_no_reproducible_gain": K2_STOP}.get(st, K2_MIXED)
    agg = c8.get("k2_agg", {})
    return gate(G3, status, k2_status=st, reproduced_endpoints=k2.get("reproduced_endpoints"),
                worst_domain_bacc=agg.get("worst_domain_bacc"), worst_domain_nll=agg.get("worst_domain_nll"))
