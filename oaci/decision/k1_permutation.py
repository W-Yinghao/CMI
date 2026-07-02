"""K1 observed statistic + grouped-permutation null on the HELD-OUT AUDIT split.

Observed ``T = L_Q^ov,audit(OACI) - L_Q^ov,audit(ERM)`` (the audit grouped-max-probe extractable-leakage
POINT estimate; lower is better). The null (``oaci.leakage.permutation``) exchanges the paired ERM/OACI
representations within each ``(Y, recording_group)`` stratum. The support graph, fold plan and probe config
are held FIXED across every permutation (a permutation only re-labels which representation feeds which arm),
so the null isolates the ERM-vs-OACI difference; ``Y``/``D``/support/folds/target are never touched.
"""
from __future__ import annotations

import numpy as np

from ..leakage.cache import critic_config_hash
from ..leakage.crossfit import FrozenFeatures, feat_population_hash
from ..leakage.estimate import estimate_extractable_leakage
from ..leakage.permutation import (build_paired_arms, make_paired_permutation_plan, permutation_p_values,
                                   strata_of_rows)

_STATISTIC = "grouped_max_probe_extractable_LQ_ov_OACI_minus_ERM"


def assert_paired(feat_erm: FrozenFeatures, feat_oaci: FrozenFeatures) -> None:
    """The two representations must be the SAME audit rows (identical id/y/d/group/mass order) so the swap
    is a genuine paired exchange. Only ``Z`` may differ."""
    if feat_erm.Z.shape != feat_oaci.Z.shape:
        raise ValueError("ERM/OACI audit features differ in shape; not paired")
    if tuple(feat_erm.sample_id) != tuple(feat_oaci.sample_id):
        raise ValueError("ERM/OACI audit features have different sample_id order; not paired")
    if feat_population_hash(feat_erm) != feat_population_hash(feat_oaci):
        raise ValueError("ERM/OACI audit features have different populations (y/d/group/mass); not paired")


def _arm_feat(ref: FrozenFeatures, Z_arm: np.ndarray) -> FrozenFeatures:
    """A FrozenFeatures carrying the arm's ``Z`` but the SHARED (identical) labels/ids of the paired rows."""
    return FrozenFeatures(Z=Z_arm, y=ref.y, d=ref.d, group=ref.group, sample_mass=ref.sample_mass,
                          sample_id=ref.sample_id)


def k1_delta_for_bit_row(feat_erm, feat_oaci, stratum_index, support_graph, fold_plan, cfg, bit_row) -> float:
    """One permutation's Δ: swap each stratum's rows per ``bit_row`` (per-stratum bits), then
    ``LQ(OACI arm) - LQ(ERM arm)`` with the FIXED support/fold/probe config. ``bit_row`` all-False = the
    observed statistic."""
    swap_row = np.asarray(bit_row, dtype=bool)[np.asarray(stratum_index, dtype=int)]
    Z_oaci_arm, Z_erm_arm = build_paired_arms(feat_oaci.Z, feat_erm.Z, swap_row)
    lq_oaci = estimate_extractable_leakage(_arm_feat(feat_oaci, Z_oaci_arm), support_graph, fold_plan, cfg)
    lq_erm = estimate_extractable_leakage(_arm_feat(feat_erm, Z_erm_arm), support_graph, fold_plan, cfg)
    return float(lq_oaci["extractable_LQ_ov"] - lq_erm["extractable_LQ_ov"])


def compute_k1_permutation(feat_erm, feat_oaci, support_graph, fold_plan, cfg, *, n_permutations: int,
                           seed: int, alpha: float, parallel_n_jobs: int = 1,
                           parallel_backend: str = "sequential") -> dict:
    """Observed Δ + the permutation null + p-values and the identity hashes. ``parallel_backend='process'``
    with ``parallel_n_jobs>1`` is PURE acceleration — the null values and order are identical to the
    sequential loop (proven in the tests)."""
    assert_paired(feat_erm, feat_oaci)
    stratum_index, _keys = strata_of_rows(feat_erm.y, feat_erm.group)
    plan = make_paired_permutation_plan(feat_erm.y, feat_erm.group, n_permutations, seed)
    zero = np.zeros(plan.n_strata, dtype=bool)
    observed = k1_delta_for_bit_row(feat_erm, feat_oaci, stratum_index, support_graph, fold_plan, cfg, zero)

    bit_rows = [plan.bits[p] for p in range(plan.n_permutations)]
    if parallel_backend == "process" and int(parallel_n_jobs) > 1 and bit_rows:
        from ..leakage.parallel import parallel_paired_permutation_deltas
        null = parallel_paired_permutation_deltas(bit_rows, feat_erm, feat_oaci, stratum_index,
                                                  support_graph, fold_plan, cfg, int(parallel_n_jobs))
    else:
        null = [k1_delta_for_bit_row(feat_erm, feat_oaci, stratum_index, support_graph, fold_plan, cfg, br)
                for br in bit_rows]
    null = np.asarray(null, dtype=np.float64)

    pv = permutation_p_values(observed, null, plan.n_permutations)
    qs = [0.005, 0.025, 0.05, 0.5, 0.95, 0.975, 0.995]
    return {
        "statistic": _STATISTIC,
        "split_role": "source_audit",
        "observed_delta": float(observed),
        "null": null,
        "null_quantiles": {str(q): float(np.quantile(null, q)) for q in qs},
        "p_lower": pv["p_lower"],
        "p_upper": pv["p_upper"],
        "p_two_sided": pv["p_two_sided"],
        "alpha": float(alpha),
        "n_permutations": int(plan.n_permutations),
        "permutation_plan_hash": plan.plan_hash,
        "audit_support_hash": support_graph.support_hash(),
        "audit_population_hash": feat_population_hash(feat_erm),
        "probe_config_hash": critic_config_hash(cfg),
    }
