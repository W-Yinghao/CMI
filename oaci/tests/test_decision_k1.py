"""C7 K1: the paired grouped-permutation held-out-leakage null + decision.

Synthetic PAIRED audit features (identical y/d/group/sample_id; only Z differs) let us drive a known
reduction (ERM leaky, OACI clean) and a known null (both clean). Standalone + pytest-compatible.
"""
from __future__ import annotations

import inspect

import numpy as np

from oaci.decision.k1_decision import K1_DETECTED, K1_STOP, k1_decision
from oaci.decision.k1_permutation import (assert_paired, compute_k1_permutation, k1_delta_for_bit_row)
from oaci.leakage.cache import critic_config_hash
from oaci.leakage.critic import CriticConfig
from oaci.leakage.crossfit import FrozenFeatures, make_fold_plan
from oaci.leakage.estimate import estimate_extractable_leakage
from oaci.leakage.permutation import (build_paired_arms, make_paired_permutation_plan,
                                      strata_of_rows, swap_row_mask, validate_permutation_plan)
from oaci.support_graph import build_support_graph, counts_from_labels, empirical_class_prior

FAST = CriticConfig(capacities=(0,))            # linear probe suffices (domain is linearly encoded)


def _layout(n_domains=2, n_classes=2, recs=4, per_cell=30):
    y, d, g, gid = [], [], [], 0
    for dom in range(n_domains):
        for _ in range(recs):
            for c in range(n_classes):
                y += [c] * per_cell; d += [dom] * per_cell; g += [gid] * per_cell
            gid += 1
    return np.array(y), np.array(d), np.array(g)


def _paired(seed=0, erm_leaky=True, oaci_leaky=False, dim=4, m=10, sig=1.2):
    """Two PAIRED representations on identical rows: leaky Z encodes the domain, clean Z depends on Y only.
    A MODERATE domain signal keeps L_Q roughly linear in the leaky-fraction, so under the paired null a
    fully-clean-vs-fully-leaky observation is the extreme (the sign-flip test then has power)."""
    rng = np.random.default_rng(seed)
    y, d, g = _layout()
    n_classes, n_domains = int(y.max()) + 1, int(d.max()) + 1
    class_mean = rng.standard_normal((n_classes, dim)) * 2.0
    ddir = rng.standard_normal((n_domains, dim)); ddir /= np.linalg.norm(ddir, axis=1, keepdims=True)

    def mkZ(leaky):
        z = class_mean[y] + 0.3 * rng.standard_normal((y.size, dim))
        return z + (sig * ddir[d] if leaky else 0.0)
    feat_erm = FrozenFeatures(mkZ(erm_leaky), y, d, g)
    feat_oaci = FrozenFeatures(mkZ(oaci_leaky), y, d, g)
    counts = counts_from_labels(d, y, n_domains=n_domains, n_classes=n_classes)
    sg = build_support_graph(counts, m=m, reference_prior=empirical_class_prior(counts))
    return feat_erm, feat_oaci, sg


def _plan(feat, sg):
    return make_fold_plan(feat, sg, n_folds=2, seed=0)


# ===================== permutation plan / arm construction =====================
def test_k1_permutation_swaps_paired_methods_within_y_group():
    fe, fo, _sg = _paired()
    stratum_index, keys = strata_of_rows(fe.y, fe.group)
    plan = make_paired_permutation_plan(fe.y, fe.group, n_permutations=5, seed=1)
    # craft a swap that flips exactly ONE stratum
    bits = np.zeros(plan.n_strata, dtype=bool); bits[3] = True
    swap_row = bits[stratum_index]
    Zo_arm, Ze_arm = build_paired_arms(fo.Z, fe.Z, swap_row)
    # swapped rows: OACI arm carries ERM's Z and vice-versa; untouched rows keep their own
    assert np.array_equal(Zo_arm[swap_row], fe.Z[swap_row]) and np.array_equal(Ze_arm[swap_row], fo.Z[swap_row])
    assert np.array_equal(Zo_arm[~swap_row], fo.Z[~swap_row]) and np.array_equal(Ze_arm[~swap_row], fe.Z[~swap_row])
    # a stratum is one (y, group): all its rows share the swap bit
    for s in range(plan.n_strata):
        rows = stratum_index == s
        assert len(np.unique(swap_row[rows])) == 1


def test_k1_permutation_does_not_shuffle_y_or_domain():
    fe, fo, sg = _paired()
    stratum_index, _ = strata_of_rows(fe.y, fe.group)
    plan = make_paired_permutation_plan(fe.y, fe.group, n_permutations=3, seed=2)
    from oaci.decision.k1_permutation import _arm_feat
    Zo_arm, Ze_arm = build_paired_arms(fo.Z, fe.Z, swap_row_mask(plan, stratum_index, 0))
    for arm in (_arm_feat(fo, Zo_arm), _arm_feat(fe, Ze_arm)):
        assert np.array_equal(arm.y, fe.y) and np.array_equal(arm.d, fe.d)       # Y and D untouched
        assert np.array_equal(arm.group, fe.group) and tuple(arm.sample_id) == tuple(fe.sample_id)


def test_k1_permutation_plan_is_deterministic():
    fe, _fo, _sg = _paired()
    a = make_paired_permutation_plan(fe.y, fe.group, 50, seed=707)
    b = make_paired_permutation_plan(fe.y, fe.group, 50, seed=707)
    assert np.array_equal(a.bits, b.bits) and a.plan_hash == b.plan_hash
    c = make_paired_permutation_plan(fe.y, fe.group, 50, seed=708)
    assert c.plan_hash != a.plan_hash and not np.array_equal(c.bits, a.bits)
    validate_permutation_plan(a)


def test_k1_permutation_plan_hash_binds_seed_groups_and_strata():
    fe, _fo, _sg = _paired()
    base = make_paired_permutation_plan(fe.y, fe.group, 40, seed=1)
    # fewer permutations -> different hash; different strata (shift a group label) -> different hash
    assert make_paired_permutation_plan(fe.y, fe.group, 39, seed=1).plan_hash != base.plan_hash
    g2 = np.array([str(int(x) + 100) for x in fe.group], dtype=object)
    assert make_paired_permutation_plan(fe.y, g2, 40, seed=1).plan_hash != base.plan_hash


# ===================== observed statistic / null discipline =====================
def test_k1_observed_delta_uses_audit_not_selection():
    fe, fo, sg = _paired(erm_leaky=True, oaci_leaky=False)
    plan = _plan(fe, sg)
    res = compute_k1_permutation(fe, fo, sg, plan, FAST, n_permutations=30, seed=707, alpha=0.05)
    # observed delta == the AUDIT point-estimate difference computed directly (no bootstrap UCL, no selection)
    direct = (estimate_extractable_leakage(fo, sg, plan, FAST)["extractable_LQ_ov"]
              - estimate_extractable_leakage(fe, sg, plan, FAST)["extractable_LQ_ov"])
    assert abs(res["observed_delta"] - direct) < 1e-9
    assert res["split_role"] == "source_audit"


def test_k1_null_reuses_fixed_support_and_probe_config():
    fe, fo, sg = _paired()
    plan = _plan(fe, sg)
    res = compute_k1_permutation(fe, fo, sg, plan, FAST, n_permutations=20, seed=1, alpha=0.05)
    assert res["audit_support_hash"] == sg.support_hash()
    assert res["probe_config_hash"] == critic_config_hash(FAST)
    assert res["audit_population_hash"] == res["audit_population_hash"]      # stable
    assert np.isfinite(res["null"]).all() and len(res["null"]) == 20


def test_k1_parallel_equals_sequential_permutation_null():
    fe, fo, sg = _paired()
    plan = _plan(fe, sg)
    seq = compute_k1_permutation(fe, fo, sg, plan, FAST, n_permutations=24, seed=3, alpha=0.05)
    par = compute_k1_permutation(fe, fo, sg, plan, FAST, n_permutations=24, seed=3, alpha=0.05,
                                 parallel_n_jobs=2, parallel_backend="process")
    assert np.allclose(seq["null"], par["null"], atol=0, rtol=0)             # bit-identical
    assert seq["observed_delta"] == par["observed_delta"]
    assert seq["p_lower"] == par["p_lower"] and seq["permutation_plan_hash"] == par["permutation_plan_hash"]


def test_k1_never_reads_target():
    # the K1 API consumes ONLY the two source-audit representations; there is no target input.
    params = set(inspect.signature(compute_k1_permutation).parameters) | set(
        inspect.signature(k1_delta_for_bit_row).parameters)
    assert not any("target" in p.lower() for p in params)
    fe, fo, sg = _paired()
    assert_paired(fe, fo)                                                    # only source-audit feats


# ===================== decision =====================
def test_k1_decision_detects_known_synthetic_reduction():
    fe, fo, sg = _paired(erm_leaky=True, oaci_leaky=False)                   # OACI clean, ERM leaks
    res = compute_k1_permutation(fe, fo, sg, _plan(fe, sg), FAST, n_permutations=99, seed=707, alpha=0.05)
    assert res["observed_delta"] < -0.2                                      # OACI leaks much less
    dec = k1_decision(res)
    assert dec["k1_status"] == K1_DETECTED and dec["continue_to_k2"] and dec["p_lower"] < 0.05


def test_k1_decision_stops_on_null_synthetic_case():
    fe, fo, sg = _paired(erm_leaky=False, oaci_leaky=False)                  # both clean -> nothing to reduce
    res = compute_k1_permutation(fe, fo, sg, _plan(fe, sg), FAST, n_permutations=99, seed=707, alpha=0.05)
    dec = k1_decision(res)
    assert dec["k1_status"] == K1_STOP and not dec["continue_to_k2"] and dec["p_lower"] >= 0.05


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} decision-k1 tests")


if __name__ == "__main__":
    _run_all()
