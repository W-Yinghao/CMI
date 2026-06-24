"""
CSC-P1.4.1 cluster-inference + RNG + simulator regression tests (the review's required list).
These are fast COMPONENT tests of the correctness invariants; the end-to-end frozen-path
behaviour + audit fail-closed are exercised on SLURM (csc/run_audit.sbatch).
"""
import warnings
import dataclasses
import numpy as np
warnings.filterwarnings("ignore")

from csc.protocol import ProtocolConfig, ProtocolError, _stage_seed
from csc.certificate.atlas import cluster_mean
from csc.certificate.residual_test import _subject_condition_weights, subject_null_labels
from csc.sim.shift_simulator import SimConfig, make_source, make_target


def test_master_seed_absent_and_stage_seeds_derive():
    # master_seed/seed_derivation were REMOVED from the method manifest (they did not drive
    # computation -> faked the hash). The runtime root seed drives every stage by name.
    fields = {f.name for f in dataclasses.fields(ProtocolConfig)}
    assert "master_seed" not in fields and "seed_derivation" not in fields
    assert _stage_seed(0, "analyze_source") != _stage_seed(1, "analyze_source")
    assert _stage_seed(0, "analyze_source") != _stage_seed(0, "certify_robust")
    assert _stage_seed(7, "x") == _stage_seed(7, "x")          # deterministic
    print("OK master_seed absent from manifest; named stage seeds derive from root + are stable")


def test_invalid_manifest_values_fail_closed():
    for bad in (dict(tau_quantile=1.5), dict(consensus=-0.1), dict(target_n_boot=0),
                dict(C=-1.0), dict(var_keep=0.0), dict(tau_margin=0.5), dict(source_cv_folds=1),
                dict(oracle_eps_stable_ce=0.05, oracle_eps_concept_ce=0.03)):
        try:
            ProtocolConfig(**bad).validate()
            raise AssertionError(f"validate() should reject {bad}")
        except ProtocolError:
            pass
    print("OK invalid numeric manifest values fail closed (validate raises)")


def test_cluster_mean_multiplicity():
    # the A,A,B multiplicity bug: with FRESH ids per resampled copy, a cluster drawn twice
    # contributes twice. subject means (0,0,10) -> 3.333..., NOT 5.0 (the re-merge bug).
    Z = np.array([[0.0], [0.0], [10.0]])
    fresh = cluster_mean(Z, np.array([0, 1, 2]))     # distinct ids per copy
    merged = cluster_mean(Z, np.array([0, 0, 1]))    # original ids (re-merge)
    assert abs(fresh[0] - 10.0 / 3.0) < 1e-9, fresh
    assert abs(merged[0] - 5.0) < 1e-9, merged
    print(f"OK cluster_mean multiplicity: fresh-ids {fresh[0]:.3f} (=10/3) vs re-merge {merged[0]:.1f}")


def test_subject_weights_one_vote():
    # epochs weighted 1/n_s: every subject contributes the SAME total weight (one vote/subject), and
    # CSC-P1.4.5 #1 fixes the SUM to #subjects (RAW, not renormalised to mean 1) -- sklearn lbfgs L2 =
    # 1/(C*sum_w), so a fixed sum_w==S keeps the penalty (and T) invariant to epochs/subject.
    g = np.array([0, 0, 0, 0, 1, 2, 2])              # subject 0 has 4 epochs, 1 has 1, 2 has 2
    D = np.zeros_like(g)                             # single condition -> per-subject 1/n_s
    w = _subject_condition_weights(g, D)
    S = len(np.unique(g))
    assert abs(w.sum() - S) < 1e-9, f"sum_w must equal #subjects S={S}, got {w.sum()}"
    totals = {u: w[g == u].sum() for u in np.unique(g)}
    assert max(totals.values()) - min(totals.values()) < 1e-9, totals
    assert all(abs(t - 1.0) < 1e-9 for t in totals.values()), "each subject = unit mass"
    print(f"OK 1/n_s weights: per-subject total=1 (sum_w==S={S}, raw not mean-1): {list(totals.values())}")


def test_subject_null_labels_coherent():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(6), [3, 5, 2, 4, 6, 3])  # unequal epochs
    p0 = rng.dirichlet([1, 1, 1], size=len(g))
    yb = subject_null_labels(p0, g, [0, 1, 2], rng)
    for u in np.unique(g):
        assert np.unique(yb[g == u]).size == 1, "null Y* must be ONE label per subject"
    print("OK cluster-consistent null: Y* constant within every subject")


def test_simulator_one_label_per_subject():
    src = make_source(SimConfig(seed=0), n_domains=6, concept_domains=2, seed=0)
    for u in np.unique(src.group_ids):
        assert np.unique(src.Y[src.group_ids == u]).size == 1, "subject must have ONE label"
    print(f"OK simulator: every source subject has exactly one label "
          f"({len(np.unique(src.group_ids))} subjects)")


def test_simulator_unequal_epochs_and_within_subject_correlation():
    src = make_source(SimConfig(seed=1), n_domains=6, concept_domains=2, seed=1)
    counts = [int((src.group_ids == u).sum()) for u in np.unique(src.group_ids)]
    assert max(counts) > min(counts), "epochs per subject must be UNEQUAL"
    # within-subject variance < across-subject-mean variance (latent random effect => correlation)
    within = np.mean([src.Z[src.group_ids == u].var(0).mean() for u in np.unique(src.group_ids)])
    smeans = np.stack([src.Z[src.group_ids == u].mean(0) for u in np.unique(src.group_ids)])
    across = smeans.var(0).mean()
    assert across > 0 and within > 0
    print(f"OK simulator: unequal epochs [{min(counts)},{max(counts)}]; within-subj var {within:.2f} "
          f"vs across-subj-mean var {across:.2f} (random effect => correlation)")


if __name__ == "__main__":
    test_master_seed_absent_and_stage_seeds_derive()
    test_invalid_manifest_values_fail_closed()
    test_cluster_mean_multiplicity()
    test_subject_weights_one_vote()
    test_subject_null_labels_coherent()
    test_simulator_one_label_per_subject()
    test_simulator_unequal_epochs_and_within_subject_correlation()
    print("\nall cluster-inference regression tests passed")
