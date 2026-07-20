"""P0-1: a source subset must train on a SOURCE-ONLY DAG with contiguous levels."""
from __future__ import annotations

import numpy as np

from h2cmi.data.eeg_simulator import EEGSimulator, ShiftSpec, train_target_split
from h2cmi.domains import compact_domain_labels


def test_compact_removes_target_levels_and_is_contiguous():
    sim = EEGSimulator(2, 8, 64, shift=ShiftSpec(cov=1.0), seed=0).sample(5, 3, 2, 16)
    src_idx, tgt_idx = train_target_split(sim, 1, seed=0)
    raw = sim.domains.subset(src_idx)
    dag, compact, level_maps = compact_domain_labels(raw)

    for j, f in enumerate(dag.factors):
        col = compact.levels[:, j]
        # contiguous 0..K-1 and n_levels == observed cardinality
        assert col.min() == 0 and col.max() == len(np.unique(col)) - 1
        assert f.n_levels == len(np.unique(col))
        # the compact cardinality is strictly smaller than the full DAG for every factor
        assert f.n_levels <= sim.dag.factors[j].n_levels

    # the held-out target site's levels are absent from the source-only DAG
    full_site = sim.domains.factor("site")
    tgt_sites = set(np.unique(full_site[tgt_idx]).tolist())
    src_sites = set(level_maps["site"].tolist())
    assert tgt_sites.isdisjoint(src_sites), "target site leaked into source DAG"
    # the site factor shrank by exactly the number of target sites
    assert dag.get("site").n_levels == sim.dag.get("site").n_levels - len(tgt_sites)


def test_critic_output_matches_source_cardinality():
    from h2cmi.cmi.hierarchical import HierarchicalCMI
    from h2cmi.config import CMIConfig
    sim = EEGSimulator(2, 8, 64, shift=ShiftSpec(cov=1.0), seed=1).sample(4, 2, 2, 16)
    src_idx, _ = train_target_split(sim, 1, seed=1)
    dag, compact, _ = compact_domain_labels(sim.domains.subset(src_idx))
    ys = sim.y[src_idx]
    hcmi = HierarchicalCMI(16, sim.n_classes, dag, compact, ys, CMIConfig())
    for f in dag.penalised_factors():
        crit = hcmi.critics[f.name]
        assert crit.net[-1].out_features == f.n_levels, \
            f"critic for {f.name} outputs {crit.net[-1].out_features} != source {f.n_levels}"


if __name__ == "__main__":
    test_compact_removes_target_levels_and_is_contiguous()
    test_critic_output_matches_source_cardinality()
    print("test_source_dag_remap PASSED")
