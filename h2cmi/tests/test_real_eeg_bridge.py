"""Bridge-semantics test for the Project B real-EEG adapter (Step-3A).

Uses a FAKE metadata DataFrame only — it does NOT require MOABB data (the loader wrapper is not
exercised here). Verifies the subject->session DAG, contiguous DomainLabels, label-safe LOSO
splits, and evaluation-unit levels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from h2cmi.data.real_eeg_bridge import (
    make_subject_session_dag, make_source_domain_labels, split_loso_by_subject,
    target_domain_levels, loso_subjects, source_pseudo_levels_from_domains,
)


def _fake_meta():
    rows = []
    for s in (1, 2, 3):
        for sess in ("0test", "1train"):
            for run in ("run0", "run1"):
                rows.append(dict(subject=s, session=sess, run=run))
    return pd.DataFrame(rows)               # 3 subjects x 2 sessions x 2 runs = 12 trials


def run():
    meta = _fake_meta()

    # DAG names + contiguous levels
    dag, labels, info = make_subject_session_dag(meta)
    assert dag.names == ["subject", "session"], dag.names
    assert labels.levels.shape == (len(meta), 2)
    for f in dag.names:
        col = labels.factor(f)
        assert set(col.tolist()) == set(range(int(col.max()) + 1)), (f, col.tolist())
    assert info["n_subjects"] == 3 and info["n_sessions"] == 6, info

    # LOSO split is disjoint, covers all, and holds out exactly the target subject
    assert loso_subjects(meta) == [1, 2, 3]
    src, tgt = split_loso_by_subject(meta, 2)
    assert len(np.intersect1d(src, tgt)) == 0
    assert len(src) + len(tgt) == len(meta)
    assert set(meta.loc[tgt, "subject"].tolist()) == {2}
    assert set(meta.loc[src, "subject"].tolist()) == {1, 3}

    # evaluation-unit levels for the held-out target (1 subject, 2 sessions, 4 runs)
    mt = meta.loc[tgt].reset_index(drop=True)
    assert int(target_domain_levels(mt, eval_unit="subject").max()) + 1 == 1
    assert int(target_domain_levels(mt, eval_unit="session").max()) + 1 == 2
    assert int(target_domain_levels(mt, eval_unit="run").max()) + 1 == 4
    try:
        target_domain_levels(mt, eval_unit="bogus")
        raise AssertionError("bad eval_unit should raise")
    except ValueError:
        pass

    # source domain labels + pseudo levels
    sdag, slabels, sinfo = make_source_domain_labels(meta.loc[src].reset_index(drop=True))
    assert sdag.names == ["subject", "session"]
    assert sinfo["n_subjects"] == 2 and sinfo["n_sessions"] == 4
    pl = source_pseudo_levels_from_domains(slabels, level="subject")
    assert int(pl.max()) + 1 == sinfo["n_subjects"]

    print("real_eeg_bridge self-test passed:",
          f"dag={sdag.names} n_src_subj={sinfo['n_subjects']} n_src_sess={sinfo['n_sessions']}")


def test_real_eeg_bridge():
    run()


if __name__ == "__main__":
    run()
