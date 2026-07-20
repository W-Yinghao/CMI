"""P0-7: grouped leakage split must (a) detect subject-generalising site leakage and
(b) NOT raise a false positive when the apparent site info is pure subject memorisation."""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from h2cmi.data.eeg_simulator import EEGSimulator, ShiftSpec
from h2cmi.domains import compact_domain_labels
from h2cmi.eval.leakage import crossfit_conditional_leakage


def _sim():
    sim = EEGSimulator(3, 8, 64, shift=ShiftSpec(cov=1.0, prior=0.3), seed=0).sample(4, 3, 4, 30)
    dag, dom, _ = compact_domain_labels(sim.domains)
    return sim, dag, dom


def test_site_shared_signal_is_detected():
    sim, dag, dom = _sim()
    rng = np.random.default_rng(0)
    site = dom.factor("site")
    Z = np.zeros((sim.n, 12), dtype=np.float32)
    Z[np.arange(sim.n), sim.y] = 2.0
    Z[np.arange(sim.n), 3 + site] += 2.0                     # SITE-shared -> subject-generalising
    Z += 0.3 * rng.standard_normal(Z.shape).astype(np.float32)
    leak = crossfit_conditional_leakage(Z, sim.y, dom, dag, 3, n_perm=12, seed=0)
    assert leak["site"]["excess"] > 0.1, leak["site"]


def test_subject_memorisation_is_not_site_leakage():
    """z encodes SUBJECT identity only; with the subject-grouped split (train/eval subjects
    disjoint) this must NOT register as site leakage."""
    sim, dag, dom = _sim()
    rng = np.random.default_rng(1)
    subj = dom.factor("subject")
    nsub = int(subj.max()) + 1
    Z = np.zeros((sim.n, 4 + nsub), dtype=np.float32)
    Z[np.arange(sim.n), sim.y] = 2.0
    Z[np.arange(sim.n), 4 + subj] += 2.0                     # subject identity, NOT site
    Z += 0.3 * rng.standard_normal(Z.shape).astype(np.float32)
    leak = crossfit_conditional_leakage(Z, sim.y, dom, dag, 3, n_perm=12, seed=1)
    assert leak["site"]["excess"] < 0.5, ("subject memorisation leaked as site", leak["site"])


if __name__ == "__main__":
    test_site_shared_signal_is_detected()
    test_subject_memorisation_is_not_site_leakage()
    print("test_leakage_group_split PASSED")
