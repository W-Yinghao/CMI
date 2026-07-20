"""The factorial requires M0/M2 to share Source-A and M1/M3 to share Source-B, and A,B to
start from the same init. This is guaranteed by (a) identical init at a fixed seed and
(b) deterministic training -- both verified here."""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

try:
    import pytest
    pytestmark = pytest.mark.integration
except ImportError:
    pass

from h2cmi.config import H2Config, core_config
from h2cmi.data.paired_simulator import PairedEEGSimulator
from h2cmi.domains import compact_domain_labels
from h2cmi.train.trainer import train_h2, H2Model, reference_prior
from h2cmi.run_shift_grid import hash_state


def _tiny_cfg(enabled):
    cfg = core_config(H2Config(n_classes=3))
    cfg.encoder.n_chans = 12; cfg.encoder.n_times = 64
    cfg.train.epochs = 2; cfg.train.seed = 0; cfg.cmi.enabled = enabled
    cfg.cmi.critic_inner = 1
    return cfg


def _source():
    sim = PairedEEGSimulator(3, 12, 64, data_seed=0)
    full = sim.sample(3, 2, 1, 10, target_site=0, scenario="no_shift")
    src = full.site != 0
    dag, dom, _ = compact_domain_labels(full.domains.subset(np.where(src)[0]))
    return full.X[src], full.y[src], dom, dag


def test_same_init_at_fixed_seed():
    X, y, dom, dag = _source()
    pi = reference_prior(y, 3, "uniform")
    from h2cmi.train.trainer import set_seed
    set_seed(0); m1 = H2Model(_tiny_cfg(False), pi)
    set_seed(0); m2 = H2Model(_tiny_cfg(True), pi)   # CMI on/off share the encoder+head init
    assert hash_state(m1) == hash_state(m2), "A and B do not share the initial state"


def test_cmi_off_training_is_deterministic():
    X, y, dom, dag = _source()
    mA1, *_ = train_h2(X, y, dom, dag, _tiny_cfg(False))
    mA2, *_ = train_h2(X, y, dom, dag, _tiny_cfg(False))
    assert hash_state(mA1) == hash_state(mA2), "Source-A (CMI off) training is non-deterministic"


def test_cmi_on_training_is_deterministic():
    X, y, dom, dag = _source()
    mB1, *_ = train_h2(X, y, dom, dag, _tiny_cfg(True))
    mB2, *_ = train_h2(X, y, dom, dag, _tiny_cfg(True))
    assert hash_state(mB1) == hash_state(mB2), "Source-B (CMI on) training is non-deterministic"


if __name__ == "__main__":
    test_same_init_at_fixed_seed()
    test_cmi_off_training_is_deterministic()
    test_cmi_on_training_is_deterministic()
    print("test_factorial_checkpoint_sharing PASSED")
