"""No-shift => identity rollback, in the grid pipeline context.

A genuine null for the TTA is target embeddings drawn from the trained model's OWN
class-conditional density (so there is no fittable mean/cov discrepancy). The cross-fitted
held-out evidence gate must then roll back to identity. (At the raw-embedding level even the
paired simulator's ``no_shift`` target carries an unseen-subject gap + density under-fit, so
it is not a true null -- that confound is exactly what the oracle diagnostics surface.)"""
from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

try:
    import pytest
    pytestmark = pytest.mark.integration
except ImportError:
    pass

from h2cmi.config import H2Config, core_config
from h2cmi.data.paired_simulator import PairedEEGSimulator
from h2cmi.domains import compact_domain_labels
from h2cmi.train.trainer import train_h2, reference_prior
from h2cmi.tta.class_conditional import ClassConditionalTTA


@torch.no_grad()
def _sample_from_density(dens, y):
    out = []
    for c in y:
        mu, L = dens.mu[c, 0], dens.L[c, 0]
        diag = torch.sqrt(F.softplus(dens.log_s[c, 0]) + dens.eig_floor)
        out.append(mu + L @ torch.randn(L.shape[1]) + diag * torch.randn(mu.shape[0]))
    return torch.stack(out)


def test_true_null_rolls_back_to_identity():
    torch.manual_seed(0)
    sim = PairedEEGSimulator(3, 12, 96, data_seed=0)
    full = sim.sample(4, 2, 2, 24, target_site=0, scenario="no_shift")
    src = full.site != 0
    Xs, ys = full.X[src], full.y[src]
    dag, dom, _ = compact_domain_labels(full.domains.subset(np.where(src)[0]))
    cfg = core_config(H2Config(n_classes=3))
    cfg.encoder.n_chans = 12; cfg.encoder.n_times = 96; cfg.train.epochs = 6; cfg.cmi.enabled = False
    model, *_ = train_h2(Xs, ys, dom, dag, cfg)

    pi = reference_prior(ys, 3, "uniform")
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 3, 600)
    U = _sample_from_density(model.head.density, yt)        # true null for the TTA
    tta = ClassConditionalTTA(model.head.density, pi, cfg.tta, 3)
    res = tta.fit(U, pseudo_labels=yt)
    assert not res.adapted, \
        f"adapted under a true null (gain={res.diagnostics.get('crossfit_evidence_gain')})"
    assert res.diagnostics.get("reason") == "no_heldout_evidence"


if __name__ == "__main__":
    test_true_null_rolls_back_to_identity()
    print("test_no_shift_identity PASSED")
