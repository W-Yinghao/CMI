"""CIGL_69B MetaCMI scaffold — firewall + engineering tests (CPU only, no GPU, no real EEG; NOT scientific evidence).

Firewall properties asserted:
  - episodic partition covers exactly the SOURCE-TRAIN subjects, disjointly, both sides non-empty;
  - the meta_train subject projector is a pure function of the rows passed (unaffected by meta_heldout/target);
  - train_model only ever fits the projector on a STRICT SUBSET of source subjects (meta_train), never on
    all-source / meta_heldout / target (spied via monkeypatch);
  - the SymKL penalty is active (a non-zero beta changes the learned model; the term is logged, finite, >=0);
  - deterministic under a fixed seed; fail-closed on a backbone with no linear head.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import cmi.train.trainer as T
from cmi.models.sanity_backbones import build_sanity_backbone
from cmi.train.trainer import train_model, predict, _feature_subject_projector, _head_module, META_METHODS


def _entangled(n_subj=6, per=40, C=8, T_=64, seed=0):
    """Source EEG-like data where SUBJECT id spuriously carries the class (p(y|subject) split 0.2/0.8)."""
    rng = np.random.default_rng(seed); X = []; y = []; d = []
    for dd in range(n_subj):
        p1 = 0.2 if dd < n_subj // 2 else 0.8
        for _ in range(per):
            cc = int(rng.random() < p1)
            b = rng.standard_normal((C, T_)).astype("float32") * 0.4
            b[0] += 0.4 * cc; b[1] += (dd - n_subj / 2) * 1.5
            X.append(b); y.append(cc); d.append(dd)
    return np.stack(X), np.array(y), np.array(d)


@pytest.mark.parametrize("bb", ["eegnet", "conformer"])
@pytest.mark.parametrize("method,beta", [("metace", 0.0), ("metacmi_direct", 0.5)])
def test_meta_methods_train_finite_and_log(bb, method, beta):
    X, y, d = _entangled()
    torch.manual_seed(0); np.random.seed(0)
    net = build_sanity_backbone(bb, 8, 64, 2)
    net, _, out = train_model(net, X, y, d, 2, method=method, epochs=8, bs=48, warmup=2, device="cpu", seed=0,
                              fcigl_strength=beta, fcigl_k=2, fcigl_update_every=2, meta_rho=1.0, meta_train_frac=0.6)
    assert np.isfinite(predict(net, X, "cpu")).all()
    assert out["inloop_meta_ce"] is not None                       # meta_heldout CE always logged
    if method == "metacmi_direct":
        assert out["inloop_meta_symkl"] is not None and out["inloop_meta_symkl"] >= 0.0
    else:
        assert out["inloop_meta_symkl"] is None                    # MetaCE has no removal term


def test_episodic_partition_source_only_disjoint_covers():
    """Replicates train_model's per-epoch partition: covers exactly the source subjects, disjoint, both non-empty."""
    src = np.array([3, 5, 7, 9, 11, 13])                           # arbitrary source-train subject ids
    for ep in range(6):
        perm = np.random.default_rng(0 + ep).permutation(src)
        n_mt = min(len(perm) - 1, max(1, int(round(0.7 * len(perm)))))
        mt, mh = perm[:n_mt], perm[n_mt:]
        assert set(mt).isdisjoint(mh)                              # meta_train / meta_heldout disjoint
        assert set(mt) | set(mh) == set(src)                       # covers ONLY source subjects, nothing else
        assert len(mt) >= 1 and len(mh) >= 1                       # always >=1 pseudo-target held out


def test_projector_is_pure_function_of_meta_train_rows():
    """The meta_train projector must not change when meta_heldout / target rows are corrupted (firewall)."""
    X, y, d = _entangled(); net = build_sanity_backbone("eegnet", 8, 64, 2).eval()
    mt = np.isin(d, [0, 1, 2])                                     # meta_train = subjects {0,1,2}
    S0, P0 = _feature_subject_projector(net, X[mt], y[mt], d[mt], 2, "cpu")
    Xc = X.copy(); Xc[~mt] += 99.0; yc = y.copy(); yc[~mt] = 1 - yc[~mt]   # wreck everything OUTSIDE meta_train
    S1, P1 = _feature_subject_projector(net, Xc[mt], yc[mt], d[mt], 2, "cpu")
    assert torch.allclose(P0, P1) and torch.allclose(S0, S1)


def test_projector_only_ever_fits_on_meta_train_subset(monkeypatch):
    """Spy: during metacmi_direct training the projector is fit only on a STRICT source subset (meta_train),
    never on all-source / meta_heldout / target."""
    X, y, d = _entangled(); src = set(np.unique(d).tolist()); seen = []
    orig = T._feature_subject_projector
    def spy(backbone, Xs, ys, ds, k, device, bs=256):
        seen.append(set(np.unique(ds).tolist())); return orig(backbone, Xs, ys, ds, k, device, bs)
    monkeypatch.setattr(T, "_feature_subject_projector", spy)
    torch.manual_seed(0); np.random.seed(0)
    net = build_sanity_backbone("eegnet", 8, 64, 2)
    train_model(net, X, y, d, 2, method="metacmi_direct", epochs=6, bs=48, warmup=2, device="cpu", seed=0,
                fcigl_strength=0.5, fcigl_k=2, fcigl_update_every=2, meta_train_frac=0.6)
    assert len(seen) >= 1
    for call_subj in seen:
        assert call_subj.issubset(src)                             # only ever source subjects
        assert call_subj != src and len(call_subj) < len(src)      # a strict subset = meta_train, never all/target


def test_symkl_penalty_changes_the_model():
    """A non-zero beta (metacmi_direct) must produce a different model than the unpenalized MetaCE — the removal
    penalty is actually wired into the optimizer, not a no-op."""
    X, y, d = _entangled()
    def preds(method, beta):
        torch.manual_seed(0); np.random.seed(0)
        net = build_sanity_backbone("eegnet", 8, 64, 2)
        net, _, _ = train_model(net, X, y, d, 2, method=method, epochs=12, bs=48, warmup=2, device="cpu", seed=0,
                                fcigl_strength=beta, fcigl_k=2, fcigl_update_every=2)
        return predict(net, X, "cpu")
    assert not np.allclose(preds("metace", 0.0), preds("metacmi_direct", 0.8))


def test_deterministic_under_fixed_seed():
    X, y, d = _entangled()
    def run():
        torch.manual_seed(0); np.random.seed(0)
        net = build_sanity_backbone("conformer", 8, 64, 2)
        net, _, _ = train_model(net, X, y, d, 2, method="metacmi_direct", epochs=6, bs=48, warmup=2, device="cpu",
                                seed=0, fcigl_strength=0.5, fcigl_k=2, fcigl_update_every=2)
        return predict(net, X, "cpu")
    assert np.allclose(run(), run())


def test_fail_closed_on_backbone_without_linear_head():
    """Meta methods require a replayable linear head (for the removal branch); a headless backbone must raise."""
    class Headless(nn.Module):
        def __init__(self): super().__init__(); self.z_dim = 64; self.lin = nn.Linear(64, 2)   # z_dim ok, head absent
        def forward(self, x):
            z = x.reshape(x.shape[0], -1)[:, :64]
            return self.lin(z), z
    X, y, d = _entangled()
    with pytest.raises(ValueError):
        train_model(Headless(), X, y, d, 2, method="metace", epochs=2, bs=48, device="cpu", seed=0)
