"""ShallowConvNet backbone: returns (logits, z); the dummy dimension-inference forward does not
mutate BatchNorm running stats (eval + no_grad + mode restore).

Standalone (``python -m oaci.tests.test_backbone``) and pytest-compatible.
"""
from __future__ import annotations

import torch

from oaci.models import ModelOutput, build_model
from oaci.models.shallow import ShallowConvNet


def test_shallow_forward_returns_logits_and_z():
    m = ShallowConvNet(in_chans=22, in_times=385, n_classes=4, temporal_filters=8,
                       temporal_kernel_samples=25, pool_kernel_samples=35, pool_stride_samples=7)
    out = m(torch.randn(3, 22, 385))
    assert isinstance(out, ModelOutput)
    assert out.logits.shape == (3, 4)
    assert out.z.ndim == 2 and out.z.shape[0] == 3 and out.z.shape[1] == m.feat_dim > 0
    assert torch.isfinite(out.logits).all() and torch.isfinite(out.z).all()


def test_dummy_forward_does_not_mutate_batchnorm():
    m = ShallowConvNet(in_chans=22, in_times=385, n_classes=4, temporal_filters=8)
    rm0 = m.bn.running_mean.clone(); rv0 = m.bn.running_var.clone()
    nb0 = m.bn.num_batches_tracked.clone()
    m.train()                                                    # put in TRAIN mode
    was_training = m.training
    # a SAFE (eval+no_grad) dummy forward must not update BN running stats, and must restore mode
    feat = m._infer_feat_dim(22, 385)
    assert feat == m.feat_dim
    assert torch.equal(m.bn.running_mean, rm0) and torch.equal(m.bn.running_var, rv0)
    assert torch.equal(m.bn.num_batches_tracked, nb0)
    assert m.training == was_training                            # mode restored


def test_factory_builds_shallow_and_mlp():
    sc = build_model("shallow_convnet", in_chans=22, in_times=385, n_classes=4, temporal_filters=8)
    assert sc(torch.randn(2, 22, 385)).logits.shape == (2, 4)
    mlp = build_model("mlp", in_dim=6, n_classes=2)
    out = mlp(torch.randn(5, 6))
    assert out.logits.shape == (5, 2) and out.z.shape == (5, 8)
    try:
        build_model("nope", in_dim=4)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown backbone must raise")


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} backbone tests")


if __name__ == "__main__":
    _run_all()
