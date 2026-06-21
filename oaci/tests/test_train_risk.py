"""Primal-risk metrics: balanced_ce is full-set / batch-partition invariant; balanced_err is
rejected as a primal metric (guard/report only).

Standalone (``python -m oaci.tests.test_train_risk``) and pytest-compatible.
"""
from __future__ import annotations

import torch

from oaci.train.risk import (
    assert_differentiable_primal,
    balanced_ce,
    balanced_error,
    per_class_ce_sums,
    source_risk,
)


def test_full_dataset_balanced_ce_is_batch_partition_invariant():
    torch.manual_seed(0)
    n, nc = 210, 3
    logits = torch.randn(n, nc)
    y = torch.randint(0, nc, (n,))
    full = balanced_ce(logits, y, nc).item()
    # accumulate per-class (sum CE, count) over an UNEVEN partition, then mean_c(sum/count)
    perm = torch.randperm(n)
    sums = torch.zeros(nc)
    counts = torch.zeros(nc)
    for batch in torch.chunk(perm, 7):
        s, c = per_class_ce_sums(logits[batch], y[batch], nc)
        sums += s
        counts += c
    present = counts > 0
    agg = (sums[present] / counts[present]).mean().item()
    assert abs(full - agg) < 1e-5, (full, agg)


def test_balanced_err_rejected_as_primal_metric():
    logits = torch.randn(12, 2)
    y = torch.randint(0, 2, (12,))
    for bad in ("balanced_err", "err", "acc", "0-1"):
        try:
            source_risk(logits, y, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"primal metric {bad!r} should be rejected")
    # balanced_err is fine as a (non-differentiable) report metric
    assert 0.0 <= balanced_error(logits, y, 2) <= 1.0
    assert_differentiable_primal("ce")
    assert_differentiable_primal("balanced_ce")


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} train-risk tests")


if __name__ == "__main__":
    _run_all()
