"""The termination gate: a real signal yields a stable selected subspace across seeds;
a pure-noise world yields no selection at all (identity everywhere)."""
import torch

from tos_cmi.data.synthetic import SynthSpec, make
from tos_cmi.subspace import SubspaceSelector
from tos_cmi.eval.stability import selection_stability


def _bases(spec_kw, seeds=(0, 1, 2, 3, 4), struct_seed=0):
    """Fix the planted geometry (struct_seed) and vary only the finite-sample draw (seed),
    so this measures whether the SELECTOR is stable across draws of the same distribution."""
    bases = []
    for s in seeds:
        data = make(SynthSpec(n=4000, **spec_kw), seed=s, struct_seed=struct_seed)
        sp = data["spec"]
        sel = SubspaceSelector(sp.d, sp.n_cls, sp.n_dom)
        sel.refresh(torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))
        bases.append(sel.report.basis)
    return bases


def test_stable_under_clear_signal():
    st = selection_stability(_bases(dict(overlap=0.0)))
    assert st["n_identity"] == 0, st
    assert st["mean_overlap"] > 0.80, st
    print("test_stable_under_clear_signal: OK", st)


def test_no_selection_under_pure_noise():
    # no class or domain structure -> nothing is domain-rich -> identity everywhere
    st = selection_stability(_bases(dict(sep_label=0.0, sep_dom=0.0)))
    assert st["n_identity"] == 5, st
    print("test_no_selection_under_pure_noise: OK", st)


if __name__ == "__main__":
    test_stable_under_clear_signal()
    test_no_selection_under_pure_noise()
    print("ALL STABILITY TESTS PASSED")
