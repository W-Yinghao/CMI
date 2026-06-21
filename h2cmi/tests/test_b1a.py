"""Stage-B1a tests: deterministic per-unit TTA seeding (variant-order invariance + same-seed
reproducibility), per-subject LOSO grouped evidence, pooled-moment baseline, the standard
difficulty reproducing the B0 source data exactly, the hard preset lowering SNR, and an
end-to-end b1a shard -> exact-key merge."""
from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

try:
    import pytest
    pytestmark = pytest.mark.integration
except ImportError:
    pass

from h2cmi.config import DensityConfig, TTAConfig
from h2cmi.density.student_t_mixture import ClassConditionalDensity
from h2cmi.tta.class_conditional import (ClassConditionalTTA, B1A_VARIANTS, B1A_VARIANTS_BY_NAME,
                                         pooled_source_moments)
from h2cmi.data.paired_simulator import PairedEEGSimulator, DIFFICULTY_PRESETS, difficulty_kwargs


def _toy_tta(n=240, d=8, K=3, seed=0):
    torch.manual_seed(seed)
    dens = ClassConditionalDensity(d, K, DensityConfig(n_components=1, cov_rank=2))
    with torch.no_grad():
        dens.mu[:, 0] = torch.eye(K, d)[:, :d] * 3.0
    rng = np.random.default_rng(seed)
    yt = rng.choice(K, size=n, p=[0.5, 0.3, 0.2])
    U = (dens.mu[yt, 0] + 0.3 * torch.randn(n, d)).detach()
    subj = rng.integers(0, 3, size=n)
    pi_S = np.full(K, 1.0 / K)
    return ClassConditionalTTA(dens, pi_S, TTAConfig(em_iters=8), K), U, yt, subj


# ---- determinism ----
def test_variant_order_invariance():
    tta, U, yt, _ = _toy_tta()

    def fit_all(order, ambient):
        torch.manual_seed(ambient)                                 # ambient RNG must not matter
        out = {}
        for v in order:
            ol = yt if v.responsibility == "oracle" else None
            T, pi, _ = tta.fit_variant(U, v, oracle_labels=ol, tta_seed=4242 + len(v.name))
            out[v.name] = (T.matrix().detach().clone(), pi.clone())
        return out

    a = fit_all(list(B1A_VARIANTS), 1)
    b = fit_all(list(reversed(B1A_VARIANTS)), 99999)
    for name in a:
        assert torch.allclose(a[name][0], b[name][0], atol=1e-6), f"order-variant transform: {name}"
        assert torch.allclose(a[name][1], b[name][1], atol=1e-6), f"order-variant pi: {name}"


def test_same_unit_same_tta_seed():
    tta, U, yt, _ = _toy_tta()
    v = B1A_VARIANTS_BY_NAME["oracle_oneshot_lowrank"]               # lowrank => randomness present
    torch.manual_seed(3)
    T1, p1, _ = tta.fit_variant(U, v, oracle_labels=yt, tta_seed=42)
    torch.manual_seed(123456)                                       # perturb ambient between fits
    T2, p2, _ = tta.fit_variant(U, v, oracle_labels=yt, tta_seed=42)
    assert torch.allclose(T1.matrix(), T2.matrix(), atol=1e-6)
    # a different seed CAN move the low-rank init (sanity that the seed is actually used)
    T3, _, _ = tta.fit_variant(U, v, oracle_labels=yt, tta_seed=43)
    assert not torch.allclose(T1.matrix(), T3.matrix(), atol=1e-6)


# ---- grouped LOSO ----
def test_grouped_heldout_loso():
    tta, U, yt, subj = _toy_tta()
    idv = B1A_VARIANTS_BY_NAME["identity"]
    g_id = tta.grouped_heldout(U, subj, idv, true_labels=yt, seed_parts=(0,))
    assert np.isnan(g_id["grouped_oof_bacc"]) and g_id["grouped_n_groups"] == 3   # identity -> NaN
    v = B1A_VARIANTS_BY_NAME["gen_oneshot_diag"]
    g = tta.grouped_heldout(U, subj, v, true_labels=yt, seed_parts=(0, 1, "cov"))
    assert g["grouped_n_groups"] == 3
    assert 0.0 <= g["grouped_oof_bacc"] <= 1.0
    assert np.isfinite(g["grouped_crossfit_evidence_gain"])
    # single target subject -> LOSO undefined -> NaN, n_groups=1
    g1 = tta.grouped_heldout(U, np.zeros_like(subj), v, true_labels=yt, seed_parts=(0,))
    assert np.isnan(g1["grouped_oof_bacc"]) and g1["grouped_n_groups"] == 1


def test_pooled_diag_matches_source_moments():
    tta, U, yt, _ = _toy_tta()
    mu_S, sd_S = pooled_source_moments(tta.density, tta.pi_S)
    T, pi, _ = tta.fit_variant(U, B1A_VARIANTS_BY_NAME["pooled_diag"], tta_seed=0)
    z = T.apply(U)                                                  # transformed target
    assert torch.allclose(z.mean(0), mu_S, atol=1e-3), "pooled mean not matched"
    assert torch.allclose(z.std(0, unbiased=False), sd_S, atol=1e-3), "pooled std not matched"


# ---- difficulty preset ----
def test_standard_difficulty_reproduces_b0_source():
    # class_signal_scale=1.0 (standard) must be byte-identical to the default B0 simulator
    a = PairedEEGSimulator(3, 16, 128, data_seed=0)
    dkw = difficulty_kwargs("standard")
    b = PairedEEGSimulator(3, 16, 128, base_noise=dkw["base_noise"], subj_anatomy=dkw["subj_anatomy"],
                           class_signal_scale=dkw["class_signal_scale"], data_seed=0)
    sa = a.sample(3, 2, 1, 16, target_site=0, scenario="population_null")
    sb = b.sample(3, 2, 1, 16, target_site=0, scenario="population_null")
    src = sa.site != 0
    assert np.array_equal(sa.X[src], sb.X[src]), "standard difficulty perturbed the B0 source data"


def test_hard_difficulty_lowers_snr():
    assert DIFFICULTY_PRESETS["hard"]["class_signal_scale"] < 1.0
    assert DIFFICULTY_PRESETS["hard"]["base_noise"] > DIFFICULTY_PRESETS["standard"]["base_noise"]
    # hard matched_domain_null still reuses SOURCE anatomy (true identity-null), just noisier
    h = PairedEEGSimulator(3, 16, 128, **{k: v for k, v in DIFFICULTY_PRESETS["hard"].items()}, data_seed=0)
    m = h.sample(3, 2, 1, 16, target_site=0, scenario="matched_domain_null")
    assert (m.site == 0).sum() > 0


# ---- end-to-end shard -> merge ----
def _run_b1a(extra):
    from h2cmi import run_b1a_grid
    base = ["run_b1a_grid", "--scenarios", "population_null,cov", "--difficulty", "standard",
            "--grid-seeds", "0", "--grid-target-sites", "all", "--sites", "2", "--subjects", "3",
            "--sessions", "1", "--trials", "16", "--epochs", "1", "--fast", "--allow-dirty"]
    old = sys.argv
    sys.argv = base + extra
    try:
        run_b1a_grid.main()
    finally:
        sys.argv = old


def test_b1a_shards_share_signature_and_merge():
    from h2cmi.grid_io import manifest_path
    from h2cmi.merge_grid_shards import merge_shards
    with tempfile.TemporaryDirectory() as d:
        sd = Path(d) / "shards"; sd.mkdir()
        br = str(Path(d) / "bundles")
        common = ["--bundle-root", br]
        _run_b1a(common + ["--shard-target-sites", "0", "--out", str(sd / "s0.jsonl")])
        _run_b1a(common + ["--shard-target-sites", "1", "--out", str(sd / "s1.jsonl")])
        m0 = json.load(open(manifest_path(str(sd / "s0.jsonl"))))
        m1 = json.load(open(manifest_path(str(sd / "s1.jsonl"))))
        assert m0["experiment_signature"] == m1["experiment_signature"]
        assert m0["shard_spec"] != m1["shard_spec"] and m0["item_field"] == "variant"
        out = str(Path(d) / "merged.jsonl")
        info = merge_shards(str(sd), out)
        # 1 seed x 2 sites x 2 scenarios x 7 variants x 1 cmi = 28
        assert info["rows"] == 28 and info["unique_keys"] == 28


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  {name} PASSED")
    print("test_b1a PASSED")
