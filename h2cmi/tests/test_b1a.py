"""Stage-B1a tests: deterministic per-unit TTA seeding (variant-order invariance + same-seed
reproducibility, incl. the low-rank restarts), the responsibilities ACTUALLY used (r_initial/
r_last_used/r_final), per-subject LOSO with identity scored as an action + unsupervised folds
that never read fit labels, the EMPIRICAL pooled baseline, the standard difficulty reproducing
B0 source data, the hard preset lowering SNR, and an end-to-end b1a shard -> exact-key merge."""
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

from sklearn.metrics import balanced_accuracy_score

from h2cmi.config import DensityConfig, TTAConfig
from h2cmi.density.student_t_mixture import ClassConditionalDensity
from h2cmi.tta.class_conditional import (ClassConditionalTTA, B1A_VARIANTS, B1A_VARIANTS_BY_NAME,
                                         VariantFit, reference_weighted_source_moments)
from h2cmi.data.paired_simulator import PairedEEGSimulator, DIFFICULTY_PRESETS, difficulty_kwargs


def _toy(n=240, d=8, K=3, seed=0):
    torch.manual_seed(seed)
    dens = ClassConditionalDensity(d, K, DensityConfig(n_components=1, cov_rank=2))
    with torch.no_grad():
        dens.mu[:, 0] = torch.eye(K, d)[:, :d] * 3.0
    rng = np.random.default_rng(seed)
    pi_star = np.full(K, 1.0 / K)
    ys_src = rng.integers(0, K, 300)                              # synthetic SOURCE embeddings
    Us_src = (dens.mu[ys_src, 0] + 0.3 * torch.randn(300, d)).detach()
    pooled_ref = reference_weighted_source_moments(Us_src, ys_src, pi_star)
    yt = rng.choice(K, size=n, p=[0.5, 0.3, 0.2])
    U = (dens.mu[yt, 0] + 0.3 * torch.randn(n, d)).detach()
    subj = rng.integers(0, 3, size=n)
    return ClassConditionalTTA(dens, pi_star, TTAConfig(em_iters=8), K), U, yt, subj, pooled_ref


def _ol(v, yt):
    return yt if v.responsibility == "oracle" else None


# ---- determinism ----
def test_variant_order_invariance():
    tta, U, yt, _, pref = _toy()

    def fit_all(order, ambient):
        torch.manual_seed(ambient)                               # ambient RNG must not matter
        out = {}
        for v in order:
            f = tta.fit_variant(U, v, oracle_labels=_ol(v, yt), pooled_ref=pref, tta_seed=4242 + len(v.name))
            out[v.name] = (f.transform.matrix().detach().clone(), f.pi_T.clone())
        return out

    a = fit_all(list(B1A_VARIANTS), 1)
    b = fit_all(list(reversed(B1A_VARIANTS)), 99999)
    for name in a:
        assert torch.allclose(a[name][0], b[name][0], atol=1e-6), f"order-variant transform: {name}"
        assert torch.allclose(a[name][1], b[name][1], atol=1e-6), f"order-variant pi: {name}"


def test_same_unit_same_tta_seed():
    tta, U, yt, _, pref = _toy()
    v = B1A_VARIANTS_BY_NAME["oracle_oneshot_lowrank"]            # lowrank + 3 restarts
    assert v.restarts == 3
    torch.manual_seed(3)
    f1 = tta.fit_variant(U, v, oracle_labels=yt, pooled_ref=pref, tta_seed=42)
    torch.manual_seed(123456)
    f2 = tta.fit_variant(U, v, oracle_labels=yt, pooled_ref=pref, tta_seed=42)
    assert torch.allclose(f1.transform.matrix(), f2.transform.matrix(), atol=1e-6)
    f3 = tta.fit_variant(U, v, oracle_labels=yt, pooled_ref=pref, tta_seed=43)
    assert not torch.allclose(f1.transform.matrix(), f3.transform.matrix(), atol=1e-6)


# ---- responsibilities actually used ----
def test_responsibilities_exposed():
    tta, U, yt, _, pref = _toy()
    f = tta.fit_variant(U, B1A_VARIANTS_BY_NAME["gen_oneshot_diag"], pooled_ref=pref, tta_seed=0)
    assert f.r_initial is not None and torch.allclose(f.r_initial, f.r_last_used)   # one-shot: frozen
    fo = tta.fit_variant(U, B1A_VARIANTS_BY_NAME["oracle_oneshot_diag"], oracle_labels=yt,
                         pooled_ref=pref, tta_seed=0)
    assert balanced_accuracy_score(yt, fo.r_last_used.argmax(1).cpu().numpy()) == 1.0   # == true labels
    for nm in ("identity", "pooled_empirical_diag"):
        fp = tta.fit_variant(U, B1A_VARIANTS_BY_NAME[nm], pooled_ref=pref, tta_seed=0)
        assert fp.r_initial is None and fp.r_last_used is None and isinstance(fp, VariantFit)
        assert fp.r_final.shape == (len(yt), 3)


# ---- grouped LOSO ----
def test_grouped_identity_scored_as_action():
    tta, U, yt, subj, pref = _toy()
    g = tta.grouped_heldout(U, subj, B1A_VARIANTS_BY_NAME["identity"], true_labels=yt, seed_parts=(0,))
    assert not np.isnan(g["grouped_oof_bacc"])                    # identity IS scored now
    assert g["grouped_crossfit_evidence_gain"] == 0.0
    assert g["grouped_oof_coverage"] == 1.0 and g["grouped_n_groups_scored"] == g["grouped_n_groups_total"]
    v = B1A_VARIANTS_BY_NAME["gen_oneshot_diag"]
    g2 = tta.grouped_heldout(U, subj, v, true_labels=yt, pooled_ref=pref, seed_parts=(0, 1, "cov"))
    assert 0.0 <= g2["grouped_oof_bacc"] <= 1.0 and np.isfinite(g2["grouped_crossfit_evidence_gain"])
    g1 = tta.grouped_heldout(U, np.zeros_like(subj), v, true_labels=yt, pooled_ref=pref, seed_parts=(0,))
    assert np.isnan(g1["grouped_oof_bacc"]) and g1["grouped_n_groups_total"] == 1   # single subject


def test_unsupervised_fold_does_not_read_labels():
    # 2 subjects: subject 0 single-class, subject 1 mixed. Holding out subj1 leaves a single-class
    # fit set -> oracle SKIPS that fold (needs >=2 classes), gen must NOT (no label conditioning).
    tta, _, _, _, pref = _toy()
    d, K = 8, 3
    yt = np.concatenate([np.zeros(20, int), np.array([0, 1, 2] * 7)[:20]])
    subj = np.concatenate([np.zeros(20, int), np.ones(20, int)])
    U = (tta.density.mu[yt, 0] + 0.3 * torch.randn(40, d)).detach()
    g_gen = tta.grouped_heldout(U, subj, B1A_VARIANTS_BY_NAME["gen_oneshot_diag"], true_labels=yt,
                                pooled_ref=pref, seed_parts=(0,))
    g_ora = tta.grouped_heldout(U, subj, B1A_VARIANTS_BY_NAME["oracle_oneshot_diag"], true_labels=yt,
                                oracle_labels=yt, pooled_ref=pref, seed_parts=(0,))
    assert g_gen["grouped_n_groups_scored"] == 2          # gen scores BOTH folds (never reads yt[fit])
    assert g_ora["grouped_n_groups_scored"] == 1          # oracle skips the single-class fit fold


def test_source_moments_accepts_numpy_and_tensor():
    # _embed returns a tensor on args.device (CUDA in production) -> the helper must accept a
    # tensor, not only numpy (regression: np.asarray on a CUDA tensor raises).
    rng = np.random.default_rng(0)
    ys = rng.integers(0, 3, 200)
    Us_t = torch.randn(200, 6)
    mu_t, sd_t = reference_weighted_source_moments(Us_t, ys, np.full(3, 1 / 3))
    mu_n, sd_n = reference_weighted_source_moments(Us_t.numpy(), ys, np.full(3, 1 / 3))
    assert torch.allclose(mu_t, mu_n, atol=1e-5) and torch.allclose(sd_t, sd_n, atol=1e-5)


def test_pooled_empirical_diag_matches_source_moments():
    tta, U, yt, _, pref = _toy()
    mu_S, sd_S = pref
    f = tta.fit_variant(U, B1A_VARIANTS_BY_NAME["pooled_empirical_diag"], pooled_ref=pref, tta_seed=0)
    z = f.transform.apply(U)
    assert torch.allclose(z.mean(0), mu_S, atol=1e-3), "pooled mean not matched"
    assert torch.allclose(z.std(0, unbiased=False), sd_S, atol=1e-3), "pooled std not matched"


# ---- difficulty preset ----
def test_standard_difficulty_reproduces_b0_source():
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
    h = PairedEEGSimulator(3, 16, 128, **DIFFICULTY_PRESETS["hard"], data_seed=0)
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


def test_variants_preflight_subset_is_own_experiment():
    from h2cmi.grid_io import manifest_path
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "pre.jsonl")
        _run_b1a(["--variants", "identity", "--out", out])
        man = json.load(open(manifest_path(out)))
        assert man["items"] == ["identity"]
        rows = [json.loads(l) for l in open(out)]
        # 1 seed x 2 sites x 2 scenarios x 1 variant = 4
        assert len(rows) == 4 and all(r["variant"] == "identity" for r in rows)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  {name} PASSED")
    print("test_b1a PASSED")
