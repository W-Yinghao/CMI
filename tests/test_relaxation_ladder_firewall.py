"""Relaxation Ladder Stage 9 — per-level FIREWALL + rank + head-separation + gate source-only tests."""
import numpy as np
import pytest

from tos_cmi.eeg import relaxation_ladder as RL
from tos_cmi.eeg import selective_erasure as SE


def _feat(n_src_subj=4, per=40, n_cls=2, dim=12, seed=0):
    rng = np.random.default_rng(seed)
    cm = rng.standard_normal((n_cls, dim)) * 2
    subj_dirs = rng.standard_normal((n_src_subj + 1, dim)); subj_dirs /= np.linalg.norm(subj_dirs, axis=1, keepdims=True)
    Zs, ys, ds = [], [], []
    for s in range(n_src_subj):
        for c in range(n_cls):
            Zs.append(cm[c] + 2.5 * subj_dirs[s] + 0.4 * rng.standard_normal((per, dim))); ys += [c] * per; ds += [s] * per
    Zt, yt = [], []
    for c in range(n_cls):
        Zt.append(cm[c] + 2.5 * subj_dirs[n_src_subj] + 0.4 * rng.standard_normal((per, dim))); yt += [c] * per
    return dict(Z_source=np.vstack(Zs), y_source=np.array(ys), subj_source=np.array(ds),
                Z_target=np.vstack(Zt), y_target=np.array(yt), subj_target=np.full(len(np.hstack(yt)) if False else per * n_cls, n_src_subj),
                n_cls=n_cls, dataset="D", backbone="bb", feature_object="z", training_method="erm",
                outer_fold=0, heldout_subject="T", seed=seed, head_W=None, head_b=None)


# ---- 1: L1 fit data uses ONLY source (no target X, no target Y)
def test_L1_fit_data_source_only():
    feat = _feat()
    Zfit, yfit, subj_fit = RL._fit_data("L1_STRICT_SOURCE_FRESH_HEAD", feat)
    assert Zfit.shape[0] == feat["Z_source"].shape[0]
    assert np.array_equal(Zfit, feat["Z_source"])          # no target rows appended
    assert not np.any(yfit < 0)                             # all source labels (no -1 placeholder)


# ---- 2: L2 uses target X + group but NEVER target Y
def test_L2_uses_target_x_not_target_y():
    feat = _feat()
    ns = feat["Z_source"].shape[0]; nt = feat["Z_target"].shape[0]
    Zfit, yfit, subj_fit = RL._fit_data("L2_TARGET_X_UNLABELED_FRESH_HEAD", feat)
    assert Zfit.shape[0] == ns + nt                         # target X included
    assert np.all(yfit[ns:] == -1)                          # target Y is the -1 placeholder (never used)
    assert len(np.unique(subj_fit[ns:])) == 1               # target grouped as ONE subject id
    assert np.array_equal(Zfit[ns:], feat["Z_target"])


# ---- 3: L3 always tagged oracle / non-DG
def test_L3_metadata_oracle_non_dg():
    m = RL.LEVEL_META["L3_ORACLE_GLOBAL_GEOMETRY_FRESH_HEAD"]
    assert m["is_oracle_diagnostic"] is True and m["is_source_only_dg"] is False
    assert RL.LEVEL_META["L2_TARGET_X_UNLABELED_FRESH_HEAD"]["is_transductive"] is True
    for lv in ("L0_STRICT_SOURCE_ORIGINAL_HEAD", "L1_STRICT_SOURCE_FRESH_HEAD"):
        assert RL.LEVEL_META[lv]["uses_target_x"] is False and RL.LEVEL_META[lv]["is_source_only_dg"] is True


# ---- 4: LW-LEACE rank = (#subjects present) - 1 where valid
def test_lw_leace_rank_is_k_minus_1():
    feat = _feat(n_src_subj=4, dim=32)                      # 4 subjects, dim 32 -> rank should be 3
    _, rank = RL.lw_leace_full(feat["Z_source"], feat["subj_source"])
    assert rank == 3
    # capped at feature dim
    small = _feat(n_src_subj=8, dim=4)
    _, r2 = RL.lw_leace_full(small["Z_source"], small["subj_source"])
    assert r2 == min(8 - 1, 4)


# ---- 5: random removal matches the informed eraser rank exactly (in run_cell rows)
def test_random_matches_informed_rank():
    feat = _feat()
    rows = RL.run_cell(feat, "h", "s", n_random=5, seed=0)
    lw = [r for r in rows if r["eraser"] == "lw_leace_full" and r["fit_regime"].startswith("L1")][0]
    rnd = [r for r in rows if r["eraser"] == "random_k" and r["fit_regime"].startswith("L1")]
    assert all(r["eraser_rank"] == lw["eraser_rank"] for r in rnd)
    assert len(rnd) == 5                                    # n_random draws


# ---- 6: original-head and fresh-head outcomes are not mixed (distinct effect_kind per row)
def test_head_effect_kinds_not_mixed():
    feat = _feat(); feat["head_W"] = np.random.default_rng(0).standard_normal((feat["n_cls"], feat["Z_source"].shape[1]))
    feat["head_b"] = np.zeros(feat["n_cls"])
    rows = RL.run_cell(feat, "h", "s", n_random=2, seed=0)
    kinds = {r["fit_regime"][:2]: r["effect_kind"] for r in rows}
    assert kinds["L0"] == "original_head_replay"
    assert kinds["L1"] == "fresh_head" and kinds["L2"] == "fresh_head"
    assert kinds["L3"] == "oracle_grouped_cv"
    # every row carries exactly one effect_kind (no mixing)
    for r in rows:
        assert r["effect_kind"] in ("original_head_replay", "original_head_probe_fallback", "fresh_head", "oracle_grouped_cv")


# ---- 7: gate decisions are SOURCE-ONLY (poisoning target does not change source diagnostics)
def test_gate_diagnostics_source_only():
    feat = _feat()
    d1 = SE.source_diagnostics(feat, seed=0)
    feat2 = dict(feat); feat2["Z_target"] = 1e6 * np.random.default_rng(9).standard_normal(feat["Z_target"].shape)
    feat2["y_target"] = np.zeros(len(feat["y_target"]), int)
    d2 = SE.source_diagnostics(feat2, seed=0)
    for k in ("source_task_full", "subject_decode_full", "subject_decode_after_erasure", "subject_reduction_vs_random"):
        assert d1[k] == pytest.approx(d2[k], nan_ok=True), f"{k} changed when target was poisoned"


# ---- run_cell metadata matches LEVEL_META (firewall flags propagate to every row)
def test_run_cell_metadata_matches_level_meta():
    feat = _feat()
    for r in RL.run_cell(feat, "h", "s", n_random=1, seed=0):
        m = RL.LEVEL_META[r["fit_regime"]]
        for k in ("uses_target_x", "uses_target_subject_group", "uses_target_y", "is_source_only_dg",
                  "is_transductive", "is_oracle_diagnostic"):
            assert r[k] == m[k]
        assert r["uses_target_y"] is False                 # target Y NEVER used at any level (scoring only)
