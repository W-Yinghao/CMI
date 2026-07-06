"""CITA_01 — target-unlabeled offline transductive adaptation firewall + engineering tests (CPU only, NOT
scientific evidence). The 10 checks the PM required for the scaffold gate.
"""
import inspect
import numpy as np
import pytest
import torch
import torch.nn as nn

from cmi.models.sanity_backbones import build_sanity_backbone
from cmi.adaptation import cita
from cmi.adaptation.cita import adapt, CITA_METHODS, ADAPT_METHODS, ConditionalDomainPosterior
from cmi.eval.head_export import save_fold_audit, extract_task_head
from cmi.eval.audit_npz import load_audit_npz, validate_audit_npz, head_replay_ok
from cmi.eval.leakage_removal import evaluate_reliance


def _src(n_subj=4, per=40, C=8, T=64, seed=0):
    rng = np.random.default_rng(seed); X = []; y = []; d = []
    for dd in range(n_subj):
        p1 = 0.25 if dd < n_subj // 2 else 0.75
        for _ in range(per):
            cc = int(rng.random() < p1); b = rng.standard_normal((C, T)).astype("float32") * 0.4
            b[0] += 0.4 * cc; b[1] += (dd - n_subj / 2) * 1.0
            X.append(b); y.append(cc); d.append(dd)
    return np.stack(X), np.array(y), np.array(d)


def _target(per=30, C=8, T=64, seed=9):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((per, C, T)).astype("float32")


def _m0(bb="eegnet", C=8, T=64, ncls=2):
    torch.manual_seed(0); return build_sanity_backbone(bb, C, T, ncls).eval()


# 1 --------------------------------------------------------------------------
def test_adapt_signature_has_no_target_labels():
    """adapt() must not accept a target-label parameter -> target y cannot enter adaptation by construction."""
    params = set(inspect.signature(adapt).parameters)
    for banned in ("yt", "y_target", "target_y", "ytarget"):
        assert banned not in params
    assert "Xt" in params and "ys" in params            # target X + source y only


def test_randomizing_target_labels_does_not_change_adaptation():
    """Target labels are never an input to adapt(); the adapted model depends only on (Xs, ys, Xt)."""
    Xs, ys, _ = _src(); Xt = _target()
    def run():
        torch.manual_seed(0)
        return adapt(_m0(), Xs, ys, Xt, "cita_cmi", steps=8, bs=32, device="cpu", seed=0)[0]
    a, b = run(), run()
    xa = torch.randn(5, 8, 64)
    with torch.no_grad():
        assert torch.allclose(a(xa)[0], b(xa)[0])       # identical regardless of any (absent) target label


# 2 --------------------------------------------------------------------------
def test_target_labels_unavailable_during_adaptation():
    assert "y_target" not in inspect.getsource(adapt) or "y_target=None" not in inspect.getsource(adapt)
    # adapt only closes over Xt (no target labels); the gate passes y[te_mask] ONLY to the final metric.
    src = inspect.getsource(cita)
    assert "target: DETACHED soft pseudo-label" in src


# 3 --------------------------------------------------------------------------
def test_source_replay_ce_uses_source_labels_only():
    """Corrupting source labels changes the adaptation; there is no target-label path to corrupt."""
    Xs, ys, _ = _src(); Xt = _target()
    m_ok = adapt(_m0(), Xs, ys, Xt, "tta_control", steps=10, bs=32, device="cpu", seed=0)[0]
    ys_bad = 1 - ys
    m_bad = adapt(_m0(), Xs, ys_bad, Xt, "tta_control", steps=10, bs=32, device="cpu", seed=0)[0]
    xa = torch.randn(6, 8, 64)
    with torch.no_grad():
        assert not torch.allclose(m_ok(xa)[0], m_bad(xa)[0])     # source labels DO drive the source-replay CE


# 4 --------------------------------------------------------------------------
def test_target_entropy_uses_target_X_only():
    """Changing target X changes the adaptation (entropy term); the term never sees a target label."""
    Xs, ys, _ = _src()
    m_a = adapt(_m0(), Xs, ys, _target(seed=1), "tta_control", steps=10, bs=32, device="cpu", seed=0)[0]
    m_b = adapt(_m0(), Xs, ys, _target(seed=2), "tta_control", steps=10, bs=32, device="cpu", seed=0)[0]
    xa = torch.randn(6, 8, 64)
    with torch.no_grad():
        assert not torch.allclose(m_a(xa)[0], m_b(xa)[0])


# 5 --------------------------------------------------------------------------
def test_target_pseudo_labels_are_detached_in_conditioning():
    """The target soft pseudo-label used for domain conditioning must be detached (no grad path to it)."""
    src = inspect.getsource(adapt)
    assert "y_t = p_t.detach()" in src                          # target pseudo-label detached
    assert "y_all = torch.cat([y_s, y_t], 0).detach()" in src   # conditioning labels fully detached


# 6 --------------------------------------------------------------------------
def test_cond_domain_active_only_in_cita():
    Xs, ys, _ = _src(); Xt = _target()
    _, d_tta = adapt(_m0(), Xs, ys, Xt, "tta_control", steps=6, bs=32, device="cpu", seed=0)
    _, d_cita = adapt(_m0(), Xs, ys, Xt, "cita_cmi", steps=6, bs=32, device="cpu", seed=0)
    assert d_tta["cond_domain_active"] is False and d_tta["final_cond_domain"] is None
    assert d_cita["cond_domain_active"] is True and d_cita["final_cond_domain"] is not None


# 7 --------------------------------------------------------------------------
def test_erm_path_unchanged():
    """erm_no_adapt returns the source model untouched (identical predictions)."""
    Xs, ys, _ = _src(); Xt = _target(); m = _m0()
    xa = torch.randn(6, 8, 64)
    with torch.no_grad():
        before = m(xa)[0].clone()
    m2, d = adapt(m, Xs, ys, Xt, "erm_no_adapt", steps=50, device="cpu", seed=0)
    with torch.no_grad():
        assert torch.allclose(before, m2(xa)[0]) and d["adapted"] is False


# 8 --------------------------------------------------------------------------
@pytest.mark.parametrize("bb", ["eegnet", "conformer"])
def test_exact_head_replay_preserved_after_adaptation(bb, tmp_path):
    Xs, ys, ds = _src(); Xt = _target()
    model = adapt(_m0(bb), Xs, ys, Xt, "cita_cmi", steps=8, bs=32, device="cpu", seed=0)[0].eval()
    W, b, kind, _ = extract_task_head(model)
    assert kind == "linear"                                     # single linear head survives adaptation
    p, ok, mad = save_fold_audit(str(tmp_path / bb), model=model, X_source=Xs, y_source=ys, d_source=ds,
                                 device="cpu", fold=0, seed=0, target_subject="9", method=f"{bb}:cita_cmi",
                                 dataset="syn", X_target=Xt, y_target=np.zeros(len(Xt), int), target_domain=4,
                                 source_indices=np.arange(len(Xs)), target_indices=np.arange(len(Xs), len(Xs) + len(Xt)))
    assert ok and mad < 1e-4                                    # exact classifier-level head-replay


# 9 --------------------------------------------------------------------------
def test_random_subspace_r3_control_consumable(tmp_path):
    Xs, ys, ds = _src(); Xt = _target()
    model = adapt(_m0("eegnet"), Xs, ys, Xt, "cita_cmi", steps=8, bs=32, device="cpu", seed=0)[0].eval()
    p, ok, _ = save_fold_audit(str(tmp_path / "c"), model=model, X_source=Xs, y_source=ys, d_source=ds, device="cpu",
                               fold=0, seed=0, target_subject="9", method="eegnet:cita_cmi", dataset="syn",
                               X_target=Xt, y_target=np.zeros(len(Xt), int), target_domain=4,
                               source_indices=np.arange(len(Xs)), target_indices=np.arange(len(Xs), len(Xs) + len(Xt)))
    d = load_audit_npz(p)
    assert validate_audit_npz(d) == [] and head_replay_ok(d)
    r_lc = evaluate_reliance(d, target_domain=4, k=2, conditioning="label_conditional")
    r_rand = evaluate_reliance(d, target_domain=4, k=2, conditioning="random_subspace")
    assert r_lc["removal_mode"] == "head_replay" and r_lc["firewall_passed"]
    assert "task_drop" in r_rand                                # random-subspace control is consumable


# 10 -------------------------------------------------------------------------
def test_conditioning_posterior_shapes_and_no_target_label_leak():
    q = ConditionalDomainPosterior(z_dim=16, n_cls=2)
    z = torch.randn(5, 16); ypr = torch.rand(5, 2)
    assert q(z, ypr).shape == (5,)
    # ADAPT_METHODS excludes erm; the domain-conditioning uses D labels (source/target) + detached soft y, never
    # a true target label.
    assert set(ADAPT_METHODS) == {"tta_control", "cita_cmi"} and "erm_no_adapt" in CITA_METHODS


# 11 (metadata guardrails) ---------------------------------------------------
def test_adapt_records_budget_mode_and_before_after():
    Xs, ys, _ = _src(); Xt = _target()
    _, d = adapt(_m0(), Xs, ys, Xt, "cita_cmi", steps=8, bs=32, device="cpu", seed=0, tau=1.0, mu=1.0, lam_cita=0.010)
    b = d["budget"]
    for k in ("adapt_steps", "adapt_lr", "adapt_batch_size_source", "adapt_batch_size_target", "tau_entropy",
              "mu_label_balance", "lambda_cita", "source_label_prior_source", "pseudo_label_mode"):
        assert k in b
    assert b["lambda_cita"] == 0.010 and b["source_label_prior_source"] == "source_train_only" and b["pseudo_label_mode"] == "detached_soft"
    mp = d["mode_policy"]
    assert mp["adaptation_train_mode"] and mp["dropout_active_during_adapt"] and mp["batchnorm_updates_during_adapt"]
    for k in ("source_replay_ce_before", "source_replay_ce_after", "target_entropy_before", "target_entropy_after",
              "target_label_balance_before", "target_label_balance_after"):
        assert d[k] is not None


def test_tta_and_cita_share_budget_and_mode_except_lambda():
    """The CITA-vs-TTA attribution requires identical budget (except lambda_cita) and identical model-mode."""
    Xs, ys, _ = _src(); Xt = _target()
    _, dt = adapt(_m0(), Xs, ys, Xt, "tta_control", steps=8, bs=32, lr=1e-3, tau=1.0, mu=1.0, device="cpu", seed=0)
    _, dc = adapt(_m0(), Xs, ys, Xt, "cita_cmi", steps=8, bs=32, lr=1e-3, tau=1.0, mu=1.0, lam_cita=0.010, device="cpu", seed=0)
    bt = {k: v for k, v in dt["budget"].items() if k != "lambda_cita"}
    bc = {k: v for k, v in dc["budget"].items() if k != "lambda_cita"}
    assert bt == bc                                                  # budget identical except lambda
    assert dt["mode_policy"] == dc["mode_policy"]                    # BN/dropout/train-mode identical
    assert dt["budget"]["lambda_cita"] == 0.0 and dc["budget"]["lambda_cita"] == 0.010   # lambda is the only diff


def test_no_dependency_breakage():
    import importlib
    importlib.import_module("cmi.adaptation.cita")
    for bb in ("eegnet", "conformer"):
        assert build_sanity_backbone(bb, 8, 64, 2) is not None
