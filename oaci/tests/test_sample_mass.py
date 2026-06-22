"""Sample-mass full-chain propagation: every estimand (task risk, domain CE, reference entropy,
probe fit, OOF NLL, extractable_LQ_ov, bootstrap) is invariant to duplicating a window inside a
mass unit and splitting its base mass. Plus the unit/mass contract and the raw-fingerprint fix.

Standalone (``python -m oaci.tests.test_sample_mass``) and pytest-compatible.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

from oaci.config import SamplerConfig
from oaci.data.batch import effective_prior_domain_given_y, effective_prior_y, fixed_prior_domain_given_y
from oaci.data.eeg.moabb import raw_file_fingerprint
from oaci.data.eeg.units import base_mass, cell_mass, eligibility_counts
from oaci.data.sampler import RareCellSampler
from oaci.leakage.critic import CriticConfig, DomainProbe
from oaci.leakage.crossfit import FrozenFeatures, make_fold_plan, oof_nll_by_class
from oaci.leakage.estimate import estimate_extractable_leakage
from oaci.leakage.ucb import _group_to_rows, _rebuild
from oaci.support_graph import build_support_graph
from oaci.train.adversary import ConditionalDomainAdversary
from oaci.train.risk import balanced_ce, balanced_error, source_risk, task_ce


def _base(seed=0, per_cell=6, dim=4):
    """2 domains x 2 classes; each (d,y) has `per_cell` single-window units; group == unit."""
    rng = np.random.default_rng(seed)
    Z, y, d, g, b, unit = [], [], [], [], [], []
    uid = 0
    for dom in (0, 1):
        for cls in (0, 1):
            for _ in range(per_cell):
                Z.append(rng.standard_normal(dim)); y.append(cls); d.append(dom)
                g.append(uid); b.append(1.0); unit.append(f"u{uid}"); uid += 1
    return (np.array(Z), np.array(y), np.array(d), np.array(g), np.array(b, float),
            np.array(unit, dtype=object))


def _dup(Z, y, d, g, b, unit, which="u0"):
    """Duplicate one window of unit `which`, splitting its base mass between the two copies."""
    i = int(np.where(unit == which)[0][0])
    Z2 = np.vstack([Z, Z[i:i + 1]]); y2 = np.append(y, y[i]); d2 = np.append(d, d[i])
    g2 = np.append(g, g[i]); u2 = np.append(unit, unit[i])
    b2 = b.copy(); b2[i] /= 2; b2 = np.append(b2, b2[i])
    return Z2, y2, d2, g2, b2, u2


def _sg(y, d, b, unit):
    nelig = eligibility_counts(d, y, unit, 2, 2)
    M = cell_mass(d, y, b, 2, 2)
    return build_support_graph(nelig, m=2, cell_mass=M)


# ---- contract ----
def test_fixed_prior_uses_cell_mass_not_eligibility_counts():
    sg = build_support_graph(np.array([[5, 5], [5, 5]]), m=2, cell_mass=np.array([[3.0, 3.0], [1.0, 1.0]]))
    fp = fixed_prior_domain_given_y(sg)
    assert np.allclose(fp[0][1], [0.75, 0.25])                  # from MASS 3:1, not eligibility 5:5


def test_unit_mass_sums_to_one():
    eu = np.array(["A", "A", "B"], dtype=object); b = base_mass(eu)
    assert abs(b[eu == "A"].sum() - 1) < 1e-12 and abs(b[eu == "B"].sum() - 1) < 1e-12


def test_mass_unit_must_be_nested_in_one_cell_and_group():
    Z, y, d, g, b, unit = _base()
    unit2 = unit.copy(); unit2[0] = unit2[6]                    # merge a unit across two cells
    sg = _sg(y, d, b, unit)
    try:
        RareCellSampler(y, d, g, sg, SamplerConfig(min_per_eligible_cell=2), sample_mass=b, mass_unit_id=unit2)
    except ValueError:
        pass
    else:
        raise AssertionError("a unit spanning two cells/groups must fail")


def test_sampler_cell_mass_mismatch_fails_loudly():
    Z, y, d, g, b, unit = _base()
    sg = _sg(y, d, b, unit)
    bad = b.copy(); bad[0] += 0.5                                # break cell-mass / unit-mass consistency
    try:
        RareCellSampler(y, d, g, sg, SamplerConfig(min_per_eligible_cell=2), sample_mass=bad, mass_unit_id=unit)
    except ValueError:
        pass
    else:
        raise AssertionError("inconsistent cell mass must fail loudly")


# ---- risk invariance ----
def _logits(y, seed=0):
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((len(y), 2))
    return torch.tensor(L, dtype=torch.float64)


def test_task_ce_is_invariant_to_window_duplication():
    Z, y, d, g, b, unit = _base()
    L = _logits(y); yt = torch.tensor(y)
    r1 = source_risk(L, yt, "ce", 2, weight=torch.tensor(b)).item()
    i = 0
    L2 = torch.vstack([L, L[i:i + 1]]); y2 = torch.tensor(np.append(y, y[i]))
    b2 = b.copy(); b2[i] /= 2; b2 = np.append(b2, b2[i])
    r2 = source_risk(L2, y2, "ce", 2, weight=torch.tensor(b2)).item()
    assert abs(r1 - r2) < 1e-9


def test_balanced_ce_is_invariant_to_window_duplication():
    Z, y, d, g, b, unit = _base()
    L = _logits(y); yt = torch.tensor(y)
    r1 = balanced_ce(L, yt, 2, weight=torch.tensor(b)).item()
    i = 0
    L2 = torch.vstack([L, L[i:i + 1]]); y2 = torch.tensor(np.append(y, y[i]))
    b2 = b.copy(); b2[i] /= 2; b2 = np.append(b2, b2[i])
    r2 = balanced_ce(L2, y2, 2, weight=torch.tensor(b2)).item()
    assert abs(r1 - r2) < 1e-9


def test_balanced_error_is_mass_weighted():
    L = torch.tensor([[9.0, -9.0], [-9.0, 9.0], [-9.0, 9.0]])   # preds: 0,1,1 ; y: 0,0,0
    y = torch.tensor([0, 0, 0])
    # mass-weighted class-0 error = (0*1 + 1*5 + 1*4)/(1+5+4) with masses [1,5,4]
    e = balanced_error(L, y, 2, weight=torch.tensor([1.0, 5.0, 4.0]))
    assert abs(e - 0.9) < 1e-6                                  # (0·1 + 1·5 + 1·4)/10, float32


# ---- adversary invariance ----
def _adv(sg, dim=4, seed=0):
    torch.manual_seed(seed)
    return ConditionalDomainAdversary(dim, sg, hidden=0).double()


def test_adversary_ce_is_invariant_to_window_duplication():
    Z, y, d, g, b, unit = _base(); sg = _sg(y, d, b, unit); adv = _adv(sg)
    Zt = torch.tensor(Z)
    c1 = adv.domain_ce(Zt, y, d, importance_weight=torch.tensor(b)).item()
    Z2, y2, d2, g2, b2, u2 = _dup(Z, y, d, g, b, unit)
    c2 = adv.domain_ce(torch.tensor(Z2), y2, d2, importance_weight=torch.tensor(b2)).item()
    assert abs(c1 - c2) < 1e-8


def test_streamed_and_full_domain_ce_match_under_sample_mass():
    Z, y, d, g, b, unit = _base(); sg = _sg(y, d, b, unit); adv = _adv(sg)
    Zt = torch.tensor(Z)
    full = adv.domain_ce(Zt, y, d, importance_weight=torch.tensor(b)).item()
    # stream ALL eligible rows in chunks, each contribution normalised by fixed N_ov
    elig = np.where(np.isin(d, [0, 1]))[0]
    chunks = np.array_split(elig, 5)
    streamed = sum(adv.domain_ce_contribution(Zt[c], y[c], d[c], torch.tensor(b[c])) for c in chunks).item()
    assert abs(full - streamed) < 1e-8


# ---- sampler measure ----
def test_task_sampler_restores_unit_equal_measure():
    Z, y, d, g, b, unit = _base(per_cell=8); sg = _sg(y, d, b, unit)
    s = RareCellSampler(y, d, g, sg, SamplerConfig(min_per_eligible_cell=4, task_batch_size=8),
                        sample_mass=b, mass_unit_id=unit)
    tb = s.task_batch()
    ep = effective_prior_y(tb.idx, tb.weight, y, 2)
    assert np.allclose(ep, [0.5, 0.5], atol=1e-9)               # weighted -> true p(y)


def test_adversary_sampler_restores_unit_equal_measure():
    Z, y, d, g, b, unit = _base(per_cell=8); sg = _sg(y, d, b, unit)
    s = RareCellSampler(y, d, g, sg, SamplerConfig(min_per_eligible_cell=4), sample_mass=b, mass_unit_id=unit)
    lb = s.adv_logical_batch()
    wtd = effective_prior_domain_given_y(lb.idx, lb.weight, y, d, sg)
    fixed = fixed_prior_domain_given_y(sg)
    for yy in sg.comparable_classes:
        assert np.allclose(wtd[yy][1], fixed[yy][1], atol=1e-12)


def test_sampler_queue_operates_on_units_not_windows():
    # 2-window units: unit count, not window count, drives U_cell and the queue
    Z, y, d, g, b, unit = _base(per_cell=4)
    Z, y, d, g, b, unit = _dup(Z, y, d, g, b, unit, "u0")       # u0 now has 2 windows
    sg = _sg(y, d, b, unit)
    s = RareCellSampler(y, d, g, sg, SamplerConfig(min_per_eligible_cell=2), sample_mass=b, mass_unit_id=unit)
    assert s.U_cell[(0, 0)] == 4                                # 4 UNITS (not 5 windows)


def test_ineligible_units_remain_in_task_but_not_adversary():
    Z, y, d, g, b, unit = _base(per_cell=6)
    # add a tiny third domain unit (ineligible: below m)
    Z = np.vstack([Z, np.zeros((1, Z.shape[1]))]); y = np.append(y, 0); d = np.append(d, 2)
    g = np.append(g, 999); b = np.append(b, 1.0); unit = np.append(unit, "uX")
    nelig = eligibility_counts(d, y, unit, 3, 2); M = cell_mass(d, y, b, 3, 2)
    sg = build_support_graph(nelig, m=3, cell_mass=M)
    assert 2 not in sg.support_of_class[0]
    s = RareCellSampler(y, d, g, sg, SamplerConfig(min_per_eligible_cell=2, task_batch_size=400),
                        sample_mass=b, mass_unit_id=unit)
    for _ in range(10):
        assert np.sum(d[s.adv_logical_batch().idx] == 2) == 0   # never in adversary
    seen = set()
    for _ in range(40):
        seen |= set(d[s.task_batch().idx].tolist())
    assert 2 in seen                                            # present in task


# ---- leakage chain invariance ----
def test_fold_assignment_is_invariant_to_window_duplication():
    Z, y, d, g, b, unit = _base(); sg = _sg(y, d, b, unit)
    p1 = make_fold_plan(FrozenFeatures(Z, y, d, g, b), sg, n_folds=2, seed=0)
    Z2, y2, d2, g2, b2, u2 = _dup(Z, y, d, g, b, unit)
    p2 = make_fold_plan(FrozenFeatures(Z2, y2, d2, g2, b2), sg, n_folds=2, seed=0)
    assert p1.fold_of_group == p2.fold_of_group                 # mass-balanced -> identical plan


def test_weighted_probe_standardization_is_train_fold_only():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((20, 3)); lab = rng.integers(0, 2, 20); w = rng.uniform(0.5, 2, 20)
    pr = DomainProbe(0, 2, CriticConfig()).fit(Z, lab, sample_weight=w)
    assert np.allclose(pr._mean, (w[:, None] * Z).sum(0) / w.sum())   # weighted mean from THIS (train) set


def test_probe_fit_is_invariant_to_split_weight_duplicates():
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((30, 3)); lab = rng.integers(0, 2, 30); w = np.ones(30)
    p1 = DomainProbe(0, 2, CriticConfig()).fit(Z, lab, sample_weight=w)
    Z2 = np.vstack([Z, Z[0:1]]); lab2 = np.append(lab, lab[0]); w2 = w.copy(); w2[0] = 0.5; w2 = np.append(w2, 0.5)
    p2 = DomainProbe(0, 2, CriticConfig()).fit(Z2, lab2, sample_weight=w2)
    Zt = rng.standard_normal((8, 3))
    assert np.allclose(p1.predict_proba(Zt), p2.predict_proba(Zt), atol=1e-5)


def test_weighted_oof_nll_is_invariant_to_window_duplication():
    Z, y, d, g, b, unit = _base(per_cell=6); sg = _sg(y, d, b, unit)
    plan = make_fold_plan(FrozenFeatures(Z, y, d, g, b), sg, n_folds=2, seed=0)
    n1 = oof_nll_by_class(FrozenFeatures(Z, y, d, g, b), sg, plan, 0, CriticConfig(capacities=(0,)))
    Z2, y2, d2, g2, b2, u2 = _dup(Z, y, d, g, b, unit)
    plan2 = make_fold_plan(FrozenFeatures(Z2, y2, d2, g2, b2), sg, n_folds=2, seed=0)
    n2 = oof_nll_by_class(FrozenFeatures(Z2, y2, d2, g2, b2), sg, plan2, 0, CriticConfig(capacities=(0,)))
    for yy in sg.comparable_classes:
        assert abs(n1[yy]["nll"] - n2[yy]["nll"]) < 1e-5


def test_extractable_LQ_ov_is_invariant_to_window_duplication():
    Z, y, d, g, b, unit = _base(per_cell=8); sg = _sg(y, d, b, unit)
    cfg = CriticConfig(capacities=(0, 8))
    e1 = estimate_extractable_leakage(FrozenFeatures(Z, y, d, g, b),
                                      sg, make_fold_plan(FrozenFeatures(Z, y, d, g, b), sg, 2, 0), cfg)
    Z2, y2, d2, g2, b2, u2 = _dup(Z, y, d, g, b, unit)
    f2 = FrozenFeatures(Z2, y2, d2, g2, b2)
    e2 = estimate_extractable_leakage(f2, sg, make_fold_plan(f2, sg, 2, 0), cfg)
    assert abs(e1["extractable_LQ_ov"] - e2["extractable_LQ_ov"]) < 1e-5


# ---- bootstrap ----
def test_bootstrap_multiplicity_multiplies_base_mass():
    Z, y, d, g, b, unit = _base(per_cell=4)
    feat = FrozenFeatures(Z, y, d, g, b)
    rebuilt = _rebuild(feat, _group_to_rows(feat), ["0", "0", "1"])   # group "0" drawn twice (str id)
    rows0 = feat.group == "0"
    base_mass0 = float(feat.sample_mass[rows0].sum())
    rep_mass0 = float(rebuilt.sample_mass[rebuilt.group == "0"].sum())
    assert abs(rep_mass0 - 2 * base_mass0) < 1e-12              # multiplicity 2 -> 2x base mass


def test_bootstrap_replicates_are_invariant_to_window_duplication():
    Z, y, d, g, b, unit = _base(per_cell=6); sg = _sg(y, d, b, unit)
    cfg = CriticConfig(capacities=(0,))
    from oaci.leakage.ucb import bootstrap_ucb
    f1 = FrozenFeatures(Z, y, d, g, b)
    r1 = bootstrap_ucb(f1, sg, make_fold_plan(f1, sg, 2, 0), cfg, alpha=0.1, n_bootstrap=20, seed=0)
    Z2, y2, d2, g2, b2, u2 = _dup(Z, y, d, g, b, unit)
    f2 = FrozenFeatures(Z2, y2, d2, g2, b2)
    r2 = bootstrap_ucb(f2, sg, make_fold_plan(f2, sg, 2, 0), cfg, alpha=0.1, n_bootstrap=20, seed=0)
    assert abs(r1["extractable_LQ_ov"] - r2["extractable_LQ_ov"]) < 1e-5


# ---- raw fingerprint fix ----
def test_raw_fingerprint_changes_when_bytes_change_at_same_size():
    d = tempfile.mkdtemp(); p = os.path.join(d, "r.fif")
    open(p, "wb").write(b"a" * 100); fp1 = raw_file_fingerprint([p])
    open(p, "wb").write(b"b" * 100); fp2 = raw_file_fingerprint([p])   # SAME size, different bytes
    assert fp1 != fp2


def test_raw_fingerprint_is_stable_across_datalake_mount_roots():
    d1 = tempfile.mkdtemp(); d2 = tempfile.mkdtemp()
    for d in (d1, d2):
        open(os.path.join(d, "r.fif"), "wb").write(b"same-bytes")
    assert raw_file_fingerprint([os.path.join(d1, "r.fif")]) == raw_file_fingerprint([os.path.join(d2, "r.fif")])


def test_confirmatory_fingerprint_rejects_empty_or_unreadable_files():
    for arg in ([], ["/no/such/raw.fif"]):
        try:
            raw_file_fingerprint(arg, confirmatory=True)
        except ValueError:
            pass
        else:
            raise AssertionError("confirmatory fingerprint must reject empty/unreadable inputs")


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} sample-mass tests")


if __name__ == "__main__":
    _run_all()
