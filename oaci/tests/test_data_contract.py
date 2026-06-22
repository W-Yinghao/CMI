"""Real-EEG data contract: schema invariants, unit/mass separation, eval-unit aggregation,
strict split roles + seed separation, audit hashes, cache, offline guards, no cmi/h2cmi import.

Standalone (``python -m oaci.tests.test_data_contract``) and pytest-compatible. No real data.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile

import numpy as np

from oaci.data.eeg.audit import split_manifest_hash, tensor_hash
from oaci.data.eeg.cache import atomic_write_bytes, cache_key
from oaci.data.eeg.clinical_bids import scan_bids
from oaci.data.eeg.preprocess import PreprocessSpec, assert_fit_excludes_target
from oaci.data.eeg.registry import OfflineDownloadError, ensure_offline_available
from oaci.data.eeg.schema import EEGBundle
from oaci.data.eeg.splits import apply_missing_cell_mask, make_loso_split
from oaci.data.eeg.units import aggregate_mean_prob, base_mass, cell_mass, eligibility_counts
from oaci.eval.artifacts import PredictionBundle
from oaci.eval.sweep import assert_fixed_audit_population
from oaci.support_graph import build_support_graph
from oaci.leakage.estimate import reference_conditional_entropy


def _clinical(n_sites=2, subj_per_site=3, windows=5, C=4, T=8, seed=0):
    """Clinical-style: site=domain, subject=support=eval unit, many windows per subject."""
    rng = np.random.default_rng(seed)
    rows = dict(X=[], y=[], sid=[], site=[], subj=[], sess=[], run=[], rec=[], trial=[], su=[], eu=[])
    for s in range(n_sites):
        for j in range(subj_per_site):
            sub = f"site{s}_sub{j}"
            lab = (s + j) % 2                      # 2 classes (HC/PD)
            for w in range(windows):
                rows["X"].append(rng.standard_normal((C, T)))
                rows["y"].append(lab); rows["sid"].append(f"{sub}_w{w}")
                rows["site"].append(f"site{s}"); rows["subj"].append(sub)
                rows["sess"].append("0"); rows["run"].append("0"); rows["rec"].append(sub)
                rows["trial"].append(f"{sub}_t{w}"); rows["su"].append(sub); rows["eu"].append(sub)
    return EEGBundle(
        X=np.array(rows["X"]), y=np.array(rows["y"]), sample_id=np.array(rows["sid"], dtype=object),
        dataset_id="synthetic_clin", site_id=np.array(rows["site"], dtype=object),
        subject_id=np.array(rows["subj"], dtype=object), session_id=np.array(rows["sess"], dtype=object),
        run_id=np.array(rows["run"], dtype=object), recording_id=np.array(rows["rec"], dtype=object),
        trial_id=np.array(rows["trial"], dtype=object), support_unit_id=np.array(rows["su"], dtype=object),
        eval_unit_id=np.array(rows["eu"], dtype=object), sfreq=128.0, ch_names=[f"c{i}" for i in range(C)],
        class_names=["HC", "PD"],
    )


# ---- schema ----
def test_sample_ids_are_unique_and_stable():
    b = _clinical().validate()
    assert np.unique(b.sample_id).size == b.n
    assert all(isinstance(s, str) for s in b.sample_id)          # stable strings, not load-order ints


def test_group_is_nested_in_domain():
    b = _clinical().validate()
    for g in np.unique(b.recording_id):
        m = b.recording_id == g
        assert np.unique(b.site_id[m]).size == 1                 # a recording -> one site (domain)


def test_eval_unit_has_one_label():
    b = _clinical()
    b.y[0] = 1 - b.y[0]                                          # corrupt: a window's label flipped
    try:
        b.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("eval unit with two labels must be rejected")


# ---- units / mass ----
def test_support_gate_counts_unique_units_not_windows():
    b = _clinical(n_sites=2, subj_per_site=3, windows=10)
    dom = b.domain("site_id")
    nelig = eligibility_counts(dom, b.y, b.support_unit_id, 2, 2)
    # site0 subjects: sub0(y0),sub1(y1),sub2(y0) -> cell (0,0)=2 subjects, NOT 20 windows
    assert nelig.max() <= 3 and nelig[0, 0] == 2


def test_cell_mass_drives_reference_entropy():
    elig = np.array([[5, 5], [5, 5]])                            # uniform eligibility (would give ln2)
    mass = np.array([[3.0, 3.0], [1.0, 1.0]])                    # mass 3:1 -> p(d|y)=(.75,.25)
    sg = build_support_graph(elig, m=2, cell_mass=mass)
    H = reference_conditional_entropy(sg)
    exp = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    assert abs(H[0] - exp) < 1e-9
    assert abs(H[0] - math.log(2)) > 0.05                        # NOT the eligibility-uniform value


def test_subject_equal_mass_is_invariant_to_window_count():
    # subject A: 2 windows; subject B: 8 windows; same cell -> mass 2 (one per subject), not 10
    eu = np.array(["A", "A", "B", "B", "B", "B", "B", "B", "B", "B"], dtype=object)
    dom = np.zeros(10, int); y = np.zeros(10, int)
    M = cell_mass(dom, y, base_mass(eu), 1, 1)
    assert abs(M[0, 0] - 2.0) < 1e-12                            # 2 subjects, invariant to 2 vs 8 windows


def test_mean_probability_aggregation_matches_known_value():
    logits = np.array([[2.0, 0.0], [0.0, 2.0]])                 # one unit; softmaxes average to [.5,.5]
    units, agg, _ = aggregate_mean_prob(logits, np.array(["u", "u"], dtype=object))
    p = np.exp(agg[0] - agg[0].max()); p /= p.sum()
    assert np.allclose(p, [0.5, 0.5], atol=1e-9) and len(units) == 1


# ---- splits ----
def _toy_split_arrays():
    dom = np.array([0, 0, 1, 1, 2, 2])                           # 3 domains, 2 rows each
    subj = np.array(["s0", "s1", "s2", "s3", "s4", "s5"], dtype=object)
    return dom, subj


def test_split_roles_are_disjoint():
    dom, subj = _toy_split_arrays()
    sp = make_loso_split(dom, subj, target_domain=2, split_seed=0, mode="across_source_domains")
    assert sp.roles_disjoint()


def test_target_never_enters_fit_or_preprocessing_statistics():
    dom, subj = _toy_split_arrays()
    sp = make_loso_split(dom, subj, target_domain=2, split_seed=0, mode="across_source_domains")
    fit = set(sp.source_train.tolist()) | set(sp.source_audit.tolist())
    assert not (fit & set(sp.target_audit.tolist()))
    assert_fit_excludes_target(sp.source_train, sp.target_audit)   # preprocessing guard agrees


def test_split_and_deletion_do_not_depend_on_model_seed():
    dom, subj = _toy_split_arrays()
    # make_loso_split has NO model_seed parameter; same split_seed -> identical regardless of model
    a = make_loso_split(dom, subj, 2, split_seed=7, mode="across_source_domains")
    b = make_loso_split(dom, subj, 2, split_seed=7, mode="across_source_domains")
    assert np.array_equal(a.source_train, b.source_train) and np.array_equal(a.source_audit, b.source_audit)


def test_missing_cell_mask_changes_source_train_only():
    dom = np.array([0, 0, 1, 1, 2, 2]); y = np.array([0, 1, 0, 1, 0, 1])
    subj = np.array([f"s{i}" for i in range(6)], dtype=object)
    sp = make_loso_split(dom, subj, 2, split_seed=0, mode="across_source_domains")
    sp2 = apply_missing_cell_mask(sp, dom, y, deleted_cells={(0, 0)})
    assert np.array_equal(sp2.source_audit, sp.source_audit)         # audit unchanged
    assert np.array_equal(sp2.target_audit, sp.target_audit)         # target unchanged
    assert len(sp2.source_train) <= len(sp.source_train)
    for i in sp2.source_train:
        assert not (dom[i] == 0 and y[i] == 0)                       # deleted cell gone from train


def test_single_source_domain_fold_is_flagged_method_inactive():
    dom = np.array([0, 0, 1, 1]); subj = np.array(["a", "b", "c", "d"], dtype=object)
    sp = make_loso_split(dom, subj, target_domain=1, split_seed=0, mode="within_each_source_domain")
    assert sp.n_active_source_domains == 1 and sp.method_inactive    # one source domain -> no-op


def test_seed_reproducibility():
    # within-site clinical split with several subjects per site (so an audit subject is actually held)
    dom = np.repeat([0, 1, 2], 4)
    subj = np.array([f"site{d}_sub{i}" for d in range(3) for i in range(4)], dtype=object)
    a = make_loso_split(dom, subj, 2, split_seed=5, mode="within_each_source_domain")
    b = make_loso_split(dom, subj, 2, split_seed=5, mode="within_each_source_domain")
    c = make_loso_split(dom, subj, 2, split_seed=6, mode="within_each_source_domain")
    assert np.array_equal(a.source_audit, b.source_audit) and a.audit_units == b.audit_units
    assert a.audit_units != c.audit_units or len(a.audit_units) == 0


def test_mi_audit_split_holds_out_complete_source_domains():
    # MI: domain==subject; across mode audits WHOLE source subjects and keeps >=2 train domains
    dom = np.arange(6).repeat(3)                                  # 6 subjects (domains), 3 trials each
    subj = np.array([f"s{d}" for d in range(6) for _ in range(3)], dtype=object)
    sp = make_loso_split(dom, subj, target_domain=5, split_seed=0, mode="across_source_domains")
    assert sp.source_audit.size > 0                              # NOT degenerate-empty
    audit_doms = set(dom[sp.source_audit].tolist())
    train_doms = set(dom[sp.source_train].tolist())
    assert audit_doms and not (audit_doms & train_doms)         # whole-domain audit, disjoint from train
    assert len(train_doms) >= 2


def test_clinical_audit_split_holds_subjects_within_each_site():
    dom = np.repeat([0, 1, 2], 6)                                # 3 sites, 3 subjects/site, 2 rows each
    subj = np.array([f"site{d}_sub{i}" for d in range(3) for i in range(3) for _ in range(2)], dtype=object)
    sp = make_loso_split(dom, subj, target_domain=2, split_seed=0, mode="within_each_source_domain")
    for dd in (0, 1):
        site_subj = set(subj[sp.source_train][dom[sp.source_train] == dd].tolist())
        assert len(site_subj) >= 1                               # each source site keeps a TRAIN subject
    assert set(dom[sp.source_train].tolist()) == {0, 1}         # both sites still trained on


def test_source_audit_has_two_domains_when_feasible():
    dom = np.arange(6).repeat(3); subj = np.array([f"s{d}" for d in range(6) for _ in range(3)], dtype=object)
    sp = make_loso_split(dom, subj, target_domain=5, split_seed=1, mode="across_source_domains", audit_frac=0.5)
    assert sp.n_audit_domains >= 2 and sp.source_audit_estimable


def test_same_local_subject_id_in_two_sites_does_not_collide():
    # 'sub01' appears in both sites; composite (site,subject) must not chain them
    dom = np.array([0, 0, 1, 1]); subj = np.array(["sub01", "sub02", "sub01", "sub02"], dtype=object)
    sp = make_loso_split(dom, subj, target_domain=1, split_seed=0, mode="within_each_source_domain")
    # only site-0 'sub01' could be audited; site-1 'sub01' is in the (target) fold -> never confused
    assert all(dom[i] == 0 for i in sp.source_audit)


def test_method_activity_depends_on_comparable_support():
    from oaci.methods import method_activity
    # 1 source domain -> OACI inactive (no comparable class) AND alignment baselines inactive
    sg1 = build_support_graph(np.array([[5, 5]]), m=2)            # single domain
    act1 = method_activity(sg1, n_source_domains=1)
    assert act1["ERM"]["active"] and not act1["OACI"]["active"] and not act1["global_lpc"]["active"]
    # 2 domains both eligible -> all active
    sg2 = build_support_graph(np.array([[5, 5], [5, 5]]), m=2)
    act2 = method_activity(sg2, n_source_domains=2)
    assert act2["OACI"]["active"] and act2["global_lpc"]["active"] and act2["uniform"]["active"]


# ---- audit hashes ----
def test_audit_tensor_hash_is_fixed_across_methods_and_levels():
    sid = np.arange(8); y = np.array([0, 1] * 4); dom = np.array([0, 0, 0, 0, 1, 1, 1, 1]); grp = dom
    th = tensor_hash(np.ones((8, 3)))

    def pb(method, level):
        return PredictionBundle(sid, np.zeros((8, 2)), y, dom, grp, method, 0, "s", "target_audit",
                                level, ["A", "B"], audit_tensor_hash=th)
    assert_fixed_audit_population({0: {"ERM": pb("ERM", 0), "OACI": pb("OACI", 0)},
                                   1: {"ERM": pb("ERM", 1), "OACI": pb("OACI", 1)}})
    bad = pb("ERM", 1); bad.audit_tensor_hash = "DIFFERENT"
    try:
        assert_fixed_audit_population({0: {"ERM": pb("ERM", 0)}, 1: {"ERM": bad}})
    except ValueError:
        pass
    else:
        raise AssertionError("a changed audit tensor hash must be rejected")


# ---- cache ----
def test_cache_key_changes_with_channel_order_or_preprocessing():
    spec = PreprocessSpec().to_dict()
    base = cache_key("fp1", ["Cz", "C3", "C4"], spec, "v1")
    assert cache_key("fp1", ["C3", "Cz", "C4"], spec, "v1") != base    # channel ORDER matters
    spec2 = PreprocessSpec(l_freq=1.0).to_dict()
    assert cache_key("fp1", ["Cz", "C3", "C4"], spec2, "v1") != base    # preprocessing matters
    assert cache_key("fp1", ["Cz", "C3", "C4"], spec, "v1") == base     # reproducible


def test_cache_write_is_atomic():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "x.bin")
    atomic_write_bytes(p, b"hello-cache")
    assert open(p, "rb").read() == b"hello-cache"
    assert not any(f.startswith(".tmp-cache-") for f in os.listdir(d))   # no leftover temp file


# ---- offline guards ----
def test_offline_loader_never_downloads():
    try:
        ensure_offline_available("/no/such/offline/path")
    except OfflineDownloadError:
        pass
    else:
        raise AssertionError("missing offline data must raise, never download")
    assert scan_bids("/no/such/bids")["available"] is False


def test_no_oaci_runtime_import_from_cmi_or_h2cmi():
    code = (
        "import oaci, oaci.data.eeg, oaci.protocol, oaci.eval, oaci.train, oaci.leakage, "
        "oaci.data, oaci.support_graph, sys; "
        "bad=[m for m in sys.modules if m in ('cmi','h2cmi') or m.startswith('cmi.') "
        "or m.startswith('h2cmi.')]; "
        "assert not bad, bad; print('clean')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "clean" in out.stdout, out.stderr


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} data-contract tests")


if __name__ == "__main__":
    _run_all()
