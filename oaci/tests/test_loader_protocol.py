"""MOABB loader semantics (confirmatory-grade IDs / class map / fingerprint / channels /
normalization), canonical-SEED identity, protocol v2 (heterogeneous + smoke rejection), and the
self-deriving confirmatory gate. No real data (loader helpers are pure functions).

Standalone (``python -m oaci.tests.test_loader_protocol``) and pytest-compatible.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

from oaci.data.eeg.moabb import (
    _data_paths,
    map_classes,
    paradigm_kwargs,
    raw_file_fingerprint,
    resolve_channels,
    scan_subject_channels,
    subject_logical_paths,
    trial_ids,
    validate_channel_order,
    validate_epoch_n_times,
)
from oaci.data.eeg.preprocess import PreprocessSpec
from oaci.data.eeg.registry import OfflineDownloadError
from oaci.protocol.manifest_v2 import DatasetBlock
from oaci.data.eeg.preprocess import PreprocessSpec, apply_normalization
from oaci.data.eeg.seed import scan_seed
from oaci.protocol.confirmatory import RunEvidence, _git_tree_clean, collect_evidence, confirmatory_refusals
from oaci.protocol.freeze import default_confirmatory_path, freeze, load_yaml_manifest
from oaci.protocol.manifest_v2 import load_v2


# ---- blocker 2: loader semantics ----
def test_moabb_trial_ids_stable_across_subject_subset_and_order():
    def ids(subjs):  # build meta for the given subject ORDER, two recordings each
        s, se, r = [], [], []
        for sub in subjs:
            for run in ("0", "1"):
                for _ in range(3):
                    s.append(sub); se.append("T"); r.append(run)
        _, t = trial_ids("BNCI", s, se, r)
        return {sub: [x for x in t if f"|s{sub}|" in x] for sub in subjs}
    both = ids(["1", "2"]); rev = ids(["2", "1"]); one = ids(["1"])
    assert both["1"] == rev["1"] == one["1"]                     # subject 1's IDs independent of the set/order


def test_class_map_comes_from_registry_not_loaded_subset():
    frozen = ["left_hand", "right_hand", "feet", "tongue"]
    idx = map_classes(np.array(["feet", "left_hand"], dtype=object), frozen)
    assert idx.tolist() == [2, 0]                                # frozen order, not subset sort (->{feet:0,left:1})
    try:
        map_classes(np.array(["unknown"], dtype=object), frozen)
    except ValueError:
        pass
    else:
        raise AssertionError("an out-of-map label must be rejected")


def test_raw_fingerprint_tracks_raw_bytes_not_preprocess_spec():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "raw.fif"); open(p, "wb").write(b"x" * 100)
    fp1 = raw_file_fingerprint([p])
    open(p, "wb").write(b"x" * 200)                              # raw bytes change -> fingerprint changes
    assert raw_file_fingerprint([p]) != fp1
    # fingerprint takes no PreprocessSpec -> preprocessing cannot move it (separated hashes)
    assert PreprocessSpec().hash() != raw_file_fingerprint([p])


def test_declared_sample_normalization_is_actually_applied():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 4, 64)).astype(np.float32) * 7 + 3
    Z = apply_normalization(X, None, PreprocessSpec(normalization="zscore_sample"))
    assert np.allclose(Z.mean(axis=2), 0, atol=1e-5) and np.allclose(Z.std(axis=2), 1, atol=1e-3)


def test_confirmatory_moabb_loader_rejects_generic_channel_fallback():
    assert resolve_channels(["Cz", "C3"], 2, confirmatory=True) == ["Cz", "C3"]
    for bad in (None, ["Cz"]):                                   # unreadable / inconsistent
        try:
            resolve_channels(bad, 2, confirmatory=True)
        except ValueError:
            pass
        else:
            raise AssertionError("confirmatory mode must refuse a generic channel fallback")
    assert resolve_channels(None, 2, confirmatory=False) == ["ch0", "ch1"]   # non-confirmatory fallback allowed


def test_seed_and_seed_v_are_not_aliases():
    assert scan_seed("/x/SEED-V")["available"] is False         # SEED-V is NOT canonical SEED
    assert scan_seed("/no/such/SEED")["is_canonical_seed"] is True and scan_seed("/no/such/SEED")["available"] is False


def test_eval_and_support_units_are_globally_namespaced():
    _, a = trial_ids("dsA", ["1"], ["0"], ["0"])
    _, b = trial_ids("dsB", ["1"], ["0"], ["0"])
    assert set(a.tolist()).isdisjoint(set(b.tolist()))          # dataset-prefixed -> no cross-dataset collision


# ---- blocker 3: protocol v2 ----
def test_smoke_manifest_is_rejected_in_confirmatory_mode():
    here = os.path.dirname(default_confirmatory_path())
    smoke = load_v2(os.path.join(here, "smoke_v1.yaml"))
    assert smoke.status == "smoke"
    try:
        smoke.assert_confirmatory()
    except ValueError:
        pass
    else:
        raise AssertionError("a status='smoke' manifest must be rejected in confirmatory mode")
    # the v2 confirmatory manifest is heterogeneous (per-dataset blocks) and freezes
    conf = load_v2(os.path.join(here, "confirmatory_v2.yaml"))
    assert "BNCI2014_001" in conf.datasets and conf.datasets["BNCI2014_001"].support_m == 8
    assert conf.datasets["BNCI2014_001"].domain_factor != conf.datasets.get("PD_cross_site").domain_factor
    assert len(conf.freeze()["sha256"]) == 64


def test_v1_manifest_still_freezes():
    fr = freeze(load_yaml_manifest(default_confirmatory_path()))
    assert len(fr["sha256"]) == 64


# ---- blocker 6: self-deriving gate ----
def test_confirmatory_gate_derives_evidence_internally():
    # target-in-fit is DERIVED from the actual id sets, not trusted from a caller boolean
    ev = collect_evidence(repo_dir="/no/such/repo", manifest_frozen=True, expected_manifest_sha="a",
                          actual_manifest_sha="a", fit_sample_ids=["t1", "x2"], target_sample_ids=["t1"],
                          expected_cache_fingerprint="c", actual_cache_fingerprint="c",
                          n_active_source_domains=3)
    assert ev.target_in_fit is True
    assert "target appears in a fit statistic" in confirmatory_refusals(ev)
    # git tree cleanliness is computed, not asserted by the caller
    d = tempfile.mkdtemp()
    subprocess.run(["git", "-C", d, "init", "-q"], check=True)
    subprocess.run(["git", "-C", d, "config", "user.email", "x@x"], check=True)
    subprocess.run(["git", "-C", d, "config", "user.name", "x"], check=True)
    open(os.path.join(d, "f"), "w").write("a")
    subprocess.run(["git", "-C", d, "add", "-A"], check=True)
    subprocess.run(["git", "-C", d, "commit", "-qm", "init"], check=True)
    assert _git_tree_clean(d, ".") is True
    open(os.path.join(d, "f"), "w").write("b")                   # dirty
    assert _git_tree_clean(d, ".") is False


def test_moabb_applies_tmin_tmax_events_and_frozen_channels():
    spec = PreprocessSpec(l_freq=4.0, h_freq=38.0, resample_sfreq=128.0,
                          epoch_tmin=0.5, epoch_tmax=3.5, channels=["Fz", "C3", "Cz"])
    classes = ["left_hand", "right_hand", "feet", "tongue"]
    kw = paradigm_kwargs(spec, classes)
    assert kw["tmin"] == 0.5 and kw["tmax"] == 3.5 and kw["channels"] == ["Fz", "C3", "Cz"]
    assert kw["events"] == classes and kw["n_classes"] == 4
    assert validate_epoch_n_times(385, 128, 0.5, 3.5, tol=1) == 384      # 3.0 s @128 Hz, ±1
    try:
        validate_epoch_n_times(420, 128, 0.5, 3.5, tol=1)
    except ValueError:
        pass
    else:
        raise AssertionError("a wrong epoch length must be rejected")


def test_all_selected_subjects_share_exact_channel_order():
    assert validate_channel_order(["Fz", "C3", "Cz"], ["Fz", "C3", "Cz"]) is True
    try:
        validate_channel_order(["C3", "Fz", "Cz"], ["Fz", "C3", "Cz"])    # reordered -> reject
    except ValueError:
        pass
    else:
        raise AssertionError("a different channel order must be rejected in confirmatory mode")


def test_manifest_rejects_channel_token_in_frozen_smoke():
    common = dict(enabled=True, cohort_ids=["x"], class_names=["a", "b"], outer_target_factor="subject_id",
                  domain_factor="subject_id", group_factor="recording_id", support_unit_factor="trial_id",
                  eval_unit_factor="trial_id", support_m=8, preprocessing={"fmin": 4.0})
    bad = DatasetBlock(channels="frozen_common_native", **common)        # a TOKEN, not a list
    assert any("channels" in m for m in bad.missing())
    ok = DatasetBlock(channels=["Fz", "C3"], **common)
    assert ok.missing() == []


def test_raw_fingerprint_distinguishes_same_basename_files():
    d = tempfile.mkdtemp()
    a, b = os.path.join(d, "a_r.fif"), os.path.join(d, "b_r.fif")
    open(a, "wb").write(b"AAAA"); open(b, "wb").write(b"BBBB")          # same basename pattern, diff bytes
    lp = ["sub-01/r.fif", "sub-02/r.fif"]
    fp = raw_file_fingerprint([a, b], logical_paths=lp)
    fp_swap = raw_file_fingerprint([b, a], logical_paths=lp)            # which logical file holds which bytes
    assert fp != fp_swap                                               # logical identity is bound to content
    # mount-root stability: same logical paths + same bytes at a different root -> same fingerprint
    d2 = tempfile.mkdtemp()
    a2, b2 = os.path.join(d2, "a_r.fif"), os.path.join(d2, "b_r.fif")
    open(a2, "wb").write(b"AAAA"); open(b2, "wb").write(b"BBBB")
    assert raw_file_fingerprint([a2, b2], logical_paths=lp) == fp


class _FakeRaw:
    def __init__(self, chs): self._chs = list(chs)
    def copy(self): return self
    def pick(self, kind): return self
    @property
    def ch_names(self): return self._chs


class _FakeDS:
    def __init__(self, ch_by_subj=None, fail=()):
        self.ch_by_subj = ch_by_subj or {}
        self.fail = set(fail)
    def data_path(self, s):
        if s in self.fail:
            raise RuntimeError("file absent")
        return [f"/d/s{s}/a.gdf"]
    def get_data(self, subjects=None):
        s = subjects[0]
        if s in self.fail:
            raise RuntimeError("header absent")
        return {s: {"0": {"0": _FakeRaw(self.ch_by_subj.get(s, ["Fz", "C3"]))}}}


def test_loader_scans_all_subjects_and_binds_logical_paths():
    assert subject_logical_paths("BNCI2014_001", 4, ["/x/a.gdf", "/y/a.gdf"]) == \
        ["BNCI2014_001/subject-4/0/a.gdf", "BNCI2014_001/subject-4/1/a.gdf"]
    # the loader requires an IDENTICAL channel order across ALL subjects, not just the first
    assert scan_subject_channels(_FakeDS({1: ["Fz", "C3"], 2: ["Fz", "C3"]}), [1, 2]) == ["Fz", "C3"]
    try:
        scan_subject_channels(_FakeDS({1: ["Fz", "C3"], 2: ["C3", "Fz"]}), [1, 2])   # subject 2 differs
    except ValueError:
        pass
    else:
        raise AssertionError("a cross-subject channel mismatch must fail")
    # _data_paths must NOT swallow a missing subject in confirmatory mode
    miss = _FakeDS({1: ["Fz"]}, fail={2})
    try:
        _data_paths(miss, "BNCI2014_001", [1, 2], confirmatory=True)
    except OfflineDownloadError:
        pass
    else:
        raise AssertionError("a missing data_path must hard-fail in confirmatory mode")
    paths, logical = _data_paths(miss, "BNCI2014_001", [1, 2], confirmatory=False)     # non-conf: skip
    assert len(paths) == 1 and logical == ["BNCI2014_001/subject-1/0/a.gdf"]


def _smoke_path():
    return os.path.join(os.path.dirname(default_confirmatory_path()), "smoke_v1.yaml")


def test_smoke_manifest_roundtrip_preserves_all_blocks():
    m = load_v2(_smoke_path())
    for blk in ("seeds", "backbone", "optimizer", "training", "sampler", "probe", "smoke", "methods"):
        assert getattr(m, blk) is not None, f"smoke manifest dropped block {blk}"
    assert m.backbone.name == "shallow_convnet" and m.training.stage2_bn_mode == "frozen_erm_running_stats"
    assert m.smoke.deleted_cell_level1 == {"domain_subject": 4, "class": "feet"}
    canon = m.to_canonical_json()
    for key in ("lr_encoder", "guard_chunk_size", "selection_score_tolerance", "deleted_cell_level1"):
        assert key in canon                                            # blocks are folded into the hash input


def test_manifest_hash_binds_training_and_probe_blocks():
    base = load_v2(_smoke_path()).freeze()["sha256"]
    m = load_v2(_smoke_path()); m.training.steps_per_epoch += 1         # a training change...
    assert m.freeze()["sha256"] != base                                # ...moves the manifest hash
    m = load_v2(_smoke_path()); m.probe.folds += 1                      # a probe change...
    assert m.freeze()["sha256"] != base
    m = load_v2(_smoke_path()); m.optimizer.lr_encoder *= 2            # a learning-rate change...
    assert m.freeze()["sha256"] != base


def test_manifest_rejects_unknown_or_misspelled_scientific_key():
    d = tempfile.mkdtemp(); p = os.path.join(d, "typo.yaml")
    open(p, "w").write("protocol_id: t\nstatus: smoke\ntraining: {stage1_epoch: 5}\n")   # stage1_epoch[s]
    try:
        load_v2(p)
    except ValueError as e:
        assert "stage1_epoch" in str(e)
    else:
        raise AssertionError("a misspelled scientific key must be rejected, not silently filtered")


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} loader/protocol tests")


if __name__ == "__main__":
    _run_all()
