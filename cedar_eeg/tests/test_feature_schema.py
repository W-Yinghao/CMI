from __future__ import annotations

import numpy as np

from cedar_eeg.data.feature_inventory import inventory_paths
from cedar_eeg.data.load_frozen_features import (
    FrozenFeatureSchemaError,
    load_frozen_feature_npz,
    verify_manifest_immutability,
)
from cedar_eeg.probes.crossfit_grouped import make_folds
from cedar_eeg.runners.run_01f_source_erm_feature_dump import build_plan


def _write_npz(path, *, groups=None, y=None, z=None, role=None, sample_id=None):
    n = 12
    if z is None:
        z = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    if y is None:
        y = np.tile([0, 1], n // 2)
    domain = np.repeat(["s0", "s1", "s2"], 4)
    if groups is None:
        groups = np.repeat(["r0", "r1", "r2", "r3"], 3)
    if role is None:
        role = np.array(["source_train"] * 8 + ["target_audit"] * 4)
    payload = {
        "z": z,
        "y": y,
        "domain": domain,
        "groups": np.asarray(groups),
        "role": np.asarray(role),
        "dataset": np.array("BNCI2014_001"),
        "backbone": np.array("EEGNetMini"),
        "seed": np.array(0),
    }
    if sample_id is not None:
        payload["sample_id"] = np.asarray(sample_id)
    np.savez(path, **payload)


def test_feature_loader_accepts_compliant_npz(tmp_path):
    path = tmp_path / "features.npz"
    _write_npz(path)
    bundle = load_frozen_feature_npz(path)
    full = bundle.diagnostic_full_view()
    source = bundle.source_selection_view()
    assert full["z"].shape == (12, 3)
    assert source["z"].shape[0] == 8
    assert "role" not in source
    assert bundle.metadata["dataset"] == "BNCI2014_001"


def test_feature_loader_hard_fails_missing_groups(tmp_path):
    path = tmp_path / "missing_groups.npz"
    n = 12
    np.savez(
        path,
        z=np.ones((n, 3), dtype=np.float32),
        y=np.tile([0, 1], n // 2),
        domain=np.repeat(["s0", "s1", "s2"], 4),
        role=np.array(["source_train"] * n),
    )
    try:
        load_frozen_feature_npz(path)
    except FrozenFeatureSchemaError:
        return
    raise AssertionError("missing groups must hard-fail")


def test_feature_loader_hard_fails_len_mismatch(tmp_path):
    path = tmp_path / "bad_len.npz"
    _write_npz(path, y=np.array([0, 1]))
    try:
        load_frozen_feature_npz(path)
    except FrozenFeatureSchemaError:
        return
    raise AssertionError("length mismatch must hard-fail")


def test_feature_loader_hard_fails_nan_z(tmp_path):
    path = tmp_path / "nan_z.npz"
    z = np.ones((12, 3), dtype=np.float32)
    z[0, 0] = np.nan
    _write_npz(path, z=z)
    try:
        load_frozen_feature_npz(path)
    except FrozenFeatureSchemaError:
        return
    raise AssertionError("NaN z must hard-fail")


def test_source_selection_view_quarantines_target_rows(tmp_path):
    path = tmp_path / "features.npz"
    role = np.array(["source_audit"] * 6 + ["target_audit"] * 6)
    _write_npz(path, role=role)
    bundle = load_frozen_feature_npz(path)
    source = bundle.source_selection_view()
    assert source["z"].shape[0] == 6
    assert source["y"].shape[0] == 6
    assert "target" not in source
    assert "role" not in source


def test_grouped_folds_are_disjoint_for_loaded_groups(tmp_path):
    path = tmp_path / "features.npz"
    _write_npz(path)
    bundle = load_frozen_feature_npz(path)
    source = bundle.source_selection_view()
    folds = make_folds(len(source["groups"]), groups=source["groups"], n_splits=2, seed=1)
    assert folds
    for tr, ev in folds:
        assert set(source["groups"][tr]).isdisjoint(set(source["groups"][ev]))


def test_manifest_immutability_detects_changed_dump(tmp_path):
    path = tmp_path / "features.npz"
    _write_npz(path)
    bundle = load_frozen_feature_npz(path)
    verify_manifest_immutability(path, bundle.metadata["file_sha256"])
    _write_npz(path, z=np.ones((12, 3), dtype=np.float32))
    try:
        verify_manifest_immutability(path, bundle.metadata["file_sha256"])
    except FrozenFeatureSchemaError:
        return
    raise AssertionError("changed dump must fail manifest immutability")


def test_inventory_marks_legacy_split_as_adapter_possible(tmp_path):
    path = tmp_path / "audit_PD_ds004584_erm_0.npz"
    np.savez(
        path,
        z_se=np.ones((8, 4), dtype=np.float32),
        y_se=np.tile([0, 1], 4),
        subject_id_se=np.repeat(["s0", "s1"], 4),
        group_id_se=np.repeat(["r0", "r1", "r2", "r3"], 2),
    )
    records = inventory_paths([path], include_archive=True)
    assert records[0].status == "ADAPTER_POSSIBLE"
    assert records[0].has_groups
    assert records[0].has_subject


def test_route_c_plan_freezes_feature_supply_without_selection(tmp_path):
    class Args:
        dataset = "BNCI2014_001"
        backbones = ["EEGNetMini", "EEGConformerMini"]
        seed = 0
        target_subjects = ["1", "2"]
        out_dir = str(tmp_path)
        source_audit_fraction = 0.2

    plan = build_plan(Args())
    assert plan["phase"] == "CEDAR_01F_feature_supply_route_c"
    assert plan["selection_run"] is False
    assert plan["scientific_readout_run"] is False
    assert plan["deployable"] is False
    assert len(plan["items"]) == 4
    assert {x["backbone"] for x in plan["items"]} == {"EEGNetMini", "EEGConformerMini"}
    assert plan["plan_hash"]
