from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from oaci.conditioned_ceiling_coverage import c75_protocol


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_c75_protocol_hash_and_parent_are_locked():
    protocol = json.loads(c75_protocol.PROTOCOL_PATH.read_text())
    assert _sha256(c75_protocol.PROTOCOL_PATH) == c75_protocol.PROTOCOL_SHA_PATH.read_text().strip()
    assert protocol["parent_C74_result_commit"].startswith("fe467b9")
    assert protocol["execution_boundary"]["forward_passes"] is False
    assert protocol["execution_boundary"]["T3_HO_z_Wz_access"] is False
    assert protocol["data_role"]["T2_units"] == 216
    assert protocol["data_role"]["T3_HO_units"] == 1052


def test_c75_registered_blocks_are_low_dimensional_and_availability_labeled():
    rows = list(csv.DictReader(open(c75_protocol.TABLE_DIR / "feature_block_registry.csv")))
    assert [row["block"] for row in rows] == ["F0", "F1", "F2", "F3", "F4", "F5"]
    assert [int(row["dimension"]) for row in rows] == [9, 25, 25, 18, 35, 15]
    assert {row["block"] for row in rows if row["qualification_candidate"] == "1"} == {"F2", "F4"}
    availability = {row["block"]: row for row in csv.DictReader(open(c75_protocol.TABLE_DIR / "feature_availability_ledger.csv"))}
    assert availability["F2"]["available_strict_DG"] == "1"
    assert availability["F4"]["target_unlabeled"] == "1"
    assert availability["F5"]["target_label_derived"] == "1"


def test_c75_factorization_equivalence_is_exact_on_dummy_data():
    rng = np.random.default_rng(75)
    z = rng.normal(size=(64, 16))
    W = rng.normal(size=(4, 16))
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    scales = np.linspace(0.5, 2.0, 16)
    A = q @ np.diag(scales) @ q.T
    transformed_z = z @ A.T
    transformed_W = W @ np.linalg.inv(A)
    assert np.max(np.abs(transformed_z @ transformed_W.T - z @ W.T)) < 1e-10
    assert not np.allclose(np.linalg.norm(transformed_z, axis=1), np.linalg.norm(z, axis=1))


def test_c75_Wz_is_exactly_redundant_with_logits_and_bias():
    rng = np.random.default_rng(750)
    Wz = rng.normal(size=(128, 4))
    bias = rng.normal(size=4)
    logits = Wz + bias
    reconstructed = logits - bias
    assert np.max(np.abs(reconstructed - Wz)) <= 4 * np.finfo(np.float64).eps
    design = np.column_stack((reconstructed, Wz))
    assert np.linalg.matrix_rank(design) == np.linalg.matrix_rank(Wz)


def test_c75_qualification_requires_every_locked_gate():
    rows = list(csv.DictReader(open(c75_protocol.TABLE_DIR / "t3_qualification_gates.csv")))
    assert len(rows) == 14
    assert {row["candidate"] for row in rows} == {"F2_strict_source", "F4_target_unlabeled"}
    assert all(row["all_required"] == "1" for row in rows)
    assert sum(row["gate"] == "target_label_leakage" for row in rows) == 2
