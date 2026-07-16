from pathlib import Path

import numpy as np
import pytest

from oaci.multidataset.c84s_common import read_json
from oaci.multidataset.c84sr1_common import SCORE_METHODS, reject_evaluation_tokens
from oaci.multidataset.c84sr1_q0_store import synthetic_payload
from oaci.multidataset.c84sr1_stage_a_labels import run_stage_a_from_rows
from oaci.multidataset.c84sr1_stage_b_selection import run_stage_b
from oaci.multidataset.c84sr1_synthetic import (
    SyntheticLoader, synthetic_contexts, synthetic_label_rows,
    synthetic_score_provider,
)


def _stage_a(tmp_path: Path):
    registry, labels, trials = synthetic_label_rows()
    root = tmp_path / "stage_a"
    run_stage_a_from_rows(
        guard_receipt={"C84S_authorized": True},
        frozen_registry_rows=registry, label_rows=labels, output_root=root,
    )
    return root, trials


def test_stage_a_handoff_contains_no_evaluation_descriptor(tmp_path: Path):
    root, _ = _stage_a(tmp_path)
    handoff = read_json(root / "C84S_STAGE_A_HANDOFF.json")
    reject_evaluation_tokens(handoff)
    assert "construction_descriptor" in handoff
    assert "evaluation_descriptor" not in handoff
    assert (root / "C84S_STAGE_A_EVALUATION_SEAL.json").is_file()


def test_stage_a_alignment_drift_fails_before_views(tmp_path: Path):
    registry, labels, _ = synthetic_label_rows()
    labels = labels[:-1]
    with pytest.raises(RuntimeError, match="row count drift"):
        run_stage_a_from_rows(
            guard_receipt={"C84S_authorized": True},
            frozen_registry_rows=registry, label_rows=labels,
            output_root=tmp_path / "bad_stage_a",
        )


def test_stage_b_partial_failure_does_not_publish(tmp_path: Path):
    stage_a, trials = _stage_a(tmp_path)
    contexts = synthetic_contexts()[:2]
    loader = SyntheticLoader(trials)

    def q0_builder(**kwargs):
        return synthetic_payload(kwargs["identity"], kwargs["candidate_ids"], chains=kwargs["chains"])

    final = tmp_path / "stage_b"
    with pytest.raises(RuntimeError, match="injected Stage-B"):
        run_stage_b(
            stage_a_handoff_path=stage_a / "C84S_STAGE_A_HANDOFF.json",
            final_root=final, contexts=contexts, context_loader=loader,
            score_provider=synthetic_score_provider, q0_builder=q0_builder,
            chains=2, synthetic=True, failure_injection_context=1,
        )
    assert not final.exists()


def test_stage_b_rejects_evaluation_token():
    with pytest.raises(RuntimeError, match="reached Stage B"):
        reject_evaluation_tokens({"evaluation_manifest_sha256": "0" * 64})


def test_score_provider_has_exact_registered_methods():
    context = synthetic_contexts()[0]
    scores = synthetic_score_provider(context, {})
    assert tuple(scores) == SCORE_METHODS
    assert all(value.shape == (81,) and np.isfinite(value).all() for value in scores.values())
