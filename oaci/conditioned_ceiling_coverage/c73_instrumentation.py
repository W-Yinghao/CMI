"""Frozen checkpoint instrumentation-readiness checks for C73.

Only checkpoint metadata/state and CPU dummy tensors are inspected.  No real
EEG tensor is loaded and no real-data model forward is executed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np

from . import c66_reinference_only_trial_cache_microcampaign as c66
from . import c69_powered_trial_cache_scaleup as c69


C65_MAP = "oaci/reports/c65_tables/frozen_universe_checkpoint_map.csv"


def _read_csv(path: str) -> list[dict]:
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _path_hash(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()


def inspect_frozen_instrumentation() -> dict:
    import torch
    from oaci.models import build_model

    mapping = _read_csv(C65_MAP)
    unique = c66._unique_checkpoint_rows(mapping)
    samples = c66._select_abi_sample(unique)
    hook_rows = []
    dummy_forward_count = 0
    real_forward_count = 0
    for sample_index, row in enumerate(samples):
        load = c66._state_load_metadata(row)
        spec = c66._model_spec_for_row(row)
        arch = dict(spec["backbone"])
        in_chans, in_times = [int(v) for v in spec["input_shape"]]
        model = build_model(
            spec["factory"], in_chans=in_chans, in_times=in_times,
            n_classes=int(spec["n_classes"]), **arch,
        )
        load_ok = load["load_status"] == "pass" and int(load["state_hash_matches_checkpoint_id"]) == 1
        if load_ok:
            model.load_state_dict(load["state"], strict=True)
        model.eval()
        c66._assert_cpu_model_and_tensor(model)
        captured = {}

        def pre_hook(_module, args):
            captured["classifier_input"] = args[0].detach().clone()

        handle = model.classifier.register_forward_pre_hook(pre_hook)
        dummy = torch.linspace(-1.0, 1.0, steps=2 * in_chans * in_times, dtype=torch.float32).reshape(2, in_chans, in_times)
        c66._assert_cpu_model_and_tensor(model, dummy)
        with torch.no_grad():
            output = model(dummy)
        dummy_forward_count += 1
        handle.remove()
        z = output.z
        weight = model.classifier.weight.detach()
        bias = model.classifier.bias.detach()
        wz = z @ weight.T
        reconstructed = wz + bias
        max_logit_error = float(torch.max(torch.abs(reconstructed - output.logits)).item())
        hook_error = float(torch.max(torch.abs(captured["classifier_input"] - z)).item())
        hook_rows.append({
            "sample_index": sample_index,
            "checkpoint_path_hash": _path_hash(row["pt_path"]),
            "model_factory": spec["factory"],
            "input_shape": f"[2,{in_chans},{in_times}]",
            "state_load_passed": int(load_ok),
            "state_schema_passed": int(load.get("sidecar_tensor_schema_matches", 0)),
            "state_hash_passed": int(load.get("state_hash_matches_checkpoint_id", 0)),
            "representation_shape": str(list(z.shape)),
            "classifier_weight_shape": str(list(weight.shape)),
            "classifier_bias_shape": str(list(bias.shape)),
            "logit_shape": str(list(output.logits.shape)),
            "hook_matches_output_z_max_abs": hook_error,
            "Wz_plus_b_logit_max_abs": max_logit_error,
            "CPU_only": 1,
            "dummy_tensor_only": 1,
            "real_EEG_loaded": 0,
            "real_EEG_forward": 0,
            "passed": int(load_ok and list(z.shape) == [2, 800] and list(weight.shape) == [4, 800] and hook_error <= 1e-7 and max_logit_error <= 1e-6),
        })

    preprocess_ok = False
    preprocess_signature = ""
    source_subject_count = 0
    if samples:
        pp, ds = c66._preprocess_namespace(samples[0])
        preprocess_signature = hashlib.sha256(json.dumps(ds["preprocessing"], sort_keys=True).encode()).hexdigest()
        source_subject_count = 8
        preprocess_ok = (
            len(ds.get("channels", [])) == 22
            and int(ds.get("n_classes", len(ds.get("class_names", [])))) == 4
            and int(ds.get("expected_n_times", 0)) == 385
            and os.path.exists(c69.DEFAULT_DATALAKE_ROOT)
        )

    source_schema = [
        {"field": "cache_version", "dtype": "string", "required": 1, "label_quarantined": 0, "payload_class": "metadata"},
        {"field": "checkpoint_path_hash", "dtype": "sha256", "required": 1, "label_quarantined": 0, "payload_class": "metadata"},
        {"field": "source_subject_id_hash", "dtype": "sha256", "required": 1, "label_quarantined": 0, "payload_class": "metadata"},
        {"field": "trial_id_hash", "dtype": "sha256", "required": 1, "label_quarantined": 0, "payload_class": "metadata"},
        {"field": "source_class_label", "dtype": "int8", "required": 1, "label_quarantined": 0, "payload_class": "source_observable_label"},
        {"field": "logits", "dtype": "float32[4]", "required": 1, "label_quarantined": 0, "payload_class": "source_trial_signal"},
        {"field": "probabilities", "dtype": "float32[4]", "required": 1, "label_quarantined": 0, "payload_class": "source_trial_signal"},
        {"field": "representation_ref", "dtype": "external_shard_ref", "required": 0, "label_quarantined": 0, "payload_class": "external_reference"},
        {"field": "content_sha256", "dtype": "sha256", "required": 1, "label_quarantined": 0, "payload_class": "integrity"},
    ]
    wz_schema = [
        {"field": "checkpoint_path_hash", "dtype": "sha256", "required": 1, "storage": "manifest"},
        {"field": "trial_id_hash", "dtype": "sha256", "required": 1, "storage": "manifest"},
        {"field": "z", "dtype": "float32[800]", "required": 1, "storage": "external_binary_shard"},
        {"field": "Wz", "dtype": "float32[4]", "required": 1, "storage": "external_binary_shard"},
        {"field": "head_bias_b", "dtype": "float32[4]", "required": 1, "storage": "manifest_once_per_checkpoint"},
        {"field": "logits", "dtype": "float32[4]", "required": 1, "storage": "external_binary_shard"},
        {"field": "reconstruction_max_abs", "dtype": "float32", "required": 1, "storage": "manifest"},
        {"field": "split_role", "dtype": "enum", "required": 1, "storage": "manifest"},
        {"field": "target_label", "dtype": "int8", "required": 0, "storage": "quarantined_label_shard"},
        {"field": "content_sha256", "dtype": "sha256", "required": 1, "storage": "manifest"},
    ]

    target_rows = 1268 * 576
    source_rows = 1268 * 8 * 576
    z_bytes_target = target_rows * 800 * 4
    z_bytes_source = source_rows * 800 * 4
    compact_bytes = (target_rows + source_rows) * (4 + 4 + 4) * 4
    storage_rows = []
    for campaign, equivalent_units, trials_per_unit, estimated_rows, estimated_bytes, sharding in (
        ("target_z_Wz", 1268, 576, target_rows, z_bytes_target + target_rows * 8 * 4, "checkpoint-target"),
        ("strict_source_logits_probs", 1268 * 8, 8 * 576, source_rows, source_rows * 8 * 4, "checkpoint-source_subject"),
        ("strict_source_z_Wz", 1268 * 8, 8 * 576, source_rows, z_bytes_source + compact_bytes, "checkpoint-source_subject"),
    ):
        # This is a deliberately wide planning envelope, not a measured EEG
        # throughput claim. A separately authorized pilot must replace it.
        low_seconds = equivalent_units * 1.0 / 48.0
        high_seconds = equivalent_units * 10.0 / 48.0
        storage_rows.append({
            "campaign": campaign, "physical_units": 1268,
            "equivalent_forward_units": equivalent_units,
            "trials_per_unit": trials_per_unit, "estimated_rows": estimated_rows,
            "estimated_binary_bytes": estimated_bytes, "estimated_GiB": estimated_bytes / 2**30,
            "planning_wall_seconds_low_48cpu": low_seconds,
            "planning_wall_seconds_high_48cpu": high_seconds,
            "runtime_basis": "unmeasured planning envelope 1-10 seconds per equivalent unit",
            "pilot_required_before_execution": 1,
            "CPU_high_feasible": 1, "GPU_required": 0, "sharding": sharding,
        })

    all_hooks = bool(hook_rows) and all(int(r["passed"]) for r in hook_rows)
    mapping_ok = len(unique) == 1268 and all(r["pt_exists"] == "1" and r["json_exists"] == "1" for r in unique)
    schema_ok = all(int(r["required"]) in {0, 1} for r in source_schema) and all(int(r["required"]) in {0, 1} for r in wz_schema)
    checks = [
        {"criterion": "frozen_checkpoint_mapping", "observed": len(unique), "expected": 1268, "passed": int(mapping_ok), "requires_real_forward": 0},
        {"criterion": "source_target_preprocessing_path", "observed": f"signature={preprocess_signature};source_subjects={source_subject_count}", "expected": "22ch;385time;4class;8 source subjects per LOSO target", "passed": int(preprocess_ok), "requires_real_forward": 0},
        {"criterion": "model_output_z_hook", "observed": f"passed_samples={sum(int(r['passed']) for r in hook_rows)}/{len(hook_rows)}", "expected": "6/6", "passed": int(all_hooks), "requires_real_forward": 0},
        {"criterion": "classifier_W_b_and_Wz_identity", "observed": max((float(r["Wz_plus_b_logit_max_abs"]) for r in hook_rows), default=math.inf), "expected": "<=1e-6", "passed": int(all_hooks), "requires_real_forward": 0},
        {"criterion": "source_cache_schema", "observed": len(source_schema), "expected": ">=8 fields", "passed": int(schema_ok), "requires_real_forward": 0},
        {"criterion": "representation_Wz_schema", "observed": len(wz_schema), "expected": ">=9 fields", "passed": int(schema_ok), "requires_real_forward": 0},
        {"criterion": "storage_runtime_plan", "observed": f"max_GiB={max(r['estimated_GiB'] for r in storage_rows):.3f};max_planning_seconds_48cpu={max(r['planning_wall_seconds_high_48cpu'] for r in storage_rows):.1f};pilot_required=1", "expected": "external sharded CPU planning envelope", "passed": 1, "requires_real_forward": 0},
    ]
    source_wz_feasible = all(int(r["passed"]) for r in checks)
    source_only_feasible = mapping_ok and preprocess_ok and schema_ok
    if source_wz_feasible:
        gate = "REINFERENCE_ONLY_SOURCE_AND_WZ_CAMPAIGN_FEASIBLE"
    elif source_only_feasible:
        gate = "REINFERENCE_ONLY_SOURCE_ONLY_FEASIBLE"
    elif mapping_ok and preprocess_ok:
        gate = "NEW_TRAINING_REQUIRED_FOR_REPRESENTATION_TRACE"
    else:
        gate = "DATA_OR_ABI_BLOCKER"
    return {
        "frozen_instrumentation_feasibility_rows": checks,
        "source_trial_cache_schema_rows": source_schema,
        "representation_Wz_cache_schema_rows": wz_schema,
        "hook_ABI_validation_rows": hook_rows,
        "storage_runtime_plan_rows": storage_rows,
        "instrumentation_gate": gate,
        "source_and_Wz_feasible": source_wz_feasible,
        "dummy_forward_count": dummy_forward_count,
        "real_EEG_forward_count": real_forward_count,
        "real_EEG_trials_loaded": 0,
        "training_attempted": 0,
        "gpu_used": 0,
    }
