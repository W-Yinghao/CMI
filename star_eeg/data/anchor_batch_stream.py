"""Frozen full-source anchor semantics and 750-batch exposure streams."""

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from star_eeg.config import STAR01
from star_eeg.data.faced_split_contract import SOURCE_TRAIN_SUBJECTS, canonical_hash


BASE_EXPOSURES_PER_SAMPLE = 7
BONUS_EXPOSURES_PER_SUBJECT = 12
EXPOSURES_PER_SUBJECT = 600
TOTAL_EXPOSURES = 48000
ANCHOR_BATCHES = 750


def _stable_seed(*parts: object) -> int:
    token = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:16], 16)


def _histogram(labels: Sequence[int]) -> Dict[str, int]:
    counts = Counter(int(label) for label in labels)
    return {str(label): int(counts.get(label, 0)) for label in range(9)}


def _nonidentity_shuffle(labels: Sequence[int], seed: int) -> List[int]:
    output = list(int(label) for label in labels)
    random.Random(seed).shuffle(output)
    original = list(int(label) for label in labels)
    if len(set(original)) > 1 and output == original:
        for shift in range(1, len(output)):
            candidate = output[shift:] + output[:shift]
            if candidate != original:
                output = candidate
                break
    return output


def build_exposure_matched_shuffled_manifest(anchor_manifest: Mapping[str, object]) -> Dict[str, object]:
    """Freeze one within-subject permutation with matched 48k label marginals.

    Each subject's deterministic 12 bonus samples form a separate permutation
    stratum. Therefore both the seven full-corpus passes and the eighth bonus
    exposure preserve true-vs-shuffled label marginals exactly.
    """
    if anchor_manifest.get("split") != "source_train" or anchor_manifest.get("n_records") != 6720:
        raise RuntimeError("full source_train anchor manifest required")
    by_subject: Dict[int, List[Mapping[str, object]]] = defaultdict(list)
    for row in anchor_manifest["records"]:
        by_subject[int(row["subject"])].append(row)
    if sorted(by_subject) != list(SOURCE_TRAIN_SUBJECTS):
        raise RuntimeError("anchor manifest subject universe differs from 1..80")

    shuffled_records = []
    histograms = {}
    changed = 0
    bonus_ids_by_subject = {}
    for subject in SOURCE_TRAIN_SUBJECTS:
        rows = sorted(by_subject[subject], key=lambda row: str(row["sample_id"]))
        candidate_ids = [str(row["sample_id"]) for row in rows]
        random.Random(_stable_seed("bonus", STAR01.permutation_seed, subject)).shuffle(candidate_ids)
        bonus_ids = set(candidate_ids[:BONUS_EXPOSURES_PER_SUBJECT])
        if len({int(row["label"]) for row in rows if row["sample_id"] in bonus_ids}) < 2:
            # Deterministic fail-safe to keep the bonus semantic control nontrivial.
            inside = sorted(bonus_ids)
            outside = [str(row["sample_id"]) for row in rows if row["sample_id"] not in bonus_ids]
            replacement = next(
                sample_id for sample_id in outside
                if int(next(row["label"] for row in rows if row["sample_id"] == sample_id))
                != int(next(row["label"] for row in rows if row["sample_id"] == inside[0]))
            )
            bonus_ids.remove(inside[0])
            bonus_ids.add(replacement)
        bonus_ids_by_subject[str(subject)] = sorted(bonus_ids)

        subject_output = {}
        for group_name, group_rows in (
            ("bonus", [row for row in rows if row["sample_id"] in bonus_ids]),
            ("base_only", [row for row in rows if row["sample_id"] not in bonus_ids]),
        ):
            labels = [int(row["label"]) for row in group_rows]
            permuted = _nonidentity_shuffle(
                labels,
                _stable_seed("semantic", STAR01.permutation_seed, subject, group_name),
            )
            if _histogram(labels) != _histogram(permuted):
                raise AssertionError(f"subject {subject} {group_name} histogram changed")
            for row, shuffled_label in zip(group_rows, permuted):
                sample_id = str(row["sample_id"])
                subject_output[sample_id] = int(shuffled_label)
                changed += int(shuffled_label != int(row["label"]))

        true_labels = [int(row["label"]) for row in rows]
        shuffled_labels = [subject_output[str(row["sample_id"])] for row in rows]
        bonus_true = [int(row["label"]) for row in rows if row["sample_id"] in bonus_ids]
        bonus_shuffled = [subject_output[str(row["sample_id"])] for row in rows if row["sample_id"] in bonus_ids]
        histograms[str(subject)] = {
            "full_true": _histogram(true_labels),
            "full_shuffled": _histogram(shuffled_labels),
            "bonus_true": _histogram(bonus_true),
            "bonus_shuffled": _histogram(bonus_shuffled),
        }
        for row in rows:
            sample_id = str(row["sample_id"])
            shuffled_records.append({
                "sample_id": sample_id,
                "subject": subject,
                "split": "source_train",
                "true_label": int(row["label"]),
                "shuffled_label": subject_output[sample_id],
                "bonus_exposure": sample_id in bonus_ids,
            })
    if changed == 0:
        raise RuntimeError("shuffled semantic manifest is identity")
    core = {
        "schema_version": 2,
        "dataset": "FACED",
        "split": "source_train",
        "anchor_manifest_hash": anchor_manifest["anchor_manifest_hash"],
        "permutation_seed": STAR01.permutation_seed,
        "permutation_scope": "within_subject_and_exposure_stratum",
        "permutation_rng_separate_from_training_rng": True,
        "reshuffle_each_epoch": False,
        "same_semantic_manifest_for_model_seeds": [0, 1],
        "n_records": len(shuffled_records),
        "n_changed_labels": changed,
        "bonus_exposures_per_subject": BONUS_EXPOSURES_PER_SUBJECT,
        "records": shuffled_records,
        "bonus_sample_ids_by_subject": bonus_ids_by_subject,
        "within_subject_histograms": histograms,
        "source_val_participated": False,
        "test_participated": False,
        "exposure_level_label_marginal_matched_by_construction": True,
    }
    return {**core, "shuffled_manifest_hash": canonical_hash(core)}


def load_shuffled_manifest(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text())
    core = {key: value for key, value in payload.items() if key != "shuffled_manifest_hash"}
    if canonical_hash(core) != payload.get("shuffled_manifest_hash"):
        raise RuntimeError("shuffled manifest hash mismatch")
    if payload.get("n_records") != 6720 or payload.get("split") != "source_train":
        raise RuntimeError("shuffled manifest is not the full source_train contract")
    if payload.get("source_val_participated") or payload.get("test_participated"):
        raise PermissionError("non-source split participated in shuffled manifest")
    return payload


def build_anchor_batches(
    anchor_manifest: Mapping[str, object],
    shuffled_manifest: Mapping[str, object],
    model_seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if int(model_seed) not in STAR01.model_seeds:
        raise ValueError("model seed outside frozen STAR universe")
    records = list(anchor_manifest["records"])
    shuffled_rows = {str(row["sample_id"]): row for row in shuffled_manifest["records"]}
    by_subject: Dict[int, List[Mapping[str, object]]] = defaultdict(list)
    for row in records:
        by_subject[int(row["subject"])].append(row)
    exposures = []
    for subject in SOURCE_TRAIN_SUBJECTS:
        rows = sorted(by_subject[subject], key=lambda row: str(row["sample_id"]))
        subject_exposures = []
        for row in rows:
            sample_id = str(row["sample_id"])
            repeat = BASE_EXPOSURES_PER_SAMPLE + int(shuffled_rows[sample_id]["bonus_exposure"])
            subject_exposures.extend([sample_id] * repeat)
        if len(subject_exposures) != EXPOSURES_PER_SUBJECT:
            raise AssertionError(f"subject {subject} exposure count differs from 600")
        random.Random(
            _stable_seed("subject-anchor-stream", STAR01.anchor_stream_seed_offset, model_seed, subject)
        ).shuffle(subject_exposures)
        exposures.extend(subject_exposures)
    random.Random(
        _stable_seed("global-anchor-stream", STAR01.anchor_stream_seed_offset, model_seed)
    ).shuffle(exposures)
    if len(exposures) != TOTAL_EXPOSURES:
        raise AssertionError("anchor exposure count differs from 48,000")

    true_by_id = {str(row["sample_id"]): int(row["label"]) for row in records}
    shuffled_by_id = {
        sample_id: int(row["shuffled_label"]) for sample_id, row in shuffled_rows.items()
    }
    batches = []
    for batch_index in range(ANCHOR_BATCHES):
        ids = exposures[batch_index * STAR01.batch_size:(batch_index + 1) * STAR01.batch_size]
        true_labels = [true_by_id[sample_id] for sample_id in ids]
        shuffled_labels = [shuffled_by_id[sample_id] for sample_id in ids]
        batches.append({
            "batch_index": batch_index + 1,
            "sample_ids": ids,
            "true_labels": true_labels,
            "shuffled_labels": shuffled_labels,
            "x_id_hash": canonical_hash(ids),
            "true_label_hash": canonical_hash(true_labels),
            "shuffled_label_hash": canonical_hash(shuffled_labels),
        })

    exposure_rows = []
    per_subject_true = defaultdict(Counter)
    per_subject_shuffled = defaultdict(Counter)
    subject_by_id = {str(row["sample_id"]): int(row["subject"]) for row in records}
    for sample_id in exposures:
        subject = subject_by_id[sample_id]
        per_subject_true[subject][true_by_id[sample_id]] += 1
        per_subject_shuffled[subject][shuffled_by_id[sample_id]] += 1
    for subject in SOURCE_TRAIN_SUBJECTS:
        for label in range(9):
            exposure_rows.append({
                "model_seed": int(model_seed),
                "subject": subject,
                "label": label,
                "true_exposures": per_subject_true[subject][label],
                "shuffled_exposures": per_subject_shuffled[subject][label],
                "marginal_equal": per_subject_true[subject][label] == per_subject_shuffled[subject][label],
                "subject_total_exposures": sum(per_subject_true[subject].values()),
            })
    if not all(row["marginal_equal"] for row in exposure_rows):
        raise RuntimeError("true/shuffled exposure-level label marginals differ")
    return batches, exposure_rows


def anchor_stream_hash_artifact(
    anchor_manifest: Mapping[str, object],
    shuffled_manifest: Mapping[str, object],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    streams = {}
    all_exposure_rows = []
    for seed in STAR01.model_seeds:
        batches, exposure_rows = build_anchor_batches(anchor_manifest, shuffled_manifest, seed)
        all_exposure_rows.extend(exposure_rows)
        x_hashes = [row["x_id_hash"] for row in batches]
        streams[f"s{seed}"] = {
            "model_seed": seed,
            "n_batches": len(batches),
            "batch_size": STAR01.batch_size,
            "n_exposures": len(batches) * STAR01.batch_size,
            "x_batch_hashes": x_hashes,
            "true_label_batch_hashes": [row["true_label_hash"] for row in batches],
            "shuffled_label_batch_hashes": [row["shuffled_label_hash"] for row in batches],
            "x_stream_hash": canonical_hash(x_hashes),
            "true_label_stream_hash": canonical_hash([row["true_label_hash"] for row in batches]),
            "shuffled_label_stream_hash": canonical_hash([row["shuffled_label_hash"] for row in batches]),
            "c_d_x_stream_identical": True,
            "subject_exposures_exact_600": all(row["subject_total_exposures"] == 600 for row in exposure_rows),
            "true_shuffled_exposure_marginals_equal": all(row["marginal_equal"] for row in exposure_rows),
        }
    core = {
        "schema_version": 1,
        "anchor_manifest_hash": anchor_manifest["anchor_manifest_hash"],
        "shuffled_manifest_hash": shuffled_manifest["shuffled_manifest_hash"],
        "algorithm": "seven_full_exposures_plus_twelve_balanced_bonus_per_subject_then_seeded_shuffle",
        "anchor_batches": ANCHOR_BATCHES,
        "batch_size": STAR01.batch_size,
        "total_exposures": TOTAL_EXPOSURES,
        "corpus_exposure_equivalent": TOTAL_EXPOSURES / 6720,
        "streams": streams,
    }
    return {**core, "anchor_batch_stream_hashes_hash": canonical_hash(core)}, all_exposure_rows
