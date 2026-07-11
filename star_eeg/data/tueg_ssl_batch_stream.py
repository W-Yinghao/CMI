"""Exact H200 Route-B common/replacement SSL window streams for STAR."""

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from star_eeg.config import STAR01
from star_eeg.data.faced_split_contract import canonical_hash


TUEG_ROOT = Path("/projects/EEG-foundation-model/datalake/processed/4704743c/TUEG")
WINDOW_SAMPLES = 6000
N_PATCH = 30
PATCH_SIZE = 200
CHANNELS = 33
COMMON_SSL_PASSES = 8
REPLACEMENT_SSL_PASSES = 2


def _stable_seed(*parts: object) -> int:
    token = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:16], 16)


def _route_loader(repo_root: Path):
    scripts = repo_root / "s2p/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        import route_b_33ch_loader
    finally:
        sys.dont_write_bytecode = previous

    return route_b_33ch_loader


def build_h200_window_universe(
    repo_root: Path,
    model_seed: int,
    contract_dir: Path,
) -> Tuple[List[str], Dict[str, Mapping[str, object]], Mapping[str, object]]:
    if int(model_seed) not in STAR01.model_seeds:
        raise ValueError("model seed outside frozen STAR universe")
    loader = _route_loader(repo_root)
    cell = loader.build_route_b_cell(200, int(model_seed), contract_dir=str(contract_dir))
    if int(cell["manifest"]["WT"]) != 24000:
        raise RuntimeError("H200 Route-B window universe is not 24,000")
    mapping = {}
    universe = []
    for row in cell["train"]:
        for window_index in range(int(row["take_windows"])):
            window_id = (
                f"rec{int(row['recording_id']):08d}:win{window_index:06d}:"
                f"group={row['group_id']}"
            )
            if window_id in mapping:
                raise RuntimeError(f"duplicate Route-B window ID: {window_id}")
            mapping[window_id] = {**row, "window_index": window_index}
            universe.append(window_id)
    if len(universe) != 24000:
        raise RuntimeError(f"H200 Route-B window ID count {len(universe)} != 24000")
    return universe, mapping, cell["manifest"]


def _epoch_stream(universe: Sequence[str], model_seed: int, stream: str, passes: int) -> List[str]:
    output = []
    for epoch in range(passes):
        order = list(universe)
        random.Random(
            _stable_seed("STAR-TUEG", STAR01.ssl_stream_seed_offset, model_seed, stream, epoch)
        ).shuffle(order)
        output.extend(order)
    return output


def _batches(exposures: Sequence[str]) -> List[List[str]]:
    if len(exposures) % STAR01.batch_size:
        raise RuntimeError("SSL stream is not divisible by frozen batch size")
    return [
        list(exposures[start:start + STAR01.batch_size])
        for start in range(0, len(exposures), STAR01.batch_size)
    ]


def build_ssl_batches(
    repo_root: Path,
    model_seed: int,
    contract_dir: Path,
) -> Tuple[List[List[str]], List[List[str]], Dict[str, Mapping[str, object]]]:
    universe, mapping, _ = build_h200_window_universe(repo_root, model_seed, contract_dir)
    common = _batches(_epoch_stream(universe, model_seed, "common", COMMON_SSL_PASSES))
    replacement = _batches(
        _epoch_stream(universe, model_seed, "replacement", REPLACEMENT_SSL_PASSES)
    )
    if len(common) != 3000 or len(replacement) != 750:
        raise AssertionError("frozen SSL stream batch counts differ from 3000/750")
    return common, replacement, mapping


def ssl_stream_hash_artifact(repo_root: Path, contract_dir: Path) -> Dict[str, object]:
    streams = {}
    for seed in STAR01.model_seeds:
        universe, _, route_manifest = build_h200_window_universe(repo_root, seed, contract_dir)
        common = _batches(_epoch_stream(universe, seed, "common", COMMON_SSL_PASSES))
        replacement = _batches(_epoch_stream(universe, seed, "replacement", REPLACEMENT_SSL_PASSES))
        common_hashes = [canonical_hash(batch) for batch in common]
        replacement_hashes = [canonical_hash(batch) for batch in replacement]
        streams[f"s{seed}"] = {
            "model_seed": seed,
            "window_universe_count": len(universe),
            "window_universe_hash": canonical_hash(sorted(universe)),
            "route_manifest_hash": canonical_hash(route_manifest),
            "common_passes": COMMON_SSL_PASSES,
            "common_batches": len(common),
            "common_batch_hashes": common_hashes,
            "common_stream_hash": canonical_hash(common_hashes),
            "replacement_passes": REPLACEMENT_SSL_PASSES,
            "replacement_batches": len(replacement),
            "replacement_batch_hashes": replacement_hashes,
            "replacement_stream_hash": canonical_hash(replacement_hashes),
            "same_source_pool": True,
        }
    core = {
        "schema_version": 1,
        "dataset": "TUEG_processed_exact_33ch",
        "budget_h": 200,
        "batch_size": STAR01.batch_size,
        "common_ssl_steps": 3000,
        "replacement_ssl_steps": 750,
        "common_stream_shared_by_b_c_d_within_seed": True,
        "replacement_stream_used_only_by_b_anchor_slots": True,
        "streams": streams,
    }
    return {**core, "ssl_batch_stream_hashes_hash": canonical_hash(core)}


class TUEGRouteBWindowLoader:
    """Load exact frozen Route-B window IDs using native normalization."""

    def __init__(self, mapping: Mapping[str, Mapping[str, object]], contract_dir: Path):
        self.mapping = dict(mapping)
        specification = json.loads(
            (Path(contract_dir) / "route_b_canonical_channel_order.json").read_text()
        )
        self.orders = {row["group_id"]: list(row["channels"]) for row in specification["groups"]}

    def load_batch(self, window_ids: Sequence[str]):
        import numpy as np

        if not window_ids:
            raise ValueError("SSL batch is empty")
        grouped = defaultdict(list)
        for position, window_id in enumerate(window_ids):
            row = self.mapping.get(str(window_id))
            if row is None:
                raise PermissionError(f"window ID outside frozen H200 source pool: {window_id}")
            grouped[str(row["filepath"])].append((position, row))
        output = [None] * len(window_ids)
        for filepath, requests in grouped.items():
            array = np.load(str(TUEG_ROOT / filepath), mmap_mode="r")
            for position, row in requests:
                channels = json.loads(row["channels"])
                target = self.orders[row["group_id"]]
                channel_indices = [channels.index(channel) for channel in target]
                window_index = int(row["window_index"])
                start = window_index * WINDOW_SAMPLES
                stop = start + WINDOW_SAMPLES
                if stop > array.shape[0]:
                    raise RuntimeError(f"Route-B window exceeds recording: {row['recording_id']}")
                values = np.asarray(array[start:stop, channel_indices], dtype=np.float32)
                values = values.reshape(N_PATCH, PATCH_SIZE, CHANNELS).transpose(2, 0, 1)
                values = (values - values.mean(-1, keepdims=True)) / (
                    values.std(-1, keepdims=True) + 1e-6
                )
                output[position] = values.astype(np.float32)
        return np.stack(output).astype(np.float32)
