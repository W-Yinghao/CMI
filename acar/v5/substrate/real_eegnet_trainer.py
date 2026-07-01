"""ACAR V5 Stage-1B real EEGNet + source-state training core and label-free embedding dumper (NO heavy import at module load;
torch is LAZY, inside the backend). This is the numeric seam bound to `training_config`:

  * `train_encoder_and_source_state(...)` receives ALREADY-READ FIT (train/val) (subject_key, SubjectWindows, label) triples — it
    NEVER receives raw roots, a reader, or the dataset view (the FIT view read happens in real_trainer, the view boundary). It sets
    determinism/seed via the backend, fits under TRAINING_CONFIG, and emits the 4 model files + the pinned preprocessing_config +
    the training_config sidecar into the per-ref output dir. It DOES NOT emit feat_dump.
  * `dump_fold_embeddings(...)` runs AFTER the encoder/source-state are frozen, over ALL fold subjects, driven ONLY by a
    label-free `AuthorizedEmbeddingDatasetView` (asserted). It emits feat_dump. It never reads a label.

The numeric backend is injectable (a synthetic FakeBackend drives the whole path in tests); the default `TorchEegnetBackend`
lazy-imports torch and leaves the actual EEGNet fit / embedding as the seam wired at the authorized Stage-1B run.
"""
from __future__ import annotations
import os
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import training_config as TC
from acar.v5.substrate import stage1b_embedding_dump as ED


class RealEegnetError(RuntimeError):
    pass


# backend.fit(...) must return bytes for exactly these logical artifacts (mapped to the registry file path keys)
_MODEL_FILE_KEYS = {
    "encoder_state_dict": "encoder_state_dict_path",
    "encoder_checkpoint_file": "encoder_checkpoint_file_path",
    "source_state_artifact": "source_state_artifact_path",
    "source_state_file": "source_state_file_path",
}


class TorchEegnetBackend:
    """Default numeric backend. torch is imported LAZILY here; the EEGNet fit + frozen-encoder embedding are the remaining seam."""

    def set_deterministic(self, seed):
        import torch  # lazy — never imported at module load
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(int(seed))
        torch.set_num_threads(1)

    def fit(self, train, val, training_config):
        import torch  # noqa: F401  (lazy)
        raise NotImplementedError("EEGNet + source-state fit under training_config wired at the authorized Stage-1B run")

    def embed(self, windows_by_subject, training_config):
        import torch  # noqa: F401  (lazy)
        raise NotImplementedError("frozen-encoder embedding wired at the authorized Stage-1B run")


def _as_bytes(x, what):
    if not isinstance(x, (bytes, bytearray)):
        raise RealEegnetError(f"{what} must be bytes (got {type(x).__name__})")
    return bytes(x)


def _write_file(output_dir, name, data):
    path = os.path.join(output_dir, name)
    with open(path, "wb") as f:
        f.write(data)
    return path


def train_encoder_and_source_state(disease, fold, seed, train, val, *, output_dir, backend):
    """Fit under TRAINING_CONFIG on ALREADY-READ FIT data; emit the 4 model files + pinned preprocessing_config + training_config
    sidecar into output_dir. Returns a raw build output with the 5 registry file paths (NO feat_dump) + the training_config sidecar."""
    if not train:
        raise RealEegnetError(f"{disease}/fold{fold}/seed{seed}: no FIT-train subjects")
    for triple in list(train) + list(val):                    # (subject_key, SubjectWindows, label)
        if not (isinstance(triple, tuple) and len(triple) == 3):
            raise RealEegnetError("each FIT record must be (subject_key, windows, label)")
    backend.set_deterministic(seed)
    fitted = backend.fit(train, val, TC.TRAINING_CONFIG)       # {model file key: bytes}
    if not isinstance(fitted, dict):
        raise RealEegnetError("backend.fit must return a dict of model-artifact bytes")
    missing = [k for k in _MODEL_FILE_KEYS if k not in fitted]
    if missing:
        raise RealEegnetError(f"backend.fit missing model artifacts {missing}")
    os.makedirs(output_dir, exist_ok=True)
    ref = f"{disease}/fold{fold}/seed{seed}"
    raw = {"ref": ref, "disease": disease, "fold": fold, "seed": seed}
    for content_key, path_key in _MODEL_FILE_KEYS.items():
        raw[path_key] = _write_file(output_dir, path_key + ".bin", _as_bytes(fitted[content_key], content_key))
    # the pinned config files are written FROM code (canonical JSON) — the writer hashes them, finalize validates their content
    raw["preprocessing_config_path"] = _write_file(output_dir, "preprocessing_config.json", PC.canonical_json().encode())
    raw["training_config_path"] = _write_file(output_dir, "training_config.json", TC.canonical_json().encode())
    if "feat_dump_path" in raw or "feat_dump_bytes" in raw:
        raise RealEegnetError("train_encoder_and_source_state must NOT emit feat_dump (that is the label-free dumper's job)")
    return raw


def dump_fold_embeddings(disease, fold, seed, embedding_view, all_fold_subject_keys, train_result, *, output_dir, backend):
    """Label-free feature dump over ALL fold subjects (train∪val∪cal∪eval), driven ONLY by the label-free embedding view."""
    ED.assert_view_is_label_free(embedding_view)              # must be AuthorizedEmbeddingDatasetView; has NO read_label
    if not all_fold_subject_keys:
        raise RealEegnetError("embedding dump requires the fold's subject keys")
    windows_by_subject = {k: embedding_view.read_windows(k) for k in all_fold_subject_keys}   # SIGNAL ONLY (no labels)
    feat = _as_bytes(backend.embed(windows_by_subject, TC.TRAINING_CONFIG), "embedding features")
    os.makedirs(output_dir, exist_ok=True)
    ref = f"{disease}/fold{fold}/seed{seed}"
    raw = {"ref": ref, "disease": disease, "fold": fold, "seed": seed,
           "feat_dump_path": _write_file(output_dir, "feat_dump.bin", feat)}
    ED.validate_embedding_dump_label_free(raw)                # the dump manifest carries NO label-like field
    return raw


class RealEmbeddingDumper:
    """View-facing label-free dumper. Built by its factory with the gate-issued context; per-ref output dir from the context."""

    def __init__(self, context, backend=None):
        if context is None:
            raise RealEegnetError("RealEmbeddingDumper requires a gate-issued Stage1BExecutionContext")
        self._ctx = context
        self._backend = backend if backend is not None else TorchEegnetBackend()

    def dump_embeddings(self, disease, fold, seed, embedding_view, all_fold_subject_keys, train_result):
        from acar.v5.substrate import stage1b_output_layout as LO
        ref = f"{disease}/fold{fold}/seed{seed}"
        out = LO.ref_output_dir(self._ctx.output_root, self._ctx.run_id, ref)
        return dump_fold_embeddings(disease, fold, seed, embedding_view, all_fold_subject_keys, train_result,
                                    output_dir=out, backend=self._backend)


def make_real_embedding_dumper(context):
    """Factory — construct AFTER the full-build gate, bound to the run's execution context."""
    return RealEmbeddingDumper(context)
