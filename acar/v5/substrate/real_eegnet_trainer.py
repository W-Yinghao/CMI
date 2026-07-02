"""ACAR V5 Stage-1B real EEGNet + source-state training core and label-free embedding dumper (NO heavy import at module load;
torch is LAZY, inside the backend). This is the numeric seam bound to `training_config`:

  * `train_encoder_and_source_state(...)` receives ALREADY-READ FIT (train/val) (subject_key, SubjectWindows, label) triples — it
    NEVER receives raw roots, a reader, or the dataset view. It VALIDATES the FIT records (each windows is a validated SubjectWindows,
    each label ∈ {0,1}, train/val subject-disjoint, no dups, val non-empty) BEFORE the backend sees anything, sets determinism/seed,
    fits under TRAINING_CONFIG, and emits the 4 model files + the pinned preprocessing_config + the training_config sidecar into the
    per-ref output dir. It DOES NOT emit feat_dump.
  * `dump_fold_embeddings(...)` runs AFTER the encoder/source-state are frozen, over ALL fold subjects, driven ONLY by a label-free
    AuthorizedEmbeddingDatasetView. The embedding is produced by loading the FROZEN artifacts from the trainer's output (a
    FrozenSubstrateHandle bound to the SAME ref/disease/fold/seed) — the dumper does NOT rely on sharing a backend object with the
    trainer. The dump is written in the PINNED, parseable, label-free feature-dump schema.

The numeric backend is injectable (a synthetic FakeEegnetBackend drives it in tests); the default `TorchEegnetBackend` lazy-imports
torch and leaves the actual EEGNet fit / embedding-from-frozen-artifacts as the seam wired at the authorized Stage-1B run.
"""
from __future__ import annotations
import hashlib
import os
from dataclasses import dataclass
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import training_config as TC
from acar.v5.substrate import stage1b_embedding_dump as ED
from acar.v5.substrate import stage1b_feature_dump_writer as FDW
from acar.v5.substrate import subject_windows as SW


class RealEegnetError(RuntimeError):
    pass


# backend.fit(...) must return bytes for exactly these logical artifacts (mapped to the registry file path keys)
_MODEL_FILE_KEYS = {
    "encoder_state_dict": "encoder_state_dict_path",
    "encoder_checkpoint_file": "encoder_checkpoint_file_path",
    "source_state_artifact": "source_state_artifact_path",
    "source_state_file": "source_state_file_path",
}


@dataclass(frozen=True)
class FrozenSubstrateHandle:
    """A bound reference to a fold's FROZEN substrate artifacts (the files the trainer emitted). The dumper loads the encoder /
    source-state FROM these files, so the embedding provably comes from the same registered substrate — not from an incidentally
    shared backend object."""
    ref: str
    disease: str
    fold: int
    seed: int
    encoder_checkpoint_file_path: str
    source_state_file_path: str
    preprocessing_config_path: str
    training_config_path: str

    @classmethod
    def from_train_result(cls, train_result):
        need = ("ref", "disease", "fold", "seed", "encoder_checkpoint_file_path", "source_state_file_path",
                "preprocessing_config_path", "training_config_path")
        missing = [k for k in need if k not in train_result]
        if missing:
            raise RealEegnetError(f"train_result missing frozen-artifact fields {missing}")
        for k in ("encoder_checkpoint_file_path", "source_state_file_path", "preprocessing_config_path", "training_config_path"):
            if not (isinstance(train_result[k], str) and os.path.isfile(train_result[k])):
                raise RealEegnetError(f"train_result['{k}'] is not an existing file")
        return cls(ref=train_result["ref"], disease=train_result["disease"], fold=int(train_result["fold"]),
                   seed=int(train_result["seed"]), encoder_checkpoint_file_path=train_result["encoder_checkpoint_file_path"],
                   source_state_file_path=train_result["source_state_file_path"],
                   preprocessing_config_path=train_result["preprocessing_config_path"],
                   training_config_path=train_result["training_config_path"])

    def assert_matches(self, disease, fold, seed):
        if not (self.disease == disease and int(self.fold) == int(fold) and int(self.seed) == int(seed)
                and self.ref == f"{disease}/fold{fold}/seed{seed}"):
            raise RealEegnetError(f"frozen substrate handle {self.ref} does not match dump target {disease}/fold{fold}/seed{seed}")


class TorchEegnetBackend:
    """Default numeric backend. torch is imported LAZILY here; the EEGNet fit + embedding-from-frozen-artifacts are the seam."""

    def set_deterministic(self, seed):
        import torch  # lazy — never imported at module load
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(int(seed))
        torch.set_num_threads(1)

    def fit(self, train, val, training_config):
        import torch  # noqa: F401  (lazy)
        raise NotImplementedError("EEGNet + source-state fit under training_config wired at the authorized Stage-1B run")

    def embed_from_artifacts(self, windows_by_subject, frozen, training_config):
        import torch  # noqa: F401  (lazy)
        raise NotImplementedError("frozen-encoder embedding (load from FrozenSubstrateHandle) wired at the authorized Stage-1B run")


def _as_bytes(x, what):
    if not isinstance(x, (bytes, bytearray)):
        raise RealEegnetError(f"{what} must be bytes (got {type(x).__name__})")
    return bytes(x)


def _write_file(output_dir, name, data):
    path = os.path.join(output_dir, name)
    with open(path, "wb") as f:
        f.write(data)
    return path


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_fit_records(disease, fold, seed, train, val):
    """Fail-closed FIT-record validation before the numeric backend sees anything."""
    if not train:
        raise RealEegnetError(f"{disease}/fold{fold}/seed{seed}: no FIT-train subjects")
    if not val:
        raise RealEegnetError(f"{disease}/fold{fold}/seed{seed}: FIT-val must be non-empty")
    for triple in list(train) + list(val):
        if not (isinstance(triple, tuple) and len(triple) == 3):
            raise RealEegnetError("each FIT record must be (subject_key, windows, label)")
    tkeys = [t[0] for t in train]
    vkeys = [t[0] for t in val]
    if len(set(tkeys)) != len(tkeys) or len(set(vkeys)) != len(vkeys):
        raise RealEegnetError("duplicate subject key within the FIT split")
    if set(tkeys) & set(vkeys):
        raise RealEegnetError("FIT train and val are not subject-disjoint")
    for sk, windows, label in list(train) + list(val):
        if not (isinstance(sk, str) and sk.count("/") == 2 and sk.split("/")[0] == disease):
            raise RealEegnetError(f"non-canonical FIT subject_key {sk!r} (must be {disease}/<cohort>/<raw>)")
        if not isinstance(windows, SW.SubjectWindows):
            raise RealEegnetError(f"{sk}: FIT windows must be a validated SubjectWindows")
        SW.validate_subject_windows(windows)
        if isinstance(label, bool) or label not in (0, 1):
            raise RealEegnetError(f"{sk}: FIT label must be int 0 or 1 (got {label!r})")


def train_encoder_and_source_state(disease, fold, seed, train, val, *, output_dir, backend):
    """Validate FIT records, fit under TRAINING_CONFIG, emit the 4 model files + pinned preprocessing_config + training_config
    sidecar into output_dir. Returns a raw build output with the 5 registry file paths (NO feat_dump) + the training_config sidecar."""
    _validate_fit_records(disease, fold, seed, train, val)
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


def dump_fold_embeddings(disease, fold, seed, embedding_view, all_fold_subject_keys, train_result, role_by_subject,
                         *, output_dir, backend):
    """Label-free feature dump over ALL fold subjects, driven ONLY by the label-free embedding view, using the FROZEN artifacts from
    train_result. Writes the pinned, parseable, label-free feature-dump schema."""
    ED.assert_view_is_label_free(embedding_view)              # must be AuthorizedEmbeddingDatasetView; has NO read_label
    if not all_fold_subject_keys:
        raise RealEegnetError("embedding dump requires the fold's subject keys")
    frozen = FrozenSubstrateHandle.from_train_result(train_result)
    frozen.assert_matches(disease, fold, seed)               # the dump provably comes from THIS ref's frozen substrate
    windows_by_subject = {k: embedding_view.read_windows(k) for k in all_fold_subject_keys}   # SIGNAL ONLY (no labels)
    emb_by_subject = backend.embed_from_artifacts(windows_by_subject, frozen, TC.TRAINING_CONFIG)
    if not isinstance(emb_by_subject, dict) or set(emb_by_subject) != set(all_fold_subject_keys):
        raise RealEegnetError("backend.embed_from_artifacts must return an embedding for EXACTLY every fold subject")
    records = []
    for sk in sorted(emb_by_subject):
        role = role_by_subject.get(sk)
        if role is None:
            raise RealEegnetError(f"no split role for fold subject {sk}")
        mat = emb_by_subject[sk]
        for wid in range(len(mat)):
            records.append((sk, role, wid, mat[wid]))
    os.makedirs(output_dir, exist_ok=True)
    ref = f"{disease}/fold{fold}/seed{seed}"
    feat_path = os.path.join(output_dir, "feat_dump.npz")
    FDW.write_feature_dump(feat_path, ref=ref, disease=disease, fold=fold, seed=seed,
                           preprocessing_config_sha256=_sha256_file(frozen.preprocessing_config_path),
                           training_config_sha256=_sha256_file(frozen.training_config_path),
                           encoder_checkpoint_file_sha256=_sha256_file(frozen.encoder_checkpoint_file_path),
                           source_state_file_sha256=_sha256_file(frozen.source_state_file_path), records=records)
    raw = {"ref": ref, "disease": disease, "fold": fold, "seed": seed, "feat_dump_path": feat_path}
    ED.validate_embedding_dump_label_free(raw)               # the dump manifest carries NO label-like field
    return raw


class RealEmbeddingDumper:
    """View-facing label-free dumper. Built by its factory with the gate-issued context; per-ref output dir from the context."""

    def __init__(self, context, backend=None):
        if context is None:
            raise RealEegnetError("RealEmbeddingDumper requires a gate-issued Stage1BExecutionContext")
        self._ctx = context
        self._backend = backend if backend is not None else TorchEegnetBackend()

    def dump_embeddings(self, disease, fold, seed, embedding_view, all_fold_subject_keys, train_result, role_by_subject):
        from acar.v5.substrate import stage1b_output_layout as LO
        ref = f"{disease}/fold{fold}/seed{seed}"
        out = LO.ref_output_dir(self._ctx.output_root, self._ctx.run_id, ref)
        return dump_fold_embeddings(disease, fold, seed, embedding_view, all_fold_subject_keys, train_result, role_by_subject,
                                    output_dir=out, backend=self._backend)


def make_real_embedding_dumper(context):
    """Factory — construct AFTER the full-build gate, bound to the run's execution context."""
    return RealEmbeddingDumper(context)
