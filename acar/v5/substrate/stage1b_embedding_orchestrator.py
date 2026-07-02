"""ACAR V5 Stage-1B embedding-dump ORCHESTRATION (pure/stdlib at import). Encodes the mandatory two-phase per-fold build so that
Stage-2's fold feature dump is produced correctly and safely:

  Phase A — FIT-only training: the trainer sees ONLY the FIT (train∪val) canonical keys through an AuthorizedFitDatasetView (labels
            are readable, CAL/EVAL are not). It returns frozen encoder/source-state artifacts and MUST NOT emit feat_dump.
  Phase B — label-free feature dump: AFTER freezing, an AuthorizedEmbeddingDatasetView is built over ALL fold subjects
            (train∪val∪cal∪eval) and handed to a SEPARATE dumper, which reads windows ONLY (no read_label) and produces feat_dump.

This guarantees: (1) CAL/EVAL are reachable ONLY through the label-free embedding view; (2) the feature dump covers every fold
subject; (3) the FIT training path cannot emit the feature dump (so it can never leak labels into it or restrict it to FIT-only).
"""
from __future__ import annotations
from acar.v5.substrate import fit_dataset_view as FV
from acar.v5.substrate import embedding_dataset_view as EV
from acar.v5.substrate import stage1b_embedding_dump as ED

# a feat dump may be reported as a file path (real) or bytes (synthetic bytes-writer)
FEAT_DUMP_KEYS = ("feat_dump_path", "feat_dump_bytes")
# the 5 non-feat model/config artifacts the trainer emits (path or bytes form)
_MODEL_KEYS = ("encoder_state_dict_path", "encoder_checkpoint_file_path", "source_state_artifact_path",
               "source_state_file_path", "preprocessing_config_path", "encoder_state_dict_bytes",
               "encoder_checkpoint_bytes", "source_state_artifact_bytes", "source_state_file_bytes",
               "preprocessing_config_bytes")


class Stage1bOrchestratorError(RuntimeError):
    pass


class EmbeddingDumperNotWiredError(RuntimeError):
    """Raised when a real dump is attempted without an authorized, wired embedding dumper."""


class UnwiredEmbeddingDumper:
    def dump_embeddings(self, disease, fold, seed, embedding_view, all_fold_subject_keys, train_result, role_by_subject=None):
        raise EmbeddingDumperNotWiredError("real Stage-1B embedding dumper not wired")


def require_dumper(dumper):
    if dumper is None:
        raise EmbeddingDumperNotWiredError("Stage-1B execute requires an authorized embedding dumper (none supplied)")
    if not callable(getattr(dumper, "dump_embeddings", None)):
        raise EmbeddingDumperNotWiredError("dumper is missing required method dump_embeddings()")
    return dumper


def all_fold_subject_keys(split):
    """Every subject in the fold = train ∪ val ∪ cal ∪ eval (label-free embedding coverage)."""
    return sorted(set(split["train"]) | set(split["val"]) | set(split["cal"]) | set(split["eval"]))


def split_role_by_subject(split):
    """Map each fold subject to its split role (train/val/cal/eval) — attached to every feature-dump record."""
    role = {}
    for r in ("train", "val", "cal", "eval"):
        for k in split[r]:
            role[k] = r
    return role


def _assert_no_feat_dump(train_result):
    bad = [k for k in FEAT_DUMP_KEYS if k in train_result]
    if bad:
        raise Stage1bOrchestratorError(f"train_fold must NOT emit feat_dump {bad} — only the label-free dumper produces it")


def _assert_dump_is_feat_only(dump_result):
    if not any(k in dump_result for k in FEAT_DUMP_KEYS):
        raise Stage1bOrchestratorError("the dumper must emit a feat_dump (path or bytes)")
    leaked_model = [k for k in _MODEL_KEYS if k in dump_result]
    if leaked_model:
        raise Stage1bOrchestratorError(f"the dumper must emit ONLY feat_dump, not model artifacts {leaked_model}")
    ED.validate_embedding_dump_label_free(dump_result)        # no label-like field in the dump manifest


def build_fold_raw(disease, fold, seed, ref, index, split, dev_reader, trainer, dumper, cohort_paths):
    """Run the two-phase build for one fold and return (merged raw build output, sidecars). The merged raw carries the trainer's 5
    model/config artifacts + the dumper's feat_dump; sidecars carries the (non-registry) training_config_path when present."""
    train_keys, val_keys = list(split["train"]), list(split["val"])
    fit_keys = set(train_keys) | set(val_keys)
    all_keys = all_fold_subject_keys(split)

    # Phase A: FIT-only training (labels via the FIT view; CAL/EVAL unreachable)
    fit_view = FV.AuthorizedFitDatasetView(index, fit_keys, dev_reader, cohort_paths)
    train_result = trainer.train_fold(disease, fold, seed, train_keys, val_keys, fit_view)
    if not isinstance(train_result, dict) or train_result.get("ref") != ref:
        raise Stage1bOrchestratorError(f"{ref}: train_fold returned a bad/mismatched raw output")
    _assert_no_feat_dump(train_result)

    # Phase B: label-free feature dump over ALL fold subjects (encoder now frozen). The embedding view is built from a WINDOWS-ONLY
    # reader facade (no read_subject_label capability) so labels are unreachable even via closure introspection; fail-closed if the
    # reader can't produce one.
    emb_reader = dev_reader.windows_only() if hasattr(dev_reader, "windows_only") else dev_reader
    emb_view = EV.AuthorizedEmbeddingDatasetView(index, set(all_keys), emb_reader, cohort_paths)
    ED.assert_view_is_label_free(emb_view)                    # AuthorizedEmbeddingDatasetView, no read_label
    role_by_subject = split_role_by_subject(split)            # every dump record carries its split role
    dump_result = dumper.dump_embeddings(disease, fold, seed, emb_view, all_keys, train_result, role_by_subject)
    if not isinstance(dump_result, dict) or dump_result.get("ref") != ref:
        raise Stage1bOrchestratorError(f"{ref}: dump_embeddings returned a bad/mismatched raw output")
    _assert_dump_is_feat_only(dump_result)

    raw = dict(train_result)                                   # model/config artifacts
    for k in FEAT_DUMP_KEYS:
        if k in dump_result:
            raw[k] = dump_result[k]                            # + the label-free feat dump
    sidecars = {"training_config_path": train_result.get("training_config_path"),
                "n_windows_by_subject": dump_result.get("n_windows_by_subject")}   # authoritative per-subject window counts
    return raw, sidecars
