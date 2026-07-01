"""ACAR V5 Stage-1B label-free fold feature/embedding dump (pure/stdlib). Runs AFTER the encoder/source-state are frozen, over ALL
fold subjects (train/val/cal/eval) via the AuthorizedEmbeddingDatasetView (no labels). The dump schema is fail-closed against any
label-like field; feat_dump_sha256 is the hash of this label-free dump. The actual embedding computation is the numeric seam (real
run); here we validate the schema + isolation.
"""
from __future__ import annotations
from acar.v5.substrate.embedding_dataset_view import AuthorizedEmbeddingDatasetView
from acar.v5.substrate.subject_windows import has_label_field
from acar.v5.substrate.fit_dataset_view import DatasetViewAccessError

FORBIDDEN_DUMP_FIELDS = ("label", "y", "y_te", "diagnosis", "target", "case_control", "labels")


class Stage1bEmbeddingDumpError(RuntimeError):
    pass


def validate_embedding_dump_label_free(dump):
    """Fail-closed: a feature/embedding dump record (or list of records) must carry NO label-like field."""
    records = dump if isinstance(dump, list) else [dump]
    for r in records:
        if not isinstance(r, dict):
            raise Stage1bEmbeddingDumpError("dump record must be a dict")
        bad = [k for k in r if k in FORBIDDEN_DUMP_FIELDS]
        if bad:
            raise Stage1bEmbeddingDumpError(f"embedding dump must be label-free; found forbidden fields {bad}")
    return True


def assert_view_is_label_free(view):
    """The embedding dump MUST be driven by a label-free view (no read_label method)."""
    if not isinstance(view, AuthorizedEmbeddingDatasetView):
        raise Stage1bEmbeddingDumpError("embedding dump requires an AuthorizedEmbeddingDatasetView")
    if hasattr(view, "read_label"):
        raise Stage1bEmbeddingDumpError("embedding-dump view must NOT expose read_label")
    return True
