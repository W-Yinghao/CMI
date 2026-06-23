"""Versioned, deterministic on-disk artifacts for a completed fold run.

Kept intentionally import-light (no heavy submodule imports at package load) so that
``oaci.runner.finalize`` can import ``result_payload`` without pulling in the writer/verifier and risking
an import cycle. Import the writer/verifier directly:
``from oaci.artifacts.writer import write_artifact_tree_atomic`` /
``from oaci.artifacts.verify import verify_artifact_tree``.
"""
from __future__ import annotations

from .schema import ARTIFACT_PROFILE, ARTIFACT_SCHEMA_VERSION

__all__ = ["ARTIFACT_SCHEMA_VERSION", "ARTIFACT_PROFILE"]
