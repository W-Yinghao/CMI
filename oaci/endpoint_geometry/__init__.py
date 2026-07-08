"""C31 — Endpoint-Axis / Accuracy-Calibration Geometry Audit (read-only, diagnostic-only).

Are the source-visible RANK axis and the target-specific GAUGE axis (C22-C30) endpoint-specific? Do accuracy-good /
NLL-good / ECE-good / joint-good checkpoints coincide? Does the source rank predict accuracy or calibration? Is the
C16 accuracy↔calibration "trade-off" a Pareto geometry in checkpoint space? No training, no probe tuning (frozen C19
config hash), no feature selection, no selector; every oracle endpoint is explicitly non-deployable.
"""
from . import (artifact_loader, endpoint_labels, gauge_endpoint, overlap_conflict, pareto_geometry, report, schema,
               source_rank_endpoint, taxonomy)

__all__ = ["artifact_loader", "endpoint_labels", "gauge_endpoint", "overlap_conflict", "pareto_geometry", "report",
           "schema", "source_rank_endpoint", "taxonomy"]
