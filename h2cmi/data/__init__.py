"""Data: a controllable EEG mechanism simulator with a hierarchical domain DAG.

The simulator (review R0 / section 10.1) lets us orthogonally manipulate the kinds of
shift the method must handle -- covariance, label-prior, concept (decision-boundary),
montage and noise shift, plus an optional site-dependent label mechanism -- so each
component (covariance alignment, reference-prior marginal, hierarchical CMI, selective
TTA, safety gate, latent-Ystar label model) can be validated where its assumptions
hold and shown to fail gracefully where they do not.
"""
from __future__ import annotations

from h2cmi.data.eeg_simulator import EEGSimulator, ShiftSpec, SimulatedEEG

__all__ = ["EEGSimulator", "ShiftSpec", "SimulatedEEG"]
