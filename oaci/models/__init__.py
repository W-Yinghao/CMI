"""OACI model backbones (self-contained; no import of cmi/h2cmi). Fixed ``ModelOutput(logits, z)``
interface so the trainer is model-agnostic. First backbone: ShallowConvNet for BNCI MI."""
from __future__ import annotations

from .output import ModelOutput, RepresentationClassifier
from .shallow import ShallowConvNet
from .factory import MLPClassifier, build_model

__all__ = ["ModelOutput", "RepresentationClassifier", "ShallowConvNet", "MLPClassifier", "build_model"]
