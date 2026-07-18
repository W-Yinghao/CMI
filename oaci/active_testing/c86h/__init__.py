"""C86H terminal untouched-confirmation — gated preparation package.

Locks the executable bindings (``contract``), the two-level output taxonomy and
confirmatory inference (``analysis``), and the gated entrypoint (``entrypoint``). No
real EEG/label is touched; real execution requires a separate ``授权 C86H`` and a
separately authorized field generation. See ``oaci/reports/C86H_CONTRACT.md``.
"""
from . import analysis, contract, entrypoint, field_spec, held_eval, runner  # noqa: F401

__all__ = ["analysis", "contract", "entrypoint", "field_spec", "held_eval", "runner"]
