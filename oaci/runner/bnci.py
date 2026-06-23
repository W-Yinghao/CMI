"""In-memory two-level run over the real BNCI fold (used by the B2 GPU smoke; B1b only preflights)."""
from __future__ import annotations

import torch

from ..methods.activity import METHODS
from .finalize import assemble_fold_run
from .fold import run_level_complete
from .keys import RunKey
from .plans import build_level_plans
from .scope import build_level_population
from .support import build_level_support, level0_reference_prior

DEFAULT_METHOD_ORDER = ("ERM", "OACI", "global_lpc", "uniform")


def run_bnci_two_level_in_memory(bnci_fold, *, model_seed, method_order=DEFAULT_METHOD_ORDER, device="cpu"):
    if int(model_seed) not in [int(s) for s in (bnci_fold.manifest.seeds.model or [])]:
        raise ValueError(f"model_seed {model_seed} is not a declared manifest seed")
    if tuple(sorted(method_order)) != tuple(sorted(METHODS)):
        raise ValueError("method_order must be a permutation of the four methods")
    fd, maps, schedule, fs = (bnci_fold.fold_data, bnci_fold.maps, bnci_fold.deletion_schedule,
                              bnci_fold.fold_scope)
    exec_cfg, model_spec, cfg = bnci_fold.execution_config, bnci_fold.model_spec, bnci_fold.scope_config
    support_m = int(bnci_fold.manifest.enabled_datasets()["BNCI2014_001"].support_m)
    ref = level0_reference_prior(fd, maps)
    dev = torch.device(device)
    levels = {}
    for level in (0, 1):
        ss = build_level_support(fd, maps, level, schedule, ref, support_m=support_m)
        lp = build_level_population(fd, maps, ss)
        plans = build_level_plans(fs, level, ss, lp, cfg, model_seed=model_seed)
        rk = RunKey(fs.fold_key, level, int(model_seed))
        levels[level] = run_level_complete(rk, fd, ss, lp, fs, plans, exec_cfg, model_spec,
                                           bnci_fold.model_factory(), dev, method_order=tuple(method_order))
    return assemble_fold_run(fs, levels)
