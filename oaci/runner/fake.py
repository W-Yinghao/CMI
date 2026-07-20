"""In-memory two-level run over the deterministic FAKE fixture (A2b-2b-i).

The FoldScope (and its frozen source-audit plans / target signatures) is built once in the FakeFold and
reused for both levels; each level runs the full train -> select -> audit -> predict pipeline and the
two levels are assembled. No artifact write, no CLI, no GPU.
"""
from __future__ import annotations

import torch

from ..methods.activity import METHODS
from .fold import run_level_complete
from .finalize import assemble_fold_run
from .keys import RunKey
from .scope import build_level_population
from .support import build_level_support, level0_reference_prior

DEFAULT_METHOD_ORDER = ("ERM", "OACI", "global_lpc", "uniform")


def run_fake_two_level_in_memory(fake_fold, *, model_seed, method_order=DEFAULT_METHOD_ORDER, device="cpu"):
    if int(model_seed) not in [int(s) for s in (fake_fold.manifest.seeds.model or [])]:
        raise ValueError(f"model_seed {model_seed} is not in the manifest's declared seeds")
    if tuple(sorted(method_order)) != tuple(sorted(METHODS)):
        raise ValueError("method_order must be a permutation of the four methods")

    fd, maps, schedule, fs = fake_fold.fold_data, fake_fold.maps, fake_fold.deletion_schedule, fake_fold.fold_scope
    exec_cfg, model_spec = fake_fold.execution_config, fake_fold.model_spec
    support_m = int(fake_fold.manifest.enabled_datasets()["FAKE_TWO_LEVEL"].support_m)
    ref = level0_reference_prior(fd, maps)
    dev = torch.device(device)

    from .plans import build_level_plans
    levels = {}
    for level in (0, 1):
        ss = build_level_support(fd, maps, level, schedule, ref, support_m=support_m)
        lp = build_level_population(fd, maps, ss)
        plans = build_level_plans(fs, level, ss, lp, fake_fold.scope_config, model_seed=model_seed)
        rk = RunKey(fs.fold_key, level, int(model_seed))
        levels[level] = run_level_complete(rk, fd, ss, lp, fs, plans, exec_cfg, model_spec,
                                           fake_fold.model_factory(), dev, method_order=tuple(method_order))
    return assemble_fold_run(fs, levels)
