"""Parse + validate the confirmatory_v2 DRAFT protocol (distinct from the runnable manifest_v2).

This only READS and checks the protocol; it never executes anything. materialize.py turns one enabled
dataset + one held-out target into a runnable manifest_v2.
"""
from __future__ import annotations

from dataclasses import dataclass

_REQUIRED_TOP = ("protocol_id", "status", "datasets", "risk", "backbone", "optimizer", "training",
                 "sampler", "probe", "methods", "evaluation", "seeds", "k1", "k2")
_REQUIRED_DATASET = ("enabled", "cohort_ids", "class_names", "outer_target_factor", "domain_factor",
                     "group_factor", "support_unit_factor", "eval_unit_factor", "support_m", "channels",
                     "preprocessing")
_REQUIRED_PP = ("fmin", "fmax", "resample_sfreq", "epoch_tmin", "epoch_tmax", "normalization")


@dataclass(frozen=True)
class ConfirmatoryProtocol:
    raw: dict
    path: str

    @property
    def protocol_id(self) -> str:
        return str(self.raw["protocol_id"])

    @property
    def status(self) -> str:
        return str(self.raw["status"])

    def enabled_datasets(self) -> dict:
        return {k: v for k, v in self.raw["datasets"].items() if v.get("enabled")}

    def dataset(self, name: str) -> dict:
        ds = self.raw["datasets"].get(name)
        if ds is None:
            raise KeyError(f"dataset {name!r} not in protocol {self.path}")
        if not ds.get("enabled"):
            raise ValueError(f"dataset {name!r} is not enabled in {self.path}")
        return ds

    def block(self, name: str) -> dict:
        return self.raw[name]


def load_confirmatory(path: str) -> ConfirmatoryProtocol:
    import yaml
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    for k in _REQUIRED_TOP:
        if k not in d or d[k] in (None, {}, ""):
            raise ValueError(f"confirmatory protocol {path}: missing required top-level block {k!r}")
    if not isinstance(d["datasets"], dict) or not d["datasets"]:
        raise ValueError(f"confirmatory protocol {path}: 'datasets' must be a non-empty mapping")
    enabled = {k: v for k, v in d["datasets"].items() if v.get("enabled")}
    if not enabled:
        raise ValueError(f"confirmatory protocol {path}: no enabled datasets")
    for name, ds in enabled.items():
        for f in _REQUIRED_DATASET:
            if f not in ds or ds[f] in (None, [], {}, ""):
                raise ValueError(f"confirmatory dataset {name!r}: missing {f!r}")
        pp = ds["preprocessing"]
        for f in _REQUIRED_PP:
            if f not in pp or pp[f] is None:
                raise ValueError(f"confirmatory dataset {name!r} preprocessing: missing {f!r}")
        if pp["normalization"] != "zscore_sample":
            raise ValueError(f"confirmatory dataset {name!r}: only normalization=zscore_sample is supported here")
        if not (0 <= pp["fmin"] < pp["fmax"] < pp["resample_sfreq"] / 2):
            raise ValueError(f"confirmatory dataset {name!r}: require 0 <= fmin < fmax < resample_sfreq/2")
        if pp["epoch_tmax"] <= pp["epoch_tmin"]:
            raise ValueError(f"confirmatory dataset {name!r}: require epoch_tmax > epoch_tmin")
    if not d["seeds"].get("model"):
        raise ValueError(f"confirmatory protocol {path}: seeds.model must list the model seeds")
    return ConfirmatoryProtocol(raw=d, path=str(path))
