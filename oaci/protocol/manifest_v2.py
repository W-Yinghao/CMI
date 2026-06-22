"""Protocol v2 — heterogeneous, machine-expressible, with STRICT typed blocks.

Every scientific block (seeds, backbone, optimizer, training, sampler, probe, methods, evaluation,
risk, k1, k2, and per-dataset blocks) is a dataclass folded into the canonical JSON, so the
manifest SHA-256 is sensitive to any learning rate / epoch / channel order / bootstrap seed /
deleted-cell change. Parsing is STRICT: an unknown or misspelled key (top-level or inside any
block) is a hard error — never silently filtered. A ``status: smoke`` manifest must carry a
``smoke`` block and is rejected in confirmatory mode (and vice-versa).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields

DATASET_REQUIRED = ["enabled", "cohort_ids", "class_names", "outer_target_factor", "domain_factor",
                    "group_factor", "support_unit_factor", "eval_unit_factor", "support_m",
                    "channels", "preprocessing"]


def _strict(cls, d: dict, where: str):
    """Build a dataclass from ``d``, REJECTING unknown keys (typo detection)."""
    allowed = {f.name for f in fields(cls)}
    unknown = set(d) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown/misspelled key(s) {sorted(unknown)} (allowed: {sorted(allowed)})")
    return cls(**d)


@dataclass
class SeedBlock:
    split: int | None = None
    deletion: int | None = None
    model: list | None = None
    selection_bootstrap: int | None = None
    audit_bootstrap: int | None = None


@dataclass
class BackboneBlock:
    name: str | None = None
    temporal_filters: int | None = None
    temporal_kernel_samples: int | None = None
    pool_kernel_samples: int | None = None
    pool_stride_samples: int | None = None
    dropout: float | None = None
    safe_log_eps: float | None = None


@dataclass
class OptimizerBlock:
    name: str | None = None
    lr_stage1: float | None = None
    lr_encoder: float | None = None
    lr_critic: float | None = None
    weight_decay: float | None = None
    dual_lr: float | None = None
    lambda_init: float | None = None
    lambda_max: float | None = None
    lambda_floor: float | None = None
    gradient_clip: float | None = None
    critic_gradient_clip: float | None = None


@dataclass
class TrainingBlock:
    stage1_epochs: int | None = None
    stage2_epochs: int | None = None
    stage1_steps_per_epoch: int | None = None
    stage2_steps_per_epoch: int | None = None
    task_batch_size: int | None = None
    warmup_steps: int | None = None
    critic_steps: int | None = None
    checkpoint_every: int | None = None
    guard_chunk_size: int | None = None
    numerical_tol: float | None = None
    stage2_bn_mode: str | None = None
    selection_score_tolerance: float | None = None
    deterministic_algorithms: bool | None = None


@dataclass
class SamplerBlock:
    min_per_eligible_cell: int | None = None
    adv_microbatch_size: int | None = None
    adv_accumulation_steps: int | None = None
    replacement_mode: str | None = None


@dataclass
class ProbeBlock:
    capacities: list | None = None
    folds: int | None = None
    selection_bootstrap: int | None = None
    audit_bootstrap: int | None = None
    l2_C: float | None = None
    max_iter: int | None = None
    prob_floor: float | None = None
    feature_seed_base: int | None = None
    max_candidate_multiplier: int | None = None
    max_invalid_draw_rate: float | None = None


@dataclass
class MethodBlock:
    names: list | None = None
    global_lpc_laplace_smoothing: float | None = None
    critic_capacity: int | None = None


@dataclass
class RiskBlock:
    metric: str | None = None
    epsilon: float | None = None


@dataclass
class EvaluationBlock:
    alpha: float | None = None
    delta_bacc: float | None = None
    ece_bins: int | None = None
    paired_bootstrap: int | None = None
    invalid_draw_threshold: float | None = None
    min_clusters: int | None = None


@dataclass
class K1Block:
    statistic: str | None = None
    grouped_permutation_scheme: str | None = None
    n_permutations: int | None = None
    decision_rule: str | None = None


@dataclass
class K2Block:
    endpoints: list | None = None
    decision_rule: str | None = None


@dataclass
class SmokeBlock:
    subjects: list | None = None
    target_subjects: list | None = None
    source_audit_subjects: list | None = None
    source_train_subjects: list | None = None
    deletion_levels: list | None = None
    deleted_cell_level1: dict | None = None


@dataclass
class DatasetBlock:
    enabled: bool | None = None
    cohort_ids: list | None = None
    class_names: list | None = None
    outer_target_factor: str | None = None
    domain_factor: str | None = None
    group_factor: str | None = None
    support_unit_factor: str | None = None
    eval_unit_factor: str | None = None
    support_m: int | None = None
    channels: list | None = None
    preprocessing: dict | None = None
    expected_sfreq: float | None = None
    expected_epoch_seconds: float | None = None
    expected_n_times: int | None = None

    def missing(self) -> list:
        miss = [f for f in DATASET_REQUIRED if getattr(self, f) in (None, [], {}, "")]
        if self.channels is not None and not isinstance(self.channels, list):
            miss.append("channels(must be an explicit ordered list)")
        if self.support_m is not None and (isinstance(self.support_m, bool) or not isinstance(self.support_m, int)):
            miss.append("support_m(must be int)")
        return miss


# blocks every RUNNABLE manifest must specify explicitly (no Python defaults)
RUNNABLE_BLOCKS = ["seeds", "backbone", "optimizer", "training", "sampler", "probe", "methods",
                   "evaluation", "risk"]


@dataclass
class ProtocolManifestV2:
    protocol_id: str | None = None
    status: str | None = None
    seeds: SeedBlock | None = None
    backbone: BackboneBlock | None = None
    optimizer: OptimizerBlock | None = None
    training: TrainingBlock | None = None
    sampler: SamplerBlock | None = None
    probe: ProbeBlock | None = None
    methods: MethodBlock | None = None
    smoke: SmokeBlock | None = None
    datasets: dict | None = None          # name -> DatasetBlock
    risk: RiskBlock | None = None
    evaluation: EvaluationBlock | None = None
    k1: K1Block | None = None
    k2: K2Block | None = None

    def enabled_datasets(self) -> dict:
        return {k: v for k, v in (self.datasets or {}).items() if getattr(v, "enabled", False)}

    def validate_complete(self) -> "ProtocolManifestV2":
        for f in ["protocol_id", "status", "datasets", "k1", "k2"] + RUNNABLE_BLOCKS:
            if getattr(self, f) in (None, {}, ""):
                raise ValueError(f"protocol v2 missing required field/block: {f}")
        if self.status == "smoke" and self.smoke is None:
            raise ValueError("a status='smoke' manifest must carry a 'smoke' block")
        en = self.enabled_datasets()
        if not en:
            raise ValueError("protocol v2 has no enabled datasets")
        for name, blk in en.items():
            miss = blk.missing()
            if miss:
                raise ValueError(f"dataset block {name!r} incomplete: {miss}")
        self.validate_ranges()
        return self

    def validate_ranges(self) -> None:
        def _chk(cond, msg):
            if not cond:
                raise ValueError(f"manifest value out of range: {msg}")
        r, e, o, t, p, mb = self.risk, self.evaluation, self.optimizer, self.training, self.probe, self.methods
        _chk(r.metric in ("ce", "balanced_ce"), f"risk.metric={r.metric!r}")
        _chk(r.epsilon is not None and r.epsilon >= 0, "epsilon >= 0")
        _chk(e.alpha is not None and 0 < e.alpha < 1, "0 < alpha < 1")
        _chk(e.delta_bacc is not None and e.delta_bacc >= 0, "delta_bacc >= 0")
        _chk(e.ece_bins is not None and e.ece_bins >= 2, "ece_bins >= 2")
        _chk(e.paired_bootstrap is not None and e.paired_bootstrap >= 1, "paired_bootstrap >= 1")
        _chk(p.folds is not None and p.folds >= 2, "folds >= 2")
        caps = list(p.capacities or [])
        _chk(len(caps) > 0 and len(set(caps)) == len(caps) and all(c >= 0 for c in caps),
             "capacities non-empty/unique/>=0")
        _chk(p.l2_C is not None and p.l2_C > 0, "l2_C > 0")
        _chk(p.max_iter is not None and p.max_iter >= 1, "max_iter >= 1")
        _chk(p.prob_floor is not None and 0 < p.prob_floor < 1, "0 < prob_floor < 1")
        _chk(p.max_candidate_multiplier is not None and p.max_candidate_multiplier >= 1, "max_candidate_multiplier >= 1")
        _chk(p.max_invalid_draw_rate is not None and 0 <= p.max_invalid_draw_rate < 1, "0 <= max_invalid_draw_rate < 1")
        _chk(mb.global_lpc_laplace_smoothing is not None and mb.global_lpc_laplace_smoothing > 0,
             "global_lpc_laplace_smoothing > 0")
        _chk(o.lambda_floor == 0, "lambda_floor == 0 (main protocol)")
        _chk(t.stage2_bn_mode == "frozen_erm_running_stats", "stage2_bn_mode == frozen_erm_running_stats")

    def assert_confirmatory(self) -> "ProtocolManifestV2":
        if self.status == "smoke":
            raise ValueError("a status='smoke' manifest cannot be used in confirmatory mode "
                             "(use a separate smoke manifest, not a shortened confirmatory one)")
        return self.validate_complete()

    def to_canonical_json(self) -> str:
        d = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "datasets":
                d[f.name] = {k: asdict(b) for k, b in (v or {}).items()}
            elif hasattr(v, "__dataclass_fields__"):
                d[f.name] = asdict(v)
            else:
                d[f.name] = v
        return json.dumps(d, sort_keys=True, default=str)

    def freeze(self) -> dict:
        self.validate_complete()
        canon = self.to_canonical_json()
        return {"canonical_json": canon, "sha256": hashlib.sha256(canon.encode()).hexdigest()}


_BLOCK_TYPES = {"seeds": SeedBlock, "backbone": BackboneBlock, "optimizer": OptimizerBlock,
                "training": TrainingBlock, "sampler": SamplerBlock, "probe": ProbeBlock,
                "methods": MethodBlock, "smoke": SmokeBlock, "risk": RiskBlock,
                "evaluation": EvaluationBlock, "k1": K1Block, "k2": K2Block}
_PASSTHROUGH = {"protocol_id", "status"}


def load_v2(path: str) -> ProtocolManifestV2:
    try:
        import yaml
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"PyYAML required to load {path}: {e}")
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    allowed = set(_BLOCK_TYPES) | _PASSTHROUGH | {"datasets"}
    unknown = set(d) - allowed
    if unknown:
        raise ValueError(f"protocol v2 {path}: unknown top-level key(s) {sorted(unknown)}")
    kw = {k: d.get(k) for k in _PASSTHROUGH}
    for name, cls in _BLOCK_TYPES.items():
        if name in d and d[name] is not None:
            kw[name] = _strict(cls, d[name], f"{path}:{name}")
    kw["datasets"] = {name: _strict(DatasetBlock, blk or {}, f"{path}:datasets.{name}")
                      for name, blk in (d.get("datasets") or {}).items()}
    return ProtocolManifestV2(**kw)
