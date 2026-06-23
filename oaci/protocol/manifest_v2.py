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
    mlp_z_dim: int | None = None            # MLP backbone: frozen, no Python default
    mlp_hidden: int | None = None


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
    feature_chunk_size: int | None = None       # selection/audit feature extraction (affects float path)
    prediction_chunk_size: int | None = None     # prediction forward (A2b-1b-ii)
    numerical_tol: float | None = None
    stage2_bn_mode: str | None = None
    selection_score_tolerance: float | None = None
    deterministic_algorithms: bool | None = None


@dataclass
class SamplerBlock:
    min_per_eligible_cell: int | None = None    # OACI alignment per-cell coverage
    min_per_observed_cell: int | None = None    # full-domain alignment per-cell coverage (separate)
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
    prediction_prob_floor: float | None = None   # eval-unit aggregation floor (affects logits/NLL/ECE)


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
class DeletedCellBlock:
    domain_id: str | None = None
    class_name: str | None = None


@dataclass
class MIPreprocessingBlock:
    kind: str | None = None                  # "moabb_motor_imagery"
    fmin: float | None = None
    fmax: float | None = None
    resample_sfreq: float | None = None
    epoch_tmin: float | None = None
    epoch_tmax: float | None = None
    baseline: object | None = None           # null | [lo, hi]
    normalization: str | None = None
    normalization_eps: float | None = None
    channel_interpolation: bool | None = None
    code_version: str | None = None


@dataclass
class SyntheticPreprocessingBlock:
    kind: str | None = None                  # "deterministic_synthetic_identity"


_PREPROCESSING_KINDS = {"moabb_motor_imagery": MIPreprocessingBlock,
                        "deterministic_synthetic_identity": SyntheticPreprocessingBlock}


@dataclass
class FakeFixtureBlock:
    source_domain_ids: list | None = None
    target_domain_ids: list | None = None
    source_train_groups_per_domain: int | None = None
    source_audit_groups_per_domain: int | None = None
    target_groups_per_domain: int | None = None
    windows_per_unit_cycle: list | None = None
    input_dim: int | None = None
    class_signal_scale: float | None = None
    domain_signal_scale: float | None = None
    recording_signal_scale: float | None = None
    window_noise_scale: float | None = None
    data_seed: int | None = None


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
    fake_fixture: FakeFixtureBlock | None = None
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
        self._validate_fake()
        self._validate_smoke_subjects()
        self._validate_mi_preprocessing()
        return self

    def _validate_mi_preprocessing(self) -> None:
        for name, ds in self.enabled_datasets().items():
            pp = ds.preprocessing
            if not isinstance(pp, MIPreprocessingBlock):
                continue
            for f in fields(pp):
                if f.name != "baseline" and getattr(pp, f.name) is None:
                    raise ValueError(f"dataset {name} MI preprocessing missing {f.name}")
            if not (0 <= pp.fmin < pp.fmax < pp.resample_sfreq / 2):
                raise ValueError(f"{name}: 0 <= fmin < fmax < resample_sfreq/2")
            if pp.epoch_tmax <= pp.epoch_tmin:
                raise ValueError(f"{name}: epoch_tmax > epoch_tmin")
            if pp.normalization != "zscore_sample":
                raise ValueError(f"{name}: this smoke requires normalization == zscore_sample")
            if pp.channel_interpolation is not False:
                raise ValueError(f"{name}: channel_interpolation must be False (no target-driven interpolation)")
            if pp.normalization_eps <= 0:
                raise ValueError(f"{name}: normalization_eps > 0")
            if ds.expected_sfreq is not None and float(ds.expected_sfreq) != float(pp.resample_sfreq):
                raise ValueError(f"{name}: expected_sfreq must equal resample_sfreq")
            if ds.expected_epoch_seconds is not None \
                    and abs(float(ds.expected_epoch_seconds) - (pp.epoch_tmax - pp.epoch_tmin)) > 1e-9:
                raise ValueError(f"{name}: expected_epoch_seconds must equal epoch_tmax - epoch_tmin")
            exp_nt = int(round((pp.epoch_tmax - pp.epoch_tmin) * pp.resample_sfreq)) + 1
            if ds.expected_n_times is not None and int(ds.expected_n_times) != exp_nt:
                raise ValueError(f"{name}: expected_n_times must be {exp_nt} (MNE includes both endpoints)")

    def _validate_smoke_subjects(self) -> None:
        sm = self.smoke
        if sm is None or not sm.subjects:                  # only the real-data smoke pins subject roles
            return
        subs = [int(x) for x in sm.subjects]
        roles = {"target": [int(x) for x in (sm.target_subjects or [])],
                 "source_audit": [int(x) for x in (sm.source_audit_subjects or [])],
                 "source_train": [int(x) for x in (sm.source_train_subjects or [])]}
        flat = roles["target"] + roles["source_audit"] + roles["source_train"]
        if len(set(flat)) != len(flat):
            raise ValueError("smoke subject roles overlap (target / source_audit / source_train must be disjoint)")
        if set(flat) != set(subs):
            raise ValueError("the union of smoke subject roles must equal smoke.subjects")
        if sorted(roles["target"]) != [1] or sorted(roles["source_audit"]) != [2, 3] \
                or sorted(roles["source_train"]) != [4, 5, 6]:
            raise ValueError("smoke roles must be target=[1], source_audit=[2,3], source_train=[4,5,6]")

    def _validate_fake(self) -> None:
        en = self.enabled_datasets()
        is_fake = (self.status == "smoke" and set(en) == {"FAKE_TWO_LEVEL"})
        if self.fake_fixture is not None and not is_fake:
            raise ValueError("fake_fixture is only allowed in a status=smoke FAKE_TWO_LEVEL manifest")
        if is_fake and self.fake_fixture is None:
            raise ValueError("a status=smoke FAKE_TWO_LEVEL manifest requires a fake_fixture block")
        if self.fake_fixture is None:
            return
        ff, ds, sm = self.fake_fixture, en["FAKE_TWO_LEVEL"], self.smoke
        for f in fields(ff):
            if getattr(ff, f.name) is None:
                raise ValueError(f"fake_fixture missing required field: {f.name}")
        if len(ds.channels) != int(ff.input_dim):
            raise ValueError("len(dataset.channels) must equal fake_fixture.input_dim")
        if list(sm.deletion_levels or []) != [0, 1]:
            raise ValueError("fake smoke.deletion_levels must be [0, 1]")
        dc = sm.deleted_cell_level1
        if not isinstance(dc, DeletedCellBlock) or dc.domain_id != "S0" or dc.class_name != "c1":
            raise ValueError("fake deleted_cell_level1 must be {domain_id: S0, class_name: c1}")
        for f in ("subjects", "target_subjects", "source_audit_subjects", "source_train_subjects"):
            if getattr(sm, f):
                raise ValueError(f"fake smoke.{f} must be empty (no real-EEG subject semantics)")
        if list(ff.windows_per_unit_cycle) != [1, 2, 3] or int(ff.input_dim) < 1:
            raise ValueError("fake fixture window cycle / input_dim out of range")

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
        _chk(e.prediction_prob_floor is not None and 0 < e.prediction_prob_floor < 1, "0 < prediction_prob_floor < 1")
        for nm, blk in self.enabled_datasets().items():          # the floor must leave room for every class
            _chk(e.prediction_prob_floor * len(blk.class_names or []) < 1,
                 f"prediction_prob_floor * n_classes < 1 for dataset {nm}")
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
        bb = self.backbone
        _shallow = (bb.temporal_filters, bb.temporal_kernel_samples, bb.pool_kernel_samples,
                    bb.pool_stride_samples, bb.dropout, bb.safe_log_eps)
        if bb.name == "mlp":                                   # MLP arch fully frozen, shallow fields absent
            _chk(bb.mlp_z_dim is not None and bb.mlp_z_dim >= 1, "mlp_z_dim >= 1")
            _chk(bb.mlp_hidden is not None and bb.mlp_hidden >= 0, "mlp_hidden >= 0")
            _chk(all(x is None for x in _shallow), "mlp backbone must leave ShallowConvNet fields unset")
        elif bb.name == "shallow_convnet":                     # shallow fully specified, mlp fields absent
            _chk(all(x is not None for x in _shallow), "shallow_convnet requires all structural fields")
            _chk(bb.mlp_z_dim is None and bb.mlp_hidden is None, "shallow_convnet must leave mlp fields unset")
        else:
            _chk(False, f"unknown backbone name {bb.name!r}")

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


def manifest_logical_payload(manifest: "ProtocolManifestV2") -> dict:
    """The parsed canonical manifest mapping (NOT a {'canonical_json': '...'} wrapper)."""
    return json.loads(manifest.to_canonical_json())


def manifest_payload_hash(payload: dict) -> str:
    """Re-encode the payload exactly as the manifest does -> the same manifest SHA-256."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


_BLOCK_TYPES = {"seeds": SeedBlock, "backbone": BackboneBlock, "optimizer": OptimizerBlock,
                "training": TrainingBlock, "sampler": SamplerBlock, "probe": ProbeBlock,
                "methods": MethodBlock, "smoke": SmokeBlock, "fake_fixture": FakeFixtureBlock,
                "risk": RiskBlock, "evaluation": EvaluationBlock, "k1": K1Block, "k2": K2Block}
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
    m = ProtocolManifestV2(**kw)
    if m.smoke is not None and isinstance(m.smoke.deleted_cell_level1, dict):   # strict nested deleted cell
        m.smoke.deleted_cell_level1 = _strict(DeletedCellBlock, m.smoke.deleted_cell_level1,
                                              f"{path}:smoke.deleted_cell_level1")
    for name, ds in (m.datasets or {}).items():                                # strict nested preprocessing by kind
        pp = ds.preprocessing
        if isinstance(pp, dict) and pp.get("kind") in _PREPROCESSING_KINDS:
            ds.preprocessing = _strict(_PREPROCESSING_KINDS[pp["kind"]], pp, f"{path}:datasets.{name}.preprocessing")
    return m
