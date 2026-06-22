"""Source-train leakage selection: dedup feasible checkpoints, extract features once per state,
score each unique checkpoint once, hand the selector a pure dict lookup. The ERM leakage score is
computed once and read by the three Stage-2 sessions (cache request/compute/hit = 4/1/3). On a
structurally non-estimable level every method conservatively selects ERM with NO feature/probe work.
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from dataclasses import dataclass

from ..leakage.cache import LeakageScoreCache, LeakageScoreKey, critic_config_hash, frozen_feature_hash
from ..train.rng import derive_seed
from ..train.selector import Selection, select_checkpoint
from .features import extract_frozen_features
from .results import SelectedMethod
from .scoring import SelectionScoringSession, compute_leakage_score, overlap_probe_sample_ids

_STAGE2 = ("OACI", "global_lpc", "uniform")


def unique_feasible_records(train_result, *, numerical_tol):
    """Risk-feasible Stage-2 records, deduped by model_hash, ERM duplicates dropped, in trajectory
    order. A repeated hash with inconsistent R_src fails. Infeasible records are never returned (so
    they never trigger feature extraction)."""
    tau = train_result.erm_stage.tau
    erm_hash = train_result.erm_record.model_hash
    out, seen = [], {}
    for c in train_result.trajectory:
        if c.R_src > tau + numerical_tol or c.model_hash == erm_hash:
            continue
        if c.model_hash in seen:
            if abs(seen[c.model_hash] - c.R_src) > max(numerical_tol, 1e-9):
                raise ValueError(f"checkpoint {c.model_hash} has inconsistent R_src across duplicates")
            continue
        seen[c.model_hash] = c.R_src
        out.append(c)
    return tuple(out)


@dataclass(frozen=True)
class FeatureArtifactKey:
    model_hash: str
    tensor_hash: str
    population_hash: str
    model_spec_hash: str
    feature_chunk_size: int | None


class FeatureArtifactCache:
    def __init__(self):
        self._store = {}
        self._req = defaultdict(int); self._comp = defaultdict(int); self._hit = defaultdict(int)

    def get_or_extract(self, key: FeatureArtifactKey, extract_fn):
        self._req[key] += 1
        if key in self._store:
            self._hit[key] += 1
            return self._store[key]
        self._comp[key] += 1
        self._store[key] = extract_fn()
        return self._store[key]

    def request_count(self, key): return int(self._req[key])
    def compute_count(self, key): return int(self._comp[key])
    def hit_count(self, key): return int(self._hit[key])
    def total_requests(self): return int(sum(self._req.values()))
    def total_computes(self): return int(sum(self._comp.values()))
    def total_hits(self): return int(sum(self._hit.values()))


def make_leakage_score_key(feature_artifact, support_graph, fold_plan, bootstrap_plan, critic_cfg) -> LeakageScoreKey:
    return LeakageScoreKey(model_hash=feature_artifact.model_hash,
                           frozen_feature_hash=frozen_feature_hash(feature_artifact.features.Z),
                           population_hash=feature_artifact.population_hash,
                           support_hash=support_graph.support_hash(), fold_plan_hash=fold_plan.plan_hash,
                           bootstrap_plan_hash=bootstrap_plan.plan_hash,
                           critic_config_hash=critic_config_hash(critic_cfg))


def _erm_only_selection(erm_stage, score):
    c = erm_stage.checkpoint
    return Selection(used_erm_fallback=True, selected_erm=True, selection_reason="erm_best",
                     selected_epoch=-1, model_state=c.model_state, model_hash=c.model_hash,
                     R_src=erm_stage.R_ERM_hat, selection_score=score, n_feasible=0,
                     score_name="selection_bootstrap_ucl")


def select_methods(run_key, trained_methods, stage1_run, level_population, level_plans, support_state,
                   fold_scope, execution_cfg, model_spec, model_factory, device, provenance):
    erm_stage = stage1_run.erm_stage
    erm_hash = erm_stage.checkpoint.model_hash
    tol = execution_cfg.engine_template.numerical_tol

    if level_plans.selection_status != "estimable":                       # ---- non-estimable ----
        provenance.record_fit("selection", [])                            # empty overlap-probe fit
        selected = {n: SelectedMethod(tm, _erm_only_selection(erm_stage, None),
                                      level_plans.selection_status, None, ())
                    for n, tm in trained_methods.items()}
        return selected, {}, {}

    # ---- estimable ----
    sg = support_state.support_graph
    design, fold, boot = level_plans.selection_design, level_plans.selection_fold_plan, level_plans.selection_bootstrap_plan
    critic = execution_cfg.critic
    data = level_population.training_data
    leakage_cache, feat_cache = LeakageScoreCache(), FeatureArtifactCache()
    fseed = derive_seed(run_key.model_seed, "selection_feature", run_key.run_key_hash)

    def extract(model_hash, model_state):
        key = FeatureArtifactKey(model_hash, level_population.tensor_hash, design.population_hash,
                                 model_spec.model_spec_hash, execution_cfg.feature_chunk_size)
        return feat_cache.get_or_extract(key, lambda: extract_frozen_features(
            model_state, model_hash, model_factory, data, design, factory_seed=fseed,
            chunk_size=execution_cfg.feature_chunk_size, device=device))

    erm_feat = extract(erm_hash, erm_stage.checkpoint.model_state)
    erm_key = make_leakage_score_key(erm_feat, sg, fold, boot, critic)
    erm_scorer = lambda: compute_leakage_score(erm_feat.features, sg, fold, boot, critic)
    erm_prefetch = leakage_cache.get_or_compute(erm_key, erm_scorer)       # request 1 / compute 1

    selected = {"ERM": SelectedMethod(trained_methods["ERM"],                # reuses prefetch, no new request
                                      _erm_only_selection(erm_stage, float(erm_prefetch["bootstrap_ucl"])),
                                      "estimable", erm_prefetch, ())}

    for name in _STAGE2:
        tm = trained_methods[name]
        feasible = unique_feasible_records(tm.train_result, numerical_tol=tol)

        def ck_scorer(record):
            feat = extract(record.model_hash, record.model_state)
            key = make_leakage_score_key(feat, sg, fold, boot, critic)
            return leakage_cache.get_or_compute(key, lambda: compute_leakage_score(feat.features, sg, fold, boot, critic))

        session = SelectionScoringSession(name, leakage_cache, erm_key, erm_hash, erm_scorer, ck_scorer)
        sub = dataclasses.replace(tm.train_result, trajectory=list(feasible))
        sel = select_checkpoint(sub, score_fn=session.score, numerical_tol=tol,
                                score_name="selection_bootstrap_ucl",
                                selection_score_tolerance=execution_cfg.selection_score_tolerance)
        selected[name] = SelectedMethod(tm, sel, "estimable", session.result(sel.model_hash),
                                        tuple(c.model_hash for c in feasible))

    provenance.record_fit("selection", overlap_probe_sample_ids(design, sg))
    stats = {"erm_request": leakage_cache.request_count(erm_key), "erm_compute": leakage_cache.compute_count(erm_key),
             "erm_hit": leakage_cache.hit_count(erm_key)}
    return selected, stats, {"erm_feature_computes": feat_cache.total_computes()}
