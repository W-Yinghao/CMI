"""C86H F1 training coordinator — faithful 11-channel cell training via the FROZEN engine.

One cell = 81 candidates: 1 ERM stage-1 anchor + 40 OACI stage-2 checkpoints + 40 SRC stage-2
checkpoints. The coordinator reuses the frozen ``oaci.train.engine`` (train_stage1/train_stage2)
and the frozen ``OACIObjective``/``SRCObjective`` unchanged — only the model factory is the
11-channel ``build_c86h_model`` (in_chans 20 -> 11), which is the sole retarget. The one
real-data coupling in the C84 canary (``apply_level_intervention``, bound to the locked subject
panel registry) is bypassed by building the training bundle directly from synthetic/adapter
``(X, y, domain)`` arrays, exactly replicating ``materialize_training_bundle``'s array body.

Nothing here is changed from the registered training: optimizer=adam, wd=0, lr_stage1=0.005,
lr_encoder=0.01, lr_critic=0.01, dual_lr=0.5, lambda_init/max/floor, epochs 200/200, steps 20,
warmup 60, critic 5, checkpoint_every 5, OACI/SRC objectives, cadence range(4,200,5) -> 40.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from . import contract as K
from .field_spec import c86_candidate_id
from .f1f2_field import IN_CHANS, IN_TIMES, build_c86h_model

_REGIME_EPOCHS = tuple(range(4, 200, 5))             # canonical OACI/SRC checkpoint epochs (40)


def _canonical_epoch(regime: str, order: int) -> int:
    """Genealogy-position -> canonical checkpoint epoch (the candidate-ID uses this, not the
    actual per-run epoch, so a tiny run maps to the same canonical positions)."""
    if regime == "ERM":
        return 199
    return _REGIME_EPOCHS[order - 1]

# Faithful preset -> 81 candidates (matches the registered EngineConfig + cadence).
FAITHFUL = dict(S1_EP=200, S2_EP=200, SPE=20, WU=60, CS=5, PC=8, MB=256, ACC=4, BS=256,
                m=8, checkpoint_every=5)
# Tiny preset -> 5 candidates; exercises the whole engine/objective/plan path in seconds.
TINY = dict(S1_EP=2, S2_EP=10, SPE=2, WU=4, CS=2, PC=2, MB=256, ACC=None, BS=16,
            m=1, checkpoint_every=5)


def build_synthetic_source(n_domains: int = 2, per_cell: int = 8, seed: int = 0,
                           sep: float = 2.0) -> tuple:
    """Synthetic 11-channel source: (X[n,11,480], y[n], domain[n]) with n_domains >= 2 and each
    (domain, class) cell populated so OACI (comparable class) and SRC (>=2 domains) are active."""
    rng = np.random.default_rng(seed)
    Xs, ys, ds = [], [], []
    for d in range(n_domains):
        for c in range(2):
            for _ in range(per_cell):
                x = rng.standard_normal((IN_CHANS, IN_TIMES))
                x[:, :IN_TIMES // 2] += sep * (2 * c - 1)          # a class-dependent signal
                Xs.append(x); ys.append(c); ds.append(d)
    return (np.asarray(Xs, dtype=np.float64),
            np.asarray(ys, dtype=int), np.asarray(ds, dtype=int))


def train_cell(X, y, domain, seed: int, preset: dict = TINY, device=None) -> list:
    """Train one cell and return 81 (or preset) candidate records
    ``[(regime, CheckpointRecord, trajectory_order), ...]`` in canonical order (ERM, OACI, SRC)
    via the frozen engine with the 11-channel model."""
    import torch
    from oaci.train.data import TrainingData, population_signature_hash
    from oaci.train.engine import EngineConfig, train_stage1, train_stage2, InvocationRegistry
    from oaci.train.rng import forked_rng, derive_seed
    from oaci.methods.oaci import OACIObjective
    from oaci.methods.source_robust import SRCObjective
    from oaci.data.plan_sampler import UnitIndex
    from oaci.data.plan_materialize import (
        materialize_stage1_task_plan, materialize_stage2_task_plan,
        materialize_oaci_alignment_plan, materialize_full_domain_alignment_plan)
    from oaci.support_graph import build_support_graph, counts_from_labels, empirical_class_prior

    P = preset
    device = device or torch.device("cpu")
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=int)
    domain = np.asarray(domain, dtype=int)
    n = len(y)
    sample_ids = tuple(f"syn|d{int(domain[i])}|c{int(y[i])}|r{i}" for i in range(n))
    groups = tuple(f"g{int(domain[i])}" for i in range(n))
    mass = np.ones(n)
    data = TrainingData(
        X=torch.as_tensor(X, dtype=torch.float32),
        y=torch.as_tensor(y, dtype=torch.long),
        sample_id=sample_ids,
        sample_mass=torch.as_tensor(mass, dtype=torch.float32),
        n_classes=2,
        d=torch.as_tensor(domain, dtype=torch.long),
        group=groups).validate()

    n_dom = int(domain.max()) + 1
    counts = counts_from_labels(domain, y, n_domains=n_dom, n_classes=2)
    support = build_support_graph(
        counts, P["m"], cell_mass=counts.astype(float),
        reference_prior=empirical_class_prior(counts),
        domain_names=[str(i) for i in range(n_dom)],
        class_names=["left_hand", "right_hand"]).validate()

    index = UnitIndex(data.sample_id, y, domain, data.group, data.sample_id, mass)
    pop = population_signature_hash(data)
    total_inner = P["S2_EP"] * P["SPE"]
    stage1 = materialize_stage1_task_plan(index, pop, P["S1_EP"], 1, P["BS"], seed, "auto")
    stage2 = materialize_stage2_task_plan(index, pop, P["S2_EP"], P["SPE"], P["BS"], seed, "auto")
    oaci_plan = materialize_oaci_alignment_plan(
        index, support, pop, P["WU"], total_inner, P["CS"], P["PC"], P["MB"], seed,
        accumulation_steps=P["ACC"], replacement_mode="auto")
    full_plan = materialize_full_domain_alignment_plan(
        index, pop, P["WU"], total_inner, P["CS"], P["PC"], P["MB"], seed,
        accumulation_steps=P["ACC"], replacement_mode="auto")

    cfg = EngineConfig(
        metric="balanced_ce", epsilon=0.03, numerical_tol=1e-4,
        stage1_epochs=P["S1_EP"], stage1_steps_per_epoch=1,
        stage2_epochs=P["S2_EP"], steps_per_epoch=P["SPE"],
        warmup_steps=P["WU"], critic_steps=P["CS"], checkpoint_every=P["checkpoint_every"],
        guard_chunk_size=1024, optimizer_name="adam", weight_decay=0.0,
        lr_stage1=0.005, lr_encoder=0.01, lr_critic=0.01, dual_lr=0.5,
        lambda_init=0.3, lambda_max=20.0, lambda_floor=0.0,
        gradient_clip=0.0, critic_gradient_clip=0.0,
        deterministic_algorithms=False, stage2_bn_mode="frozen_erm_running_stats",
        base_seed=seed)

    factory = build_c86h_model
    with forked_rng(derive_seed(seed, "c86h", "model_init"), device):
        model = factory()
    erm = train_stage1(model, data, stage1, cfg, device, InvocationRegistry(), "C86H|cell")
    oaci_res = train_stage2(factory, erm, data, OACIObjective(support, adv_hidden=16),
                            stage2, oaci_plan, cfg, device)
    src_res = train_stage2(factory, erm, data, SRCObjective(2, support.n_domains,
                                                            smooth_temperature=0.1),
                           stage2, full_plan, cfg, device)
    candidates = [("ERM", erm.checkpoint, 0)]
    candidates += [("OACI", r, i) for i, r in enumerate(oaci_res.trajectory, start=1)]
    candidates += [("SRC", r, i) for i, r in enumerate(src_res.trajectory, start=1)]
    return candidates


# ------------------------------------------------------------------- F1: 648-model zoo
def f1_train_zoo(source_provider, out_root, preset: dict = TINY, cell_trainer=None,
                 contexts=None, save_weights: bool = True) -> dict:
    """Train the 11-ch candidate zoo (2 panels x 2 seeds x 2 levels x 81 = 648 under FAITHFUL).
    ``source_provider(panel, seed, level) -> (X, y, domain)`` supplies each cell (real adapter or
    synthetic). Returns the zoo manifest and writes weights + C86H_ZOO_MANIFEST.json."""
    from oaci.train.checkpoint import state_hash
    trainer = cell_trainer or train_cell
    contexts = contexts or [(p, s, lv) for p in K.PANELS for s in K.TRAINING_SEEDS
                            for lv in K.LEVELS]
    os.makedirs(out_root, exist_ok=True)
    zoo, by_context = {}, {}
    for (panel, seed, level) in contexts:
        X, y, domain = source_provider(panel, seed, level)
        cands = trainer(X, y, domain, seed=int(seed), preset=preset)
        ck = f"panel={panel}|seed={seed}|level={level}"
        ids = []
        for (regime, rec, order) in cands:
            epoch = _canonical_epoch(regime, order)
            cid = c86_candidate_id(K.COMMON_INTERFACE_ID, K.FIELD_TRAINING_MANIFEST_SHA256,
                                   panel, int(seed), int(level), regime, epoch)
            wsha = getattr(rec, "model_hash", "") or state_hash(rec.model_state)
            entry = {"candidate_id": cid, "regime": regime, "trajectory_order": int(order),
                     "canonical_epoch": int(epoch), "panel": panel, "seed": int(seed),
                     "level": int(level), "interface_id": K.COMMON_INTERFACE_ID,
                     "weight_sha256": wsha, "context_key": ck,
                     "genealogy": f"{ck}|{regime}|order{order}|epoch{epoch}"}
            if save_weights:
                import torch
                wp = os.path.join(out_root, f"{cid}.pt")
                torch.save(rec.model_state, wp)
                entry["weight_path"] = os.path.basename(wp)
            zoo[cid] = entry
            ids.append(cid)
        by_context[ck] = ids
    manifest = {"schema": "c86h_zoo_manifest_v1", "n_models": len(zoo),
                "interface_id": K.COMMON_INTERFACE_ID,
                "field_training_manifest_sha256": K.FIELD_TRAINING_MANIFEST_SHA256,
                "candidate_ids_by_context": by_context, "candidates": zoo}
    with open(os.path.join(out_root, "C86H_ZOO_MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


# ------------------------------------------------------------------- F2: prediction helper
def predict_probs(model_state, X) -> np.ndarray:
    """Softmax probabilities [n,2] from a candidate's saved 11-channel model_state on X[n,11,480]."""
    import torch
    model = build_c86h_model()
    model.load_state_dict(model_state)
    return _predict_model(model, X)


def _predict_model(model, X) -> np.ndarray:
    import torch
    model.eval()
    with torch.no_grad():
        out = model(torch.as_tensor(np.asarray(X), dtype=torch.float32))
        logits = out.logits if hasattr(out, "logits") else out
        return torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)


# ------------------------------------------------------------------- F2: field generation
def f2_generate_predictions(zoo_manifest, zoo_root, target_provider, field_root) -> dict:
    """Generate the C86H field (pool/oracle/contrib/held) + the content-addressed real-field
    manifest from the trained zoo. ``target_provider()`` yields
    ``(cohort, target_int, X[n,11,480], y[n], trial_ids[n], raw_sha)`` per target. Semantics B is
    honoured (one label per physical trial across the 8 contexts). C86-E on split/support failure."""
    import csv
    import torch
    from oaci.theory.c86_active_program import canonical_trial_split
    from . import field_spec
    from .f1f2 import C86EError, REAL_FIELD_MANIFEST_NAME
    pool_root = os.path.join(field_root, "acquisition_unlabeled_pool")
    oracle_root = os.path.join(field_root, "acquisition_label_oracle")
    contrib_root = os.path.join(field_root, "query_contribution_store")
    held_root = os.path.join(field_root, "held_evaluation_field")
    for d in (pool_root, oracle_root, contrib_root, held_root):
        os.makedirs(d, exist_ok=True)
    contexts = field_spec.field_context_keys()
    by_ctx = zoo_manifest["candidate_ids_by_context"]
    ctx_models = {}                                       # preload each context's candidates once
    for ck in contexts:
        ms = []
        for cid in by_ctx[ck]:
            m = build_c86h_model()
            m.load_state_dict(torch.load(os.path.join(
                zoo_root, zoo_manifest["candidates"][cid]["weight_path"]), weights_only=False))
            m.eval(); ms.append(m)
        ctx_models[ck] = ms
    n_cands = len(by_ctx[contexts[0]])

    label_rows, split_m, support, pred_shas, target_shas = [], {}, {}, {}, {}
    for cohort, target, X, y, tids, raw_sha in target_provider():
        ds, subj = cohort, int(target)
        y = np.asarray(y).astype(int)
        target_shas[f"{cohort}|{subj}"] = raw_sha
        pool_ids, held_ids = canonical_trial_split(ds, str(subj), list(tids), salt=K.SPLIT_SALT)
        idx = {t: i for i, t in enumerate(tids)}
        pj = [idx[t] for t in pool_ids]; hj = [idx[t] for t in held_ids]
        tkey = f"t_{cohort}_{subj}"
        for view, jj in (("pool", pj), ("held", hj)):
            cnt = {0: int(sum(y[j] == 0 for j in jj)), 1: int(sum(y[j] == 1 for j in jj))}
            if cnt[0] < K.MIN_CLASS_SUPPORT or cnt[1] < K.MIN_CLASS_SUPPORT:
                raise C86EError(f"class support fail {cohort} sub-{subj} {view}: {cnt}")
            support.setdefault(tkey, {})[view] = {"0": cnt[0], "1": cnt[1]}
        split_m[tkey] = {"pool": list(pool_ids), "held": list(held_ids)}
        for ck in contexts:
            cm = field_spec._ctx_meta(ck)
            pp = np.stack([_predict_model(m, X[pj]) for m in ctx_models[ck]], axis=1)  # [np,K,2]
            ph = np.stack([_predict_model(m, X[hj]) for m in ctx_models[ck]], axis=1)
            meta = json.dumps({"dataset": ds, "subject": subj, **cm})
            tag = f"{ds}__{subj}__p{cm['panel']}_s{cm['seed']}_l{cm['level']}"
            np.savez(os.path.join(pool_root, tag + ".npz"), meta=meta,
                     trial_ids=np.array(pool_ids), probabilities=pp)
            pc = field_spec._contribs(pp, y[pj])
            np.savez(os.path.join(contrib_root, tag + ".npz"), meta=meta,
                     trial_ids=np.array(pool_ids), true_label=y[pj], **pc)
            np.savez(os.path.join(held_root, tag + ".npz"), meta=meta,
                     trial_ids=np.array(held_ids), probabilities=ph, true_label=y[hj])
            pred_shas[f"{ds}|{subj}|{ck}"] = hashlib.sha256(pp.tobytes() + ph.tobytes()).hexdigest()
        for t in pool_ids:
            label_rows.append({"dataset": ds, "target_subject_id": subj,
                               "target_trial_id": t, "canonical_class_label": int(y[idx[t]])})
    with open(os.path.join(oracle_root, "labels.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "target_subject_id",
                                           "target_trial_id", "canonical_class_label"])
        w.writeheader(); w.writerows(label_rows)

    rfm = {"schema": "c86h_real_field_manifest_v1", "interface_id": K.COMMON_INTERFACE_ID,
           "field_training_manifest_sha256": K.FIELD_TRAINING_MANIFEST_SHA256,
           "n_targets": len(split_m), "n_contexts": len(split_m) * len(contexts),
           "n_candidates_per_context": n_cands,
           "n_candidate_context_slices": len(split_m) * len(contexts) * n_cands,
           "zoo": {"n_models": zoo_manifest["n_models"],
                   "weight_sha256": {c: e["weight_sha256"]
                                     for c, e in zoo_manifest["candidates"].items()}},
           "prediction_context_sha256": pred_shas, "construction_evaluation_overlap": 0,
           "split": split_m, "class_support": support, "target_raw_sha256": target_shas}
    with open(os.path.join(field_root, REAL_FIELD_MANIFEST_NAME), "w") as fh:
        json.dump(rfm, fh, indent=2)
    return rfm
