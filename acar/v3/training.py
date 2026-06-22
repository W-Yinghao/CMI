"""ACAR v3 candidate TRAINING (HSCR). DESIGN/DEV stage — SYNTHETIC only until DEV_DESIGN_LOCK.

Two registered interfaces (Amendment 4):
  fit_candidate_earlystop(train, val, ...)  -> per-fold FittedCandidateArtifact (val early-stop, best-epoch restore;
                                               sigma_min from the FULL fold FIT = TRAIN∪VAL raw σ̂)
  refit_candidate_fixed_epochs(all_dev, n_epochs, sigma_min_oof, ...) -> final artifact (NO early stop; OOF σ_min)
  final_epochs(best_epochs) = round_half_up(median_k(best_epoch_k + 1))     [best_epoch is 0-based -> +1]
Exact losses (Huber/Seitzer β-NLL with v.detach()**β / pinball), subject-balanced reduction, Adam+grad-clip,
deterministic (CPU/1-thread/seeded/use_deterministic_algorithms), FIT-only normalizers. FAIL-CLOSED: TRAIN/VAL
subject overlap, malformed eligible batch, non-finite loss/grad/param, no best epoch, or a missing action all raise.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import math
import numpy as np
import torch

from .set_features import WindowActionSet, NON_IDENTITY
from .normalizers import InputNormalizer, TargetNormalizer
from .predictors import DeepSetsNet, make_artifact, HP, env_versions, CANDIDATES
from .data import SubjectKey


@dataclass(frozen=True, slots=True)
class TrainExample:
    subject_key: SubjectKey
    deployment_batch_digest: str
    action: str
    window_action_set: WindowActionSet
    delta_r: float

    def __post_init__(self):
        if not isinstance(self.subject_key, SubjectKey):
            raise TypeError("subject_key must be a SubjectKey")
        if self.action not in NON_IDENTITY:
            raise ValueError("action must be non-identity")
        if self.window_action_set.action_name != self.action:
            raise ValueError("window_action_set.action_name != action")
        if len(self.deployment_batch_digest) != 64:
            raise ValueError("deployment_batch_digest must be a full SHA-256")
        if not math.isfinite(float(self.delta_r)):
            raise ValueError("delta_r must be finite")


def _huber(p, y, d):
    e = p - y; ae = torch.abs(e)
    return torch.where(ae <= d, 0.5 * e * e, d * (ae - 0.5 * d))


def _beta_nll(mu, sigma, y, beta):
    v = sigma * sigma
    return (0.5 * ((y - mu) ** 2 / v + torch.log(v))) * (v.detach() ** beta)


def _pinball(y, q, tau):
    e = y - q
    return torch.maximum(tau * e, (tau - 1.0) * e)


def _example_loss(net, win, ctx, action, y, candidate):
    a, b = net(win, ctx, action)
    if candidate == "C1":
        return _huber(a, y, HP["huber_delta"])
    if candidate == "C2":
        return _beta_nll(a, b, y, HP["beta_nll"])
    return 0.5 * _pinball(y, a, 0.5) + 0.5 * _pinball(y, a + b, 0.9)


def _validate(examples, disease):
    """Return {subject_key: list[TrainExample]}; FAIL-CLOSED on malformed structure."""
    if not examples:
        raise ValueError("empty example set")
    by_subj = defaultdict(list); by_batch = defaultdict(set); dup = set()
    for e in examples:
        if not isinstance(e, TrainExample):
            raise TypeError("examples must be TrainExample")
        key = (e.deployment_batch_digest, e.action)
        if key in dup:
            raise ValueError(f"duplicate (batch_digest, action): {key}")
        dup.add(key)
        by_subj[e.subject_key].append(e)
        by_batch[(e.subject_key, e.deployment_batch_digest)].add(e.action)
    for b, acts in by_batch.items():
        if acts != set(NON_IDENTITY):
            raise ValueError(f"eligible batch {b} actions {sorted(acts)} != {sorted(NON_IDENTITY)}")
    for sk, exs in by_subj.items():
        if not any(True for _ in exs):
            raise ValueError(f"subject {sk} has no complete eligible batch")
    return by_subj


def _prep(by_subj, inorm, tnorm):
    out = {}
    for sk, exs in by_subj.items():
        rows = []
        for e in exs:
            nw = inorm.transform(e.window_action_set)
            win = torch.tensor(np.concatenate([nw.values, nw.availability_mask.astype(np.float64)], 1), dtype=torch.float32)
            ctx = torch.tensor(np.concatenate([nw.context_values, nw.context_mask.astype(np.float64)]), dtype=torch.float32)
            y = torch.tensor(float(tnorm.standardize(e.delta_r)), dtype=torch.float32)
            rows.append((win, ctx, e.action, y))
        out[sk] = rows
    return out


def _subject_balanced_loss(net, by_subject, candidate, subjects):
    sl = []
    for sk in subjects:
        el = [_example_loss(net, w, c, a, y, candidate) for (w, c, a, y) in by_subject[sk]]
        sl.append(torch.stack(el).mean())
    loss = torch.stack(sl).mean()
    if not torch.isfinite(loss):
        raise ValueError("non-finite loss")
    return loss


def _opt_step(net, opt, loss):
    opt.zero_grad(); loss.backward()
    for p in net.parameters():
        if p.grad is not None and not torch.all(torch.isfinite(p.grad)):
            raise ValueError("non-finite gradient")
    torch.nn.utils.clip_grad_norm_(net.parameters(), HP["grad_clip"]); opt.step()
    for p in net.parameters():
        if not torch.all(torch.isfinite(p)):
            raise ValueError("non-finite parameter after step")


def _sigma_min(net, by_subject, tnorm):
    per = defaultdict(list)
    with torch.no_grad():
        for rows in by_subject.values():
            for (w, c, a, _y) in rows:
                _, sig = net(w, c, a)
                per[a].append(float(sig) * tnorm.sd)
    sm = {}
    for a in NON_IDENTITY:
        if not per.get(a):
            raise ValueError(f"C2 sigma_min: action {a!r} has no FIT prediction (candidate failure)")
        v = float(np.quantile(per[a], HP["sigma_min_quantile"]))
        if not (math.isfinite(v) and v > 0):
            raise ValueError(f"C2 sigma_min[{a}] non-positive ({v})")
        sm[a] = v
    return sm


def _det_ctx():
    return (torch.are_deterministic_algorithms_enabled(), torch.get_num_threads())


def fit_candidate_earlystop(candidate, disease, train_examples, val_examples, seed):
    if candidate not in CANDIDATES:
        raise ValueError("bad candidate")
    tr_by = _validate(train_examples, disease); va_by = _validate(val_examples, disease)
    if set(tr_by) & set(va_by):
        raise ValueError("TRAIN and VAL share SubjectKey(s) — early stopping must be subject-disjoint")
    pdet, pth = _det_ctx()
    try:
        torch.use_deterministic_algorithms(True); torch.set_num_threads(1); torch.manual_seed(seed)
        inorm = InputNormalizer.fit([e.window_action_set for e in train_examples])
        tnorm = TargetNormalizer.fit([e.delta_r for e in train_examples])
        tr = _prep(tr_by, inorm, tnorm); va = _prep(va_by, inorm, tnorm)
        tr_subj = sorted(tr, key=str); va_subj = sorted(va, key=str)
        net = DeepSetsNet(candidate, seed)
        opt = torch.optim.Adam(net.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
        rng = np.random.default_rng(seed)
        best_val = math.inf; best_state = None; best_epoch = -1; bad = 0
        for epoch in range(HP["max_epochs"]):
            net.train(); order = list(tr_subj); rng.shuffle(order)
            for s in range(0, len(order), HP["batch_subjects"]):
                _opt_step(net, opt, _subject_balanced_loss(net, tr, candidate, order[s:s + HP["batch_subjects"]]))
            net.eval()
            with torch.no_grad():
                vl = float(_subject_balanced_loss(net, va, candidate, va_subj))
            if not math.isfinite(vl):
                raise ValueError("non-finite validation loss")
            if vl < best_val - HP["min_delta"]:
                best_val = vl; best_state = {k: v.clone() for k, v in net.state_dict().items()}; best_epoch = epoch; bad = 0
            else:
                bad += 1
                if bad >= HP["patience"]:
                    break
        if best_state is None:
            raise ValueError("no improving epoch — candidate failure")
        net.load_state_dict(best_state); net.eval()
        fit_all = _prep({**tr_by, **va_by}, inorm, tnorm)     # full fold FIT = TRAIN∪VAL
        sm = _sigma_min(net, fit_all, tnorm) if candidate == "C2" else {}
        return make_artifact(candidate, disease, net, inorm, tnorm, sm, best_epoch, env_versions())
    finally:
        torch.use_deterministic_algorithms(pdet); torch.set_num_threads(pth)


def refit_candidate_fixed_epochs(candidate, disease, all_dev_examples, n_epochs, sigma_min_oof, seed):
    if candidate not in CANDIDATES:
        raise ValueError("bad candidate")
    if not (isinstance(n_epochs, int) and n_epochs >= 1):
        raise ValueError("n_epochs must be a positive int")
    by = _validate(all_dev_examples, disease)
    pdet, pth = _det_ctx()
    try:
        torch.use_deterministic_algorithms(True); torch.set_num_threads(1); torch.manual_seed(seed)
        inorm = InputNormalizer.fit([e.window_action_set for e in all_dev_examples])
        tnorm = TargetNormalizer.fit([e.delta_r for e in all_dev_examples])
        data = _prep(by, inorm, tnorm); subj = sorted(data, key=str)
        net = DeepSetsNet(candidate, seed)
        opt = torch.optim.Adam(net.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
        rng = np.random.default_rng(seed)
        for _epoch in range(n_epochs):                        # fixed epochs; NO early stopping / no new validation
            net.train(); order = list(subj); rng.shuffle(order)
            for s in range(0, len(order), HP["batch_subjects"]):
                _opt_step(net, opt, _subject_balanced_loss(net, data, candidate, order[s:s + HP["batch_subjects"]]))
        net.eval()
        if candidate == "C2":
            sm = dict(sigma_min_oof)
            if set(sm) != set(NON_IDENTITY) or any(not (math.isfinite(v) and v > 0) for v in sm.values()):
                raise ValueError("refit requires a complete, positive OOF sigma_min")
        else:
            sm = {}
        return make_artifact(candidate, disease, net, inorm, tnorm, sm, n_epochs - 1, env_versions())
    finally:
        torch.use_deterministic_algorithms(pdet); torch.set_num_threads(pth)


def final_epochs(best_epochs_zero_based):
    vals = [int(e) + 1 for e in best_epochs_zero_based]       # epochs trained = best_epoch(0-based)+1
    if not vals:
        raise ValueError("no fold best-epochs")
    return int(math.floor(float(np.median(vals)) + 0.5))      # round half up
