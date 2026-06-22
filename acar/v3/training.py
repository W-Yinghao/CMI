"""ACAR v3 candidate TRAINING (HSCR). DESIGN/DEV stage — SYNTHETIC only until DEV_DESIGN_LOCK.

fit_candidate() implements the three registered objectives with the pinned HP (predictors.HP):
  C1 = Huber(δ=1.0)            ; C2 = Seitzer β-NLL (β=0.5, weight = v.detach()**β) ; C3 = pinball ½ρ₀.₅+½ρ₀.₉
all on the per-disease STANDARDIZED target, with **subject-balanced reduction** (mean within subject, then mean over
subjects), Adam (lr/weight_decay), gradient clipping, early stopping with best-epoch restoration, FIT-only input +
target normalizers, deterministic execution (CPU, single thread, fixed seeds, use_deterministic_algorithms), and env
capture. Returns an immutable, hashed FittedCandidate. NO DEV cohort / lockbox access; targets are GIVEN (Phase-2 DEV).
"""
from __future__ import annotations
from collections import namedtuple, defaultdict
import numpy as np
import torch

from .set_features import WindowActionSet, NON_IDENTITY
from .normalizers import InputNormalizer, TargetNormalizer
from .predictors import DeepSetsNet, FittedCandidate, HP, env_versions, CANDIDATES

TrainExample = namedtuple("TrainExample", "was delta_r subject_key")


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


def _prep(examples, inorm, tnorm):
    """-> dict subject_key -> list of (win, ctx, action, y_std)."""
    by = defaultdict(list)
    for ex in examples:
        nw = inorm.transform(ex.was)
        win = torch.tensor(np.concatenate([nw.values, nw.availability_mask.astype(np.float64)], 1), dtype=torch.float32)
        ctx = torch.tensor(np.concatenate([nw.context_values, nw.context_mask.astype(np.float64)]), dtype=torch.float32)
        y = torch.tensor(float(tnorm.standardize(ex.delta_r)), dtype=torch.float32)
        by[ex.subject_key].append((win, ctx, ex.was.action_name, y))
    return by


def _subject_balanced_loss(net, by_subject, candidate, subjects):
    sl = []
    for sk in subjects:
        el = [_example_loss(net, w, c, a, y, candidate) for (w, c, a, y) in by_subject[sk]]
        sl.append(torch.stack(el).mean())
    return torch.stack(sl).mean()


def fit_candidate(candidate, disease, train_examples, val_examples, seed) -> FittedCandidate:
    if candidate not in CANDIDATES:
        raise ValueError("bad candidate")
    if not train_examples or not val_examples:
        raise ValueError("need non-empty train and val examples")
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_threads = torch.get_num_threads()
    try:
        torch.use_deterministic_algorithms(True); torch.set_num_threads(1); torch.manual_seed(seed)
        inorm = InputNormalizer.fit([e.was for e in train_examples])
        tnorm = TargetNormalizer.fit([e.delta_r for e in train_examples])
        tr = _prep(train_examples, inorm, tnorm); va = _prep(val_examples, inorm, tnorm)
        tr_subj = sorted(tr); va_subj = sorted(va)
        net = DeepSetsNet(candidate, seed)
        opt = torch.optim.Adam(net.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
        rng = np.random.default_rng(seed)
        best_val = float("inf"); best_state = None; best_epoch = -1; bad = 0
        for epoch in range(HP["max_epochs"]):
            net.train()
            order = list(tr_subj); rng.shuffle(order)
            for s in range(0, len(order), HP["batch_subjects"]):
                mb = order[s:s + HP["batch_subjects"]]
                opt.zero_grad()
                loss = _subject_balanced_loss(net, tr, candidate, mb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), HP["grad_clip"])
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(_subject_balanced_loss(net, va, candidate, va_subj))
            if vl < best_val - HP["min_delta"]:
                best_val = vl; best_state = {k: v.clone() for k, v in net.state_dict().items()}
                best_epoch = epoch; bad = 0
            else:
                bad += 1
                if bad >= HP["patience"]:
                    break
        net.load_state_dict(best_state)                       # best-epoch restoration
        net.eval()
        # sigma_min (C2 only): Q0.05 of FIT raw σ̂ per action
        sigma_min = ()
        if candidate == "C2":
            per = defaultdict(list)
            with torch.no_grad():
                for sk in tr_subj:
                    for (w, c, a, _y) in tr[sk]:
                        _, sig = net(w, c, a)
                        per[a].append(float(sig) * tnorm.sd)
            sm = {}
            for a in NON_IDENTITY:
                vals = per.get(a, [])
                sm[a] = float(np.quantile(vals, HP["sigma_min_quantile"])) if vals else 0.0
            sigma_min = tuple(sorted(sm.items()))
        return FittedCandidate(candidate, disease, net, inorm, tnorm, sigma_min,
                               int(best_epoch), tuple(sorted(env_versions().items())))
    finally:
        torch.use_deterministic_algorithms(prev_det); torch.set_num_threads(prev_threads)
