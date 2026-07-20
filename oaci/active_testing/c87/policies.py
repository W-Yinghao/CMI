"""C87 acquisition/selection policies (PRODUCTION; the synthetic control gate uses these exact classes).

Registry (C87P §3, v3 information-view pin): a paid query on held record r returns ONLY the single
selected-task binary label y_task(r) in {0,1}. Candidate PREDICTIONS p_{a,r} (pos-class prob) are always
visible; once a label is revealed, the per-record NLL l(a,r) is computable for EVERY candidate. Each policy
queries B held records (one label each) and returns the selected candidate index (min estimated held loss),
plus the queried record indices.

  P0                 baseline uniform-without-replacement                          NOT label-adaptive
  LUREActiveTesting  prediction-driven, FIXED surrogate proposal + LURE risk       NOT label-adaptive
  ModelSelector      best-arm posterior over min-held-loss; EIG-proxy racing query GENUINELY label-adaptive
  CODA               binary Dawid-Skene confusion; accuracy best-arm; EIG query    GENUINELY label-adaptive

Fidelity note: ModelSelector/CODA are the production adaptations of Okanovic 2025 / Kay 2025 to the single
binary-task NLL selection objective under patient-clustered querying. The reference-CODE numerical-agreement
audit (C87P §3.4 check B, D-13) against the public repos is a SEPARATE deferred gate before C87E; this module
supplies the statistical behavior the §6 control gate validates (detect / refuse / calibrate).
"""
from __future__ import annotations

import numpy as np

from .estimand import binary_nll
from .lure import lure_risk, without_replacement_proposal_sequence


class Policy:
    name = "policy"
    label_adaptive = False

    def select(self, probs, y_full, patient_of, B, rng):
        """probs:(A,n_r) always-visible pos-class prob; y_full:(n_r,) true labels (revealed only for the B
        records this policy chooses to query); returns (a_pick:int, queried:np.ndarray of record idx)."""
        raise NotImplementedError


def _empirical_pick(probs, y_full, queried):
    """Min empirical held NLL over the queried records; deterministic lowest-index tie-break."""
    if queried.size == 0:
        return 0
    loss = binary_nll(probs[:, queried], y_full[queried]).mean(axis=1)
    return int(np.argmin(loss))


class P0(Policy):
    name = "P0"
    label_adaptive = False

    def select(self, probs, y_full, patient_of, B, rng):
        n = probs.shape[1]
        B = min(B, n)
        queried = rng.choice(n, size=B, replace=False) if B > 0 else np.empty(0, int)
        return _empirical_pick(probs, y_full, queried), queried


class LUREActiveTesting(Policy):
    """Prediction-driven active testing with a FIXED surrogate proposal (does not retrain on labels, so it
    is NOT label-adaptive). Proposal targets records where candidates DISAGREE (informative for selection);
    per-candidate risk is the unbiased LURE estimate, and the pick is argmin LURE risk."""

    name = "LURE-AT"
    label_adaptive = False

    def __init__(self, floor=1e-3):
        self.floor = floor

    def _proposal(self, probs):
        # disagreement surrogate: variance across candidates of predicted pos-prob, + floor (fixed, label-free)
        w = probs.var(axis=0) + self.floor
        return w / w.sum()

    def select(self, probs, y_full, patient_of, B, rng):
        A, n = probs.shape
        B = min(B, n)
        if B == 0:
            return 0, np.empty(0, int)
        w = self._proposal(probs)
        # sample B records without replacement proportional to w (the proposal q)
        order = rng.choice(n, size=B, replace=False, p=w)
        q_seq = without_replacement_proposal_sequence(w, order)
        loss = binary_nll(probs[:, order], y_full[order])          # (A,B) losses at acquired points
        risk = np.array([lure_risk(loss[a], q_seq, n) for a in range(A)])
        return int(np.argmin(risk)), order


class ModelSelector(Policy):
    """Genuinely label-adaptive best-arm selection over the min-held-loss candidate (Okanovic-2025 family,
    adapted to NLL selection). Maintains a Normal posterior over each candidate's mean loss from queried
    labels; each step queries the unlabeled record that best DISCRIMINATES the current leading pair
    (EIG-proxy racing) -> the query target updates from labels (label-adaptive); picks argmin posterior mean."""

    name = "MODEL-SELECTOR"
    label_adaptive = True

    def __init__(self, prior_var=1.0, obs_var=0.5, warmup=2):
        self.prior_var, self.obs_var, self.warmup = prior_var, obs_var, warmup

    def select(self, probs, y_full, patient_of, B, rng):
        A, n = probs.shape
        B = min(B, n)
        labeled = np.zeros(n, bool)
        sum_loss = np.zeros(A)
        cnt = 0
        # warmup: a few uniform queries to seed the posterior
        nq = 0
        prior_mean = probs.mean()  # weakly-informative global loss scale proxy (label-free)
        while nq < B:
            unl = np.where(~labeled)[0]
            if unl.size == 0:
                break
            if cnt < self.warmup:
                r = unl[rng.integers(unl.size)]
            else:
                mean = (sum_loss + prior_mean * (self.obs_var / self.prior_var)) / (
                    cnt + (self.obs_var / self.prior_var))
                order = np.argsort(mean)
                a1, a2 = int(order[0]), int(order[1])
                # EIG proxy: query the unlabeled record whose two leading candidates most disagree
                disagree = np.abs(probs[a1, unl] - probs[a2, unl])
                r = int(unl[np.argmax(disagree)])
            labeled[r] = True
            l_r = binary_nll(probs[:, [r]], y_full[[r]])[:, 0]
            sum_loss += l_r
            cnt += 1
            nq += 1
        queried = np.where(labeled)[0]
        mean = (sum_loss + prior_mean * (self.obs_var / self.prior_var)) / max(
            cnt + (self.obs_var / self.prior_var), 1e-9)
        return int(np.argmin(mean)), queried


class CODA(Policy):
    """Genuinely label-adaptive Dawid-Skene accuracy best-arm (Kay-2025 family, binary task). Consensus
    pseudo-labels seed Beta accuracy priors; queries the record maximizing leading-pair hard-pred
    disagreement (EIG proxy); selects argmax posterior accuracy. Targets ACCURACY (reported as a
    label-adaptive robustness lens; primary-conditional on the §3.4 audit)."""

    name = "CODA"
    label_adaptive = True

    def __init__(self, prior_strength=2.0):
        self.prior_strength = prior_strength

    def select(self, probs, y_full, patient_of, B, rng):
        A, n = probs.shape
        B = min(B, n)
        hard = (probs > 0.5).astype(int)                 # (A,n) hard predictions
        consensus = (hard.mean(axis=0) > 0.5).astype(int)  # majority-vote pseudo-labels (label-free seed)
        agree = (hard == consensus[None, :]).mean(axis=1)   # prior accuracy proxy per candidate
        a_correct = self.prior_strength * agree            # Beta(a,b) pseudo-counts
        b_wrong = self.prior_strength * (1 - agree)
        labeled = np.zeros(n, bool)
        nq = 0
        while nq < B:
            unl = np.where(~labeled)[0]
            if unl.size == 0:
                break
            acc = a_correct / (a_correct + b_wrong)
            order = np.argsort(-acc)
            a1, a2 = int(order[0]), int(order[1])
            disagree = (hard[a1, unl] != hard[a2, unl]).astype(float) + 1e-6 * rng.random(unl.size)
            r = int(unl[np.argmax(disagree)])
            labeled[r] = True
            correct = (hard[:, r] == y_full[r]).astype(float)
            a_correct += correct
            b_wrong += 1 - correct
            nq += 1
        acc = a_correct / (a_correct + b_wrong)
        return int(np.argmax(acc)), np.where(labeled)[0]


class _OracleLeakSelector(Policy):
    """CHEATING selector for the NEG-M3 mutation test ONLY: peeks at the FULL held loss field (all labels)
    and picks the true argmin. Must manufacture spurious all-cohort G>0 in NEG-B -> proves the control can
    detect a leak. NEVER part of the real registry."""

    name = "ORACLE-LEAK"
    label_adaptive = True

    def select(self, probs, y_full, patient_of, B, rng):
        loss = binary_nll(probs, y_full).mean(axis=1)     # uses EVERY held label (leak)
        return int(np.argmin(loss)), np.arange(min(B, probs.shape[1]))


PRIMARY = [P0(), LUREActiveTesting(), ModelSelector(), CODA()]
LABEL_ADAPTIVE = [p.name for p in PRIMARY if p.label_adaptive]
