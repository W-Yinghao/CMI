"""Information-Regime Ladder (Track B) — selection-only + head-only few-shot, on FROZEN ERM EEGNet features.

Question: with a frozen source-trained encoder + a source-fitted readout, at what level of TARGET information
(none R0 / unlabeled cal X RX / 1,2,4 labeled cal trials-per-class R1,R2,R4 / all cal labels RF) does a useful
low-rank subspace-DELETION action first become IDENTIFIABLE (selectable), and is that threshold subspace-SPECIFIC
(informed B_cond dictionary beats matched-rank random dictionaries)?

SELECTION-ONLY primary: every regime SELECTS one action from the SAME exhaustive family (subsets rank<=3 of a
dictionary, identity included); the action's target utility is the session-macro query gain (fresh source-fitted
head per action). Target labels only SELECT (via CE on the k cal trials); they NEVER retrain the encoder or the head.
Firewall: query (X,Y) enters ONLY the final utility/oracle; cal Y enters only label-regime selection.

All whitened-metric (Ledoit-Wolf A_s); reuses the verified target-X / mechanism-subspace primitives. Pure numpy+sklearn.
Manuscript FROZEN."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss

from tos_cmi.eval import targetx_metric as TM
from tos_cmi.eval.mechanism_subspace import (_del, build_exhaustive_action_family, build_ambient_random_dictionaries,
                                             score_on_target_query, cell_seed)
from tos_cmi.eval.dg_identifiability import _fit_logreg, _bacc, _source_loso_gain, crossfit_target_oracle
from tos_cmi.eval.targetx_observability import session_split

DICT_RANK = 8            # dictionary rank cap (whitened_cond_basis / random), subsets enumerated to rank<=3
MAX_SUBSET_RANK = 3
REGIMES = ["R0", "RX", "R1", "R2", "R4", "RF"]
KSHOT = {"R1": 1, "R2": 2, "R4": 4}
EPS = 1e-9


def _U(B, S):
    """Orthonormal whitened rows for action subset S (empty -> (0,D) identity action)."""
    return TM._orthonormal(B[list(S)]) if S else np.zeros((0, B.shape[1]))


def _ce(head, U, Xcal_w, ycal, classes):
    """Cross-entropy of the source-fitted head (fit on deleted source) on deleted cal features vs cal labels.
    head = (clf, mu, sd) from _fit_logreg on _del(Zs_w, U). Defined for k>=1 cal trials; uses margins (proba)."""
    clf, mu, sd = head
    Xc = (_del(Xcal_w, U) - mu) / sd
    proba = clf.predict_proba(Xc)
    return float(log_loss(ycal, proba, labels=clf.classes_))


def _balanced_draw(ycal, k, rng):
    """k class-balanced indices per class into the cal set (fewer if a class has <k trials). Deterministic via rng."""
    idx = []
    for c in np.unique(ycal):
        ci = np.where(ycal == c)[0]
        idx.extend(rng.choice(ci, min(k, len(ci)), replace=False).tolist())
    return np.array(sorted(idx), dtype=int)


def precompute_actions(Zs_w, ys, ds, B, Xcal_w, ycal, Xq_w, yq, sq, d_white):
    """For each action (subset<=3 of dictionary B): its whitened projector U, the regime-INDEPENDENT session-macro
    query gain, the source-fitted head (for CE), the G1 unlabeled score, and the source-LOSO gain. Precomputing the
    query gain + head once lets every regime and every label draw just SELECT and look up utility."""
    r = B.shape[0]
    actions = build_exhaustive_action_family(min(r, DICT_RANK), MAX_SUBSET_RANK) if r else [[]]
    recs = []
    for S in actions:
        U = _U(B, S)
        clf, W, b, mu, sd = _fit_logreg(_del(Zs_w, U), ys)
        recs.append(dict(S=list(S), U=U,
                         query_gain=float(score_on_target_query(Zs_w, ys, U, Xq_w, yq, sq)),
                         head=(clf, mu, sd),
                         g1=float(np.sum((U @ d_white) ** 2)) if U.shape[0] else 0.0,
                         src_loso=float(_source_loso_gain(Zs_w, ys, ds, B, S)) if S else 0.0))
    return recs, actions


def select_and_utility(recs, regime, Xcal_w, ycal, classes, draws=None):
    """Return the selected action's query gain for a regime. R0=argmax source-LOSO; RX=argmax G1; RF=argmin CE on
    ALL cal; R1/R2/R4=argmin CE on each k-shot draw, averaged over draws. Identity is always a candidate (no-harm)."""
    if regime == "R0":
        j = int(np.argmax([r["src_loso"] if np.isfinite(r["src_loso"]) else -1e9 for r in recs]))
        return recs[j]["query_gain"], recs[j]["S"]
    if regime == "RX":
        j = int(np.argmax([r["g1"] for r in recs]))
        return recs[j]["query_gain"], recs[j]["S"]
    if regime == "RF":
        ces = [_ce(r["head"], r["U"], Xcal_w, ycal, classes) for r in recs]
        j = int(np.argmin(ces))
        return recs[j]["query_gain"], recs[j]["S"]
    # R1/R2/R4: average over the provided k-shot draws
    gains, sels = [], []
    for di in draws:
        ces = [_ce(r["head"], r["U"], Xcal_w[di], ycal[di], classes) for r in recs]
        j = int(np.argmin(ces)); gains.append(recs[j]["query_gain"]); sels.append(recs[j]["S"])
    return float(np.mean(gains)), sels


def _session_macro_bacc(Ztr, ytr, Xq_w, yq, sq, Utrans):
    """Fit a linear head on Ztr (deleted) and score session-macro query bAcc (deleted)."""
    if len(np.unique(ytr)) < 2:
        return float("nan")
    Zd = _del(Ztr, Utrans); mu, sd = Zd.mean(0), Zd.std(0) + 1e-8
    clf = LogisticRegression(max_iter=200, C=1.0).fit((Zd - mu) / sd, ytr)
    per = [balanced_accuracy_score(yq[sq == s], clf.predict((_del(Xq_w[sq == s], Utrans) - mu) / sd))
           for s in np.unique(sq) if (sq == s).sum() >= 4 and len(np.unique(yq[sq == s])) >= 2]
    return float(np.mean(per)) if per else float("nan")


def head_only_calibration(U_star, Zs_w, ys, Xcal_w, ycal, Xq_w, yq, sq, di):
    """SECONDARY practical ladder. Three session-macro query bAccs: `source_identity` = the FROZEN source-fitted
    readout on native features (the selection-only reference); `native` = a FRESH head fit on the k cal labels
    (encoder frozen, native features); `selected` = a fresh head fit on the k cal labels of the SELECTED-subspace-
    deleted features. Distinguishes: labels needed to SELECT a subspace (selected>native) vs to fit a new readout
    (native>source_identity). di = k-shot draw indices (None = full cal)."""
    idx = di if di is not None else np.arange(len(ycal))
    Xc, yc = Xcal_w[idx], ycal[idx]
    D = Xq_w.shape[1]
    return dict(source_identity=_session_macro_bacc(Zs_w, ys, Xq_w, yq, sq, np.zeros((0, D))),
                native=_session_macro_bacc(Xc, yc, Xq_w, yq, sq, np.zeros((0, D))),
                selected=_session_macro_bacc(Xc, yc, Xq_w, yq, sq, U_star))
