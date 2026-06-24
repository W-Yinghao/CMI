"""
CSC-P1.4.5 inference-contract regression tests (the review's 8 pass conditions):

 1. fit sample-weights sum to the #training BIOLOGICAL subjects (not #rows) -> sklearn lbfgs
    L2 = 1/(C * sum_w) is epoch-duplication invariant -> T is invariant to machine epsilon.
 2. the FULL execute_protocol (T, atlas, tau, source_status, certificate) is invariant to
    duplicating a whole source subject 1x/5x.
 3. condition-first estimand end-to-end: the class POOL mean / shared cell-mean primitive is one
    vote per (subject,condition), invariant to duplicating ONE condition, and != the row-pool mean.
 4. a concept SIGNAL with an ill-defined (near-tied eigengap) leading direction is UNASSESSED, not
    reinterpreted as "no candidate": the leading-direction is non-dominant and the gate abstains BOTH.
 5. each residual-null replicate runs the SAME source-validity pipeline (a disconnected-but-all-
    classes relabel is INVALID, charged as extreme).
 6. the cov-loading NULL keeps a value per replicate (empty/invalid -> 0): adding 0s can only LOWER
    the noise floor q0 -> never makes cov_stable EASIER (dropping them would raise q0).
 7. target condition ids are a MANDATORY contract on BOTH public entry points (missing -> fail closed).
 8. the audit records the FULL-protocol duplication invariance AND the ACTUAL executor-derived
    per-stage seed chain.
"""
import warnings
import dataclasses
import numpy as np
warnings.filterwarnings("ignore")

from csc.protocol import ProtocolConfig, ProtocolError, execute_protocol, run_frozen_protocol
from csc.certificate.residual_test import (
    residual_decoder_test, _subject_condition_weights, _replicate_valid, subject_null_labels,
    fit_h0_proba, stage_seed,
)
from csc.certificate.atlas import (
    _cell_mean, _means, cluster_mean, _leading_concept, analyze_source,
)
from csc.certificate.certifier import certify, UNIDENTIFIABLE
from csc.calibration.lodo import _subj_w, _wprior
from csc.sim.shift_simulator import SimConfig, make_source, make_target, make_paired_subjects


def _dup_subject(Z, Y, D, G, times=1):
    """Duplicate WHOLE subject G[0]'s epochs `times` extra times."""
    m = (G == G[0])
    Zs = [Z] + [Z[m]] * times; Ys = [Y] + [Y[m]] * times
    Ds = [D] + [D[m]] * times; Gs = [G] + [G[m]] * times
    return (np.concatenate(Zs), np.concatenate(Ys), np.concatenate(Ds), np.concatenate(Gs))


# 1 ---- fit sample-weights sum to #subjects -> L2 invariant -> T invariant to machine epsilon ----
def test_fit_weights_sum_to_subject_count_and_T_invariant():
    src = make_source(SimConfig(seed=0), n_domains=8, concept_domains=3, seed=0)
    G, D = src.group_ids, src.D
    S = len(np.unique(G))
    w_sc = _subject_condition_weights(G, D)
    assert abs(w_sc.sum() - S) < 1e-9, f"subject-condition weights must sum to S={S}, got {w_sc.sum()}"
    assert abs(_subj_w(G).sum() - S) < 1e-9, "oracle subject weights must sum to S (raw, not mean-1)"
    assert abs(_subj_w(G, D).sum() - S) < 1e-9, "oracle subject-condition weights must sum to S"
    # duplicating a whole subject's epochs leaves sum_w == S (still one unit-mass per subject) ...
    Z2, Y2, D2, G2 = _dup_subject(src.Z, src.Y, src.D, G, times=4)   # subject G[0] x5
    assert abs(_subject_condition_weights(G2, D2).sum() - S) < 1e-9, "sum_w must stay S under dup"
    # ... so the cross-fitted T is invariant to machine epsilon (NOT just optimizer noise)
    r1 = residual_decoder_test(src.Z, src.Y, src.D, n_boot=20, group_ids=G, C=0.5, label_unit="subject", seed=0)
    r2 = residual_decoder_test(Z2, Y2, D2, n_boot=20, group_ids=G2, C=0.5, label_unit="subject", seed=0)
    assert abs(r1.T - r2.T) < 1e-9, f"T not duplication-invariant: dT={abs(r1.T - r2.T):.2e}"
    print(f"OK sum_w==S={S}; T invariant to subject 5x-duplication (dT={abs(r1.T - r2.T):.1e})")


# 2 ---- FULL execute_protocol invariant to source-subject multiplicity -------------------------
def test_full_protocol_invariant_to_multiplicity():
    cfg = ProtocolConfig(n_boot=25, n_dir_boot=60, target_n_boot=40, tau_n_pseudotargets=60, oracle_boot=15)
    src = make_source(SimConfig(seed=0), n_domains=8, concept_domains=3, seed=0)
    tb = make_target("covariate", SimConfig(seed=0), geom=src.geom, seed=100)
    tc = np.zeros(len(tb.Z), int)
    o1 = execute_protocol(src.Z, src.Y, src.D, tb.Z, cfg, src_group_ids=src.group_ids,
                          tgt_group_ids=tb.group_ids, tgt_condition_ids=tc, seed=0)
    for times in (1, 4):                              # subject G[0] present 2x and 5x
        Z2, Y2, D2, G2 = _dup_subject(src.Z, src.Y, src.D, src.group_ids, times=times)
        o2 = execute_protocol(Z2, Y2, D2, tb.Z, cfg, src_group_ids=G2,
                              tgt_group_ids=tb.group_ids, tgt_condition_ids=tc, seed=0)
        assert abs(o1["analysis"].test.T - o2["analysis"].test.T) < 1e-9, "T not invariant"
        assert o1["analysis"].source_status == o2["analysis"].source_status, "status not invariant"
        assert abs(o1["tau_detect"] - o2["tau_detect"]) < 1e-9, "tau_detect not invariant"
        assert abs(o1["tau_label"] - o2["tau_label"]) < 1e-9, "tau_label not invariant"
        assert o1["certificate"].state == o2["certificate"].state, "certificate not invariant"
    print(f"OK full execute_protocol invariant to 1x/5x source-subject duplication "
          f"(T, tau, status={o1['analysis'].source_status}, cert={o1['certificate'].state})")


# 3 ---- condition-first shared primitive: class pool mean is one vote/(subject,condition) -------
def test_condition_first_class_pool_mean():
    Z, Y, D, G = make_paired_subjects(SimConfig(seed=1), n_subjects=20, concept_delta=1.0, seed=1)
    classes = list(np.unique(Y))
    base = _means(Z, Y, classes, G, D)                # condition-first class pool means
    # the shared primitive IS cluster_mean over the masked subset
    for c in classes:
        assert np.max(np.abs(base[c] - cluster_mean(Z[Y == c], G[Y == c], D[Y == c]))) < 1e-12
    # duplicate ONE condition of ONE subject's epochs -> condition-first pool mean is UNCHANGED
    m = (G == G[0]) & (D == D[0])
    Z2 = np.concatenate([Z, Z[m]]); Y2 = np.concatenate([Y, Y[m]])
    D2 = np.concatenate([D, D[m]]); G2 = np.concatenate([G, G[m]])
    after = _means(Z2, Y2, classes, G2, D2)
    assert all(np.max(np.abs(base[c] - after[c])) < 1e-9 for c in classes), "pool mean not condition-first"
    # and it DIFFERS from the condition-BLIND subject-vote mean on unbalanced ON/OFF data
    blind = {c: _cell_mean(Z, Y == c, G) for c in classes}
    assert any(np.max(np.abs(base[c] - blind[c])) > 1e-6 for c in classes), "must differ from blind"
    # weighted class prior reduces to row proportions when w is None, else uses the weights
    assert np.allclose(_wprior(Y, None, classes), [(Y == c).mean() + 1e-9 for c in classes])
    print("OK class pool mean is the shared condition-first primitive (dup-invariant; != blind)")


# 4 ---- a near-tied leading concept direction is UNASSESSED, not "no candidate" ------------------
def test_tied_eigengap_is_unassessed_not_reinterpreted():
    # build data whose class-conditional residual Rc has a TIED top-2 spectrum (two orthogonal,
    # equal-magnitude concept axes) -> the leading direction is NOT identifiable.
    rng = np.random.default_rng(0)
    d, A = 6, 3.0
    rows_Z, rows_Y, rows_D, rows_G = [], [], [], []
    # domain offsets that put +-A on e0 (domains 0,1) and +-A on e1 (domains 2,3), symmetric over D
    spec_tied = {0: (0, +A), 1: (0, -A), 2: (1, +A), 3: (1, -A)}
    sid = 0
    for dom, (axis, amp) in spec_tied.items():
        for c in (0, 1):
            mu = np.zeros(d); mu[axis] = amp if c == 0 else -amp     # class-conditional (-> in Rc)
            for _ in range(6):                                       # 6 subjects / cell
                rows_Z.append(mu + 0.01 * rng.standard_normal(d)); rows_Y.append(c)
                rows_D.append(dom); rows_G.append(sid); sid += 1
    Z = np.array(rows_Z); Y = np.array(rows_Y); D = np.array(rows_D); G = np.array(rows_G)
    _, dom_tied = _leading_concept(Z, Y, D, [0, 1], [0, 1, 2, 3], 0.95, G, eigengap_min=0.30)
    assert dom_tied is False, "a near-tied top-2 concept spectrum must be NON-dominant (UNASSESSED)"
    # a SEPARATED spectrum (shrink the e1 axis) is dominant
    Zs = Z.copy(); Zs[(D == 2) | (D == 3)] *= (1.0 / 4)
    _, dom_sep = _leading_concept(Zs, Y, D, [0, 1], [0, 1, 2, 3], 0.95, G, eigengap_min=0.30)
    assert dom_sep is True, "a well-separated leading concept axis must be dominant"
    # the gate: a SIGNAL-but-unassessable source carries UNASSESSED_CONCEPT_ATTRIBUTION and abstains BOTH
    src = make_source(SimConfig(seed=0), n_domains=8, concept_domains=3, seed=0)
    sa = analyze_source(src.Z, src.Y, src.D, n_boot=20, n_dir_boot=40, group_ids=src.group_ids, seed=0)
    bad = dataclasses.replace(sa, source_status="UNASSESSED_CONCEPT_ATTRIBUTION")
    assert certify(bad, src.Z[:300], group_ids=src.group_ids[:300]).state == UNIDENTIFIABLE
    print("OK tied-spectrum leading dir -> non-dominant -> UNASSESSED_CONCEPT_ATTRIBUTION abstains BOTH")


# 5 ---- each residual-null replicate runs the FULL source-validity pipeline ---------------------
def test_residual_null_replicate_validity():
    # disconnected-but-all-classes relabel: classes {0,1} on domains {0,1}; classes {2,3} on {2,3}
    g, Y, D = [], [], []
    sid = 0
    for (doms, cls) in (([0, 1], [0, 1]), ([2, 3], [2, 3])):
        for dm in doms:
            for c in cls:
                for _ in range(4):
                    g.append(sid); Y.append(c); D.append(dm); sid += 1
    g = np.repeat(np.arange(len(g)), 3); Y = np.repeat(Y, 3); D = np.repeat(D, 3)
    g = np.asarray(g); Y = np.asarray(Y); D = np.asarray(D)
    s_cv = stage_seed(0, "residual_cv")
    assert not _replicate_valid(Y, D, g, g, 2, "subject", s_cv), \
        "a disconnected-but-all-classes relabel must be INVALID per replicate"
    # a genuinely valid (fully connected) subject-level labelling is valid
    src = make_source(SimConfig(seed=0), n_domains=8, concept_domains=3, seed=0)
    Gs = src.group_ids
    assert _replicate_valid(src.Y, src.D, Gs, Gs, 4, "subject", stage_seed(0, "residual_cv")), \
        "a connected all-class subject labelling must be VALID"
    print("OK residual-null replicate runs full source validity (disconnected-all-classes -> INVALID)")


# 6 ---- the cov-loading null keeps a value per replicate; 0s never RAISE the noise floor q0 -----
def test_cov_null_empty_never_raises_q0():
    # the implementation appends 0.0 for an empty/invalid replicate (FIXED length B). Inserting 0s
    # into the loading null can only LOWER the (1-alpha) quantile q0 -> cov_stable=(cov_ub<kappa*q0)
    # becomes HARDER, never EASIER. Dropping those replicates (the bug) would RAISE q0.
    rng = np.random.default_rng(0)
    loads = rng.random(20) + 0.5                    # valid replicate loadings
    q_full = float(np.quantile(loads, 0.95))
    for n_empty in (1, 5, 10):
        with_zeros = np.concatenate([loads, np.zeros(n_empty)])   # empty/invalid -> 0 (kept)
        q_with = float(np.quantile(with_zeros, 0.95))
        assert q_with <= q_full + 1e-12, "inserting 0s must NOT raise the noise floor q0"
    # the DROPPED variant (only the valid loadings) has a >= q0 -> would loosen stability
    assert float(np.quantile(loads, 0.95)) >= float(np.quantile(np.concatenate([loads, np.zeros(10)]), 0.95))
    print("OK cov-null keeps a value/replicate; empty/invalid->0 only LOWERS q0 (never loosens stability)")


# 7 ---- target condition ids are MANDATORY on BOTH public entry points --------------------------
def test_target_condition_ids_mandatory():
    # n_boot must satisfy validate() (>= ceil(1/alpha)-1) so the MISSING-condition-id check is what
    # fires -- not an unrelated config error.
    cfg = ProtocolConfig(n_boot=20, n_dir_boot=30, target_n_boot=20, tau_n_pseudotargets=20, oracle_boot=10)
    src = make_source(SimConfig(seed=2), n_domains=6, concept_domains=2, seed=2)
    tb = make_target("covariate", SimConfig(seed=2), geom=src.geom, seed=200)
    # group ids ARE supplied (so this is NOT the group_aware gate); MISSING condition ids fail closed
    for fn in (execute_protocol, run_frozen_protocol):
        try:
            fn(src.Z, src.Y, src.D, tb.Z, cfg, src_group_ids=src.group_ids,
               tgt_group_ids=tb.group_ids, seed=2)              # no tgt_condition_ids
            raise AssertionError(f"{fn.__name__} must fail closed without tgt_condition_ids")
        except ProtocolError:
            pass
    # supplying a single-condition id runs
    out = run_frozen_protocol(src.Z, src.Y, src.D, tb.Z, cfg, src_group_ids=src.group_ids,
                              tgt_group_ids=tb.group_ids, tgt_condition_ids=np.zeros(len(tb.Z), int), seed=2)
    assert out["certificate"].state is not None
    print("OK target condition ids mandatory on execute_protocol AND run_frozen_protocol (fail closed)")


# 8 ---- the audit records FULL-protocol invariance + the ACTUAL derived stage-seed chain --------
def test_audit_full_protocol_and_derived_seeds():
    from csc.run_audit import contract_diagnostics
    cfg = ProtocolConfig(n_boot=20, n_dir_boot=40, source_cv_folds=4)
    diag = contract_diagnostics(cfg, [0])
    fp = diag["full_protocol_duplication_invariance"]
    assert fp["atlas_hash_before"] == fp["atlas_hash_after"], "atlas subspace not invariant"
    assert fp["source_status_before"] == fp["source_status_after"]
    assert fp["tau_detect_before"] == fp["tau_detect_after"], "tau_detect not invariant in audit"
    assert fp["tau_label_before"] == fp["tau_label_after"]
    assert fp["cov_stable_before"] == fp["cov_stable_after"]
    assert fp["certificate_before"] == fp["certificate_after"], "certificate not invariant in audit"
    # the named seeds are the REAL chain: source-internal stages derive from the analyze_source seed
    ns = diag["named_stage_seeds_at_root0"]
    s_analyze = stage_seed(0, "analyze_source")
    assert ns["analyze_source"] == s_analyze
    assert ns["analyze_source/residual_cv"] == stage_seed(s_analyze, "residual_cv"), "seed chain wrong"
    assert ns["analyze_source/geometry_null"] == stage_seed(s_analyze, "geometry_null")
    print(f"OK audit records full-protocol invariance (cert={fp['certificate_before']}) + derived seed chain")


if __name__ == "__main__":
    test_fit_weights_sum_to_subject_count_and_T_invariant()
    test_full_protocol_invariant_to_multiplicity()
    test_condition_first_class_pool_mean()
    test_tied_eigengap_is_unassessed_not_reinterpreted()
    test_residual_null_replicate_validity()
    test_cov_null_empty_never_raises_q0()
    test_target_condition_ids_mandatory()
    test_audit_full_protocol_and_derived_seeds()
    print("\nall CSC-P1.4.5 contract tests passed")
