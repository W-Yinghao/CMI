# CMI-Trace DG-identifiability — hardened protocol + verified verdict (supersedes DG-oracle first pass)

The DG-erasure first pass (`notes/CMI_TRACE_DG_ORACLE_EXECUTION.md`) inverted the objective to minimize
source-held-out risk (CMI as constraint) and reported "predominantly Result B". PM directed three protocol
fixes; this note records the hardened result, which was adversarially verified (5-skeptic Workflow →
high-confidence DOWNGRADE_TO_HINDSIGHT_ONLY) and self-corrected mid-analysis. Branch
`agent/cmi-trace-dg-oracle`, env `c84c-eeg2025-v3`.

## Protocol (Phase-0 fixes + Phase-1 rescue), `tos_cmi/eval/dg_identifiability.py`
- **0.1 cross-fitted existence oracle** (select on T_select target-labels, report on DISJOINT T_query). CRITICAL:
  the oracle must be MECHANISM-MATCHED — a differentiable supermask deletes an ARBITRARY coordinate set, so the
  honest existence test is `mode="greedy"` (arbitrary-coordinate forward selection), NOT `mode="prefix"`
  (top-k of the ordered basis). The initial commit used prefix and spuriously reported ~0 → premature
  NO_CONFIRMED_TICKET; caught by self-cross-check AND the adversarial oracle-strength skeptic.
- **0.2 nested rule-based source-meta** (inner LOSO; pseudo-target excluded from basis/head/rank selection).
- **Phase-1 rescue**: 4 bases {marg, cond=label-conditional, rule=decision-rule-disagreement,
  grad=task-gradient-disagreement} × {full, contested=rowspace(W_c)} × 2 objectives {mean+1SE, CVaR25} +
  no-harm gate. Tests 6/6 (`tos_cmi/tests/test_dg_identifiability.py`).
- **Greedy source-only identifiability audit** (`source_greedy_audit`): the nested selector is PREFIX-only and
  cannot express the greedy ticket, so identifiability is tested GREEDY-vs-GREEDY — a source-only greedy
  selector (max source-LOSO held-out bAcc, arbitrary coords) applied to the true target; report Δ vs
  matched-rank random + subspace alignment to the greedy target ticket. Synthetic anchor: recovers a
  source-visible ticket (balanced shortcut +0.032; strong majority +0.150, align 1.0) → machinery can say yes.

## VERIFIED VERDICT (both EEGNet datasets, subject/fold-cluster 95% CI): TARGET_HINDSIGHT_ONLY

### Existence — CONFIRMED (greedy cross-fit oracle, cond/full)
| dataset | optimistic (un-xfit) | greedy cross-fit | matched random | prefix (weak) |
|---------|----------------------|------------------|----------------|---------------|
| BNCI2014 | +0.049 [.040,.060] | **+0.021 [+0.012,+0.032]** (8/9 folds +) | −0.015 | +0.005 [−0.000,.012] |
| BNCI2015 | +0.065 [.030,.113] | **+0.045 [+0.010,+0.094]** (1–2 outlier subj) | −0.007 | +0.001 [−0.003,.006] |

~half the un-xfit gain = subset-search optimism (PM defect #1, quantified); the other half survives (LCB>0,
beats random). CAVEAT: WITHIN-TARGET hindsight (select on T_select, score on same subject's T_query) — NOT
cross-subject transfer. BNCI2014 robust; BNCI2015 concentrated in outlier subjects.

### Identifiability — FAILS (greedy source-only audit, mechanism-matched)
| dataset | marg | cond | rule | grad |
|---------|------|------|------|------|
| BNCI2014 | −0.0008 (lo<0) | **−0.0029 [−0.0055,−0.0007]** anti-align 0.30 | +0.0003 (lo<0) | −0.0004 (lo<0) |
| BNCI2015 | +0.0024 [+0.0013,+0.0036] (≈7% of oracle, align 0.42) | +0.0008 (lo<0) | −0.0015 | +0.0000 (random beats) |

Only BNCI2015/marg clears 0 & beats random, but recovers ≈7% of that basis's oracle gain, is UNREPLICATED on
BNCI2014, low alignment → SOURCE_DETECTABLE_TINY, not practical (RecoveryRatio ≈0.07 ≪ 0.25). CMI-only harmful
−0.010/−0.063 (solid). Nested prefix selector also refuses (k*=0) except isolated rule/full k*>0 that are
harmful/null (prose "k*=0 everywhere" was overstated; corrected).

## Decision (per pre-registered A/B/C-style matrix + strict P2 gate)
The 8-condition P2 gate FAILS at condition 2 (source identifiability: greedy source LCB ≤ 0 / TINY unreplicated)
and 3 (RecoveryRatio ≥ 0.25). **Do NOT build the source-only differentiable subspace supermask (P2)** — it
would inherit the unidentifiability. The measurement→action chain: **leakage amount ≠ safe removability ≠ DG
relevance ≠ source identifiability** — each real-EEG result breaks the next link. Admissible next step =
explicit information-regime change: a TARGET-X observability audit (does any unlabeled-target statistic predict
the greedy ticket / its per-direction utility on a held-out query set?) BEFORE any target-unlabeled adaptation.
Cleaning results (exact-head-null oracle + TTE V1) unchanged.
