# FSR_35 — Submission Risk Register (Phase 6A)

**Project FSR.** Reviewer-facing risks for the frozen manuscript and the pre-registered mitigation for each. The
paper must not exceed what the ledger (FSR_05/34) licenses; each risk maps to a claim-hygiene guardrail.

| # | reviewer risk | mitigation (in the paper) |
|---|---|---|
| R1 | "This is just probing / an audit, no method." | Framed as an **audit + verification framework** (C10); the contribution is the ladder + boundary findings + scoped-repair boundary, stated explicitly. Not sold as a DG method. |
| R2 | "You claim to repair EEG shortcuts." | Repair is **scoped** (C12/C17): controlled first-moment injected positive control only; **construction-matched** (73% mechanical identity), **fails leave-one-dataset-out**, **BNCI2015-carried**. Second-moment = none (4G). Natural = refused (4B). All caveats attached. |
| R3 | "The 4F 'strong' is a strong repair claim." | Reported as `strong_within_controlled_first_moment_scope`; **primary metric is the absolute netted gain +0.033 bAcc** (not the 0.93 ratio); the ratio is 73% mechanical identity (a first-moment aligner inverts a first-moment offset by algebra). Never "surgical", never "general". |
| R4 | "Only 2 datasets; not generalizable." | Explicitly disclosed. Leave-one-dataset-out is reported and **fails** for 4F; N=2 sign-consistency is descriptive only. Generality beyond the two backbones/datasets is **not claimed**; ≥3-dataset extension is future work (FSR_31). |
| R5 | "You injected the shortcut; that's circular." | PC1/4F/4G are **explicitly labeled injected positive controls**, separated from the **natural** result (4B = none). The injected controls test the *machinery*; the natural result is where the honest EEG conclusion lives. |
| R6 | "Second-moment 'none' is under-powered, not a real negative." | 4G is **mechanistically attributed**: even an **oracle** shrinking the true injected direction is sub-DELTA; the estimator recovers v_c (overlap 0.71); `fail_attribution = genuinely_weak`. A DELTA-clearing advantage exists only in the injection-dominant (α=3) near-tautology, excluded by the source-only α rule. |
| R7 | "Target-label leakage in the repair/selection." | Firewall attested per phase (TargetScorer; source-only fit/α/k-λ/veto set); independently recomputed and firewall-audited each phase. |
| R8 | "CMI / DG-method over-claim." | C8: CMI-control is a **closed negative** premise; no CMI/DG/SOTA claim anywhere. |
| R9 | "Forbidden wording slips in." | `check_paper_claims.py` (forbidden-wording gate) + the FSR_33 forbidden↔allowed list run on every build; hidden claim tags map prose to ledger IDs. |

## Hard stops (any → do not submit)
- Any forbidden sentence from FSR_33 present in the compiled PDF.
- Any repair claim without its mandatory caveat (C12/C17).
- 4F reported as unqualified "strong", or the 0.93 ratio led over the absolute gain.
- Natural (4B) and injected (PC1/4F/4G) results conflated.
- Any claim of PC2 / learned-reliance / second-moment repair as established.
