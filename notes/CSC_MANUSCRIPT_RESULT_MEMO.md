# CSC manuscript memo — identifiability/abstention boundary + negative confirmatory result

**Framing decision (reviewer): Direction A.** The paper owns an **identifiability boundary** and an
**audited negative confirmatory result**, not a positive detector. The frozen confirmatory test did
exactly what a falsifiable-certification paper needs: it *falsified* the development-observed positive
operating core on unseen clusters. This is the cleanest outcome for a paper whose premise is
falsifiable certification.

## The scientific claim (use this phrasing)

> We prove that pure conditional shift is unidentifiable from the unlabeled target marginal (Z) alone,
> motivate abstention as a necessary output, and construct a falsifiable certificate for
> source-anchored, support-visible concept evidence. In development sweeps the certificate exhibited an
> apparent identifiable core; however, a **frozen confirmatory test on unseen synthetic clusters failed
> both the false-certification and the power endpoints**. The contribution is therefore an
> **identifiability boundary** and an **audited negative result** showing how easily development-observed
> source-free concept evidence can fail to generalize.

**Do NOT claim** "we have a working concept-shift detector." **Do NOT over-claim** "no Z-only method can
ever work." The theorem covers *pure conditional unidentifiability*; the confirmatory result falsifies
*this frozen certificate and this development-selected core*, not every possible future method.

## The development → confirmatory gap (the decisive evidence)

Same operating point `P_baseline` (effect 14, 22 subj/dom, 30 target subj, balanced priors, corr 0.2,
leakage 10, 3 concept domains, 22 epochs), evaluated two ways:

| | DEVELOPMENT (P1.5 map, dev seeds) | CONFIRMATORY (frozen, unseen base_seed 900000) |
|---|---|---|
| clusters | 12/cell | G=66, N_valid=65 |
| visible-concept power | **0.83** (CP-LB 0.56) | **0.43** (28/65; CP-LB **0.326**) |
| forbidden (false-cert) | **0/12** | **1/65**, CP-UB **0.0709 > α=0.05** |
| headline | apparent favourable core | **FAIL both endpoints** (`headline_core_pass=false`) |

The development map's own caveat was that 12 clusters give only CP-UB ≈ 0.221 — a boundary locator,
**not** error control. The frozen confirmatory test (≥59 clusters, 0-failure CP-UB ≤ 0.05; power bar 0.50
with `max(conditional, unconditional)` guard) is the honest test, and it refutes the dev core. The gap is
a textbook informed-selection / generalization effect, surfaced — not hidden — by the freeze→unseen
protocol.

Provenance of the confirmatory run (all verified, frozen): SLURM 876329; `git_head == expected_code_commit
== csc-confirmatory-v1^{commit} == dee8958`; clean tree; manifest `da2c0f4309…`; freshness-verified
artifact `csc/results/confirmatory.json` (sha256 `8b07524ecc3b…`); source seeds `900000..900065` → target
seeds `1800000..1800065` (disjoint). Per protocol the result was **committed, not rerun**; no thresholds,
seeds, manifest, or tag changed; `P_strong` not run.

## Recommended paper structure

1. **Problem.** Universal source-free concept-shift detection from unlabeled target embeddings is not
   identifiable. Motivation ties to the A0 line: density/CMI support statistics can be anti-aligned with
   adaptation harm, so the right object is a *certificate with explicit abstention*, not a stronger
   universal detector.
2. **Theory.** Three shift classes — support-visible covariate; boundary coupled with support signature;
   pure conditional. Impossibility result: for any observed `Q_Z` there exist multiple compatible
   `Q(Y|Z)`, so pure conditional shift is unidentifiable from target `Z` alone → abstention is necessary.
3. **Certificate design.** Three-state output `COVARIATE_COMPATIBLE / CONCEPT_SUSPECT / UNIDENTIFIABLE`.
   `CONCEPT_SUSPECT` is **source-anchored** and conditional on a transportability/support-signature
   assumption — NOT distribution-free identification.
4. **Audit discipline (a contribution).** development → freeze (tag + manifest + fail-closed audit) →
   unseen confirmatory, with non-selection rules (fixed point list, conjunction, max-failures=0,
   `max(cond,uncond)` power guard, frozen-code + freshness provenance guards). The P1.5 development map
   showed an apparent favourable region but flagged 12 clusters → CP-UB ≈ 0.221, not control.
5. **Confirmatory result.** Report plainly (the table above; `headline_core_pass=false`, forbidden 1/65
   CP-UB 0.0709, fired 28/65 < required 41, power CP-LB 0.326).
6. **Interpretation.** The positive contribution is a formal + empirical boundary for *when* source-free
   EEG concept-shift certification is identifiable, *when* it must abstain, and *how* a plausible
   development core collapses under frozen confirmatory evaluation — stronger than a marginally positive
   detector.

## Figures (generated: `notes/figures/`, script `csc/tools/make_paper_figures.py`)

- **Fig 1** — shift taxonomy + unidentifiability/abstention boundary.
- **Fig 2** — certificate pipeline + fail-closed gates (3-state output).
- **Fig 3** — development operating map vs frozen confirmatory failure (power + forbidden, dev vs unseen).

## Status / locks

```
CSC-P1.5 DEVELOPMENT map : descriptive evidence only
Frozen confirmatory run  : COMPLETE — scientific FAIL (recorded, valid)
csc-confirmatory-v1      : USED & LOCKED — no further runs under this tag
Method-revision round B  : deferred (NOT opened)
P2 real EEG              : NOT authorized (optional future work / context only)
Next direction           : manuscript around abstention boundary + negative confirmatory evidence
```
