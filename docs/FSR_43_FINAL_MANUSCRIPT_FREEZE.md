# FSR_43 — Final Manuscript Freeze (Paper 2)

**Project FSR.** PM decision (2026-07-07): **GO to manuscript freeze.** With the 7B+7C mechanism bridge merged into
§08, Paper 2 enters proofread / submission hardening. No more experiments (FSR_42).

## What changed in this freeze patch (7B+7C bridge)
- **§08 `08_recoverability.tex`** — new `\paragraph{Head-only learned-reliance bridge.}`: two CPU-only head-only
  probes on frozen 4B latents show prevalence skew is not learned (7B) and subject-keyed task-conflict is learnable
  in-sample but not transferably subject-structure-specific (7C); both sit in the R3 (under-identified) region and
  neither licenses PC2. Explicit boundary sentence added ("do not prove learned shortcuts cannot occur").
- **`tables/table4_recoverability.tex`** — two R3-region probe rows (7B prevalence / 7C task-conflict). Also
  converted to `tabularx{\textwidth}` — this **fixes a pre-existing overfull** (the R0–R3 table was 109pt over the
  text block in the frozen version; now it wraps and fits).
- **§10 `10_limitations.tex`** — new bullet: head-only probes are **mechanism tests, not natural-label-noise
  claims** (do not read as "resists weaponization"); the "two datasets" bullet reconciled (full-backbone PC2 =
  future work; head-only is the cheap substitute).
- **§11 `11_conclusion.tex`** — closing paragraph: a shortcut claim requires an **information contract** (leakage
  must be shown task-coupled, target-harmful, transferable, and recoverable); in our audits those conditions
  **split** rather than collapse into a single leakage score.
- **`tables/table1_claim_ledger.tex`** — added **C18** (7B: prevalence does not demonstrate weaponization) and
  **C19** (7C: task-conflict learnable in-sample but not transferable structure-specific harm), both
  READY_WITH_CAVEAT, matching `docs/FSR_05`.

## Build / hygiene status (verified)
```
pdflatex main.tex ............ builds (exit 0, 2 passes + bibtex)
pages ........................ 14
undefined refs/citations ..... 0
claim-hygiene checker ........ PASS (18 files; 7 forbidden phrases, all negated/safe; 0 violations)
table4 overfull .............. FIXED (was 109pt over; now fits via tabularx)
remaining overfulls .......... pre-existing (102/90/82pt) in tables NOT touched by this patch; unchanged from the
                               frozen version; out of scope for this freeze
```

## Final science story (frozen)
> **Subject leakage in EEG is not automatically a harmful shortcut.** FSR separates leakage, reliance, target
> harm, and repairability under observability contracts. Natural spatial subject leakage is strong and
> task-coupled but not verified harmful; blind erasure hurts. Controlled first-moment shortcuts are repairable
> within a narrow construction-matched scope; controlled second-moment perturbations are not. Head-only
> learned-reliance probes show that prevalence skew and even task-conflicting subject-keyed corruption do not
> readily weaponize the natural subject signal into transferable, subject-structure-specific harm.

Three layers of boundary, not a single leakage score:
```
1. Natural EEG:                 subject signal is task-entangled, not automatically harmful (refuse).
2. Controlled injected:         first-moment repairable; second-moment not repairable by current operators.
3. Learned head-level stress:   prevalence skew inert; task-conflict learnable in-sample but not transferable
                                subject-structured reliance.
```

## Scope separation (important)
**Paper 1 (Prior-Decoupled TTA) is independent and unaffected** — it is a prior/geometry/decision-prior
identification paper (three-prior distinction + four-branch decomposition), NOT a subject-shortcut paper. Do not
merge the two lines.

## Status
Manuscript: **FROZEN** → proofread / submission hardening only (no new experiments, no new claims beyond the C1–C19
ledger). PC2 stays PAUSED (FSR_42).
