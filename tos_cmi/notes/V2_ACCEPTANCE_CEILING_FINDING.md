# V2 finding: source-only acceptance has a provable CEILING (World A is not passable as an "accept" test)

Status: **design-lock BLOCKED on a PM decision.** The V2 scaffold is written and reviewed (Worlds B/C, gate
reuse, fair router, target-leakage all clean); but the acceptance-power leg (World A) is **not constructible**
as a gate-ACCEPT test, and the reason is a genuine result, not a bug.

## What we did
1. Adversarial design review (6 lenses) flagged and one lens **simulated** that the original World A ("reversed"
   nuisance, reliability varies by subject) gives net source-LOSO benefit <= 0 for every phi -> unpassable.
2. We added two better mechanisms (`aligned_noise`, `aligned_noise_flip`: a MINORITY of source subjects carry
   the spurious shortcut so the head uses it, the MAJORITY see the nuisance as noise) and **searched** the
   construction space on REAL Lee/Cho EEGNet latents: 3 mechanisms x f_align{0.15,0.25,0.35} x
   alpha{0.5,1,2,3} x {LEACE, fair_conditional} x 2 datasets x 8 folds = 144 aggregated cells.

## What we found (tos_cmi/results/method_deepen/v2/worldA_search.csv)
* **0 / 144 cells ACCEPT** (safe AND source-LOSO benefit LCB > +0.01). Max benefit LCB across ALL cells =
  **+0.0009** (noise floor); max among SAFE cells = **+0.0009**. The gate essentially never sees a source-LOSO
  benefit from erasing an injected nuisance.
* **49 / 144 cells have a REAL target benefit** (target dbAcc LCB > +0.01), up to **+0.46**. NONE are accepted.
* **Smoking gun (8 SAFE cells with real target gain):** e.g. `aligned_noise_flip, f=0.15, alpha=1.0, LEACE`:
  task-drop UCB <= 0.02 (**safe**), target dbAcc **+0.097 [LCB +0.067]** (**genuinely beneficial at target**),
  yet source-LOSO benefit LCB = **-0.046** -> gate **ABSTAINS**.
* Unsafe cells (`reversed`): target dbAcc up to +0.46, but source task-drop UCB ~ +0.28 -> gate **REJECTS**.

## Why (mechanism, not miscalibration)
For the gate to ACCEPT it must see a benefit *within source* (source-LOSO). But a nuisance is beneficial to
erase exactly when it is **misleading at the target** while **helpful (or neutral) in source** -- i.e. the harm
lives in the **source->target shift**. Held-out SOURCE subjects share the source shortcut, so source-LOSO shows
NO benefit (indeed negative). Symmetrically, source-defined **safety cannot distinguish** "erasing a genuine
task signal" (truly unsafe) from "erasing a source-shortcut that only misleads at the target" (beneficial):
both look like a source task-drop. **Source-only information is insufficient to certify a benefit that
manifests only under the deployment shift.**

## Why this is valuable (completes the measurement->control-gap thesis)
* Track B / real EEG: erasure of real nuisances is useless/harmful -> gate correctly REFUSES.
* Phase 2: even task-preserving erasure is transfer-flat -> refusal correct.
* **V2: even when erasure WOULD genuinely help the target (+0.06 .. +0.46), source-only signals cannot
  certify it -> the gate cannot ACCEPT.** Source-only certification has a **provable acceptance ceiling**.

## Decision for the PM (do NOT lock until chosen)
1. **[recommended] Reframe World A as the acceptance-CEILING demonstration.** Keep B (REJECT) + C (ABSTAIN)
   as reject/abstain-power validators. World A instead SHOWS beneficial-at-target erasures (safe: +0.08 target;
   unsafe: +0.46) that the source-only gate correctly cannot accept, with the mechanism above. No threshold
   change, no manufacturing. V2 claim becomes a limit theorem, not an "accept works" demo.
2. **Add a target-informed acceptance branch** (transductive/UDA extension): give the gate a little target
   info -- (a) UNLABELED target (does erasing reduce source-target mismatch while preserving source task?), or
   (b) K target labels (few-shot benefit probe) -- and quantify how much target info crosses the ceiling.
   Turns the ceiling into a bridge; connects to EA-transductive / SCPS; more work; loosens strict-DG framing.
3. **Keep chasing a source-only-detectable beneficial nuisance** (leave-one-source-DOMAIN-out with
   domain-varying alignment; capacity-overfitting nuisances). Mechanistic analysis + the 144-cell search make
   this unlikely and it risks looking reverse-engineered. **Advised against.**

Recommendation: **Option 1 as primary**, optionally **Option 2** as a constructive complement.
