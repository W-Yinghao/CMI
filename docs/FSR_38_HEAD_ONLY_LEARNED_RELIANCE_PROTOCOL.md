# FSR_38 — Head-Only Learned-Reliance Protocol (Phase 7B; pre-registration, NO RUN)

**Project FSR — Phase 7B.** Pre-registration of a **head-only learned-reliance** experiment that bridges the
*injected* controls (PC1/4F/4G) and the *learned* reliance target of PC2, **without** the cost/complexity of a
full FBCSP-LGG GPU refit. It reuses the **Phase-4B frozen branch latents** and retrains only the classifier head
under a source subject↔class skew, asking: *does the natural subject signal in a frozen EEG representation become
a target-harmful learned reliance when the head is trained under source bias, and if so, is it repairable?* **This
is a protocol only. No experiment is run until PM go.** Design-red-teamed before freeze; CPU-only; no GPU, no
backbone retrain, no CMI/fbdualpc, no target-label fit.

## Why head-only (the scientific gap it fills)
PC1/4F/4G inject a *known* shortcut; PC2 wants a *learned* one but needs a costly full refit. The missing rung is:
**freeze the real EEG representation, retrain only the head under source subject-class bias.** This isolates
*learned reliance* from backbone-training confounds, is CPU-cheap, and directly tests recoverability class R3
(FSR_36/37) on a *natural* (not injected) subject signal.

## The 8 pre-registered questions (answers)

**Q1 — Which frozen latents?** The Phase-4B dumps `results/fsr_rq4_refit/latents/*` (per-fold
`{graph,temporal,spatial,fused}_z`, source `y`/`d`, target `y`) for **BNCI2014\_001 + BNCI2015\_001**, 21
leave-one-subject-out folds. Source subjects train the head; the held-out target subject scores only.

**Q2 — Head-only training CPU-feasible?** Yes. The backbone stays frozen; we retrain **only** `head3` (and,
secondary, the `_fuse3` gate) on the frozen 32-dim branch latents. This is a small linear/MLP head on a few
thousand source samples — identical cost class to the 4E/4F adapters (single-threaded torch, seconds/fold). No
GPU.

**Q3 — Induce subject↔class reliance while holding $P(y)$?** For each source subject $d$ assign a spurious class
$c_d$ (deterministic, source-only). Skew $P(y\mid \text{subject}=d)$ toward $c_d$ at stress $\rho$ by
class-stratified **subsampling** of $d$'s trials; choose the $\{c_d\}$ **complementary** across subjects so the
mixture reproduces the **global** $P(y)$ exactly (coupling in the subject×class joint, not the class marginal).
The skew is entirely source-side. **Odd training pools** (e.g. BNCI2015, 11 train subjects / 2 classes) need a
per-fold subsample-weight re-solve to restore global $P(y)$; the **achieved $P(y)$ match, effective $n$, and
per-class subject-diversity (entropy)** are logged per $\rho$ in `head_skew_manifest.csv`, and $\rho$ is **capped
where subsampled $n$ stays above a pre-registered power floor** (else data-loss confounds L6, P3).

**Q4 — Shuffled control?** `H2` = heads trained with the **subject→$c_d$ mapping shuffled** across **multiple
shuffle seeds** — a **null band** (one shuffle over 8–12 subjects retains chance coupling), at matched marginal
$P(y)$ **and matched per-class subject-diversity**. `H1` is tested against the `H2` **band**, not a point
estimate. Isolates subject-coupling from class imbalance *and* from the reduced diversity the skew induces.

**Q5 — Dose-response validation (that reliance is *learned*)?** Train heads at $\rho\in\{0,0.5,0.8\}$
(`H0`=balanced, `H1`=skewed, `H2`=shuffled). Learned reliance is confirmed iff, monotonically in $\rho$: (a) the
head's **subject-reliance** rises (L4 head-alignment to the subject subspace; L5 subject-subspace replay on the
head output) and (b) $\text{H1} > \text{H0}$ and $\text{H1} > \text{H2}$ (**null band**, Q4). Because L6 harm can
also come from data loss (P3), reliance is attributed **through the L4/L5 weaponization signature**, not L6 alone.

**Q5a — LEARNABILITY GATE (freeze-blocker; design-red-team w0vwrsqsp).** On **frozen** latents a *flat* H1 is
confounded with "the subject signal is not head-decodable from *this* $z$ by the `head3` architecture" or "the
head is too weak" or "high-$\rho$ subsampling left too few trials." So a null is only interpretable after a
positive control that the shortcut **could and did** get learned:
(i) measure subject decodability from the frozen $z$ using the **same architecture class as `head3`** (not an
unmatched nonlinear probe); (ii) confirm on **source held-in** data that `H1` exploits $c_d$ (held-in source
subject→class reliance strictly above `H0`). The "**resists head-level weaponization**" reading is licensed
**only** if this gate passes (head demonstrably could and did learn the source shortcut) yet target harm stays
flat. If the gate **fails**, the verdict is "**representation not head-decodable / underpowered**," **not** a
physiology claim.

**Q6 — Repair arms (test E4/E4b/erasure + a training-time control)?** On the skewed head `H1` at deployment,
apply **target-$X$-only** repairs to the target latents fed to the head: **E4** (first-moment mean alignment),
**E4b** (second-moment covariance-shrink), **ERASE** (subject-subspace projection). Plus one **source-only
training-time** arm: **H1$_{\text{reg}}$**, a head trained on the *same* skewed source with a subject-invariance
penalty (source-only; no target labels). Recovery = target bAcc raised toward `H0`, netted where applicable. This
maps the learned reliance onto the FSR_36 recoverability classes (does it look R0-repairable, R1/R2-refused, or
R3-needs-training-time?). **All repair strengths are pinned source-only and never selected on target bAcc (P5,
STOP 1):** E4 $\lambda=1$ (theory-set), fixed ERASE subspace dimensionality, fixed `H1_reg` penalty weight. **L1**
(subject decode from frozen $z$) is a **$\rho$-invariant context metric** — it is a property of the frozen latent,
so any drift with $\rho$ is a recompute bug, not signal (only L4/L5/L6, which are head-dependent, vary with $\rho$).

**Q7 — Target-label firewall?** Target labels enter **only** final scoring (all target bAcc via a `TargetScorer`
guard). Head training (H0/H1/H2/H1$_\text{reg}$), the skew design, $c_d$ assignment, repair fits, and $\rho$/
hyperparameter selection are **source-only**. Recorded per fold in `head_target_label_firewall.json`.

**Q8 — When to escalate to PC2?** Head-only skew is a *learned* reliance = **class R3** (FSR_36/37), which the
theory predicts the deployable target-$X$ arms E4/E4b/ERASE will **fail** on — so **their failure confirms the
class, it is not a veto on escalation** (decoupled from L6/deployable-repair result). PC2 (full GPU refit) becomes
worth considering only if: (a) head-only reliance is confirmed (Q5 dose-response + the Q5a learnability gate +
target-harmful L6); **AND** (b) **either** the source-only **training-time** arm `H1_reg` reaches $\ge$ partial
**or** the reliance is confirmed-but-**unrepairable by every target-$X$ arm** (the scientifically strong *refuse*
case that a controlled learned-reliance study would sharpen); **AND** (c) $\ge 3$ preset-ready datasets (FSR_31);
**AND** (d) explicit PM go. If the learnability gate (Q5a) fails, PC2 is **not** pursued. Head-only cannot, by
itself, authorize PC2 GPU.

## Design (frozen once red-teamed)
- **Heads:** `H0`(ρ0) / `H1`(ρ) / `H2`(shuffled-ρ) / `H1_reg`(ρ + subject-invariance penalty), all source-only,
  fixed architecture (mirror the frozen `head3`; no sweep). Seeds: 8 fresh (mirror 4F/4G) for the confirmatory
  claim; dev seed 0 for mechanism.
- **Ladder per head:** L1 subject decode (frozen latents); L4 head-alignment to subject subspace; L5 subject-
  subspace replay on the head output; **L6 target harm** = `H0` target bAcc $-$ `H1` target bAcc.
- **Gate (inherits 4F/4G corrections):** clustered bootstrap over $(\text{dataset},\text{subject})$ folds;
  structural veto set / task-safety on repair comparators; netting where a clean-target arm applies; report
  all-pooled + per-dataset + dose-response. With **2 datasets**, leave-one-DATASET-out is a **bidirectional
  consistency check, not generalization** (no distribution/CI over datasets); the $\ge 3$-dataset requirement in
  Q8 is the actual generalization guard, and FSR_39 language will label the 2-dataset LODO a consistency check.
- **Pass/fail (repair):** a repair arm is `partial`/`strong` only if it beats its random control on pooled
  clustered CI, is task-safe, and (for `strong`) survives leave-one-dataset-out — identical bar to 4F/4G.

## Outputs (when/if approved)
```
docs/FSR_39_HEAD_ONLY_LEARNED_RELIANCE_RESULTS.md
results/fsr_head_only_learned_reliance/
  head_skew_manifest.csv            # per fold/seed/rho: c_d, achieved P(y) match, eff-n, subject-diversity, ids
  head_learnability_gate.csv        # Q5a: head3-class subject decode; H1 source-held-in c_d reliance vs H0
  head_reliance_dose_response.csv   # L4/L5 vs rho for H0/H1/H2-band (L1 = rho-invariant context)
  head_target_harm.csv              # L6 target harm vs rho (attributed THROUGH L4/L5)
  head_repair_results.csv           # E4/E4b/ERASE (target-X, expected R3-fail) + H1_reg (training-time) on H1
  head_target_label_firewall.json
  head_verdict.json                 # learned_reliance_induced, imbalance_confound_ruled_out,
                                     #   recoverability_class, repair_claim_level, escalate_pc2
```

## STOP rules
```text
1  target labels used for head training / skew design / c_d / repair fit / rho selection, OR for SELECTION of any
   repair strength (E4 lambda / ERASE dim / H1_reg penalty) -- all pinned source-only.
2  GPU / backbone retrain / CMI / fbdualpc / architecture or hyperparameter sweep beyond the frozen head.
3  global P(y) not held fixed across rho (imbalance confound), OR rho pushed below the pre-registered n power
   floor / subject-diversity not reported+matched -> re-design.
4  learned reliance claimed without the shuffled-stress (H2) imbalance control AND a rho dose-response.
5  a repair pass claimed without clustered CI + leave-one-dataset-out (the 4F/4G bar).
6  head-only result used to authorize PC2 GPU by itself (Q8 requires >=3 datasets + PM go).
7  CLAIM-LOCK: written as natural/general/DG/SOTA repair, or as "natural subject leakage is harmful"
   (head-only induces a CONTROLLED learned reliance; it is not a natural-harm claim).
8  "resists head-level weaponization" claimed WITHOUT the Q5a learnability gate passing (source-held-in: head
   could + did learn the source shortcut) -> a flat H1 is then "not head-decodable/underpowered", NOT physiology.
9  Q8 escalation vetoed by a deployable target-X arm failing -- their failure CONFIRMS class R3, it is not a veto
   (escalation is decoupled from the deployable-repair result; see Q8).
```

## Framing (fixed)
A head-only positive would show natural subject signal *can* be turned into a target-harmful learned reliance by
source bias (a controlled learned shortcut, not a natural-harm claim); a head-only negative would show the
natural signal resists head-level weaponization (supporting the 4B task-physiology reading). Either way it maps
the learned reliance onto the FSR_36 recoverability classes and decides — honestly and cheaply — whether the
expensive PC2 GPU is worth it. **Awaiting PM go; nothing runs from this document.**
