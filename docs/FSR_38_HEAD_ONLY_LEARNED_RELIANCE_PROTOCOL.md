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
The skew is entirely source-side.

**Q4 — Shuffled control?** `H2` = head trained with the **subject→$c_d$ mapping shuffled** at matched marginal
(spurious structure destroyed, same $P(y)$). Isolates subject-coupling from class imbalance.

**Q5 — Dose-response validation (that reliance is *learned*)?** Train heads at $\rho\in\{0,0.5,0.8\}$
(`H0`=balanced, `H1`=skewed, `H2`=shuffled). Learned reliance is confirmed iff, monotonically in $\rho$: (a) the
head's **subject-reliance** rises (L4 head-alignment to the subject subspace; L5 subject-subspace replay on the
head output) and (b) $\text{H1} > \text{H0}$ and $\text{H1} > \text{H2}$. A flat curve ⇒ no learned reliance ⇒
report and stop (informative negative: natural subject signal resists head-level weaponization).

**Q6 — Repair arms (test E4/E4b/erasure + a training-time control)?** On the skewed head `H1` at deployment,
apply **target-$X$-only** repairs to the target latents fed to the head: **E4** (first-moment mean alignment),
**E4b** (second-moment covariance-shrink), **ERASE** (subject-subspace projection). Plus one **source-only
training-time** arm: **H1$_{\text{reg}}$**, a head trained on the *same* skewed source with a subject-invariance
penalty (source-only; no target labels). Recovery = target bAcc raised toward `H0`, netted where applicable. This
maps the learned reliance onto the FSR_36 recoverability classes (does it look R0-repairable, R1/R2-refused, or
R3-needs-training-time?).

**Q7 — Target-label firewall?** Target labels enter **only** final scoring (all target bAcc via a `TargetScorer`
guard). Head training (H0/H1/H2/H1$_\text{reg}$), the skew design, $c_d$ assignment, repair fits, and $\rho$/
hyperparameter selection are **source-only**. Recorded per fold in `head_target_label_firewall.json`.

**Q8 — When to escalate to PC2?** PC2 (full GPU refit) becomes worth considering **only if all** hold: (a) head-
only learned reliance is confirmed (Q5 dose-response, target-harmful L6); (b) at least one repair arm reaches
$\ge$ partial under the corrected gate (clustered CI, leave-one-dataset-out); (c) $\ge 3$ preset-ready datasets
(FSR_31). If head-only reliance does **not** appear, PC2 is **not** pursued (the natural signal is task-
physiological, not head-weaponizable). Head-only cannot, by itself, authorize PC2 GPU.

## Design (frozen once red-teamed)
- **Heads:** `H0`(ρ0) / `H1`(ρ) / `H2`(shuffled-ρ) / `H1_reg`(ρ + subject-invariance penalty), all source-only,
  fixed architecture (mirror the frozen `head3`; no sweep). Seeds: 8 fresh (mirror 4F/4G) for the confirmatory
  claim; dev seed 0 for mechanism.
- **Ladder per head:** L1 subject decode (frozen latents); L4 head-alignment to subject subspace; L5 subject-
  subspace replay on the head output; **L6 target harm** = `H0` target bAcc $-$ `H1` target bAcc.
- **Gate (inherits 4F/4G corrections):** clustered bootstrap over $(\text{dataset},\text{subject})$ folds;
  **leave-one-DATASET-out binding**; structural veto set / task-safety on repair comparators; netting where a
  clean-target arm applies; report all-pooled + LODO + per-dataset + dose-response.
- **Pass/fail (repair):** a repair arm is `partial`/`strong` only if it beats its random control on pooled
  clustered CI, is task-safe, and (for `strong`) survives leave-one-dataset-out — identical bar to 4F/4G.

## Outputs (when/if approved)
```
docs/FSR_39_HEAD_ONLY_LEARNED_RELIANCE_RESULTS.md
results/fsr_head_only_learned_reliance/
  head_skew_manifest.csv            # per fold/seed/rho: c_d, P(y) match, train subject ids
  head_reliance_dose_response.csv   # L1/L4/L5 vs rho for H0/H1/H2
  head_target_harm.csv              # L6 target harm vs rho
  head_repair_results.csv           # E4/E4b/ERASE/H1_reg recovery on H1
  head_target_label_firewall.json
  head_verdict.json                 # learned_reliance_induced, imbalance_confound_ruled_out,
                                     #   recoverability_class, repair_claim_level, escalate_pc2
```

## STOP rules
```text
1  target labels used for head training / skew design / c_d / repair fit / rho or hyperparameter selection.
2  GPU / backbone retrain / CMI / fbdualpc / architecture or hyperparameter sweep beyond the frozen head.
3  global P(y) not held fixed across rho (imbalance confound) -> re-design.
4  learned reliance claimed without the shuffled-stress (H2) imbalance control AND a rho dose-response.
5  a repair pass claimed without clustered CI + leave-one-dataset-out (the 4F/4G bar).
6  head-only result used to authorize PC2 GPU by itself (Q8 requires >=3 datasets + PM go).
7  CLAIM-LOCK: written as natural/general/DG/SOTA repair, or as "natural subject leakage is harmful"
   (head-only induces a CONTROLLED learned reliance; it is not a natural-harm claim).
```

## Framing (fixed)
A head-only positive would show natural subject signal *can* be turned into a target-harmful learned reliance by
source bias (a controlled learned shortcut, not a natural-harm claim); a head-only negative would show the
natural signal resists head-level weaponization (supporting the 4B task-physiology reading). Either way it maps
the learned reliance onto the FSR_36 recoverability classes and decides — honestly and cheaply — whether the
expensive PC2 GPU is worth it. **Awaiting PM go; nothing runs from this document.**
