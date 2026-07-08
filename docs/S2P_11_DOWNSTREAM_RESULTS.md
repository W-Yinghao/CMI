# S2P_11 — P1 Downstream Results (neutral; D0/D0.5/D1 complete)

**Project S2P — P1 fixed-budget subject-vs-depth frontier.** Neutral downstream results for the 15-run CBraMod
frontier (T=200 h, N in {128,256,512,1024,2048}, seeds {0,1,2}), plus single random-init and released-CBraMod
references. This doc reports the completed `shumi_downstream_audit.py` / `frontier_summary.py` path on SHU-MI LMDB
200 Hz. It supersedes the stale S2P_09 handoff state for downstream status, but does not delete or reinterpret the
parallel S2P_09 scaffold.

Machine artifacts:

- `results/s2p_p1_downstream/p1_task_and_frontier_raw.csv`
- `results/s2p_p1_downstream/p1_frontier_summary.json`
- `results/s2p_p1_downstream/p1_D0_probe_gate.json`
- `results/s2p_p1_downstream/d0p5_decodability_sanity/d0p5_decodability_verdict.json`
- `results/s2p_p1_downstream/p1_target_label_firewall.json`

Code / provenance commits:

- P1 pretraining results: `2ce30cf`
- D0 probe gate: `463c5dd`
- D0.5 decodability verdict: `1965f44`
- D1 downstream raw results: `abc9fbc`
- D1 verified frontier summary: `745337c`
- Current handoff: `ee07985`

## Gate status

| gate | status | notes |
|---|---|---|
| D0 probe gate | **PASS** | 8/8 checks true: channel map, native 4-patch forward, deterministic embeddings, source-only pipeline, metrics, variance null, firewall, output schemas |
| D0 task status | **AT FLOOR** | probe target bAcc 0.492 vs random-init 0.503 for the probe cell |
| D0.5 decodability sanity | **D1 allowed** | raw-CSP strict within-subject 0.60 gate missed, but label/split/channel audit passed and released-checkpoint sanity reproduced above-floor SHU-MI transfer |
| D1 fleet | **complete** | 15 P1 cells + random-init control + released reference |
| target-label firewall | **clean** | target labels final scoring only; not used for PCA, head fit, selection, subject subspace, rank choice, or checkpoint selection |

Primary D1 norm is patch normalization for P1 and random-init. Released CBraMod is reported under both patch and
window normalization because the released reference is normalization-sensitive.

## D1 per-N means

Patch-normalized P1 means over 3 seeds, spatial embedding. Random-init is a single-seed floor; released is a reference
checkpoint, not part of the P1 frontier.

| N | L1 subject sep | target bAcc | source-val bAcc | L4 | L5 z |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.832 | 0.518 | 0.564 | 0.010 | +2.5 |
| 256 | 0.839 | 0.517 | 0.565 | 0.004 | +4.8 |
| 512 | 0.854 | 0.519 | 0.559 | 0.008 | +1.5 |
| 1024 | 0.844 | 0.528 | 0.576 | 0.005 | +7.0 |
| 2048 | 0.836 | 0.518 | 0.575 | 0.006 | +0.8 |
| random-init | 0.747 | 0.503 | 0.576 | 0.013 | -6.4 |
| released, patch | 0.893 | 0.553 | 0.621 | 0.002 | -4.3 |
| released, window | 0.899 | 0.590 | 0.671 | 0.001 | +0.4 |

## Verified conclusions

1. **L1 is flat across the allocation frontier.** The verified D1 summary reports perm-ANOVA F=0.809, p=0.569;
   slope permutation p=0.713; slope +0.0012 per log2(N). The apparent N=512 maximum is not reported as a peak or
   optimum: it is driven by the single N512_s0 global-max cell, and leave-one-N / seed sensitivity does not support
   curvature language.

2. **Target MI transfer is at a near-chance floor for P1.** Mean P1 target bAcc is approximately 0.520 against the
   random-init floor 0.503. The margin is directionally positive, but all P1 cells score the same five SHU-MI target
   subjects, so the pooled result is treated as practically near chance. The target-transfer slope is not interpreted:
   the endpoint has not cleared the random floor enough to support an allocation-effect test. The released reference
   clears the floor under both patch (0.553) and window (0.590) normalization.

3. **Pretrained P1 checkpoints are numerically above random-init on L1 at every N.** This is the only practically
   meaningful positive signal in D1. It is a pretraining / architecture contrast, not an allocation effect. The
   random-init L1 floor is high (0.747) and is a single uncertified seed with no CI.

4. **L4/L5/L6 remain weak task diagnostics only.** Only 4/15 P1 cells pass the source-val >= 0.58 task gate, so the
   task-dependent mechanism diagnostics are not interpreted as primary effects. Correction carried forward from the
   adversarial review: L5z is systematically positive on pretrained P1 cells, but the underlying subject-removal drop
   is only about 0.005 bAcc and is practically trivial.

5. **"Flat" means no detectable allocation effect under this measurement design, not proven zero.** The 3-seed design
   is underpowered for small effects, and the frontier remains confounded with population: N=128 draws from a deep,
   more clinical pool while N=2048 draws from a much broader general pool.

6. **P2 is not recommended by the verified D1 summary.** A second budget would multiply an underpowered near-null and
   repeat the N-to-population confound. If the line continues, the verified recommendation is more seeds plus
   population de-confounding at the existing 200 h budget, not a new allocation axis without PM approval.

## Required caveats

- SHU-MI is weak / low-ceiling under this protocol: raw-CSP strict within-subject median is 0.54 and misses the 0.60
  gate, although cross-subject CSP, label/channel audit, and released-checkpoint reproduction support D1 validity.
- Released-checkpoint transfer is above the random floor but with a limited margin and no new CI in this audit.
- Released-checkpoint pretraining provenance is not certified here; upstream overlap with SHU-MI target subjects cannot
  be ruled out at the pretraining-data level. The probe firewall is clean.
- Released patch and window normalization must both be reported; window normalization also lifts the random encoder.
- The P1 estimand remains the bundled fixed-budget allocation frontier, not a pure subject-diversity or population-
  adjusted causal effect.

## Claim boundary

Allowed neutral headline:

> At a fixed 200 h pretraining budget, from-scratch CBraMod learns subject-identifiable structure (L1 >> random-init)
> but not frozen-probe MI-transferable structure (target transfer remains at the floor). The L1 allocation frontier is
> flat, while the target-transfer allocation slope is not meaningful because the endpoint has not risen above floor.

Do not claim from these results that subject diversity does not matter in general, that TUEG pretraining solves
cross-subject transfer, that 200 h proves a zero allocation effect, or that CBraMod cannot learn MI-transferable
representations at larger pretraining scale.
