# S2P_10 — Codex Handoff (CURRENT, authoritative) — supersedes S2P_09_CODEX_HANDOFF.md

**Written 2026-07-08 by the Claude session that ran D0 → D0.5 → D1 to completion.** Read top-to-bottom before
touching anything. The older `docs/S2P_09_CODEX_HANDOFF.md` is **STALE** (it says "blocked at D0, don't launch D1") —
D1 is **done and adversarially verified**. Where the two disagree, THIS file wins.

---

## 0. TL;DR — the work is essentially COMPLETE; result is a well-characterized MOSTLY-NULL

- **Project S2P** = does controlling subject-count DURING EEG foundation pretraining change the learned representation?
- **P1 = fixed-budget subject-vs-depth FRONTIER**: fixed **T=200 h** pretraining budget, vary N∈{128,256,512,1024,2048}
  subjects × seeds{0,1,2} = 15 CBraMod checkpoints from scratch, + a random-init floor + the released CBraMod as a
  reference. Estimand = **bundled fixed-budget allocation trend** (coverage⊕depth⊕population⊕redundancy; DESCRIPTIVE,
  NOT causal, NOT pure-diversity — the `T=N·e` identifiability triangle makes pure diversity unidentifiable; settled).
- **DONE:** pretraining (15/15 clean) → downstream **D0** probe-gate (pipeline valid, but transfer at chance) →
  **D0.5** decodability sanity (adversarially verified: pipeline reproduces Phase-8B, SHU-MI decodable, P1=valid
  negative) → **D1** 16-cell fleet → frontier summary → **adversarial verification of the D1 conclusion.**
- **THE RESULT (verified):** At the fixed 200 h budget the **coverage-vs-depth allocation axis has NO detectable
  effect on any endpoint** — L1 subject-separability is FLAT (perm-ANOVA F=0.809, p=0.569; slope perm p=0.713),
  target MI transfer is a near-chance floor. The only **practically meaningful** signal is **pretrained > single-seed
  random-init on L1 at every N** (15/15, min gap ~0.077) — but that's an architecture/pretraining effect, not an
  allocation effect (random-init already captures ~72.5 % of the above-chance separability). **P2 = NOT recommended.**
- **WHAT'S LEFT (small):** (1) write the neutral results doc (§8); (2) **return to the PM** with the verified result +
  proposed framing and **WAIT** for the final steer (report-then-wait; the PM decides close/park vs any follow-up).
  The science is done; do **not** launch new compute without a PM go.

---

## 1. Mission & non-negotiable discipline (skill `preregistered-experiments` + PM feedback)

Negative-result-driven, provenance-bound experimental science; the PM steers framing tightly. Optimize for
**defensible, reproducible, honestly-reported** — not speed, not method-search.

**Loop:** Design → **FREEZE `*_FROZEN.md` pre-reg before compute** → **PROBE one unit, gate on QC** → **fleet** →
**Analyze → neutral `*_RESULTS.md` → propose framing → WAIT for PM steer.**

- **Disclose surprising cells BEFORE concluding; under-claim.** (This session's D1 verification caught me calling L5z
  "erratic noise" — wrong — and over-claiming "only L1 is robust." Both corrected. Expect the same scrutiny.)
- **Adversarially verify findings** (multi-skeptic) before finalizing — used twice here (D0.5 verdict, D1 conclusion).
- **Two-commit:** result-only commit first, interpretation second — never mixed. `Co-Authored-By` trailer.
- **Report the gate-checklist, then WAIT for explicit "go" before consuming GPU.** PM decisions as **prose options +
  trade-offs + WAIT** — never a multiple-choice picker (the PM rejects those).
- **SLURM only** for GPU (no login-node training); **set `--time` under the smallest partition cap** (A100/H100 =
  1 day; V100 = 2 day; A40 = 4 day → a no-`--time` job defaults to A40's 4-day and is REJECTED by A100/H100/V100 with
  `PartitionTimeLimit`; use `--time=8:00:00` etc.). **`scancel <jobid>` only — never `ps`/`kill`** (Bash may run on a
  SLURM node). Don't let a job pend on a busy partition — scancel + resubmit on a free one; V100 usually idle.
- **Role: experiments-only.** No manuscript/`paper/` edits; the paper narrative is owner-controlled.
- **Auto-push after each commit** IS the norm here — but note the old handoff claimed pushes were "gated/held". In THIS
  session pushes went to origin normally; `origin/project/s2p-subject-scaling` == local HEAD. Keep pushing after commits.

---

## 2. GitHub — repo / branch / commits

- **Repo:** `git@github.com:W-Yinghao/CMI.git` · **Branch:** `project/s2p-subject-scaling`
- **Worktree (SHARED — two agents committed here):** `/home/infres/yinwang/CMI_AAAI_s2p`
- **origin is UP TO DATE** at the latest commit (this session pushed everything). `git log --oneline -1` should show
  the D1-summary commit (`745337c` at handoff time, or later).
- **Key commits (newest first):**

  | SHA | what |
  |---|---|
  | `745337c` | D1 frontier summary: verified conclusion + corrections (p2=False) |
  | `abc9fbc` | **D1 downstream frontier RESULTS** (result-only): flat frontier + transfer null |
  | `1965f44` | D0.5 verdict: `d1_allowed=TRUE` (adversarially verified) — P1 = valid negative |
  | `47aa969` | D0.5 sanity (partial): CSP + label-audit |
  | `463c5dd` | D0 probe-gate: pipeline PASS but transfer AT CHANCE (return-for-review) |
  | `0a0fcee`,`bccd013`,`95c00e2`,`e010160` | **CONCURRENT worker's S2P_09 downstream line** (see §3) |
  | `2ce30cf` | S2P_08 P1 pretraining results (15/15 clean) |
  | `e58038b` | P1 pretraining runner + LAUNCH · `1441096`/`3f8a708` P1 protocol v4 frontier · `c98aecb` CodeBrain smoke |

---

## 3. ⚠️ TWO parallel downstream implementations exist on this branch — use the COMPLETED one

This is the single most important thing to understand:

| | **THIS session (COMPLETED)** | Concurrent worker's S2P_09 (stopped at D0) |
|---|---|---|
| runner | `s2p/scripts/shumi_downstream_audit.py` + `frontier_summary.py` | `s2p/scripts/run_downstream_frontier.py` |
| SLURM | `downstream_audit.slurm`, `csp_sanity.slurm` | `downstream_d0_probe.slurm`, `downstream_d1_array.slurm` |
| pre-reg / handoff | (results in `docs/` S2P_08 + this file) | `docs/S2P_09_DOWNSTREAM_AUDIT_FROZEN.md`, `docs/S2P_09_CODEX_HANDOFF.md` |
| **SHU-MI data** | **LMDB `/projects/EEG-foundation-model/tdoan-24/SHUMI_200hz` (200 Hz, native)** | `.mat` `/projects/EEG-foundation-model/SHU-MI-cbramod/mat/` (250 Hz) |
| status | **D0 + D0.5 + D1 DONE + verified** | D0 PASS only, then blocked |

Both converged on the **same science** (transfer ≈ chance ≈ random floor; L1 strong). **Continue with THIS session's
`shumi_downstream_audit.py`/`frontier_summary.py` path** — it has the completed, verified D1 results. The S2P_09
`run_downstream_frontier.py` line is a valid independent replication scaffold (kept, not deleted); reconcile/retire it
with the PM if desired, but do not confuse its stale handoff/pre-reg for the current state.

---

## 4. Files that matter (paths relative to the worktree root)

**Active (this session):**
| File | Role |
|---|---|
| `s2p/scripts/tueg_subject_loader.py` | **Upstream loader.** `COMMON19` canonical channel order; `build_frontier_cell(N,subset_seed,total_hours=200)` (fixed-budget frontier, exact 24000-window budget, fixed global val, floored-window eligibility); per-PATCH z-score. |
| `s2p/scripts/run_frontier_cbramod.py` | **P1 pretraining runner** (thin adapter over native CBraMod `Trainer_valid`; masked-recon MSE 0.5; per-patch z-score, HBN 129-ch normalizer neutralized, no `/100`; best-by-pretrain-val checkpoint saved as `{"model_state":...}`). |
| `s2p/scripts/shumi_downstream_audit.py` | **Downstream audit.** SHU-MI LMDB loader (19-common via hashed idx, native 4-patch, `--norm patch|window`), frozen CBraMod `--embedding spatial|mean`, source-only PCA(64)+logistic head, L1 (session-held-out pairwise), subject_subspace, L4, **L5 low-rank subspace-removal vs variance-matched null** (optimized), firewall. `resolve_ckpt`: `N{N}_s{S}`→P1, `random`, `released`. `--mode D0` (probe-gate) / `D1` (fleet). |
| `s2p/scripts/frontier_summary.py` | Post-processes the D1 raw CSV → the 12 required outputs + `p1_frontier_summary.json` (L1 frontier slope/curvature/leave-one-N, task-gate, released ref, population join). |
| `s2p/scripts/shumi_csp_sanity.py` | D0.5-A raw-CSP sanity (manual CSP+LDA, band-pass 8–30 Hz, within/cross-subject + shuffle control). |
| `s2p/scripts/downstream_audit.slurm` | GPU audit job (`--export MODE,CELLS,EMB,NORM,OUT`; `--time=8h`; `A40,A100,H100,V100`). |
| `s2p/scripts/csp_sanity.slurm` | CPU CSP job. |
| `results/s2p_p1_cbramod/N{N}_s{S}/best.pth` | 15 P1 checkpoints (gitignored, on-disk). `run_summary.json` has `best_val_loss`. |
| `results/s2p_p1_downstream/` | Downstream outputs (see §7). `p1_frontier_summary.json` = the verdict. `d0p5_decodability_sanity/` = D0.5. |

**Released CBraMod reference weights:** `/home/infres/yinwang/eeg2025/NIPS/Cbramod_pretrained_weights.pth`
(flat state_dict, 211 tensors, 4.88 M params, loads clean into `CBraMod(...seq_len=4...)` strict). Reproduces
Phase-8B SHU-MI band (target 0.590 window / 0.553 patch vs published 0.598 CI[0.561,0.638]).

---

## 5. Environment & data (verified 2026-07-08)

- **Conda:** `source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh && conda activate eeg2025`
  (python `…/envs/eeg2025/bin/python`; torch+sklearn+scipy+numpy+lmdb+mne). `icml` env also has pyarrow (for parquet).
  Base anaconda python LACKS pyarrow/lmdb — use `eeg2025`/`icml`.
- **CBraMod source:** `/home/infres/yinwang/eeg2025/CBraMod` (on `sys.path` inside runners → `from models.cbramod import CBraMod`).
  Arch `CBraMod(in_dim=200,out_dim=200,d_model=200,dim_feedforward=800,seq_len=4,n_layer=12,nhead=8)` ~4.92 M params;
  native 4-patch forward `(B,19,4,200)→(B,19,4,200)` (encoder out is **4-D**: `(B,19,4,200)` → pool all tokens).
- **Pretraining corpus (TUEG):** `/projects/EEG-foundation-model/datalake/processed/4704743c/TUEG` (200 Hz, metadata.parquet,
  19-common subset 6,535 subj / 3,440 usable-h). Fixed-budget frontier verified feasible; N=128 deep pool only 201
  (clinical), N=2048 pool 6,388 (general) → the disclosed N↔population confound.
- **SHU-MI (this session's path):** LMDB `/projects/EEG-foundation-model/tdoan-24/SHUMI_200hz` — 200 Hz, keys
  `sub-XXX_ses-YY_task_motorimagery_eeg-{trial}`, value dict `{sample:(32,800), label:0/1}`. 11,988 trials, 25 subjects,
  balanced. Channel order verified vs EDF (`SHU-MI-cbramod/edf/…`) at 20.8 SD over permutation null; 19-common row idx
  `[0,1,3,4,12,13,23,24,30,31,5,6,14,15,25,26,2,11,22]` hashed. Split: source-train 1–15, source-val 16–20, target-test 21–25.
- **CodeBrain** = **P2-READY infrastructure only, NOT in P1** (9B-0C native smoke passed on V100). Do not add to P1.
- **SLURM:** partitions `A40,A100,H100,V100`; QOS ~8 GPUs/user concurrent; `--gres=gpu:1`; **always `--time` ≤ 1 day**.

---

## 6. The result in full (verified) — copy into any results doc

**Per-N means (patch-norm, spatial embedding; `results/s2p_p1_downstream/p1_task_and_frontier_raw.csv`):**

| N | L1 subj-sep | target bAcc | source-val | L4 | L5 z |
|---|---|---|---|---|---|
| 128 | 0.832 | 0.518 | 0.564 | 0.010 | +2.5 |
| 256 | 0.839 | 0.517 | 0.565 | 0.004 | +4.8 |
| 512 | 0.854 | 0.519 | 0.559 | 0.008 | +1.5 |
| 1024 | 0.844 | 0.528 | 0.576 | 0.005 | +7.0 |
| 2048 | 0.836 | 0.518 | 0.575 | 0.006 | +0.8 |
| **random-init** | **0.747** | 0.503 | 0.576 | 0.013 | −6.4 |
| **released (patch/window)** | 0.893/0.899 | 0.553/0.590 | 0.621/0.671 | ~0.002 | −4.3/+0.4 |

**Verified conclusions (adversarial workflow wf_17621f8d, 4 skeptics):**
1. **L1 FLAT across allocation** — perm-ANOVA F=0.809 p=0.569; 15-pt slope perm p=0.713; slope +0.0012/log₂N. **The
   apparent N=512 peak is a single-seed-0 artifact** (N512_s0=0.888 is the global max); drop seed 0 → weak monotone
   rise, no peak. **DROP all curvature/peak/"optimal-allocation" language.**
2. **Target transfer = near-chance floor** — mean 0.520 vs random 0.503; "near-chance-but-significant" (13/15 above
   floor, t=4.28 p=8e-4) BUT all 15 cells score the SAME 5 target subjects (non-independent) → pooled margin
   practically chance. Released (0.55–0.59) clears it → **200 h from-scratch is insufficient for cross-subject MI transfer.**
3. **Pretrained > random-init on L1 at every N** (15/15, min gap ~0.077) — the only **practically meaningful** signal,
   but an architecture/pretraining effect, NOT allocation. Caveat: random-init already captures ~72.5 % of the
   above-chance separability (learned increment ~27.5 %); the 0.747 floor is a single uncertified seed with no CI.
4. **L4/L5/L6 = WEAK_TASK_DIAGNOSTIC_ONLY** — only 4/15 cells pass the source-val ≥0.58 task gate. (Correction: L5z is
   **not** "erratic noise" — it is systematically positive on pretrained, 12/15 z>0, sign-flipped from random-init −6.4,
   but the underlying subject-removal drop is only ~0.005 bAcc = practically trivial. Keep the not-interpretable verdict,
   fix the descriptor.)
5. **"Flat" = signal buried in measurement + 3-seed noise, NOT proven zero.** N↔population confound (clinical 201 @
   N=128 → general 6,388 @ N=2048) entangles allocation with population — even a monotone hint couldn't be cleanly
   attributed to allocation. This underpowered 3-seed design cannot rule out a small real effect.
6. **P2 NOT recommended** — a second budget multiplies underpowered nulls and the population confound recurs; if the
   line continues at all: MORE SEEDS + population de-confounding at the existing 200 h budget, not a new budget/axis.

**One-line honest headline:** *"At a fixed 200 h pretraining budget, from-scratch CBraMod learns subject-identifiable
structure (L1 ≫ random-init) but not MI-transferable structure (transfer at the floor), and the coverage-vs-depth
allocation axis produces no detectable change in any endpoint."*

---

## 7. Downstream outputs already written (`results/s2p_p1_downstream/`)

`p1_frontier_summary.json` (**the verdict** — verified_conclusion, verification_corrections, p2_recommended=False,
required_caveats), `p1_task_and_frontier_raw.csv` (16 cells), `p1_task_performance.csv`,
`p1_pairwise_subject_separability.csv`, `p1_l4_task_alignment.csv`, `p1_l5_replay.csv`, `p1_l6_target_consequence.csv`,
`p1_random_init_control.csv`, `p1_released_checkpoint_reference.csv`, `p1_population_frontier_diagnostics.csv`,
`p1_downstream_run_manifest.csv`, `p1_downstream_norm_manifest.csv`, `p1_channel_mapping_manifest.json`,
`p1_target_label_firewall.json`, `p1_D0_probe_gate.json`, `p1_D0_findings.json`,
`d0p5_decodability_sanity/{raw_csp_verdict.json, label_split_channel_audit.json, d0p5_decodability_verdict.json, rel_patch/, rel_window/}`.
Firewall: `target_labels_used_for_selection=false` everywhere (PCA/head/subspace/rank all source-only; target labels final-scoring only).

---

## 8. What Codex should do next (small, no new compute without PM go)

1. **Write the neutral `docs/S2P_11_DOWNSTREAM_RESULTS.md`** (result-only commit): the §6 table + the six verified
   conclusions/corrections, QC (D0 8/8, D0.5 verified, determinism), firewall, sha/commit refs. **No interpretation
   beyond the verified conclusions.** Use the corrected wording (no peak/curvature; L5z systematic-but-trivial;
   pretrained>random = numeric/architecture not allocation).
2. **Return to the PM** with: the verified mostly-null result + P2-not-recommended + the proposed framing, and **WAIT**
   for the steer (close/park P1, or a MORE-SEEDS + population-de-confound follow-up at 200 h — NOT a new budget/dataset/
   fine-tuning/CodeBrain). Present as prose options, not a picker.
3. Optionally reconcile the two downstream implementations (§3) with the PM (retire or keep the S2P_09 scaffold as an
   independent replication). Do **not** silently delete the concurrent worker's files.

**Do NOT:** add another downstream dataset (stop-rule 8), full-fine-tune, add specialist baselines, put CodeBrain in
P1, change the SHU split/normalization, retrain P1, or launch P2 — all require an explicit PM decision.

---

## 9. Continuation commands

```bash
cd /home/infres/yinwang/CMI_AAAI_s2p
git fetch origin && git log --oneline -1            # expect 745337c or later
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh && conda activate eeg2025

# inspect the verdict
python -c "import json;print(json.dumps(json.load(open('results/s2p_p1_downstream/p1_frontier_summary.json')),indent=2))"

# re-run the frontier summary from the raw CSV (pure numpy/sklearn, CPU):
python s2p/scripts/frontier_summary.py

# re-run a single downstream cell if ever needed (GPU forward; SLURM):
sbatch --export=ALL,MODE=D1,CELLS="N512_s0 random",EMB=spatial,NORM=patch,OUT="$PWD/results/s2p_p1_downstream" s2p/scripts/downstream_audit.slurm
# monitor (scancel only, never ps/kill): squeue -u yinwang -o '%.10i %.9P %.8j %.2t %.10M %R'
```

---

## 10. Embedded memory (so Codex has it without this machine's `~/.claude`)

- **preregistered-experiments skill** governs everything (§1). Pre-reg → probe-gate → fleet → analyze → report-then-wait.
- **Disclose all cells before "PASS"; under-claim.** Adversarially verify findings (multi-skeptic) before finalizing.
- **Report gate-checklist, WAIT for explicit go before GPU.** PM decisions = **free-text prose options**, never a picker.
- **SLURM:** GPU (and heavy CPU) via `sbatch`; **`--time` under the 1-day cap** for multi-partition; **`scancel` only,
  never `ps`/`kill` own workers**; GPU partition fallback = scancel + resubmit on free (V100 usually idle).
- **Two-commit** (result-only then interpretation); `Co-Authored-By` trailer; **push after commits** (origin tracks HEAD).
- **S2P PERMANENT-FORBIDDEN claims:** "subject diversity removes/reduces leakage", "foundation encoders become
  subject-invariant", "TUEG solves cross-subject", growing-hours-read-as-diversity, pure-diversity-effect,
  population-adjusted-effect, SOTA/full-FT/leaderboard, "X converges to Y" without the metric, any deployable
  selector / oracle-as-feature. **REQUIRED** wording is the descriptive bundled-allocation-frontier framing (§6 headline).
- **The S2P identifiability arc (do NOT reopen):** single-H0 line (BL-1) → two-H0 crossed (BL-2 pop-confound/near-
  singular) → matched-exposure (BL-9 total-data confound) → **fixed-budget frontier (option C, accepted)**. `T=N·e`
  triangle ⇒ pure subject-diversity is unidentifiable at pretraining scale; P1 is a DESCRIPTIVE allocation study.
- **CodeBrain = P2-ready infra only.** **Paper 1 (Prior-Decoupled TTA) + Paper 2 (FSR) unaffected; PC2 GPU paused.**
```
