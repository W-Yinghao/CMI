# CITA_01 — Target-unlabeled CMI adaptation protocol

```
Branch project/cita-target-unlabeled-cmi. NEW information regime opened after docs/CIGL_70 closed the source-only
CMI reliance-control route as a systematic negative. This is NOT source-only DG and NOT fully-online TTA.
```

## Setting (explicit)
> This is not source-only DG. This is target-unlabeled / offline transductive adaptation. Target X is allowed.
> Target y is forbidden. No target label is used for training, adaptation, model selection, pseudo-label
> selection, early stopping, or hyperparameter selection.

Per outer LOSO fold: source subjects are labeled (source training + source replay); the held-out target
subject's **X is available during an offline adaptation phase**, its **y is forbidden** until the final reported
metric. Adaptation runs a fixed number of steps (no target early stopping). All of target X is used
(`adaptation_uses_all_target_X=true`) and evaluation is on the same target X → **offline transductive
target-unlabeled adaptation** (recorded as such; not "source-only", not "fully online TTA").

**New question:** if the held-out subject's functional nuisance structure cannot be inferred from source
subjects (the source-only closure finding), can **unlabeled target X** make CMI control effective? A win here is
"target-unlabeled CMI adaptation works under a different information regime" — **never** "CIGL source-only works".

## Methods (minimal attribution set) — all from the SAME source-ERM M0
- **ERM-no-adapt** — M0 (source ERM) evaluated on target, no adaptation. The pre-adaptation baseline; reused
  within-fold (no separate ERM run).
- **TTA-Control** — adapt a copy of M0: `L = CE(source_replay) + τ·H(p_target) + μ·KL(mean p_target ||
  source_label_prior)`. Target-unlabeled entropy/confidence adaptation + anti-collapse label balance. **Not a CMI
  method** — the control that isolates "is target-unlabeled adaptation itself useful?"
- **CITA-CMI** — `L = L_TTA-Control + λ·L_cond_domain`. `L_cond_domain` = label-conditional source/target
  domain-confusion (posterior-KL proxy): a conditional domain posterior `q(D|z,y)` is fit to distinguish
  source(D=0) vs target(D=1) **given y**, and the encoder is penalized so `q(D|z,y) → domain prior` (i.e.
  `D ⊥ z | Y`). `Y`: source = one-hot true label; target = **detached soft** pseudo-label `softmax(logits_t)`
  (no confidence threshold, no target label). `λ_cita = 0.010` (first round, **no λ grid**).

**Primary attribution comparison: CITA-CMI vs TTA-Control** (does the CMI term add anything beyond
target-unlabeled adaptation?), not CITA vs ERM. Attribution order: 1) ERM→TTA-Control (is adaptation useful?),
2) **TTA-Control→CITA-CMI (the CMI-specific test)**, 3) CITA-CMI→ERM.

## Firewall (`cmi/adaptation/cita.py`, tests `tests/test_cita_firewall.py` — 13 CPU pass)
`adapt(model, Xs, ys, Xt, method, ...)` has **no target-label parameter** → target y cannot enter adaptation by
construction. Source replay CE uses source labels only; target entropy uses target X only; the domain-conditioning
target pseudo-label is **detached**; the conditional-domain term is present only in CITA-CMI. ERM path unchanged;
single-linear head survives adaptation → **exact classifier-level head-replay**; random-subspace R3 control
consumable. Each artifact records `meta.cita_firewall`: `target_X_allowed, target_y_used=false,
target_y_for_{training,adaptation,model_selection,hyperparameter_selection}=false, adaptation_uses_all_target_X,
adaptation_mode=offline_transductive, target_eval_same_X_as_adapt, early_stopping=false,
source_replay_uses_source_labels_only`.

## Backbones
**EEGNetMini** (controlled CNN anchor) + **EEGConformerMini** (leakage-rich, exact classifier-level replay, clean
R3). **NOT ConformerFull** (multi-seed not stronger; MLP-replay adds complexity the CMI question doesn't need).
Naming boundary held: the Minis are internal audit-compatible implementations, **not** official EEGNet / official
Conformer.

## seed0 gate (`scripts/run_cita_gate.py`)
{EEGNetMini, EEGConformerMini} × {ERM-no-adapt, TTA-Control, CITA-CMI} × {BNCI2014_001, BNCI2015_001}, full-LOSO
seed0. ERM-no-adapt reuses M0 within-fold. `λ_cita=0.010`, no grid. Per method: metrics JSON + feature_z
`.audit.npz` (classifier-level R3) + firewall metadata. seed0 = scientific SCREENING.

## Success criteria (per backbone, then pooled — don't let 2a/2015 mask each other)
- **Strong CMI pass:** CITA-CMI target_bacc > TTA-Control AND > ERM; R3 task_drop ↓; leakage/conditional-domain
  leakage does not explode.
- **Functional pass:** target retained; R3 task_drop ↓ vs TTA-Control; leakage controlled.
- **TTA-only pass:** TTA-Control improves target but CITA-CMI does not beat TTA-Control → adaptation useful, CMI
  term adds nothing (not CMI positive).
- **Fail:** TTA-Control not good and CITA not good; or CITA ≤ TTA-Control; or random_subspace/R3 invalid → **no
  λ/k/schedule sweep**.

## Expansion policy
Seeds 1/2 only if seed0 shows a CMI-specific signal (CITA-CMI beats TTA-Control on target, OR clearly reduces R3
vs TTA-Control without target cost, OR one backbone gives a clean interpretable CMI signal) — and then only the
**single best method/backbone** combination. No auto-expansion to both backbones; no auto λ sweep. If CITA-CMI
seed0 shows no signal beyond TTA-Control → stop CMI method development.
```
