# V2 ceiling-smoke --- post-smoke audit (evidence locked before full benchmark)

All numbers below are extracted by script from the committed artifacts (config + `v2_smoke_summary.json` at
commit `665fffb`), not from memory. This note is the evidence gate the PM required before V2 full.

## 1. Canonical YAML evidence (`tos_cmi/eeg/configs/v2_certificate_fixed.yaml`)
Recorded from script output (the GitHub *raw* view once rendered this as "3 lines" -- a CDN artifact; the file
is and was multi-line LF):
```
wc -l                    : 63
CR/LF check              : LF_ONLY_PASS   (no CR bytes; LF newlines)
yaml.safe_load required  : PARSE PASS     (20 top-level keys; all required keys present)
config sha256[:12]       : 169a6b2507f2
```
A regression gate for this lives in `test_v2_worlds.py::test_config_parses_and_has_required_keys` (asserts
goal=source_only_acceptance_ceiling, thresholds 0.02/0.01, world_*.acceptance_expected=False, oracle present).

## 2. Smoke provenance
```
commit id     : 665fffb  (results) ; driver/config 4d97d8a
SLURM job id  : 883280
tasks         : 1200      failures: 0
datasets      : Lee2019_MI, Cho2017
backbone      : EEGNet
seed          : 0
folds         : first 5 per dataset
worlds        : A, B, C
interventions : identity, random_k, leace_baseline, rlace, tos_vd, inlp, tp_leace, alpha_leace,
                fair_conditional_leace_disjoint_router, oracle_nuisance_eraser_DIAGNOSTIC_ONLY
alpha_grid    : 0.25, 0.5, 1.0, 2.0
```

## 3. Non-vacuity of the gate's ACCEPT branch (critical)
The ACCEPT branch is LIVE logic, not dead code:
```
gate_action(source_task_drop_ucb=0.00, source_benefit_lcb=+0.05)  =>  ACCEPT
gate_action(0.01, +0.02) => ACCEPT ;  gate_action(0.01, -0.03) => ABSTAIN ;  gate_action(0.05, +0.05) => REJECT
```
It never fired in the smoke ONLY because the source-only signal was never positive:
```
deployable cells with source-LOSO benefit LCB > +0.01 : 0 of 216
max deployable source-LOSO benefit LCB across the smoke : +0.0009  (noise floor)
```
=> "0 accepts" is a DATA property (the ceiling), not an implementation that cannot accept.

## 4. World A evidence (target-beneficial but source-uncertifiable)
```
safe + target-beneficial cells       : 21   (span BOTH datasets: Cho via fair_conditional/rlace/oracle;
                                             Lee via leace_baseline/fair_conditional/rlace/oracle)
principled source-only ACCEPTs       : 0
oracle max target dbAcc              : +0.049      random_k max target dbAcc : +0.000  (random does not reproduce)
max deployable source-LOSO benefit   : +0.0009
```
Headline cell (CORRECTED -- earlier smoke report misattributed this to Cho2017; it is Lee2019_MI):
```
Lee2019_MI, leace_baseline, alpha=1.0 :
  source task-drop UCB  +0.018   (SAFE, <= 0.02)
  source-LOSO benefit LCB  -0.035  (negative -> not certifiable source-only)
  gate action           ABSTAIN
  actual target dbAcc   +0.086 [+0.050, +0.122]   (genuinely target-beneficial)
```
Even the oracle nuisance eraser (perfect nuisance removal) has a NEGATIVE source-LOSO benefit while giving a
positive target gain -- the benefit is genuinely source-invisible.

## 5. Naive-controller table (all deployable cells; GOOD accept = actually target-beneficial)
```
controller                              accepts  false-accepts  true-accepts
domain-gain-only                          108        90            18
safety-only                               180       167            13
always-erase-if-any-domain-gain           118       100            18
OUR GATE (benefit+safety, source-only)      0         0             0
ORACLE target-informed selector [DIAG]     18         0            18   (uses target labels; diagnostic only)
```
Reading: naive source-only rules false-accept 90--167 useless/harmful erasures; our source-only gate accepts
nothing (conservative -- correct under the ceiling, and it misses all 18 beneficial cells, which is the ceiling
itself); only the target-informed oracle picks the beneficial cells -> crossing the ceiling needs target info.

## 6. C14 forbidden wording still active (claim_evidence_table.md)
Confirmed present. FORBIDDEN: "V2 shows the source-only gate can accept genuine target-beneficial erasure" /
"the source-only gate has acceptance power" / "World A is a successful accept case" / "erasure benefit is
certifiable from source-only data" / "source-only benefit is impossible in general" / oracle-as-method.
ALLOWED: "source-only acceptance ceiling; deployment-shift benefit not certifiable source-only; correct action
stays reject/abstain; crossing the ceiling requires target information."

## 7. Honest disclosures carried forward
* Our gate accepted 0, including 0 of the 18 genuinely-beneficial cells -- correct under the ceiling, but means
  the gate has no *acceptance* capability on source-only data (that is the ceiling, stated plainly).
* We did NOT construct a source-only world where the gate accepts (the 144-cell search argues it is not
  constructible for subject-erasure). The smoke demonstrates correct REFUSAL + the ceiling, not accept-vs-
  non-accept discrimination on source-only data.
* rows.csv has "nan" in the `router_acc` column for the 9 non-fair_conditional interventions (benign; router_acc
  applies only to fair_conditional), not a degeneracy.
* Smoke scope is narrow (Lee/Cho, EEGNet, seed0, first-5 folds); robustness across backbone/seed/fold/source-
  size is what the staged full benchmark (full-lite -> full) will establish.
