# Source-rich Phase 1A (Lee2019 development smoke) --- VERDICT + FROZEN params for confirmation

## Erased concept: D_nuis = z (injected nuisance label), NOT the EEG subject
In the source-rich semi-synthetic smoke, the erased concept is the KNOWN INJECTED nuisance factor `D_nuis = z`,
NOT the original EEG subject identity (D = subject). This tests whether a source-rich environment construction
can certify a beneficial intervention WHEN THE RELEVANT NUISANCE IS KNOWN. It does NOT solve nuisance discovery
in real EEG. (In code `z_src` is `D_nuis`; the environments E0-E5 are separate from `D_nuis`.)

## Development-selected params --- NOW FROZEN before Cho2017 confirmation
```
regime fractions      = aligned 0.4 / reversed 0.3 / noisy 0.3   (tuned on Lee development, disclosed)
alpha grid            = 0.5, 1.0, 2.0     (unchanged)
thresholds            = safety task-drop UCB <= 0.02 ; benefit LCB > +0.01   (unchanged, frozen)
environment definitions = E0 subject / E_oracle regime / E2 covariance_cluster / E4 margin_cluster /
                          E5 augmentation_shift / random   (unchanged)
cluster k             = 8   (unchanged)
selection rule        = source-only (unchanged)
```
**No parameter retuning on Cho2017 confirmation.** No changing fractions / alpha / thresholds / cluster k /
environment definitions after seeing Cho results.

## Phase 1A result (accepted wording)
```
Lee2019 development smoke:
  E_oracle accepts safe target-beneficial interventions.
  E2 covariance_cluster recovers oracle acceptance with 0 harmful accepts.
  E4 margin and E5 augmentation do not recover.
  Random partitions do not reproduce.
```
Adversarially verified (4 lenses, 0 confirmed material bugs): no target leak; covariance recovery is a
legitimate source-only signal; the 1 E_oracle "false accept" (RLACE alpha=2.0, target +0.023 mean, LCB +0.009)
is a BENIGN BOUNDARY = the eps_coverage slack at the strongest shift (safe + target-positive), not a harmful
accept; at that same alpha LEACE is a clean safe target-good accept, so Prop 2 holds even at the strongest shift.

## Mandatory caveats (keep all)
```
semi-synthetic ; construction-favorable ; covariance-detectable by construction ; D_nuis known ;
1 dataset / 1 backbone / 1 seed / 5 folds ; frac tuned on development set
```
Do NOT write: "real EEG source-rich environments work" / "source-only accept power is solved" /
"covariance clustering generally discovers target-beneficial shifts". Correct framing:
> first source-only positive under a CONSTRUCTED source-visible nuisance, NOT proof that real EEG shifts are
> covariance-discoverable.

## Confirmation decision rule (Cho2017)
* Case A (Cho confirms E_oracle + E2): -> Phase 1C TSMNet check.
* Case B (Cho confirms E_oracle but not E2): analyze why covariance discovery failed on Cho (not TSMNet next).
* Case C (Cho fails E_oracle): STOP the Fork-2 positive line ("demonstrated on development only; not confirmed")
  -> Fork 1 target-information frontier. Do NOT retune Cho.
Harmful false-accept by any discovered env -> STOP + audit (clustering overfit / source-label leakage).
