# C85P Regression Verification

## Accepted Implementation

```text
protocol commit:
  2449be1c24e313922688b5e957ce6d19cb75d9d6

implementation commit:
  73844601d82037cfe9b8f31cb21bc53bd9b5f334

environment:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact

Python:
  3.13.7

GPU:
  0

scheduler evidence:
  squeue only

sacct used:
  false
```

## Preserved Initial Focused Failure

The first component-focused run produced:

```text
33 passed, 1 failed
```

The only failure was a test-only hard-coded full SHA for the already committed
protocol chronology boundary. The test contained an incorrect guessed SHA:

```text
incorrect:
  2449be1c157237582bc41628e8703b425d220bfb

actual:
  2449be1c24e313922688b5e957ce6d19cb75d9d6
```

The protocol, generator contract, formulas, statuses, and tables were
unchanged. Correcting the identity assertion yielded 34/34 at that point; the
final addition of the suite-inclusion test brings the two C85 files to 35
tests. This initial failure is not counted as accepted evidence.

## Leading-Numeric Repair

The shared parser previously defaulted to milestone 84. C85P changes its upper
bound to 85 and adds both C85 files to focused. Static tests prove both files
occur in focused, C65, and C23; full continues to select the complete test
directory. The historical C34S suffix case remains covered.

## Accepted Runs

| Suite | Passed | Failed | Skipped | Deselected | Pytest runtime | stdout SHA-256 | stderr bytes |
|---|---:|---:|---:|---:|---:|---|---:|
| focused | 291 | 0 | 0 | 0 | 14.19 s | `2ab3c6cdab645f218cb03e3e3e85e194dd8eeae9b58de4bc29e4caf2c043c725` | 0 |
| C65 cumulative | 902 | 0 | 1 | 3 | 107.92 s | `f2de2714822b008627488319886d7e2c99d6dfbeab003536e0f58a8f9fffc458` | 0 |
| C23 cumulative | 1,313 | 0 | 1 | 3 | 217.48 s | `8afd9a0ae037f83b99568cde4dd6e85a090be0b3ed8981837101d814e8338db1` | 0 |
| full OACI | 2,237 | 0 | 1 | 3 | 350.68 s | `ba88351ccd95e06270a70033d7c05991892d89f202a309504cc529caf97aa775` | 0 |

Every accepted stderr file has SHA-256:

```text
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

The accepted skip is the historical finalized C78F red-team test. The three
accepted deselections are the standing C79P unauthorized-adapter checks named
in the committed wrapper.

Relative to C84A, each cumulative suite adds exactly 35 C85 tests:

```text
C65:  867 -> 902
C23: 1278 -> 1313
full: 2202 -> 2237
```

## Runtime Boundary

No test read EEG, direct label roots, target logits, source arrays, model
checkpoints, or scientific result arrays. No selector, Q0, inference,
training, forward, GPU, active-acquisition, or manuscript path ran.

Post-run `squeue` showed zero active C84/C85 jobs. `sacct` was not used.

## Disposition

```text
focused:       PASS
C65 cumulative: PASS
C23 cumulative: PASS
full OACI:      PASS
accepted stderr: EMPTY
```
