# C86D — Development Protocol (short)
## Active-policy development on the accepted C86L field; NOT execution

**Status**

```text
C86L_DEVELOPMENT_FIELD_ACCEPTED_AS_AUTHORITATIVE_C86D_INPUT
C86D_PROTOCOL_AND_CLIENT_SERVER_IMPLEMENTATION_GO
C86D_REAL_ACTIVE_POLICY_EXECUTION_NOT_AUTHORIZED
C86H_NOT_AUTHORIZED / C87_NOT_AUTHORIZED / MANUSCRIPT_NOT_AUTHORIZED
```

C86D is **development**: implement and debug the registered active policies on the
frozen C86L field, identify estimator/query failures, and freeze the final C86H
method + hyperparameters. It makes **no confirmatory claim**, treats no
trial/query/replicate as scientific N, and calls no C84 development result
"transport". Real P0/A1/A2H execution needs a separate direct `授权 C86D`. The only
gated entrypoint (`c86d.execute`) refuses without it.

## 1. Physical process boundary (the C86L→C86D addition)

Three real roles (C86L proved directory separation; C86D adds process + access-path
separation):

```text
Active client   : reads ONLY acquisition_unlabeled_pool; does not know the oracle
                  or contribution paths; holds only a query-server handle.
Query server    : a separate PROCESS that exclusively owns the label oracle and
                  contribution store; each query releases one physical trial's label
                  + that trial's 8 context-specific contribution rows; nothing else.
Held evaluator  : opens C85U only AFTER selection freeze; receives no construction
                  oracle and no query-server capability.
```

Implemented via `multiprocessing`: the server runs in a child process owning the
sealed paths; the client holds only a pipe connection (`QueryClientHandle`) with no
oracle/contribution attributes.

## 2. Locked primary policies (registry-faithful)

```text
P0  : uniform without replacement
A1  : exact registered Active Testing / LURE (acquisition = mean expected candidate
      NLL; LURE weights correct the biased without-replacement sampling)
A2H : faithful Hara general-K, score = Σ_{k<k'} E_π|loss_k − loss_k'|
```

`A2M` (max-pair heuristic) is DEVELOPMENT-only and is **never** relabelled as Hara.
All methods share the same physical trial pool, total-query budgets, candidate zoo,
construction plugin, and held-evaluation utility.

## 3. Semantics-B active score

A query is one **target-level physical-label** decision. For a trial's 8 contexts:
context-specific acquisition contributions are computed separately, then combined
into a target-level trial acquisition score by a pre-fixed **equal-weight** context
aggregation (the 8 contexts are equally weighted in the final target regret; the
fixed 8-context sum and mean are rank-equivalent — reported as **mean**).

## 4. Estimation and selection objects + claim boundary

Query responses support linear NLL / correctness / class / calibration-bin moments.
Candidate selection uses the historical **composite plugin**:

```text
estimated bAcc, estimated NLL, estimated ECE
  → within-context oriented midranks
  → equal-weight composite utility
  → first-index argmax
```

Claim boundary (unchanged, enforced in code):

```text
LURE unbiasedness            : ONLY the locked linear moments
bAcc ratio / absolute ECE /  : nonlinear plugins, NO unbiasedness claim
  midrank / composite / action
```

## 5. Selection freeze precedes held evaluation

For every `target × method × budget × stochastic replicate`, freeze BEFORE any C85U
access: the physical query sequence, per-query sampling probability / importance
weight, query receipts, final per-context candidate estimates, the 8 selected
candidate indices, and the action/regime identity. Only then is the freeze handed to
the held evaluator, which opens the C85U held-evaluation utility field.

**C85U identity (PM fix — verify, do not hardcode).** Before held evaluation the
evaluator opens and hashes the real acceptance manifest and checks its field
identity:

```text
path   : oaci-c85u-candidate-utility-v2/.../final_acceptance_bundle/C85U_RESULT_ARTIFACT_MANIFEST.json
sha256 : dfcf84569beb1b34b786cbe72233a22fd3928a4475b7e345f23b40cdb6671620
field  : 944 contexts × 81 candidates = 76,464 candidate rows; evaluation-label rows 4,848
bind   : to the current C86D attempt's selection freeze
```

## 6. Development endpoints (accepted definitions + exact CVaR)

```text
target regret        : equal-weight mean over the target's 8 contexts, first
target_near_opt_prob : P(target 8-context mean regret ≤ ε)
tail                 : EXACT upper-tail CVaR over the cohort target-regret distribution
                       WITH fractional boundary mass — NOT the shadow ceil(frac·N).
                       (e.g. Lee 22 targets, frac 0.25 → 5.5 → 5 full + 0.5 of the 6th)
```

## 7. One-time method freeze (pre-registered, deterministic)

Fixed before any real execution: A1/A2H hyperparameter candidate sets,
sampling-probability floor, uniform mixing, importance weighting, stopping/failure
rules, passive replicate count + seed schedule, and the deterministic final
hyperparameter freeze rule. C86D may use C84/C85 held outcomes for development
selection, but only through this pre-written rule. **No** method may be added, and
**no** A1/A2H dropped for poor C86D performance, after results. The C86H primary
registry stays `{P0, A1, A2H}` unless a pre-defined engineering blocker makes one
non-executable.

## 8. Implementation discipline

Only these changes return to PM: information view, query cost, method identity, held
endpoint, target population, primary claim. All other implementation issues stay in
one C86D failure ledger, no new research milestones for dtype/path/format bugs.

## 9. Files

```text
oaci/active_testing/c86d/{__init__,core,server,policies,pipeline}.py
oaci/tests/test_c86d.py   (shadow + failure tests)
```

Real execution is gated on `授权 C86D`.
