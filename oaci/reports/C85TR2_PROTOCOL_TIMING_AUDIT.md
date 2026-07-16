# C85TR2 Protocol Timing Audit

## Chronology

```text
C85TR1 repair protocol commit:
  46442b281d61d00a575fae17685648b749659263

C85TR1 implementation commit:
  f17e25d0d8dc117f7973f90743e07139eeb0c1e1

historical C85T V2 execution-lock commit:
  920c5540a6ae157b77f2acb36f227bfdc172110b

C85TR1 final HEAD entering C85TR2:
  dd75d52be4414cc893c5a2fddf0374e01e13137a

C85TR2 protocol authored:
  2026-07-16T22:37:25Z
```

The C85TR1 protocol, V2 lock, reports, tables, tests, and regression evidence
remain immutable. The V2 lock is superseded before authorization or registered
execution. This additive protocol is committed and pushed before any C85TR2
implementation byte, V3 lock, context certificate, transaction fixture, or
semantic-replay artifact is created.

## Pre-Repair State

```text
branch:
  oaci

HEAD == origin/oaci:
  dd75d52be4414cc893c5a2fddf0374e01e13137a

historical V2 lock SHA-256:
  0f6907f9b997634b48062e97e789d50ae189dcc3c01f03cb5cee8b105c379719

registered S0-S10 draws:
  0

canonical proof candidates:
  0

theorem-status transitions:
  0

C85T V2/V3 authorization records:
  absent / absent

authorization-consumption receipts:
  0
```

No prior C85T authorization carries into V3. The direct statement embedded in
the earlier pre-lock C85TR1 request was already classified as unbound and is
not reusable.

## Information Used To Design The Repair

The repair uses only static inspection of the unexecuted V2 implementation and
its frozen readiness evidence:

```text
module-level sentinel and issuance registry;
generic pre-parsed-record consumption API;
post-rename lifecycle callback;
post-rename lifecycle replay and completion write;
terminal-ledger failure masking path;
hard-coded manifest counts;
missing exact-key/action/digest/proof/context semantic links.
```

No registered random stream, synthetic result, proof conclusion, theorem
transition, empirical project array, label, or scientific outcome informed the
repair.

## Prospective Boundary

This protocol is prospective to:

```text
the committed V3 authorization schema;
the receipt-validated execution context;
all future registered exact/MC/proof-candidate dispatches;
the atomic staging transaction and one final rename;
post-rename recovery classification;
primary/secondary exception precedence;
all V3 result semantic replay;
the V3 execution lock;
every future registered S0-S10 draw.
```

C85TR2 may create only shadow fixtures and temporary committed authorization
repositories. It cannot open a registered S0-S10 stream, create a canonical
proof candidate, transition T1-T7, access real project data, run active
acquisition, create C85V/C85E authorization, execute new data/model zoos, or
modify manuscript prose.

## Non-Circular Transaction Identity

The scientific artifact manifest excludes itself, lifecycle, and completion
receipt. Those three objects are bound by an ordered hash chain:

```text
manifest SHA
  -> MANIFEST_COMPLETED lifecycle event
  -> lifecycle-prefix SHA in completion receipt
  -> completion-receipt SHA in ATOMIC_PUBLISH_COMMIT_READY
```

The final bundle validator replays both the artifact manifest and this chain.
This avoids a self-referential hash while requiring every final-bundle object
and terminal event before the one rename.

## Authorization Boundary

At protocol lock time:

```text
C85T V3 execution lock:
  absent

C85T authorization:
  false

C85V authorization:
  false

C85E authorization:
  false

active acquisition authorization:
  false
```

Only a new standalone `授权 C85T` issued after the unique V3 lock exists may be
bound to `c85t_direct_pi_authorization_record_v3`.
