# C78F Protocol Timing Audit

Status: prospective field-generation and analysis contracts locked.

```text
C78F protocol SHA-256: 85aba93fe2e232f0434162b3c6c97a30cac02047228676951c25cbab805d3d84
C78S protocol SHA-256: df85699090a65d1e1766d754bcebd9eb5648cc13e4441d8074a3f4884487c7f8
authorization mode: direct_explicit_user_authorization
direct user authorization received: true
magic token required: false
remaining-target EEG data access before lock: 0
remaining-target GPU job submission before lock: 0
remaining-target outcome access before lock: 0
seed-4 access before lock: 0
```

The PM explicitly replaced the former exact-token ceremony with direct user
authorization. Execution remains fail-closed through a committed lock binding
that authorization record to these exact protocol and implementation hashes.

C78F is generation/instrumentation only. C78S is locked prospectively, excludes
target 4 from primary inference, and has not started.
