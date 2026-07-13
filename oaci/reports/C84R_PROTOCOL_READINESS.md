# C84R Protocol Readiness

## Result

The C84P 21-channel blocker is resolved prospectively by the exact 20-channel
cross-dataset intersection. `FCz` is removed from all datasets; `Fz` substitution,
interpolation, zero filling and dataset-specific masks remain forbidden.

```text
repair protocol commit:  482a725abc6bf1f0e5d33be76ea17d37bcfaa6c3
V2 protocol commit:      a5d9fd0a0e76a7e0c6a49b87048d642eb8c0da6a
final adapter commit:    e91b71c5e0cd99d90c8ac9c44e2736a4cfc18f4f
C84C lock commit:        4eaad36cafefb2645f1d5c6e393ae5a51ff33af9
C84C lock SHA-256:       f9cabf8f362917d663e13154910085d5b105740b265789a2323dd7bc0193222b
montage SHA-256:         988e8f89c3001a5144172a10f3a8b30eb50c28d485b900210b91ed1a0cf04f04
```

All 214 subject assignments replay unchanged. All 1,944 prospective unit IDs now bind
the V2 interface and differ from the blocked plan. C84C is fixed at panel A, seed 5,
level 0, three datasets, 9 training phases and 243 units, with targets Lee 19, Cho 24
and Physionet 106. The adapter imports loaders only after direct authorization has been
bound and consumed; the target-unlabeled payload has no label field.

No direct C84C authorization record exists. No C84F/C84S lock exists. Protected access
counts remain zero. C84R therefore stops at readiness; it does not execute C84C.

```text
C84_COMMON_20_CHANNEL_MONTAGE_REPAIRED_CANARY_LOCKED_READY_FOR_PI_AUTHORIZATION
```
