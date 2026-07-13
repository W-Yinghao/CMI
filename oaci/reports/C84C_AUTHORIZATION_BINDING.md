# C84C Direct Authorization Binding

The PI directly stated `授权 C84C` in the current execution conversation.
Under authorization policy commit `3d9dd76`, this is sufficient without a token or
verbatim hash recital. The server-side record binds that statement to:

```text
C84 canary protocol V3 SHA-256:
  34cf9e9daca2578ed22c64345e014c0b9fa08b31c4c04939ba13c112c5f57dac

C84C execution lock V2 commit:
  270fbb0d9f47f9bf6a2888ee58fd7ca6eadff0ea

C84C execution lock V2 SHA-256:
  2e38dcd63c02a887b1dcf7eaa26749709dbfb5187373de7808efae21afb0285b

C84R2 repair protocol SHA-256:
  ff7c01f1760aaa19f2019672ad5426c8c28eba6c7071cd3b78b39bfe69dc8874

20-channel montage SHA-256:
  988e8f89c3001a5144172a10f3a8b30eb50c28d485b900210b91ed1a0cf04f04

243-unit canary identity SHA-256:
  4ada05be758975e7c28429819d804b4064a1bdcfd99fe7a4752a3bdbded6d396
```

Scope is the engineering-only C84C run: source panel A, training seed 5, level 0,
three datasets, nine phases and 243 candidate units. It does not authorize C84F,
C84S, construction/evaluation labels, target scientific metrics, selector scores,
the same-label oracle or any scientific taxonomy decision.
