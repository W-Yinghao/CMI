# C84C V4 Direct Authorization Binding

The PI directly stated `授权 C84C` in the current execution conversation. Under
authorization policy commit `3d9dd76`, this is sufficient without a token or
verbatim hash recital. The server-side record binds that statement to:

```text
C84 canary protocol V4 SHA-256:
  cc54b5e6f92e4b0d338bf297c92823b4d60a8628a55dcff547ef9d808ee43afb

C84C execution lock V3 commit:
  a5feff377a18283dbe050d2feaa54126e5f924a9

C84C execution lock V3 SHA-256:
  c198607fb9e46ea2353ffa57d6b71bfa966c36e8ece53fdc40292681bba8bd1a

C84R3 repair protocol SHA-256:
  cdbdb9a25dc29b6a37ac9eb65f130f44efa120042dfb7ddb140cf3db103ec196

20-channel montage SHA-256:
  988e8f89c3001a5144172a10f3a8b30eb50c28d485b900210b91ed1a0cf04f04

243-unit canary identity SHA-256:
  4ada05be758975e7c28429819d804b4064a1bdcfd99fe7a4752a3bdbded6d396
```

Scope is the replacement engineering-only C84C run: source panel A, training
seed 5, level 0, three datasets, nine phases, and 243 candidate units. It does
not authorize C84F, C84S, construction/evaluation labels, target scientific
metrics, selector scores, the same-label oracle, or a scientific taxonomy
decision. The authorization consumed by failed job `895366` is not reused.
