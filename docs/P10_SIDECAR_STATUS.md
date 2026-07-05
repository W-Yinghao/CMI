# P10 sidecar — status & claim boundary (frozen)

P10 (EEGNetMiniCSPInit) is a **sidecar**, not the main track. Frozen at seed0 (`032636a`); **seeds 1/2 NOT
run** (PM decision — resources go to the CIGL R2/R3 main line).

## Result (seed0 only)
| metric | value |
|---|---|
| BNCI2014 2a-decodable {1,3,8,9}, CSPInit − EEGNetMini | **+0.066** (positive screen) |
| BNCI2014 2a-full | +0.028 |
| BNCI2015 2-class | **−0.015** (fails the pre-set secondary ≥ −0.01 → NOT a clean pass) |

## Claim boundary (use this wording)
**Allowed:**
- P10 shows a **positive seed0 screen** for CSP-init transfer on BNCI2014 2a-decodable.

**NOT allowed:**
- ~~P10 confirms CSP-init transfers to EEGNetMini.~~
- ~~P10 is a validated mechanism finding across datasets/seeds.~~

seed0 is scientific *screening*, not method-level judgment (which needs full-LOSO × seeds 0/1/2). The
correction supersedes any stronger phrasing in the `032636a` commit message ("IS a transferable ...") — read
that as "positive seed0 screen only." Revisit seeds 1/2 only after R2/R3, or if a spatial-init control is
needed to interpret a CMI result.
