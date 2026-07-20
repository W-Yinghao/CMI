# C79E Wave-A Engineering Gate

Wave A completed under the locked target order `[8, 9, 3, 6]`.

```text
targets:                           4 / 4
retained units:                648 / 648
ERM anchors:                       8
OACI checkpoints:               320
SRC checkpoints:                320
strict-source rows:        2,985,984
target-unlabeled rows:       373,248
failed instrumented units:         0
target training-label reads:       0
scientific-outcome reads:          0
```

External gate:

```text
path: /projects/EEG-foundation-model/yinghao/oaci-c79-seed4/
      protocol_e350b7f0c4ee3dfc/implementation_dd4043ad7dd67552/
      gates/WAVE_A_ENGINEERING_VALID.json
file SHA-256:     afe4b7b42e5f8bf12ec4e995b8a223a2722de4d591c7165704c3a714d9c21286
manifest SHA-256: 811197ea7f476bdb32141eeb0bc28f22a908fbd96c3c1c80faa0bb3e6c2ccede
```

Two initial SRC attempts failed the exact source `tensor_hash` replay gate
before optimizer execution. Repair 004 retained those attempts and reran the
same locked SRC phases on the nodes that generated their frozen ERM anchors.
Both replacements passed the unchanged exact gate. No target outcome informed
the repair or Wave-A continuation decision.

Gate: `C79E_WAVE_A_ENGINEERING_GATE_PASSED`.

