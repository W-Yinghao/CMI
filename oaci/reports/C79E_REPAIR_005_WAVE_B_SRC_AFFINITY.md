# C79E Repair 005 - Prospective Wave-B SRC Replay Affinity

Repair 004 showed that independent source preprocessing processes can fail the
bitwise `tensor_hash` replay gate when the frozen ERM anchor and SRC phase run
on different Slurm nodes. The gate is intentionally exact and remains exact.

Before Wave B, lock this outcome-blind engineering schedule:

1. Run the exact OACI/ERM phase for targets `[5, 2, 7, 1]`.
2. Read only each completed predecessor's Slurm node and engineering exit
   state.
3. Submit the exact locked SRC phase on that same node.
4. Start single-process instrumentation only after that SRC manifest freezes.
5. Run the Wave-B gate only after all four instrumentation manifests pass.

Node provenance is scheduler metadata, not a target outcome. No accuracy,
calibration, label, association, transport, actionability, or checkpoint rank
is read. No implementation file, scientific registry entry, tensor tolerance,
model, null, threshold, candidate universe, or execution lock changes.

