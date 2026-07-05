"""C10 OACI failure-mode diagnostics + counterfactual selector replay (artifact-only Part 1 + epoch-level
GPU inference-replay Part 2). Reads the frozen C8 BNCI2014-001 seeds-[0,1,2] artifacts; NEVER retrains,
changes the OACI objective, the artifact schema, or uses target for selection. Imports only within `oaci`."""
