# C58 - Split-Label Cache Schema

Required future fields:

- run_id
- sample_id
- target_id
- trajectory_id
- candidate_model_id
- split_role: construction or evaluation
- target_label
- predicted_label
- logits_or_probabilities
- disjointness_group_id

The cache is missing in C58 and therefore cannot support few-label or split-label sufficiency.
