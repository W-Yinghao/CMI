from star_eeg.objectives.alternating_schedule import build_compute_match_contract
from star_eeg.red_team.compute_match import evaluate_compute_match


def test_b_c_d_are_optimizer_step_and_batch_matched():
    contract = build_compute_match_contract()
    summaries = contract["summaries"]
    assert {row["optimizer_steps"] for row in summaries.values()} == {3750}
    assert {row["total_batches"] for row in summaries.values()} == {3750}
    assert summaries["H200_SSL_CONT"]["replacement_ssl_steps"] == 750
    assert summaries["H200_STAR_TRUE"]["anchor_steps"] == 750
    assert summaries["H200_STAR_SHUFFLED"]["anchor_steps"] == 750
    assert evaluate_compute_match(contract)["status"] == "PASS"
