from star_eeg.objectives.task_anchor import synthetic_training_step_smoke


def test_one_ssl_and_one_anchor_step_have_finite_loss_and_gradients():
    first = synthetic_training_step_smoke()
    second = synthetic_training_step_smoke()
    assert first == second
    assert first["status"] == "PASS"
    assert first["ssl_steps"] == 1
    assert first["task_anchor_steps"] == 1
    assert first["finite_losses"] is True
    assert first["finite_gradients"] is True
    assert first["real_eeg_used"] is False
