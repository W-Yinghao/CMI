from star_eeg.objectives.alternating_schedule import (
    BASE,
    SSL_CONT,
    STAR_SHUFFLED,
    STAR_TRUE,
    build_schedule,
    schedule_payload,
)


def test_exact_four_ssl_one_anchor_cycle():
    schedule = build_schedule(STAR_TRUE, total_steps=10)
    assert [step.update_kind for step in schedule] == [
        "ssl", "ssl", "ssl", "ssl", "source_task_anchor",
        "ssl", "ssl", "ssl", "ssl", "source_task_anchor",
    ]
    assert build_schedule(STAR_SHUFFLED, total_steps=10) == schedule


def test_ssl_control_replaces_anchor_slots_and_base_has_no_steps():
    schedule = build_schedule(SSL_CONT, total_steps=10)
    assert all(step.update_kind == "ssl" for step in schedule)
    assert [step.ssl_stream for step in schedule if step.semantic_slot == "anchor_slot"] == [
        "replacement", "replacement"
    ]
    assert build_schedule(BASE) == []


def test_schedule_hash_is_deterministic():
    assert schedule_payload(STAR_TRUE, 25) == schedule_payload(STAR_TRUE, 25)
    assert schedule_payload(STAR_TRUE, 25)["schedule_hash"] == schedule_payload(STAR_SHUFFLED, 25)["schedule_hash"]
