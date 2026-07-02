"""Guard (Stage-1B8): subject eligibility is decided BEFORE the split — every subject in the index must have a resolvable
control/case label (checked via a BOOLEAN-only resolver, no label value leaked); an ineligible subject aborts the build before any
split/train/dump. Synthetic only."""
from __future__ import annotations
from acar.v5.substrate import stage1b_build as B
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import subject_eligibility as SE
from acar.v5.tests._util import (expect_raises, ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeDevReader, FakeTrainer,
                                 FakeDumper, stage1b_fake_subjects, stage1b_subject_index)

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL


def test_all_eligible_passes_and_resolver_is_boolean():
    subs = stage1b_fake_subjects(n_per_cohort=3)
    idx = stage1b_subject_index(subs, "PD")
    reader = FakeDevReader(subs)
    cps = {c: f"/p/{c}" for c in {idx.cohort_of(k) for k in idx.subject_keys}}
    assert SE.assert_all_eligible(idx, reader, cps) is True
    # the resolver returns a BOOLEAN, never a label value
    k = idx.subject_keys[0]
    assert reader.subject_label_resolvable("PD", idx.cohort_of(k), idx.raw_of(k), "/p") in (True, False)
    ok("all subjects resolvable → eligible; subject_label_resolvable returns a boolean only (no label value)")


def test_ineligible_subject_rejected():
    subs = stage1b_fake_subjects(n_per_cohort=3)
    idx = stage1b_subject_index(subs, "PD")
    cps = {c: f"/p/{c}" for c in {idx.cohort_of(k) for k in idx.subject_keys}}

    class _OneUnresolvable(FakeDevReader):
        def subject_label_resolvable(self, disease, cohort, subject, path):
            return not (cohort == "ds002778" and subject == "sub-000")   # one subject's label won't resolve
    expect_raises(SE.SubjectEligibilityError, lambda: SE.assert_all_eligible(idx, _OneUnresolvable(subs), cps))
    ok("an unresolvable-label subject → SubjectEligibilityError (fail-closed)")


def test_build_aborts_before_split_on_ineligible():
    subs = stage1b_fake_subjects()

    class _OneUnresolvable(FakeDevReader):
        def subject_label_resolvable(self, disease, cohort, subject, path):
            return not (cohort == "ds002778" and subject == "sub-000")
    trainer, dumper = FakeTrainer(), FakeDumper()
    expect_raises(SE.SubjectEligibilityError,
                  lambda: B.run_stage1b_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                                              stage1b_lock(protocol_tag_target_sha=FULL), execute=True,
                                              dev_reader=_OneUnresolvable(subs), trainer=trainer, dumper=dumper))
    assert trainer.received == {} and dumper.reads == {}       # aborted before ANY fold was built
    ok("an ineligible subject aborts the build BEFORE any split/train/dump (trainer + dumper never ran)")


def main():
    print("ACAR v5 Stage-1B8 guard: subject eligibility before split")
    test_all_eligible_passes_and_resolver_is_boolean()
    test_ineligible_subject_rejected()
    test_build_aborts_before_split_on_ineligible()
    print("ALL V5 STAGE1B-SUBJECT-ELIGIBILITY GUARDS PASS")


if __name__ == "__main__":
    main()
