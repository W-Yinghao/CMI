"""C8 throttled wave controller — the pure throttle + the submit loop (mocked cluster). Standalone."""
from __future__ import annotations

import io

from oaci.confirmatory.c8_wave import c8_fold_runs, run_wave, slots_to_submit

_REPO = "/home/infres/yinwang/CMI_AAAI_oaci"
_OUT = "/projects/EEG-foundation-model/yinghao/oaci-loso-seeds012"


def test_slots_to_submit_bounds_and_nonnegative():
    assert slots_to_submit(my_total=22, my_c8=0, max_total=28, max_c8=8, n_remaining=27) == 6   # overall headroom
    assert slots_to_submit(my_total=10, my_c8=6, max_total=28, max_c8=8, n_remaining=27) == 2   # c8 cap binds
    assert slots_to_submit(my_total=10, my_c8=0, max_total=28, max_c8=8, n_remaining=3) == 3     # remaining binds
    assert slots_to_submit(my_total=30, my_c8=0, max_total=28, max_c8=8, n_remaining=27) == 0    # over cap -> 0
    assert slots_to_submit(my_total=10, my_c8=8, max_total=28, max_c8=8, n_remaining=27) == 0    # c8 full -> 0


def test_c8_fold_runs_27_self_chain_and_decisions():
    fr = c8_fold_runs(_OUT, _REPO, seeds=(0, 1, 2))
    assert len(fr) == 27 and len({f["fold_uid"] for f in fr}) == 27
    assert {f["out_root"] for f in fr} and len({f["out_root"] for f in fr}) == 27
    for f in fr:
        assert f["env"]["OACI_CHAIN_PHASE_B"] == "1" and f["env"]["OACI_COMPUTE_DECISIONS"] == "1"
        assert f["env"]["OACI_MODEL_SEED"] == str(f["seed"]) and f["env"]["OACI_TARGET_SUBJECT"] == str(f["target"])
        assert f"/seed-{f['seed']}/target-{f['target']:03d}" in f["out_root"]


def test_run_wave_throttles_and_submits_all():
    seen = []
    def submit(fold, dl):
        seen.append(fold["fold_uid"]); return str(900000 + len(seen))
    r = run_wave(_OUT, _REPO, "/dl", seeds=(0, 1, 2), user="u", count_total=lambda u: 25, count_c8=lambda u: 0,
                 submit=submit, sleep=lambda s: None, done_check=lambda root: False, out=io.StringIO(),
                 max_total=28, max_c8=8, poll=1)
    assert len(r["submitted"]) == 27 and len(seen) == 27         # all trickled in
    assert [u for u, _ in r["submitted"]] == seen                # in (seed, target) order
    assert seen[0] == "seed-0/target-001" and seen[-1] == "seed-2/target-009"


def test_run_wave_respects_c8_cap_per_iteration():
    # with 3 free (max_total-my_total) and max_c8 huge, at most 3 submitted per iteration
    calls = {"n": 0}
    subs_per_iter = []
    def count_total(u):
        return 25                                                # 3 overall slots each poll
    def submit(fold, dl):
        calls["n"] += 1; return str(calls["n"])
    # instrument: count how many submits happen between sleeps via a sleep hook
    def sleep(s):
        subs_per_iter.append(calls["n"])
    run_wave(_OUT, _REPO, "/dl", user="u", count_total=count_total, count_c8=lambda u: 0, submit=submit,
             sleep=sleep, done_check=lambda root: False, out=io.StringIO(), max_total=28, max_c8=8)
    steps = [subs_per_iter[0]] + [subs_per_iter[i] - subs_per_iter[i - 1] for i in range(1, len(subs_per_iter))]
    assert all(s <= 3 for s in steps) and calls["n"] == 27       # never more than the 3 free slots per wave


def test_run_wave_backs_off_on_qos_failure_then_recovers():
    state = {"fail_next": True}
    made = []
    def submit(fold, dl):
        if state["fail_next"]:
            state["fail_next"] = False; return ""                # simulate one QOS rejection
        made.append(fold["fold_uid"]); return "1"
    r = run_wave(_OUT, _REPO, "/dl", user="u", count_total=lambda u: 20, count_c8=lambda u: 0, submit=submit,
                 sleep=lambda s: None, done_check=lambda root: False, out=io.StringIO(), max_total=28, max_c8=8)
    assert len(made) == 27 and len(r["submitted"]) == 27         # the rejected fold is retried, none lost


def test_run_wave_skips_already_done_folds():
    done = {"seed-0/target-001", "seed-1/target-005"}
    fr = c8_fold_runs(_OUT, _REPO, seeds=(0, 1, 2))
    root_to_uid = {f["out_root"]: f["fold_uid"] for f in fr}
    made = []
    r = run_wave(_OUT, _REPO, "/dl", user="u", count_total=lambda u: 22, count_c8=lambda u: 0,
                 submit=lambda f, dl: (made.append(f["fold_uid"]) or "1"), sleep=lambda s: None,
                 done_check=lambda root: root_to_uid[root] in done, out=io.StringIO(), max_total=28, max_c8=8)
    assert set(r["skipped"]) == done and len(r["submitted"]) == 25
    assert not (set(made) & done)                                # done folds are never submitted


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} c8-wave tests")


if __name__ == "__main__":
    _run_all()
