"""C8 throttled wave controller (manages both phases, no self-chain). Pure throttle + classify + the
sim-driven submit loop (mocked cluster/disk). Standalone + pytest."""
from __future__ import annotations

import io

from oaci.confirmatory.c8_wave import (_submit_a, _submit_b, c8_fold_runs, classify, run_wave, slots_to_submit)

_REPO = "/home/infres/yinwang/CMI_AAAI_oaci"
_OUT = "/projects/EEG-foundation-model/yinghao/oaci-loso-seeds012"


def test_slots_to_submit_bounds_and_nonnegative():
    assert slots_to_submit(22, 0, 28, 8, 27) == 6
    assert slots_to_submit(10, 6, 28, 8, 27) == 2
    assert slots_to_submit(10, 0, 28, 8, 3) == 3
    assert slots_to_submit(30, 0, 28, 8, 27) == 0
    assert slots_to_submit(10, 8, 28, 8, 27) == 0


def test_c8_fold_runs_27_shared_env():
    fr = c8_fold_runs(_OUT, _REPO, seeds=(0, 1, 2))
    assert len(fr) == 27 and len({f["fold_uid"] for f in fr}) == 27 and len({f["out_root"] for f in fr}) == 27
    for f in fr:
        assert f["common_env"]["OACI_MODEL_SEED"] == str(f["seed"])
        assert f["common_env"]["OACI_TARGET_SUBJECT"] == str(f["target"])
        assert "OACI_CHAIN_PHASE_B" not in f["common_env"]      # set per-phase at submit time
        assert f"/seed-{f['seed']}/target-{f['target']:03d}" in f["out_root"]


def test_submit_helpers_set_phase_flags():
    cap = {}
    def capsub(script, env, dl, extra=()):
        cap[script] = env; return "1"
    f = c8_fold_runs(_OUT, _REPO, seeds=(0,))[0]
    _submit_a(f, "/dl", submit=capsub)
    _submit_b(f, "/dl", submit=capsub)
    from oaci.confirmatory.c8_wave import _PHASE_A, _PHASE_B
    assert cap[_PHASE_A]["OACI_CHAIN_PHASE_B"] == "0" and "OACI_COMPUTE_DECISIONS" not in cap[_PHASE_A]
    assert cap[_PHASE_B]["OACI_COMPUTE_DECISIONS"] == "1"       # Phase B computes native K1/K2


def test_classify_four_states():
    f = {"fold_uid": "u", "out_root": "/r"}
    assert classify(f, set(), {}, done=lambda r: True, a_complete=lambda r: False) == "done"
    assert classify(f, set(), {}, done=lambda r: False, a_complete=lambda r: False) == "needs_a"
    assert classify(f, {"5"}, {"u": {"a_jid": "5"}}, done=lambda r: False, a_complete=lambda r: False) == "in_progress"
    assert classify(f, set(), {}, done=lambda r: False, a_complete=lambda r: True) == "needs_b"
    assert classify(f, {"9"}, {"u": {"b_jid": "9"}}, done=lambda r: False, a_complete=lambda r: True) == "in_progress"


class _Sim:
    """A/B lifecycle: submit -> (on sleep) Phase-A completes (staging) -> submit B -> (on sleep) commits."""
    def __init__(self, max_c8=99):
        self.acomplete, self.done, self.live = set(), set(), set()
        self.ajid, self.bjid, self.nj, self.max_c8 = {}, {}, 1000, max_c8
        self.peak = 0
    def _jid(self):
        self.nj += 1; return str(self.nj)
    def sub_a(self, f, dl):
        if len(self.live) >= self.max_c8:                       # simulate QOS cap on concurrent C8
            return ""
        j = self._jid(); self.ajid[f["fold_uid"]] = j; self.live.add(j); self.peak = max(self.peak, len(self.live)); return j
    def sub_b(self, f, dl):
        if len(self.live) >= self.max_c8:
            return ""
        j = self._jid(); self.bjid[f["fold_uid"]] = j; self.live.add(j); self.peak = max(self.peak, len(self.live)); return j
    def total(self, u):
        return len(self.live)
    def c8(self, u):
        return len(self.live)
    def livejids(self, u):
        return set(self.live)
    def tick(self, _):
        # complete every running Phase-A (staging appears) and every running Phase-B (commits)
        for uid, j in list(self.ajid.items()):
            if j in self.live and uid not in [u for u in self.acomplete]:
                pass
        for uid, j in list(self.ajid.items()):
            if j in self.live:
                self.acomplete.add(uid); self.live.discard(j)
        for uid, j in list(self.bjid.items()):
            if j in self.live:
                self.done.add(uid); self.live.discard(j)


def _root_uid(fr):
    return {f["out_root"]: f["fold_uid"] for f in fr}


def test_run_wave_drives_a_then_b_to_completion():
    fr = c8_fold_runs(_OUT, _REPO, seeds=(0, 1, 2))
    ru = _root_uid(fr)
    sim = _Sim(max_c8=99)
    r = run_wave(_OUT, _REPO, "/dl", seeds=(0, 1, 2), user="u", max_total=99, max_c8=99, poll=1, max_cycles=200,
                 live_jids=sim.livejids, count_total=sim.total, count_c8=sim.c8, submit_a=sim.sub_a,
                 submit_b=sim.sub_b, done=lambda root: ru[root] in sim.done,
                 a_complete=lambda root: ru[root] in sim.acomplete, load_state=lambda lr: {},
                 save_state=lambda lr, s: None, sleep=sim.tick, out=io.StringIO())
    assert r["n_done"] == 27 and sim.done == {f["fold_uid"] for f in fr}    # every fold-run: A then B then commit
    assert set(sim.ajid) == set(sim.bjid) == sim.done                      # each got exactly an A and a B


def test_run_wave_respects_c8_cap_never_exceeds():
    fr = c8_fold_runs(_OUT, _REPO, seeds=(0, 1, 2))
    ru = _root_uid(fr)
    sim = _Sim(max_c8=99)
    run_wave(_OUT, _REPO, "/dl", seeds=(0, 1, 2), user="u", max_total=99, max_c8=3, poll=1, max_cycles=400,
             live_jids=sim.livejids, count_total=sim.total, count_c8=sim.c8, submit_a=sim.sub_a,
             submit_b=sim.sub_b, done=lambda root: ru[root] in sim.done,
             a_complete=lambda root: ru[root] in sim.acomplete, load_state=lambda lr: {},
             save_state=lambda lr, s: None, sleep=sim.tick, out=io.StringIO())
    assert sim.peak <= 3 and sim.done == {f["fold_uid"] for f in fr}        # cap-3 honoured, still completes


def test_run_wave_retries_on_qos_rejection():
    fr = c8_fold_runs(_OUT, _REPO, seeds=(0,))                             # 9 fold-runs
    ru = _root_uid(fr)
    sim = _Sim(max_c8=2)                                                    # tight cap -> forces rejections + retries
    r = run_wave(_OUT, _REPO, "/dl", seeds=(0,), user="u", max_total=99, max_c8=2, poll=1, max_cycles=400,
                 live_jids=sim.livejids, count_total=sim.total, count_c8=sim.c8, submit_a=sim.sub_a,
                 submit_b=sim.sub_b, done=lambda root: ru[root] in sim.done,
                 a_complete=lambda root: ru[root] in sim.acomplete, load_state=lambda lr: {},
                 save_state=lambda lr, s: None, sleep=sim.tick, out=io.StringIO())
    assert r["n_done"] == 9                                                 # nothing lost despite rejections


def test_run_wave_resumes_committed_and_a_complete_from_disk():
    fr = c8_fold_runs(_OUT, _REPO, seeds=(0,))
    ru = _root_uid(fr)
    pre_done = {"seed-0/target-001"}
    pre_a = {"seed-0/target-002"}                                          # A finished, needs B
    sim = _Sim(max_c8=99)
    sim.done |= pre_done; sim.acomplete |= pre_a
    r = run_wave(_OUT, _REPO, "/dl", seeds=(0,), user="u", max_total=99, max_c8=99, poll=1, max_cycles=200,
                 live_jids=sim.livejids, count_total=sim.total, count_c8=sim.c8, submit_a=sim.sub_a,
                 submit_b=sim.sub_b, done=lambda root: ru[root] in sim.done,
                 a_complete=lambda root: ru[root] in sim.acomplete, load_state=lambda lr: {},
                 save_state=lambda lr, s: None, sleep=sim.tick, out=io.StringIO())
    assert r["n_done"] == 9
    assert "seed-0/target-001" not in sim.ajid                             # already committed -> never resubmitted
    assert "seed-0/target-002" in sim.bjid and "seed-0/target-002" not in sim.ajid   # only B submitted for it


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} c8-wave tests")


if __name__ == "__main__":
    _run_all()
