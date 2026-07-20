"""C8b: the multi-seed (seeds [0,1,2]) staged submitter / dry-run job graph. Standalone + pytest."""
from __future__ import annotations

import io

from oaci.confirmatory.c8_submit import build_c8_job_plan, print_c8_plan, validate_c8_launch

_REPO = "/home/infres/yinwang/CMI_AAAI_oaci"
_OUT = "/projects/EEG-foundation-model/yinghao/oaci-loso-seeds012"


def _plan():
    return build_c8_job_plan(_OUT, _REPO)


def test_c8_plan_is_27_phase_a_27_phase_b_1_aggregation():
    jobs = _plan()
    assert sum(1 for j in jobs if j["kind"] == "phase_a") == 27
    assert sum(1 for j in jobs if j["kind"] == "phase_b") == 27
    assert sum(1 for j in jobs if j["kind"] == "aggregation") == 1
    assert len(jobs) == 55


def test_c8_fold_runs_are_9_targets_x_3_seeds_distinct_roots():
    a = [j for j in _plan() if j["kind"] == "phase_a"]
    assert {j["seed"] for j in a} == {0, 1, 2}
    assert {j["target"] for j in a} == set(range(1, 10))
    roots = [j["out_root"] for j in a]
    assert len(set(roots)) == 27                                       # every (seed, target) has a distinct root
    assert all(f"/seed-{j['seed']}/target-{j['target']:03d}" in j["out_root"] for j in a)


def test_c8_phase_a_parallel_and_phase_b_requests_decisions():
    for j in _plan():
        if j["kind"] == "phase_a":
            assert j["depends_on"] is None and j["env"]["OACI_CHAIN_PHASE_B"] == "0"
        elif j["kind"] == "phase_b":
            assert j["env"]["OACI_COMPUTE_DECISIONS"] == "1" and j["k1_permutations"] == 2000


def test_c8_phase_b_depends_on_its_phase_a_with_rolling_cap3():
    b = [j for j in _plan() if j["kind"] == "phase_b"]
    uids = [j["fold_uid"] for j in b]
    for i, j in enumerate(b):
        assert j["depends_on_phase_a"] == j["fold_uid"]                # afterok on its OWN phase A
        assert j["depends_on_prior_phase_b"] == (uids[i - 3] if i >= 3 else None)   # rolling cap-3
        assert j["depends_on"].startswith(f"{j['fold_uid']}:phase_a")
    assert sum(1 for j in b if j["depends_on_prior_phase_b"] is None) == 3


def test_c8_aggregation_depends_on_all_27_phase_b():
    jobs = _plan()
    agg = next(j for j in jobs if j["kind"] == "aggregation")
    b_uids = [j["fold_uid"] for j in jobs if j["kind"] == "phase_b"]
    assert sorted(agg["depends_on_all_phase_b"]) == sorted(b_uids) and len(b_uids) == 27
    assert "C8_BNCI001_LOSO_SEEDS012_K1K2" in agg["report"][0]


def test_c8_no_duplicate_target_seed_pairs():
    a = [j for j in _plan() if j["kind"] == "phase_a"]
    pairs = [(j["seed"], j["target"]) for j in a]
    assert len(set(pairs)) == 27 == len(pairs)


def test_c8_seeds_configurable():
    a = [j for j in build_c8_job_plan(_OUT, _REPO, seeds=(0, 1)) if j["kind"] == "phase_a"]
    assert {j["seed"] for j in a} == {0, 1} and len(a) == 18


def test_c8_dry_run_prints_all_required_fields():
    buf = io.StringIO()
    print_c8_plan(build_c8_job_plan(_OUT, _REPO, k1_permutations=2000,
                                    bootstrap_budgets={"selection_bootstrap": 200, "audit_bootstrap": 2000,
                                                       "paired_bootstrap": 2000}),
                  loso_root=_OUT, file=buf)
    s = buf.getvalue()
    for tok in ("target=", "seed=", "audit=", "train=", "deleted=", "A store=", "B artifact=", "depends=",
                "K1 permutations=2000", "bootstrap_budgets=", "aggregation", "minimum-seed"):
        assert tok in s, tok
    for t in range(1, 10):
        for sd in range(3):
            assert f"seed-{sd}/target-{t:03d}" in s


def test_c8_validate_rejects_in_repo_root():
    import os
    try:
        validate_c8_launch(os.path.join(_REPO, "oaci", "inside"), _REPO, "/projects/EEG-foundation-model/datalake/raw")
    except ValueError:
        return
    raise AssertionError("an in-repo LOSO root must be rejected")


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} c8-submit tests")


if __name__ == "__main__":
    _run_all()
