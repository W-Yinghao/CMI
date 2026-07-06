"""Guard (V6-A0a): the permutation null permutes SUBJECT BLOCKS, not individual batches — whole-subject label-blocks are moved as
units; the label multiset is preserved; a subject can never receive a within-subject-scrambled label vector. The p-value uses
(1+#{null>=obs})/(1+n_valid). Torch-free; the p-value arithmetic path is exercised only when sklearn is present."""
from __future__ import annotations
import numpy as np
from acar.v5 import v6_a0_sign_predictability as SP
from acar.v5.tests._util import ok, has_sklearn


def _blocks(groups):
    subs, idx = [], {}
    for i, g in enumerate(groups):
        s = str(g)
        idx.setdefault(s, []).append(i)
        if s not in subs:
            subs.append(s)
    return subs, idx


def test_permutation_is_subject_block_unequal_sizes():
    # UNEQUAL record counts (the REAL regime) WITH size-sharing so permutation actually happens: sizes {2:[s0,s1], 4:[s2,s3]},
    # plus a UNIQUE size (s4 size 3) that must stay FIXED. Every source block is unique-tagged so fragmentation is detectable.
    groups = np.array(["s0", "s0", "s1", "s1", "s2", "s2", "s2", "s2", "s3", "s3", "s3", "s3", "s4", "s4", "s4"], dtype=object)
    y = np.array([10, 11, 20, 21, 40, 41, 42, 43, 50, 51, 52, 53, 90, 91, 92], dtype=int)
    subs, idx = _blocks(groups)
    src_block = {s: tuple(int(y[i]) for i in idx[s]) for s in subs}
    size_members = {2: {src_block["s0"], src_block["s1"]}, 4: {src_block["s2"], src_block["s3"]}, 3: {src_block["s4"]}}
    for seed in range(200):
        yp = SP.subject_block_permute(y, groups, np.random.RandomState(seed))
        assert sorted(yp.tolist()) == sorted(y.tolist()), "multiset must be preserved"
        for s in subs:
            got = tuple(int(yp[i]) for i in idx[s])
            size = len(idx[s])
            # each subject receives an INTACT source block of the SAME size — never a fragment / cross-subject mix
            assert got in size_members[size], f"seed {seed}: {s} (size {size}) got fragmented/cross-size block {got}"
        assert tuple(int(yp[i]) for i in idx["s4"]) == src_block["s4"], "singleton-size subject must stay FIXED"

    # DISCRIMINATING with PURE blocks of UNEQUAL sizes: a batch/record-level scramble could mix within a subject; the stratified
    # subject-block permutation never can. Sizes {2:[a,b], 3:[c,d]} all pure -> every subject stays pure across seeds.
    g2 = np.array(["a", "a", "b", "b", "c", "c", "c", "d", "d", "d"], dtype=object)
    y2 = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1], dtype=int)
    _, idx2 = _blocks(g2)
    for seed in range(200):
        yp = SP.subject_block_permute(y2, g2, np.random.RandomState(seed))
        for s in ("a", "b", "c", "d"):
            vals = {int(yp[i]) for i in idx2[s]}
            assert len(vals) == 1, f"seed {seed}: pure subject {s} received a MIXED block {vals} — record-level leak"
    ok("subject-block null permutes INTACT blocks within equal-size strata (unequal sizes handled); singletons fixed; multiset kept")


def test_pvalue_fields_and_failclosed_integration():
    if not has_sklearn():
        ok("sklearn absent — end-to-end p-value path skipped (subject-block structure covered above)")
        return
    r = np.random.RandomState(0)
    recs = []
    for si in range(8):                                       # 8 subjects -> underpowered subject-block null (< 20)
        sk = f"PD/ds002778/sub-{si:02d}"
        for bi in range(4):
            for a in SP.PRIMARY_ACTIONS:
                feats = {"per_action": {aa: {k: float(r.randn()) for k in
                          ("d_entropy", "d_margin", "flip_rate", "JS", "Bures", "post_sep", "n_eff")} for aa in SP.PRIMARY_ACTIONS},
                         "source_confidence": float(r.rand()), "batch_entropy": float(r.rand()), "batch_size": 32}
                recs.append({"subject_key": sk, "batch_id": bi, "action_id": a, "provenance": "native",
                             "features": feats, "beneficial": int(r.rand() < 0.5)})
    prov = ["native"]
    obs = SP.primary_sign_auroc(recs, prov, seed=0)           # subject-balanced AUROC runs end-to-end
    assert (obs != obs) or (0.0 <= obs <= 1.0)
    out = SP.permutation_pvalue(recs, prov, obs, seed=0, n_perm=12)
    assert set(out) >= {"perm_p_subject_block", "raw_p_value", "reason", "n_perm_valid", "n_permutable_subjects", "n_subjects"}
    assert out["perm_p_subject_block"] == 1.0 and out["reason"] == "permutation_null_underpowered"   # 8 subjects -> fail-closed
    ok("primary_sign_auroc + permutation_pvalue run end-to-end; 8-subject null is underpowered -> fail-closed perm_p=1.0")


def main():
    print("ACAR v5 V6-A0a guard: permutation null is subject-blocked")
    test_permutation_is_subject_block_unequal_sizes()
    test_pvalue_fields_and_failclosed_integration()
    print("ALL V6A0-PERMUTATION-SUBJECT-BLOCK GUARDS PASS")


if __name__ == "__main__":
    main()
