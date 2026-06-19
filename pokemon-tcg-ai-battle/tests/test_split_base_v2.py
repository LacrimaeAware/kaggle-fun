"""SPLIT_BASE_V2 / P3 -- golden acceptance tests for the shared state/action schema + teacher API.

Runs the seven preflight gates against tests/golden_state_action_fixtures/fixtures.json. No branch work
may start until all pass. Pure-stdlib runner (no pytest needed); also pytest-collectable.

    PYTHONIOENCODING=utf-8 python tests/test_split_base_v2.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import state_action_schema_v2 as SCH   # noqa: E402
import features as FT                   # noqa: E402
import main as M                        # noqa: E402

FIX = json.load(open(ROOT / "tests" / "golden_state_action_fixtures" / "fixtures.json", encoding="utf-8"))
FIXTURES = FIX["fixtures"]
KEYS = FIX["feature_keys"]
EXPECTED_CONSTANT = set(FIX.get("expected_constant_features", []))


def _ctx(fx):
    obs = fx["observation"]
    cur = obs["current"]
    return obs, cur, obs["select"]["option"], fx["your_index"]


def test_min_fixture_count():
    assert len(FIXTURES) >= 100, f"need >=100 fixtures, have {len(FIXTURES)}"
    assert set(FIX["covered_action_types"]) >= set(FIX["major_action_types"]), "missing major action types"


def test_trainer_equals_live_encoding():
    """The dataset-builder ('trainer') encoding and the live/schema encoding are byte-identical, and
    match what was frozen into the fixture."""
    for fx in FIXTURES:
        obs = fx["observation"]
        live = SCH.encode_vector(obs)
        trainer = FT.vectorize(FT.encode_state(obs))   # exactly build_action_dataset's expression
        assert live == trainer, f"live != trainer encoding @ {fx['source']}:{fx['step']}"
        assert live == fx["root_encoding"], f"live != frozen encoding @ {fx['source']}:{fx['step']}"


def test_all_play_actions_resolve_card_identity():
    n_play = 0
    for fx in FIXTURES:
        obs, cur, opts, me = _ctx(fx)
        players = cur["players"]; me_player = players[me]
        for j, o in enumerate(opts):
            if isinstance(o, dict) and o.get("type") == SCH.OptType.PLAY:
                n_play += 1
                cid = SCH.card_identity(o, me_player)
                assert cid is not None, f"PLAY option with no card identity @ {fx['source']}:{fx['step']} opt {j}"
                assert cid == fx["card_ids"][j], "recomputed card_id != frozen"
    assert n_play >= 10, f"too few PLAY options exercised ({n_play})"


def test_equivalent_options_collapse():
    for fx in FIXTURES:
        obs, cur, opts, me = _ctx(fx)
        eq = SCH.equivalence_classes(opts, cur, me)
        assert eq == fx["eq_classes"], f"eq classes drifted @ {fx['source']}:{fx['step']}"
        keys = [tuple(SCH.semantic_action_key(o, cur, me)) if isinstance(o, dict) else None for o in opts]
        # same key <=> same class, for every pair
        for i in range(len(opts)):
            for k in range(i + 1, len(opts)):
                same_key = keys[i] == keys[k]
                same_cls = eq[i] == eq[k]
                assert same_key == same_cls, f"collapse mismatch @ {fx['source']}:{fx['step']} opts {i},{k}"
        assert len(set(eq)) == len(set(keys)), "n classes != n distinct keys"


def test_distinct_plays_do_not_collapse():
    """Two PLAY options for DIFFERENT hand cards must be in different equivalence classes (the exact
    bug the old type-only key caused)."""
    exercised = 0
    for fx in FIXTURES:
        obs, cur, opts, me = _ctx(fx)
        eq = fx["eq_classes"]
        plays = [(j, fx["card_ids"][j]) for j, o in enumerate(opts)
                 if isinstance(o, dict) and o.get("type") == SCH.OptType.PLAY]
        for a in range(len(plays)):
            for b in range(a + 1, len(plays)):
                (ja, ca), (jb, cb) = plays[a], plays[b]
                if ca != cb:
                    exercised += 1
                    assert eq[ja] != eq[jb], (f"distinct PLAY cards {ca},{cb} collapsed @ "
                                              f"{fx['source']}:{fx['step']}")
    assert exercised >= 1, "no distinct-PLAY pair exercised; fixtures do not cover the case"


def test_option_permutation_transforms_labels():
    """Permuting the option order permutes the semantic keys identically and preserves the equivalence
    partition (the schema is order-equivariant)."""
    for fx in FIXTURES:
        obs, cur, opts, me = _ctx(fx)
        n = len(opts)
        perm = list(range(n - 1, -1, -1))                 # reverse
        ropts = [opts[p] for p in perm]
        keys = [tuple(SCH.semantic_action_key(o, cur, me)) if isinstance(o, dict) else None for o in opts]
        rkeys = [tuple(SCH.semantic_action_key(o, cur, me)) if isinstance(o, dict) else None for o in ropts]
        assert rkeys == [keys[p] for p in perm], f"keys not permutation-equivariant @ {fx['source']}:{fx['step']}"
        eq = fx["eq_classes"]
        req = SCH.equivalence_classes(ropts, cur, me)
        for i in range(n):
            for k in range(i + 1, n):
                assert (req[i] == req[k]) == (eq[perm[i]] == eq[perm[k]]), \
                    f"partition not preserved under permutation @ {fx['source']}:{fx['step']}"


def test_no_unexpected_dead_features():
    rows = [fx["root_encoding"] for fx in FIXTURES]
    constant = {KEYS[i] for i in range(len(KEYS)) if len({r[i] for r in rows}) == 1}
    unexpected = constant - EXPECTED_CONSTANT
    assert not unexpected, f"unexpectedly constant/dead features: {sorted(unexpected)}"


def test_teacher_queries_do_not_mutate_root():
    import teacher_api_v1 as T
    checked = 0
    for fx in FIXTURES:
        if checked >= 12:
            break
        obs = fx["observation"]
        if not SCH.is_single_pick_decision(obs):
            continue
        before = json.dumps(obs, sort_keys=True)
        r = T.query(obs, M.DECK, n_determ=4, time_budget=5.0, leaf_mode="hand", seed=1)
        after = json.dumps(obs, sort_keys=True)
        assert before == after, f"teacher mutated the root obs @ {fx['source']}:{fx['step']}"
        assert "applicable" in r
        checked += 1
    assert checked >= 5, f"teacher non-mutation exercised on too few fixtures ({checked})"


TESTS = [test_min_fixture_count, test_trainer_equals_live_encoding,
         test_all_play_actions_resolve_card_identity, test_equivalent_options_collapse,
         test_distinct_plays_do_not_collapse, test_option_permutation_transforms_labels,
         test_no_unexpected_dead_features, test_teacher_queries_do_not_mutate_root]


def main():
    print(f"SPLIT_BASE_V2 golden tests | {len(FIXTURES)} fixtures | schema {FIX['schema_version']}\n")
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{'ALL PASS' if not failed else str(failed) + ' FAILED'}  ({len(TESTS) - failed}/{len(TESTS)})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
