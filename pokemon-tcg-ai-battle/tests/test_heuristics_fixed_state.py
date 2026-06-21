"""Fixed-state heuristic tests: score a heuristic / leaf eval / forced-move rule on a
FROZEN game state, with no full game and (mostly) no engine. Milliseconds, deterministic.

This is the template the refactor plan calls for. To test a new heuristic or signal:
  1. add a synthetic state or a golden fixture that exercises it, and
  2. add one assertion here calling your function directly (eval.evaluate, main._forced_move,
     main._attack_value, or your new heuristic) on that frozen state.
You do NOT need to run cabt games to know whether a heuristic fires correctly.

What is covered now:
  - eval.evaluate: purity/determinism, terminal WIN/LOSS/DRAW, prize dominance, body gradient.
  - main._forced_move: must reproduce the recorded `forced_option` label on all 130 golden
    fixtures (a regression lock on the forced-move heuristic + _attack_value).
  - main._attack_value: the KO/lethal threshold contract (>=8000 any KO, >=90000 game-winning,
    <8000 non-KO). Finding A3 in the audit: this contract has diverged across modules; lock it.
  - main.agent (no-search): legal + never raises on every fixture.
  - main.agent_search (shipped agent): legal + never raises + under a per-decision wall-clock
    bound on every fixture. This is the competition's never-throw / never-timeout rule, which
    had zero test coverage before. Degrades gracefully to the heuristic fallback if the cg
    engine is absent (e.g. a clean checkout with data/ gitignored).

    PYTHONIOENCODING=utf-8 python tests/test_heuristics_fixed_state.py
Also pytest-collectable (functions named test_*).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import eval as EV     # noqa: E402
import main as M      # noqa: E402

# Help search find the bundled engine if present (best-effort; absence is handled gracefully).
_CG = ROOT / "data" / "external" / "official" / "sample_submission" / "cg"
if _CG.is_dir():
    os.environ.setdefault("CABT_CG_DIR", str(_CG))

FIX = json.load(open(ROOT / "tests" / "golden_state_action_fixtures" / "fixtures.json", encoding="utf-8"))
FIXTURES = FIX["fixtures"]

# Per-decision wall-clock bound for the shipped agent. Production DEFAULT_BUDGET is 0.6s; this
# bound is generous (engine-backed worst case measured ~0.09s) and only catches a real hang.
PER_DECISION_BOUND_S = 1.5


# --------------------------------------------------------------------------- helpers
def _legal(sel: dict, ret) -> bool:
    """A returned selection must be distinct in-range indices within [minCount, maxCount]."""
    opts = sel.get("option") or []
    n = len(opts)
    mn = sel.get("minCount") or 0
    mx = sel.get("maxCount") or 0
    if not isinstance(ret, list):
        return False
    if any((not isinstance(i, int)) or i < 0 or i >= n for i in ret):
        return False
    if len(set(ret)) != len(ret):
        return False
    if n == 0:
        return ret == []
    return mn <= len(ret) <= max(mx, mn)


def _player(prize=2, bench=(), active=None, energies=0):
    """A minimal State-dict player. prize is a count; bench/active hold {id,hp} slots."""
    p = {"prize": [0] * prize, "bench": list(bench)}
    if active is not None:
        a = dict(active)
        a.setdefault("energies", [0] * energies)
        p["active"] = [a]
    else:
        p["active"] = []
    return p


def _state(me_player, opp_player, result=-1):
    return {"result": result, "players": [me_player, opp_player]}


def _first_attack_id():
    for k, v in M.ATK.items():
        if (v.get("d") or 0) > 0:
            return k, v["d"]
    raise AssertionError("no attack with positive damage in ATK stats")


# --------------------------------------------------------------------------- eval.evaluate
def test_eval_is_deterministic():
    cur = _state(_player(prize=2, bench=[{"id": "1", "hp": 60}]), _player(prize=3))
    assert EV.evaluate(cur, 0) == EV.evaluate(cur, 0)


def test_eval_terminal_win_loss_draw():
    base = _state(_player(), _player())
    assert EV.evaluate({**base, "result": 0}, 0) == EV.WIN
    assert EV.evaluate({**base, "result": 1}, 0) == EV.LOSS
    assert EV.evaluate({**base, "result": 0}, 1) == EV.LOSS
    assert EV.evaluate({**base, "result": 2}, 0) == 0.0


def test_eval_prize_dominates():
    # With EQUAL boards, fewer prizes left for me == closer to winning == strictly higher,
    # and each prize is worth exactly W_PRIZE. (Note: W_PRIZE does NOT dominate an arbitrarily
    # large board-HP gap, so this is asserted under equal boards, where it holds exactly.)
    more = _state(_player(prize=2), _player(prize=3))
    fewer = _state(_player(prize=1), _player(prize=3))
    assert EV.evaluate(fewer, 0) > EV.evaluate(more, 0)
    assert abs((EV.evaluate(fewer, 0) - EV.evaluate(more, 0)) - EV.W_PRIZE) < 1e-9


def test_eval_body_gradient():
    two = _state(_player(prize=2, bench=[{"id": "1", "hp": 60}, {"id": "2", "hp": 60}]), _player(prize=2))
    one = _state(_player(prize=2, bench=[{"id": "1", "hp": 60}]), _player(prize=2))
    assert EV.evaluate(two, 0) > EV.evaluate(one, 0)


# --------------------------------------------------------------------------- _attack_value thresholds
def test_attack_value_ko_threshold():
    aid, dmg = _first_attack_id()
    opt = {"type": M.ATTACK, "attackId": aid}
    me = {"active": [], "prize": [0, 0, 0]}                       # 3 prizes left -> a KO does not win
    opp = {"active": [{"id": "999999", "hp": 1}], "prize": [0]}   # hp 1 <= dmg -> KO
    v = M._attack_value(opt, me, opp)
    assert v >= 8000.0, f"a KO must score >=8000, got {v}"
    assert v < 90000.0, f"a non-winning KO must score <90000, got {v}"


def test_attack_value_game_winning_threshold():
    aid, dmg = _first_attack_id()
    opt = {"type": M.ATTACK, "attackId": aid}
    me = {"active": [], "prize": [0]}                             # 1 prize left -> a 1-prize KO wins
    opp = {"active": [{"id": "999999", "hp": 1}], "prize": [0]}
    v = M._attack_value(opt, me, opp)
    assert v >= 90000.0, f"a game-winning KO must score >=90000, got {v}"


def test_attack_value_non_ko_below_threshold():
    aid, dmg = _first_attack_id()
    opt = {"type": M.ATTACK, "attackId": aid}
    me = {"active": [], "prize": [0, 0, 0]}
    opp = {"active": [{"id": "999999", "hp": dmg + 5000}], "prize": [0]}  # survives the hit
    v = M._attack_value(opt, me, opp)
    assert v < 8000.0, f"a non-KO attack must score <8000, got {v}"


# --------------------------------------------------------------------------- _forced_move on fixtures
def test_forced_move_matches_golden_label():
    bad = []
    for fx in FIXTURES:
        r = M._forced_move(fx["observation"])
        got = r[0] if isinstance(r, list) and r else None
        exp = fx.get("forced_option")
        if got != exp:
            bad.append((fx.get("step"), exp, got))
    assert not bad, f"_forced_move disagrees with golden forced_option on {len(bad)}: {bad[:6]}"


def test_forced_move_returns_none_or_legal_single():
    for fx in FIXTURES:
        obs = fx["observation"]
        r = M._forced_move(obs)
        if r is not None:
            assert _legal(obs["select"], r), f"forced move illegal at step {fx.get('step')}: {r}"
            assert len(r) == 1, f"forced move should be a single index, got {r}"


# --------------------------------------------------------------------------- agent legality / never-throw
def test_agent_legal_never_throws_all_fixtures():
    for fx in FIXTURES:
        obs = fx["observation"]
        try:
            r = M.agent(obs)
        except Exception as e:  # the agent must NEVER raise (a raise forfeits on Kaggle)
            raise AssertionError(f"agent() raised at step {fx.get('step')}: {e!r}")
        assert _legal(obs["select"], r), f"agent() illegal at step {fx.get('step')}: {r}"


def test_agent_search_legal_never_throws_within_budget():
    worst = 0.0
    worst_step = None
    for fx in FIXTURES:
        obs = fx["observation"]
        t = time.time()
        try:
            r = M.agent_search(obs)
        except Exception as e:  # shipped agent: never raise
            raise AssertionError(f"agent_search() raised at step {fx.get('step')}: {e!r}")
        dt = time.time() - t
        if dt > worst:
            worst, worst_step = dt, fx.get("step")
        assert _legal(obs["select"], r), f"agent_search() illegal at step {fx.get('step')}: {r}"
        assert dt < PER_DECISION_BOUND_S, f"agent_search() took {dt:.2f}s at step {fx.get('step')}"
    print(f"    (agent_search worst per-decision {worst:.3f}s at step {worst_step})")


TESTS = [
    test_eval_is_deterministic,
    test_eval_terminal_win_loss_draw,
    test_eval_prize_dominates,
    test_eval_body_gradient,
    test_attack_value_ko_threshold,
    test_attack_value_game_winning_threshold,
    test_attack_value_non_ko_below_threshold,
    test_forced_move_matches_golden_label,
    test_forced_move_returns_none_or_legal_single,
    test_agent_legal_never_throws_all_fixtures,
    test_agent_search_legal_never_throws_within_budget,
]


def main() -> int:
    print(f"FIXED-STATE heuristic tests | {len(FIXTURES)} fixtures | engine {'present' if _CG.is_dir() else 'ABSENT (fallback path)'}\n")
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print()
    if failed:
        print(f"FAILED  ({len(TESTS) - failed}/{len(TESTS)} passed)")
        return 1
    print(f"ALL PASS  ({len(TESTS)}/{len(TESTS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
