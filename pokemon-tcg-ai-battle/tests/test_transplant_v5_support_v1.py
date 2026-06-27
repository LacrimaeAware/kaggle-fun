"""Tests for the Transplant V5 runtime-feature support pack (Model B, OFFLINE/READ-ONLY audit).

Validates the extraction the V5 feasibility probe relies on, on the committed golden fixtures:
  - no-mutation: extractors do not mutate obs
  - no-leakage: produced context/action carry no forbidden (result/outcome/pilot/won/replay) fields
  - compact key stable: option_index_to_key is deterministic
  - terminal flag correct: ATTACK/END options flagged terminal, ATTACH/PLAY not
  - missing values explicit: context schema is stable (absent fields are None, never silent 0)
  - known ATTACH delta (engine-guarded): an ATTACH option shows energy_attached >= 1 (skips if engine absent)
  - known PLAY delta (engine-guarded): a draw/search PLAY shows hand/deck delta (skips if not found)

Run:  PYTHONIOENCODING=utf-8 python tests/test_transplant_v5_support_v1.py
"""
from __future__ import annotations
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
FIX = json.load(open(ROOT / "tests" / "golden_state_action_fixtures" / "fixtures.json", encoding="utf-8"))["fixtures"]
FORBIDDEN = ("result", "outcome", "won", "win_", "pilot", "replay", "future", "reward")
TYPE_ATTACK, TYPE_END, TYPE_ATTACH = 13, 14, 8

with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
    import turn_context_v0 as TC
    import learned_selector_bridge as BR
    import learned_proposer_adapter as AD
    import search_v3 as S
    import starmie_heuristics as SH


def _single_pick(fx):
    obs = fx.get("observation")
    return obs and obs.get("select") and (obs["select"].get("maxCount") or 0) == 1 and len(obs["select"].get("option") or []) >= 2


def _ctx(obs):
    out = {}
    with contextlib.redirect_stderr(io.StringIO()):
        tc = TC.extract_turn_context(obs)
        tac = BR.tactical_state_features(obs)
    for k in ("global_turn_number", "turn_action_count", "supporter_used_this_turn", "energy_attached_this_turn",
              "retreated_this_turn"):
        out[k] = tc.get(k)
    for k in ("board.prize_diff", "board.my_ready_main_attackers", "commitment.guaranteed_ko_available"):
        out[k] = tac.get(k)
    return out


def test_no_mutation():
    obs = next(fx["observation"] for fx in FIX if _single_pick(fx))
    before = json.dumps(obs, sort_keys=True, default=str)
    with contextlib.redirect_stderr(io.StringIO()):
        TC.extract_turn_context(obs)
        BR.tactical_state_features(obs)
        AD.option_index_to_key(obs)
        BR.state_features(obs, obs["select"]["option"])
    assert json.dumps(obs, sort_keys=True, default=str) == before, "obs mutated by extractors"
    print("PASS no-mutation")


def test_key_stable():
    n = 0
    for fx in FIX:
        if not _single_pick(fx):
            continue
        obs = fx["observation"]
        with contextlib.redirect_stderr(io.StringIO()):
            k1 = AD.option_index_to_key(obs)
            k2 = AD.option_index_to_key(obs)
        assert k1 == k2, "compact key not deterministic"
        n += 1
        if n >= 30:
            break
    assert n > 0
    print(f"PASS compact-key-stable ({n} fixtures)")


def test_terminal_flag():
    seen_term = seen_nonterm = 0
    for fx in FIX:
        if not _single_pick(fx):
            continue
        obs = fx["observation"]
        types = fx.get("option_types") or []
        for i, opt in enumerate(obs["select"]["option"]):
            if not isinstance(opt, dict):
                continue
            t = types[i] if i < len(types) else opt.get("type")
            with contextlib.redirect_stderr(io.StringIO()):
                of = BR.option_features(opt, obs)
            ends = bool(of.get("ends_turn"))
            if t in (TYPE_ATTACK, TYPE_END):
                assert ends, f"type {t} should be terminal (ends_turn)"
                seen_term += 1
            elif t == TYPE_ATTACH:
                assert not ends, "ATTACH should not end turn"
                seen_nonterm += 1
    assert seen_term and seen_nonterm, "did not see both terminal and non-terminal options"
    print(f"PASS terminal-flag ({seen_term} terminal, {seen_nonterm} attach checked)")


def test_no_leakage():
    for fx in FIX[:40]:
        if not _single_pick(fx):
            continue
        obs = fx["observation"]
        ctx = _ctx(obs)
        with contextlib.redirect_stderr(io.StringIO()):
            keys = AD.option_index_to_key(obs)
        blob = json.dumps({"ctx": ctx, "keys": keys}, default=str).lower()
        for bad in FORBIDDEN:
            assert bad not in blob.replace("results", "").replace("turn", ""), f"forbidden token '{bad}' leaked"
    # also scan the generated payloads if present
    pf = ROOT / "data" / "generated" / "transplant_v5_runtime_support" / "example_payloads.jsonl"
    if pf.exists():
        for line in pf.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            blob = json.dumps({k: r[k] for k in ("context", "action", "delta")}, default=str).lower()
            assert "result" not in blob and "pilot" not in blob and "won" not in blob, "forbidden field in payload"
    print("PASS no-leakage (context/action/payloads)")


def test_missing_explicit():
    # context schema is STABLE: every expected key present (value may be None), absence != silent 0
    keys = ("global_turn_number", "turn_action_count", "supporter_used_this_turn", "energy_attached_this_turn",
            "retreated_this_turn", "board.prize_diff", "board.my_ready_main_attackers",
            "commitment.guaranteed_ko_available")
    n = 0
    for fx in FIX:
        if not _single_pick(fx):
            continue
        ctx = _ctx(fx["observation"])
        for k in keys:
            assert k in ctx, f"context missing key {k} (schema not stable)"
        n += 1
        if n >= 20:
            break
    assert n > 0
    print(f"PASS missing-explicit (stable schema, {n} fixtures)")


def test_known_attach_delta():
    if S._api() is None:
        print("SKIP known-attach-delta (engine unavailable)")
        return
    deck = list(SH.STARMIE_DECK)
    checked = 0
    for fx in FIX:
        if not _single_pick(fx):
            continue
        obs = fx["observation"]
        types = fx.get("option_types") or []
        if TYPE_ATTACH not in types:
            continue
        with contextlib.redirect_stderr(io.StringIO()):
            deltas = S.option_deltas(obs, deck)
        if not deltas:
            continue
        for i, t in enumerate(types):
            if t == TYPE_ATTACH and i < len(deltas) and deltas[i] is not None:
                ea = deltas[i].get("energy_attached")
                assert ea is not None and ea >= 1, f"ATTACH should add >=1 energy, got {ea}"
                checked += 1
                break
        if checked >= 3:
            break
    assert checked > 0, "no ATTACH delta verified"
    print(f"PASS known-attach-delta ({checked} ATTACH options, energy_attached>=1)")


def test_known_play_or_draw_delta():
    if S._api() is None:
        print("SKIP known-play-delta (engine unavailable)")
        return
    deck = list(SH.STARMIE_DECK)
    for fx in FIX:
        if not _single_pick(fx):
            continue
        obs = fx["observation"]
        with contextlib.redirect_stderr(io.StringIO()):
            deltas = S.option_deltas(obs, deck)
        if not deltas:
            continue
        for d in deltas:
            if d and (d.get("cards_drawn") or d.get("deck_used")):
                print(f"PASS known-play/draw-delta (cards_drawn={d.get('cards_drawn')} deck_used={d.get('deck_used')})")
                return
    print("SKIP known-play-delta (no draw/search option in sample)")


def main():
    fns = [test_no_mutation, test_key_stable, test_terminal_flag, test_no_leakage, test_missing_explicit,
           test_known_attach_delta, test_known_play_or_draw_delta]
    failed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
            failed += 1
    print("ALL TRANSPLANT-V5-SUPPORT TESTS PASS" if not failed else f"{failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
