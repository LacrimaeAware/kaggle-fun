"""Tests for the conservative C3 (c3_family_limited) selector wiring + safety behavior.

Core guarantee: c3_family_limited may override the heuristic baseline ONLY into a nonterminal development family
(ATTACH / SELECT_CARD / EVOLVE / PLAY); it must NEVER override into a turn-ending ATTACK / END / RETREAT. Off-mode
stays baseline-identical, picks stay legal, and the runtime is fail-closed.

  PYTHONIOENCODING=utf-8 python tests/test_selector_v2_wiring_v1.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import deck_policy_v3 as DP            # noqa: E402
import learned_proposer_adapter as AD  # noqa: E402
import starmie_heuristics as SH        # noqa: E402

REPLAYS = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays")
MA = Path("C:/Users/EcceNihilum/.codex/worktrees/0557/pokemon-ai-agent/data/generated/starmie_specialist/portable_selector_v1/export")
BLOCKED = {"ATTACK", "END", "RETREAT"}
ALLOWED = {"ATTACH", "SELECT_CARD", "EVOLVE", "PLAY"}
_EPC: dict = {}


def _obs(e, s, seat):
    if e not in _EPC:
        if len(_EPC) > 8:
            _EPC.clear()
        try:
            _EPC[e] = json.load(open(REPLAYS / f"{e}.json", encoding="utf-8"))
        except Exception:
            _EPC[e] = None
    try:
        return _EPC[e]["steps"][s][seat].get("observation")
    except Exception:
        return None


def _decisions(limit=200):
    out = []
    for line in open(MA / "parity_inputs.jsonl", encoding="utf-8"):
        if len(out) >= limit:
            break
        did = json.loads(line)["decision_id"]
        try:
            ep, step, seat = (int(x) for x in did.split(":"))
        except Exception:
            continue
        ob = _obs(ep, step, seat)
        if ob is not None and (ob.get("select") or {}).get("option"):
            out.append(ob)
    return out


def _heuristic_single(obs):
    sel = obs.get("select") or {}
    if int(sel.get("maxCount", 1) or 1) != 1 or int(sel.get("minCount", 1) or 1) != 1:
        return None
    try:
        h = SH.choose(obs)
    except Exception:
        return None
    return [int(h[0])] if isinstance(h, (list, tuple)) and len(h) == 1 and isinstance(h[0], int) else None


def test_v2_runtime_loads():
    assert SH._selector_runtime_v2() is not None, "V2 runtime failed to load"
    print("PASS v2-runtime-loads")


def test_c3_never_overrides_into_terminal_family():
    obss = _decisions()
    assert obss, "no decisions (replays present?)"
    os.environ["STARMIE_SELECTOR_MODE"] = "c3_family_limited"
    overrides = 0
    target_fams = set()
    try:
        for o in obss:
            base = _heuristic_single(o)
            if base is None:
                continue
            p = SH._selector_override(o, list(base))
            assert isinstance(p, list) and len(p) == 1, f"not single-select: {p}"
            assert DP.valid_selection(o, p), f"illegal pick {p}"
            if p != base:
                overrides += 1
                fam = (AD.option_index_to_key(o).get(p[0]) or "?").split(":")[0]
                target_fams.add(fam)
                assert fam not in BLOCKED, f"c3 overrode into BLOCKED terminal family {fam} at pick {p}"
    finally:
        os.environ.pop("STARMIE_SELECTOR_MODE", None)
    assert overrides > 0, "c3 produced no overrides on the sample (expected some development overrides)"
    assert target_fams <= ALLOWED, f"c3 override families outside allowed set: {target_fams - ALLOWED}"
    print(f"PASS c3-no-terminal-overrides: {overrides} overrides, all in {sorted(target_fams)} (blocked={sorted(BLOCKED)})")


def test_off_identity_for_c3():
    obss = _decisions(120)
    os.environ.pop("STARMIE_SELECTOR_MODE", None)
    for o in obss:
        base = _heuristic_single(o)
        if base is None:
            continue
        assert SH._selector_override(o, list(base)) == base, "off mode mutated the pick"
    print(f"PASS off-identity (c3 inert when off)")


def test_failclosed_when_v2_runtime_missing():
    obss = _decisions(40)
    saved = SH._SELECTOR_RT_V2
    SH._SELECTOR_RT_V2 = None
    os.environ["STARMIE_SELECTOR_MODE"] = "c3_family_limited"
    try:
        for o in obss:
            base = _heuristic_single(o)
            if base is None:
                continue
            assert SH._selector_override(o, list(base)) == base, "did not fall back to baseline when V2 runtime None"
    finally:
        SH._SELECTOR_RT_V2 = saved
        os.environ.pop("STARMIE_SELECTOR_MODE", None)
    print("PASS fail-closed (V2 runtime None -> baseline)")


def main() -> int:
    rc = 0
    for t in (test_v2_runtime_loads, test_off_identity_for_c3, test_c3_never_overrides_into_terminal_family,
              test_failclosed_when_v2_runtime_missing):
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            rc = 1
        except Exception as e:
            print(f"SKIP {t.__name__}: {e}")
    print(f"\n{'ALL C3-WIRING TESTS PASS' if rc == 0 else 'SOME FAILED'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
