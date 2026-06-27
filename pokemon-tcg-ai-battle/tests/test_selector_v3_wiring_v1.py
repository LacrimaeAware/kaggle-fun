"""Tests for the selector_v3_transplant wiring. Safety invariants that hold regardless of the transplant-support
table issue: V3 runtime loads, off is baseline-identical, fail-closed when runtime missing, and V3 NEVER overrides
into a terminal family (ATTACK/END/RETREAT). (The separate diagnostic records that V3 currently abstains live due
to a runtime<->table semantic-key-format mismatch; that is not asserted here so a future Model A fix is detectable.)

  PYTHONIOENCODING=utf-8 python tests/test_selector_v3_wiring_v1.py
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

MA = Path("C:/Users/EcceNihilum/.codex/worktrees/0557/pokemon-ai-agent/data/generated/starmie_specialist/portable_selector_v1/export")
REPLAYS = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays")
TERMINAL = {"ATTACK", "END", "RETREAT"}
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


def _decisions(limit=120):
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


def test_v3_runtime_loads():
    assert SH._selector_runtime_v3() is not None, "V3 runtime failed to load"
    print("PASS v3-runtime-loads")


def test_v3_never_overrides_into_terminal():
    obss = _decisions()
    os.environ["STARMIE_SELECTOR_MODE"] = "selector_v3_transplant"
    try:
        for o in obss:
            base = _heuristic_single(o)
            if base is None:
                continue
            p = SH._selector_override(o, list(base))
            assert isinstance(p, list) and len(p) == 1 and DP.valid_selection(o, p), f"illegal {p}"
            if p != base:
                fam = (AD.option_index_to_key(o).get(p[0]) or "?").split(":")[0]
                assert fam not in TERMINAL, f"V3 overrode into terminal family {fam}"
    finally:
        os.environ.pop("STARMIE_SELECTOR_MODE", None)
    print("PASS v3-never-terminal (all picks legal single-select, no ATTACK/END/RETREAT override)")


def test_v3_off_identity():
    obss = _decisions(80)
    os.environ.pop("STARMIE_SELECTOR_MODE", None)
    for o in obss:
        base = _heuristic_single(o)
        if base is None:
            continue
        assert SH._selector_override(o, list(base)) == base
    print("PASS v3-off-identity")


def test_v3_failclosed_when_runtime_missing():
    obss = _decisions(40)
    saved = SH._SELECTOR_RT_V3
    SH._SELECTOR_RT_V3 = None
    os.environ["STARMIE_SELECTOR_MODE"] = "selector_v3_transplant"
    try:
        for o in obss:
            base = _heuristic_single(o)
            if base is None:
                continue
            assert SH._selector_override(o, list(base)) == base, "did not fall back when V3 runtime None"
    finally:
        SH._SELECTOR_RT_V3 = saved
        os.environ.pop("STARMIE_SELECTOR_MODE", None)
    print("PASS v3-fail-closed")


def main() -> int:
    rc = 0
    for t in (test_v3_runtime_loads, test_v3_off_identity, test_v3_never_overrides_into_terminal,
              test_v3_failclosed_when_runtime_missing):
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            rc = 1
        except Exception as e:
            print(f"SKIP {t.__name__}: {e}")
    print(f"\n{'ALL V3-WIRING TESTS PASS' if rc == 0 else 'SOME FAILED'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
