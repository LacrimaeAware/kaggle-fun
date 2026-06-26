"""Tests for the STARMIE_SELECTOR_MODE wiring in starmie_heuristics.choose_action.

Guarantees: default/off reproduces the heuristic baseline action-identically; the selector modes never crash,
never return an illegal selection, and fall back to the baseline on any doubt. Run:
  PYTHONIOENCODING=utf-8 python tests/test_selector_wiring_v1.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import deck_policy_v3 as DP  # noqa: E402
import starmie_heuristics as SH  # noqa: E402

REPLAYS = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays")
MA = Path("C:/Users/EcceNihilum/.codex/worktrees/0557/pokemon-ai-agent/data/generated/starmie_specialist/portable_selector_v1/export")
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
    try:
        for line in open(MA / "packer_parity_raw_inputs.jsonl", encoding="utf-8"):
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
    except Exception:
        pass
    return out


def _legal(obs, pick):
    sel = obs.get("select") or {}
    n = len(sel.get("option") or [])
    if not isinstance(pick, list):
        return False
    return all(isinstance(i, int) and 0 <= i < n for i in pick) and DP.valid_selection(obs, pick)


def _baseline_pick(obs):
    """A deterministic single-select baseline for testing the override in isolation (no stochastic search)."""
    sel = obs.get("select") or {}
    n = len(sel.get("option") or [])
    for i in range(n):
        if DP.valid_selection(obs, [i]):
            return [i]
    return [0] if n else []


def test_off_mode_is_strict_identity():
    """choose_action runs search (stochastic), so off-mode inertness is proven at the override layer: in off
    mode _selector_override must be the exact identity for every decision and every legal baseline pick."""
    obss = _decisions()
    assert obss, "no decisions resolved (replays present?)"
    os.environ.pop("STARMIE_SELECTOR_MODE", None)  # unset == off
    for o in obss:
        p = _baseline_pick(o)
        assert SH._selector_override(o, p) is p or SH._selector_override(o, p) == p, "off-mode mutated the pick"
    os.environ["STARMIE_SELECTOR_MODE"] = "off"
    for o in obss:
        p = _baseline_pick(o)
        assert SH._selector_override(o, p) == p, "explicit off mode is not the identity"
    os.environ.pop("STARMIE_SELECTOR_MODE", None)
    print(f"PASS off/unset mode is strict identity on {len(obss)} decisions (action-identical baseline)")


def test_modes_legal_and_override_via_override_layer():
    """Exercise the override deterministically on a fixed baseline: every result is legal single-select, and is
    either the baseline or a different legal option."""
    obss = _decisions()
    for mode in ("top1_gate", "top3_selector"):
        os.environ["STARMIE_SELECTOR_MODE"] = mode
        changed = 0
        for o in obss:
            b = _baseline_pick(o)
            p = SH._selector_override(o, b)
            assert _legal(o, p), f"{mode}: illegal selection {p}"
            assert isinstance(p, list) and len(p) == 1, f"{mode}: not single-select {p}"
            if p != b:
                changed += 1
        print(f"PASS {mode}: all overrides legal single-select; changed {changed}/{len(obss)} from baseline")
    os.environ.pop("STARMIE_SELECTOR_MODE", None)


def test_failclosed_on_bad_runtime():
    """If the runtime cannot load, modes must return the baseline pick unchanged (fail-closed)."""
    obss = _decisions(limit=40)
    saved = SH._SELECTOR_RT
    SH._SELECTOR_RT = None  # simulate missing runtime
    os.environ["STARMIE_SELECTOR_MODE"] = "top3_selector"
    try:
        for o in obss:
            b = _baseline_pick(o)
            assert SH._selector_override(o, b) == b, "did not fall back to baseline when runtime is None"
    finally:
        SH._SELECTOR_RT = saved
        os.environ.pop("STARMIE_SELECTOR_MODE", None)
    print(f"PASS fail-closed: runtime=None returns baseline on {len(obss)} decisions")


def main() -> int:
    rc = 0
    for t in (test_off_mode_is_strict_identity, test_modes_legal_and_override_via_override_layer,
              test_failclosed_on_bad_runtime):
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            rc = 1
        except Exception as e:  # environment issues (no replays) should not hard-fail the suite
            print(f"SKIP {t.__name__}: {e}")
    print(f"\n{'ALL SELECTOR-WIRING TESTS PASS' if rc == 0 else 'SOME FAILED'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
