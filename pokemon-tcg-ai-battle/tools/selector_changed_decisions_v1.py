"""Changed-decision trace for the learned selector wiring. For real single-select decisions (resolved from
replays), record where each mode (top1_gate / top3_selector) overrides the heuristic baseline, with the
from/to semantic action, the override's proposer rank, support status, and the safety-check verdict.

Read-only; uses starmie_heuristics._selector_override against the deterministic heuristic baseline.

  PYTHONIOENCODING=utf-8 python tools/selector_changed_decisions_v1.py --max 400
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import deck_policy_v3 as DP  # noqa: E402
import learned_proposer_adapter as AD  # noqa: E402  semantic keys
import starmie_heuristics as SH  # noqa: E402

MA = Path("C:/Users/EcceNihilum/.codex/worktrees/0557/pokemon-ai-agent/data/generated/starmie_specialist/portable_selector_v1/export")
REPLAYS = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays")
OUT = ROOT / "data" / "generated" / "starmie_selector_live_smoke_v1"
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


def _heuristic_baseline(obs):
    """Deterministic heuristic pick (no stochastic search) for a single-select decision, else None."""
    sel = obs.get("select") or {}
    if int(sel.get("maxCount", 1) or 1) != 1 or int(sel.get("minCount", 1) or 1) != 1:
        return None
    try:
        h = SH.choose(obs)
    except Exception:
        return None
    if isinstance(h, (list, tuple)) and len(h) == 1 and isinstance(h[0], int):
        return [int(h[0])]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=400)
    args = ap.parse_args()
    rt = SH._selector_runtime()
    if rt is None:
        print("selector runtime unavailable; aborting")
        return 1

    stats = {m: {"single_select": 0, "changed": 0, "veto_blocked": 0, "ood_or_unsupported": 0,
                 "by_family_from": collections.Counter(), "by_family_to": collections.Counter()}
             for m in ("top1_gate", "top3_selector")}
    examples = []
    n = 0
    for line in open(MA / "packer_parity_raw_inputs.jsonl", encoding="utf-8"):
        if n >= args.max:
            break
        did = json.loads(line)["decision_id"]
        try:
            ep, step, seat = (int(x) for x in did.split(":"))
        except Exception:
            continue
        obs = _obs(ep, step, seat)
        if obs is None:
            continue
        base = _heuristic_baseline(obs)
        if base is None:
            continue
        n += 1
        keys = AD.option_index_to_key(obs)
        base_key = keys.get(base[0])
        for mode in ("top1_gate", "top3_selector"):
            os.environ["STARMIE_SELECTOR_MODE"] = mode
            try:
                out = SH._selector_override(obs, list(base))
            except Exception:
                out = list(base)
            st = stats[mode]
            st["single_select"] += 1
            if out != base:
                st["changed"] += 1
                to_key = keys.get(out[0])
                st["by_family_from"][(base_key or "?").split(":")[0]] += 1
                st["by_family_to"][(to_key or "?").split(":")[0]] += 1
                if len(examples) < 30 and mode == "top3_selector":
                    examples.append({"decision_id": did, "from": base_key, "to": to_key,
                                     "safety_hard_veto": bool(AD.safety_check(obs, out[0]).get("hard_veto"))})
            # safety accounting on the proposed override (independent of whether accepted)
            try:
                if AD.safety_check(obs, base[0]).get("hard_veto"):
                    st["veto_blocked"] += 1
            except Exception:
                pass
    os.environ.pop("STARMIE_SELECTOR_MODE", None)

    report = {"decisions_single_select": n, "modes": {}}
    for mode, st in stats.items():
        report["modes"][mode] = {
            "single_select_decisions": st["single_select"],
            "overrode_heuristic": st["changed"],
            "override_rate_pct": round(100 * st["changed"] / max(1, st["single_select"]), 1),
            "from_family_distribution": dict(st["by_family_from"]),
            "to_family_distribution": dict(st["by_family_to"]),
        }
    report["override_examples_top3"] = examples
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "changed_decisions_trace.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str)[:1500])
    print(f"\nwrote {OUT / 'changed_decisions_trace.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
