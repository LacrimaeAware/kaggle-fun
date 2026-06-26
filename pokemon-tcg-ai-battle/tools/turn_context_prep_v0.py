"""Section 2: emit turn-context payload examples + manifest for Model A's Feature V3 packer. Read-only.

  PYTHONIOENCODING=utf-8 python tools/turn_context_prep_v0.py [--max 600]
"""
from __future__ import annotations
import argparse
import collections
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import turn_context_v0 as TC  # noqa: E402
OUT = ROOT / "data" / "generated" / "runtime_feature_audit"
REPLAYS = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays")

FIELD_SOURCE = {
    "global_turn_number": ("current.turn", "present"),
    "first_player": ("current.firstPlayer", "present (-1 during setup until resolved)"),
    "first_player_resolved": ("derived", "derived"),
    "am_i_first_player": ("yourIndex == firstPlayer", "derived; null until first_player resolved"),
    "is_setup_phase": ("turn==0 or firstPlayer==-1", "derived"),
    "turn_action_count": ("current.turnActionCount", "present"),
    "decision_index_in_turn": ("current.turnActionCount", "present"),
    "supporter_used_this_turn": ("current.supporterPlayed", "present"),
    "energy_attached_this_turn": ("current.energyAttached", "present"),
    "retreated_this_turn": ("current.retreated", "present"),
    "stadium_played_this_turn": ("current.stadiumPlayed", "present"),
    "stadium_in_play": ("bool(current.stadium)", "derived"),
    "active_appeared_this_turn": ("active.appearThisTurn", "present"),
    "bench_appeared_this_turn_count": ("sum(bench.appearThisTurn)", "derived"),
    "status_conditions": ("player.asleep/paralyzed/confused/burned/poisoned", "present"),
    "attack_available": ("scan options type==13", "derived"),
    "end_available": ("scan options type==14", "derived"),
    "terminal_legal_option_count": ("count type in {13,14}", "derived"),
    "nonterminal_legal_option_count": ("count type not in {13,14}", "derived"),
    "information_revealing_legal_count": ("count type==3 (SELECT_CARD)", "derived"),
    "safe_development_legal_count": ("count type in {7,8,9}", "derived"),
    "is_our_first_turn_best_effort": ("post-setup turn parity", "BEST_EFFORT -- verify; setup-phase caveat"),
    "is_first_player_first_turn_best_effort": ("turn==1", "BEST_EFFORT -- verify; setup-phase caveat"),
}


def _sample(max_dec):
    files = sorted(os.listdir(REPLAYS))[:200]
    out = []
    for fn in files:
        if len(out) >= max_dec:
            break
        try:
            steps = json.load(open(REPLAYS / fn, encoding="utf-8"))["steps"]
        except Exception:
            continue
        eid = fn.split(".")[0]
        for t in range(0, len(steps), max(1, len(steps) // 8)):
            for seat in (0, 1):
                try:
                    o = steps[t][seat]["observation"]
                except Exception:
                    o = None
                if o and (o.get("select") or {}).get("option"):
                    out.append((f"{eid}_{t}_{seat}", o))
                if len(out) >= max_dec:
                    break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=600)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    samples = _sample(a.max)

    mutated = leaked = 0
    coverage = collections.Counter()
    nonnull = collections.Counter()
    setup_rows = 0
    with open(OUT / "turn_context_payload_examples.jsonl", "w", encoding="utf-8") as f:
        for did, obs in samples:
            before = copy.deepcopy(obs)
            payload = TC.extract_turn_context(obs)
            if obs != before:
                mutated += 1
            blob = json.dumps(payload).lower()
            if any(b in blob for b in ("result", "future", "pilot", "outcome", "won", "replay")):
                # field_status/keys may contain 'present'... check VALUES only for the forbidden raw signals
                if any(k for k in payload if k in TC.FORBIDDEN):
                    leaked += 1
            setup_rows += 1 if payload.get("is_setup_phase") else 0
            for k in TC.FIELDS:
                coverage[k] += 1
                v = payload.get(k)
                if v is not None and v != {} and not (isinstance(v, dict) and all(x is None for x in v.values())):
                    nonnull[k] += 1
            f.write(json.dumps({"decision_id": did, "turn_context": payload}, default=str) + "\n")

    n = max(1, len(samples))
    manifest = {
        "purpose": "Public turn-context payload for Model A Feature V3 packer. PREP only -- NOT wired into the live "
                   "selector. Read-only; excludes current.result and all future/pilot/outcome data.",
        "extractor": "agent/turn_context_v0.extract_turn_context(obs)",
        "samples": len(samples), "setup_phase_rows": setup_rows,
        "fields": {k: {"source": FIELD_SOURCE.get(k, ("derived", "derived"))[0],
                       "confidence": FIELD_SOURCE.get(k, ("", "derived"))[1],
                       "nonnull_pct": round(100 * nonnull[k] / n, 1)} for k in TC.FIELDS},
        "no_mutation": mutated == 0, "forbidden_field_leaks": leaked,
        "setup_phase_nuance": "global turn 0 is a SHARED setup phase (both seats act; firstPlayer=-1 until resolved). "
                              "turn_action_count is a global counter during setup, per-turn after. Player-turn-index "
                              "(is_our_first_turn / is_first_player_first_turn) is BEST_EFFORT -- Model A should finalize.",
        "wiring_status": "NOT wired into choose_action/selector. Wire into learned_selector_bridge only AFTER Model A "
                         "Feature V3 export exists, then re-run the adapter parity gate to keep it bit-exact.",
    }
    (OUT / "turn_context_payload_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"samples {len(samples)} | mutated {mutated} | leaks {leaked} | setup_rows {setup_rows}")
    print("field nonnull %:", json.dumps({k: manifest['fields'][k]['nonnull_pct'] for k in list(TC.FIELDS)[:12]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
