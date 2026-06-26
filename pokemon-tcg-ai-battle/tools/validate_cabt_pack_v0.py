"""Validate a CABT->packed feature conversion against Model A's parity_inputs, field-by-field, on the SAME
decisions. The portable runtime's S2 model consumes only exact-ID option features + state counts + board-entity
buckets, so if these match Model A's pack, the logits match. This tells us whether a faithful live conversion is
feasible in-repo (gate for the live smoke) WITHOUT silently shipping a divergent pipeline.

Read-only. Resolves the parity decisions' raw observations from replays and compares.

  python tools/validate_cabt_pack_v0.py --max 60
"""
from __future__ import annotations
import argparse, collections, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import deck_policy_v3 as DP            # noqa
import starmie_tactical_state as TS    # noqa
MA = Path("C:/Users/EcceNihilum/.codex/worktrees/0557/pokemon-ai-agent/data/generated/starmie_specialist/portable_selector_v1/export")
REPLAYS = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays")
PLAY, ATTACH, EVOLVE, ABILITY, RETREAT, ATTACK, END, CARD = 7, 8, 9, 10, 12, 13, 14, 3
_EPC = {}


def _obs(e, s, seat):
    if e not in _EPC:
        if len(_EPC) > 12:
            _EPC.clear()
        try:
            _EPC[e] = json.load(open(REPLAYS / f"{e}.json", encoding="utf-8"))
        except Exception:
            _EPC[e] = None
    try:
        return _EPC[e]["steps"][s][seat].get("observation")
    except Exception:
        return None


def _my_option_features(o, obs):
    """Derive the model-used exact-ID option features from a raw CABT option."""
    t = o.get("type")
    try:
        src = DP.option_card_id(o, obs)
    except Exception:
        src = None
    tgt_ent = None
    try:
        tgt_ent = DP.option_target_entity(o, obs)
    except Exception:
        tgt_ent = None
    tgt = DP._cid(tgt_ent) if tgt_ent else None
    sel = DP._selection(obs) or {}
    eff = DP._get(sel, "effect", None)
    ctx = DP._get(eff, "id", None) if eff else None
    return {"type_id": t, "attack_id": o.get("attackId"), "ability_id": o.get("abilityId"),
            "source_card_id": src, "target_card_id": tgt, "context_card_id": ctx}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max", type=int, default=60); a = ap.parse_args()
    field_match = collections.Counter(); field_total = collections.Counter()
    n_dec = n_opt = 0; mismatches = collections.Counter()
    for line in open(MA / "parity_inputs.jsonl", encoding="utf-8"):
        if n_dec >= a.max:
            break
        r = json.loads(line)
        ep, step, seat = (int(x) for x in r["decision_id"].split(":"))
        obs = _obs(ep, step, seat)
        if not obs:
            continue
        raw_opts = (obs.get("select") or {}).get("option") or []
        n_dec += 1
        for po in r["legal_options"]:
            ri = po.get("raw_option_index")
            if not isinstance(ri, int) or not (0 <= ri < len(raw_opts)):
                continue
            mine = _my_option_features(raw_opts[ri], obs)
            theirs = po.get("features") or {}
            n_opt += 1
            for k in ("type_id", "attack_id", "ability_id", "source_card_id", "target_card_id", "context_card_id"):
                field_total[k] += 1
                mv, tv = mine.get(k), theirs.get(k)
                if (mv is None and tv is None) or (mv == tv):
                    field_match[k] += 1
                else:
                    mismatches[f"{k}: mine={mv} theirs={tv}"] += 1
    print(f"decisions {n_dec}, options compared {n_opt}")
    print("per-field match rate (mine vs Model A pack):")
    for k in ("type_id", "attack_id", "ability_id", "source_card_id", "target_card_id", "context_card_id"):
        print(f"  {k}: {field_match[k]}/{field_total[k]} = {round(100*field_match[k]/max(1,field_total[k]),1)}%")
    print("top mismatches:")
    for m, c in mismatches.most_common(8):
        print(f"  {c}x  {m}")


if __name__ == "__main__":
    main()
