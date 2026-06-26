"""Sections 2-4: same-turn future sequences, turn-end public-state deltas, and action-family/semantic
normalization for Model A's replay-transplant value prior. Read-only over the cohort bridge traces + replays.

Turn boundaries are delimited by the replay `status` field (a contiguous ACTIVE run for a seat = one turn).
The same-turn future is enriched from the trace's validated `future_same_turn_sequence`. No gameplay is run.

  PYTHONIOENCODING=utf-8 python tools/transplant_sequences_v0.py [--max N]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import deck_policy_v3 as DP            # noqa: E402
import starmie_tactical_state as TS    # noqa: E402
OUT = ROOT / "data" / "generated" / "starmie_transplant_support_v0"
REPLAYS = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays")
TRACES = {
    "yushin": ROOT / "data/generated/starmie_bridge_trace_v0/yushin_trace.jsonl",
    "keidroid": ROOT / "data/generated/starmie_bridge_trace_v0/keidroid_trace.jsonl",
    "old_exact": ROOT / "data/generated/starmie_bridge_trace_v0/old_exact_trace_sample.jsonl",
}
ATTACK_TOKENS = {"ATTACK", "Jetting", "Nebula", "Turbo"}
DEV_FAMILIES = {"PLAY", "ATTACH", "EVOLVE", "SELECT_CARD", "ABILITY"}
TURN_ENDING = {"ATTACK", "END"}
SUPPORTERS = {"Wally", "Boss", "Iono", "Professor", "Arven", "Night Stretcher", "Pokégear", "Earthen", "Hammer"}
_EPC: dict = {}


def _steps(ep):
    if ep not in _EPC:
        if len(_EPC) > 10:
            _EPC.clear()
        try:
            _EPC[ep] = json.load(open(REPLAYS / f"{ep}.json", encoding="utf-8"))["steps"]
        except Exception:
            _EPC[ep] = None
    return _EPC[ep]


def _status(steps, t, seat):
    try:
        return steps[t][seat].get("status")
    except Exception:
        return None


def _obs(steps, t, seat):
    try:
        return steps[t][seat].get("observation")
    except Exception:
        return None


def _turn_bounds(steps, step, seat):
    """Return (turn_end_state_step, next_own_decision_step) using the ACTIVE-run that contains `step`.
    turn_end_state_step = first INACTIVE step after the run (post-turn board); next_own = next ACTIVE run start."""
    n = len(steps)
    t = step
    while t < n and _status(steps, t, seat) == "ACTIVE":
        t += 1
    turn_end = t if t < n else None  # first non-ACTIVE after the run
    no = None
    if turn_end is not None:
        u = turn_end
        while u < n and _status(steps, u, seat) != "ACTIVE":
            u += 1
        no = u if u < n else None
    return turn_end, no


def _public_state(obs):
    if not obs:
        return None
    try:
        cur = DP._current(obs)
        me = DP._perspective(cur)
        meP, oppP = DP._player(cur, me), DP._player(cur, 1 - me)
        opts = (obs.get("select") or {}).get("option") or []
        b = TS.board_features(meP, oppP, obs, opts)
        act = DP._active(meP)
        return {
            "my_prizes_left": b["my_prizes_left"], "opp_prizes_left": b["opp_prizes_left"],
            "prize_diff": b["prize_diff"], "my_pokemon": b["my_pokemon"], "opp_pokemon": b["opp_pokemon"],
            "my_board_hp": b["my_board_hp"], "opp_board_hp": b["opp_board_hp"],
            "my_units": b["my_units"], "opp_units": b["opp_units"],
            "my_ready_attackers": b["my_ready_attackers"], "my_ready_main_attackers": b["my_ready_main_attackers"],
            "my_backup_ready": b["my_backup_ready"], "my_hand": b["my_hand_count"], "my_deck": b["my_deck_count"],
            "my_discard": len(DP._items(DP._get(meP, "discard", []))), "active_id": DP._cid(act) if act else None,
        }
    except Exception:
        return None


def _delta(a, b):
    if not a or not b:
        return None
    out = {}
    for k in a:
        if isinstance(a.get(k), (int, float)) and isinstance(b.get(k), (int, float)):
            out[k] = b[k] - a[k]
    out["active_changed"] = (a.get("active_id") != b.get("active_id"))
    return out


# ---------- action semantics ----------
def _role(family, key):
    k = key or ""
    if family == "ATTACK":
        return "attack"
    if family == "ATTACH":
        return "attach"
    if family == "EVOLVE":
        return "evolve"
    if family == "RETREAT":
        return "retreat"
    if family == "END":
        return "end"
    if family == "SELECT_CARD":
        return "search"
    if family == "PLAY":
        if "Wally" in k or "Cape" in k:
            return "heal"
        if "Boss" in k:
            return "gust"
        if "Hammer" in k:
            return "disruption"
        if any(s in k for s in ("Stretcher", "Pokégear", "Professor", "Iono", "Arven", "Earthen", "Research")):
            return "draw"
        return "setup"
    return "other"


def _action_semantics(option_map, key_for_index):
    """option_map: {raw_index(str): semantic_key}. Returns list of normalized option records."""
    out = []
    for idx_s, key in (option_map or {}).items():
        try:
            idx = int(idx_s)
        except Exception:
            continue
        fam = (key or "?").split(":")[0]
        out.append({
            "raw_option_index": idx, "semantic_action_key": key, "family": fam,
            "role": _role(fam, key), "turn_ending": fam in TURN_ENDING,
            "blocked_by_c3": fam in {"ATTACK", "END", "RETREAT"},
            "consumes_attachment": fam == "ATTACH", "consumes_supporter": fam == "PLAY" and any(s in (key or "") for s in SUPPORTERS),
            "changes_active": fam == "RETREAT",
        })
    return out


def _same_turn(em):
    seq = em.get("future_same_turn_sequence") or []
    fam = em.get("pilot_action_family")
    later = seq[1:] if seq else []
    first_atk = next((i for i, x in enumerate(seq) if x in ATTACK_TOKENS or str(x).upper() == "ATTACK"), None)
    dev_before_attack = sum(1 for x in seq[:first_atk] if str(x).split(":")[0] in DEV_FAMILIES) if first_atk is not None else None
    attacked_later = any(x in ATTACK_TOKENS or str(x).upper() == "ATTACK" for x in later)
    return {
        "future_same_turn_sequence": seq, "actions_later_in_turn": later,
        "current_action_family": fam, "current_action_terminal": fam in TURN_ENDING,
        "pilot_attacked_later_same_turn": attacked_later,
        "pilot_attacked_this_or_later": (fam == "ATTACK") or attacked_later,
        "pilot_ended_without_attacking": (not attacked_later) and (fam != "ATTACK")
        and (("END" in seq) or first_atk is None),
        "dev_actions_before_attack": dev_before_attack,
        "current_consumes_supporter": fam == "PLAY" and any(s in (em.get("pilot_action_key") or "") for s in SUPPORTERS),
        "current_consumes_attachment": fam == "ATTACH",
        "current_changes_active": fam == "RETREAT",
        "followed_by_search_tutor": any(str(x).split(":")[0] == "SELECT_CARD" for x in later),
        "followed_by_attach": any(str(x).split(":")[0] == "ATTACH" for x in later),
        "followed_by_evolve": any(str(x).split(":")[0] == "EVOLVE" for x in later),
        "followed_by_play": any(str(x).split(":")[0] == "PLAY" for x in later),
        "followed_by_attack": attacked_later,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0)  # 0 = all
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fseq = open(OUT / "same_turn_sequences.jsonl", "w", encoding="utf-8")
    fdel = open(OUT / "turn_end_deltas.jsonl", "w", encoding="utf-8")
    fsem = open(OUT / "action_semantics.jsonl", "w", encoding="utf-8")
    n = resolved = te_ok = no_ok = 0
    miss = {"no_replay": 0, "no_turn_end": 0, "no_next_own": 0}
    for cohort, path in TRACES.items():
        for line in open(path, encoding="utf-8"):
            if args.max and n >= args.max:
                break
            r = json.loads(line)
            did = r["decision_id"]
            em = r.get("eval_meta") or {}
            rt = r.get("runtime") or {}
            try:
                ep, step, seat = did.split("_")
                ep, step, seat = int(ep), int(step), int(seat)
            except Exception:
                continue
            n += 1
            # Section 2: same-turn (eval-only enrichment)
            st = _same_turn(em)
            st.update({"decision_id": did, "cohort": cohort, "episode_id": ep, "step": step, "seat": seat})
            fseq.write(json.dumps(st, default=str) + "\n")
            # Section 4: action semantics (runtime)
            fsem.write(json.dumps({"decision_id": did, "cohort": cohort,
                                   "options": _action_semantics(rt.get("option_index_to_semantic_key"), None),
                                   "agent_action_key": rt.get("current_agent_action_key")}, default=str) + "\n")
            # Section 3: turn-end deltas (from replay)
            steps = _steps(ep)
            if not steps:
                miss["no_replay"] += 1
                fdel.write(json.dumps({"decision_id": did, "cohort": cohort, "missing": "no_replay"}) + "\n")
                continue
            resolved += 1
            te_step, no_step = _turn_bounds(steps, step, seat)
            s_dec = _public_state(_obs(steps, step, seat))
            s_te = _public_state(_obs(steps, te_step, seat)) if te_step is not None else None
            s_no = _public_state(_obs(steps, no_step, seat)) if no_step is not None else None
            if te_step is None:
                miss["no_turn_end"] += 1
            else:
                te_ok += 1
            if no_step is None:
                miss["no_next_own"] += 1
            else:
                no_ok += 1
            fdel.write(json.dumps({
                "decision_id": did, "cohort": cohort, "episode_id": ep, "step": step, "seat": seat,
                "turn_end_step": te_step, "next_own_step": no_step,
                "decision_state": s_dec,
                "turn_end_delta": _delta(s_dec, s_te), "next_own_delta": _delta(s_dec, s_no),
                "missing": {"turn_end": te_step is None or s_te is None, "next_own": no_step is None or s_no is None},
            }, default=str) + "\n")
        if args.max and n >= args.max:
            break
    for f in (fseq, fdel, fsem):
        f.close()
    print(f"decisions {n}; replays resolved {resolved}; turn_end_ok {te_ok}; next_own_ok {no_ok}")
    print(f"missing: {miss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
