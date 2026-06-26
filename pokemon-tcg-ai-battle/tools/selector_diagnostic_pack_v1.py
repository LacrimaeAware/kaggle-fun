"""STARMIE SELECTOR LIVE-SMOKE DIAGNOSTIC PACK V1 -- data stage.

The live smoke saved AGGREGATE win/loss only (no per-game decision logs), and "do not run games" is in force,
so per-game-outcome attribution is impossible from existing artifacts. This builds the strongest MECHANISM
diagnostic available without running games: classify what the selector changes on the real Starmie decision
distribution (replay states the live pipeline actually processes) and quantify the premature-terminal-override
signature that explains the top3_selector regression.

Read-only over replays + Model A export. Runs the SAME pipeline the live agent uses
(learned_selector_bridge -> official packer -> portable runtime) and the SAME mode gate as
starmie_heuristics._selector_override. No games are played.

  PYTHONIOENCODING=utf-8 python tools/selector_diagnostic_pack_v1.py --max-files 500 --max-decisions 2500
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "agent" / "vendor" / "portable_selector_v1"))
import deck_policy_v3 as DP            # noqa: E402
import learned_proposer_adapter as AD  # noqa: E402  compact semantic keys + offline safety filters
import learned_selector_bridge as BR   # noqa: E402  CABT -> Feature-V2 payload
import starmie_feature_v2_packer as PK  # noqa: E402  official packer
import starmie_selector_runtime as RT   # noqa: E402  portable runtime
import starmie_heuristics as SH         # noqa: E402  heuristic baseline + the live gate

REPLAYS = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays")
OUT = ROOT / "data" / "generated" / "starmie_selector_live_smoke_v1"
MEGA, STARYU = 1031, 1030
DEVELOP = {"PLAY", "ATTACH", "EVOLVE", "SELECT_CARD", "ABILITY"}
TERMINAL = {"ATTACK", "END"}


def _family(key):
    return (key or "?").split(":")[0]


def _starmie_seat(steps, seat):
    """True if this seat ever shows the Starmie line (Mega/Staryu) in play -> it is our pilot's seat."""
    for st in steps[:: max(1, len(steps) // 12)]:
        try:
            pl = st[seat]["observation"]["current"]["players"][seat]
        except Exception:
            continue
        for zone in ("active", "bench", "hand", "discard"):
            for c in (pl.get(zone) or []):
                if isinstance(c, dict) and c.get("id") in (MEGA, STARYU):
                    return True
    return False


def _sample_decisions(max_files, max_decisions, seed=7):
    files = sorted(os.listdir(REPLAYS))
    rng = random.Random(seed)
    rng.shuffle(files)
    out = []
    for fn in files[:max_files]:
        if len(out) >= max_decisions:
            break
        if not fn.endswith(".json"):
            continue
        try:
            ep = json.load(open(REPLAYS / fn, encoding="utf-8"))
        except Exception:
            continue
        steps = ep.get("steps") or []
        eid = fn.split(".")[0]
        for seat in (0, 1):
            if not _starmie_seat(steps, seat):
                continue
            for si, st in enumerate(steps):
                if len(out) >= max_decisions:
                    break
                try:
                    obs = st[seat]["observation"]
                except Exception:
                    continue
                sel = (obs or {}).get("select") or {}
                if not sel.get("option"):
                    continue
                if int(sel.get("maxCount", 1) or 1) != 1 or int(sel.get("minCount", 1) or 1) != 1:
                    continue
                out.append((f"{eid}:{si}:{seat}", obs))
    return out


def _heuristic_baseline(obs):
    try:
        h = SH.choose(obs)
    except Exception:
        return None
    if isinstance(h, (list, tuple)) and len(h) == 1 and isinstance(h[0], int):
        return int(h[0])
    return None


def _selector_eval(obs, base_raw, mode, rt):
    """Replicate the live gate (starmie_heuristics._selector_override) and capture rich metrics in one pass."""
    sel = obs.get("select") or {}
    n = len(sel.get("option") or [])
    res = {"pick": base_raw, "changed": False, "support": None, "source": None, "entropy": None,
           "top1_margin": None, "sel_score_margin": None, "proposer_prob_of_pick": None,
           "proposer_rank_of_pick": None, "hard_veto_on_pick": None, "runtime_status": None}
    try:
        payload = BR.cabt_to_payload(obs, baseline_action={"raw_option_index": base_raw, "raw_option_indexes": [base_raw]})
        packed = PK.pack_cabt_observation(payload, payload["raw_legal_options"])
        out = rt.rank_and_select(packed, packed["packed_options"], packed.get("baseline_action"),
                                 packed.get("search_action"), 5)
    except Exception as e:
        res["runtime_status"] = f"ERROR:{e}"
        return res
    res["runtime_status"] = out.get("status")
    res["support"] = out.get("support_status")
    res["source"] = out.get("source")
    res["entropy"] = out.get("entropy")
    res["top1_margin"] = out.get("top1_margin")
    scores = sorted((out.get("selector_scores_by_packed_index") or {}).values(), reverse=True)
    res["sel_score_margin"] = round(scores[0] - scores[1], 4) if len(scores) >= 2 else None
    sel_raw = out.get("selected_raw_option_index")
    ranked = {a.get("raw_option_index"): a for a in (out.get("ranked_actions") or [])}
    # apply the live gate
    if out.get("status") != "READY" or str(out.get("support_status") or "") not in ("SUPPORTED", "SAFETY_FALLBACK"):
        return res
    if not isinstance(sel_raw, int) or not (0 <= sel_raw < n) or sel_raw == base_raw:
        res["proposer_prob_of_pick"] = (ranked.get(base_raw) or {}).get("probability")
        return res
    rank = (ranked.get(sel_raw) or {}).get("rank")
    res["proposer_prob_of_pick"] = (ranked.get(sel_raw) or {}).get("probability")
    res["proposer_rank_of_pick"] = rank
    if mode == "top1_gate" and rank != 1:
        return res
    if mode == "top3_selector" and rank not in (1, 2, 3):
        return res
    if out.get("source") == "fallback":
        return res
    try:
        if not DP.valid_selection(obs, [sel_raw]):
            return res
        veto = bool(AD.safety_check(obs, sel_raw).get("hard_veto"))
    except Exception:
        return res
    res["hard_veto_on_pick"] = veto
    if veto:
        return res
    res["pick"] = sel_raw
    res["changed"] = True
    return res


def _transition(base_fam, sel_fam, base_key, sel_key):
    if base_fam in DEVELOP and sel_fam == "ATTACK":
        return "DEVELOP_TO_ATTACK"
    if base_fam in DEVELOP and sel_fam == "END":
        return "DEVELOP_TO_END"
    if base_fam == "ATTACK" and sel_fam != "ATTACK":
        return "ATTACK_CHANGE"
    if base_fam == "ATTACH" and sel_fam == "ATTACH":
        return "ATTACH_TARGET_CHANGE"
    if base_fam == "SELECT_CARD" and sel_fam == "SELECT_CARD":
        return "SELECT_CARD_CHANGE"
    if base_fam == "PLAY" and sel_fam == "PLAY":
        return "PLAY_CHANGE"
    if "RETREAT" in (base_fam, sel_fam):
        return "RETREAT_CHANGE"
    return "OTHER"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-files", type=int, default=500)
    ap.add_argument("--max-decisions", type=int, default=2500)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rt = SH._selector_runtime()
    if rt is None:
        print("selector runtime unavailable; PIPELINE_INVALID")
        return 2

    decisions = _sample_decisions(args.max_files, args.max_decisions)
    print(f"sampled {len(decisions)} Starmie single-select decisions", flush=True)

    rows = []
    n_base = 0
    for did, obs in decisions:
        base_raw = _heuristic_baseline(obs)
        if base_raw is None:
            continue
        n_base += 1
        keys = AD.option_index_to_key(obs)
        base_key = keys.get(base_raw)
        base_fam = _family(base_key)
        tac = BR.tactical_state_features(obs)
        ko = bool(tac.get("commitment.guaranteed_ko_available"))
        gw = bool(tac.get("commitment.game_winning_attack_available"))
        safe_dev = bool(tac.get("commitment.safe_development_available"))
        nonterm_atk = bool(tac.get("commitment.nonterminal_attack_available"))
        for mode in ("top1_gate", "top3_selector"):
            ev = _selector_eval(obs, base_raw, mode, rt)
            sel_key = keys.get(ev["pick"]) if ev["changed"] else base_key
            sel_fam = _family(sel_key)
            trans = _transition(base_fam, sel_fam, base_key, sel_key) if ev["changed"] else "NONE"
            terminal = ev["changed"] and sel_fam in TERMINAL
            premature_terminal = bool(terminal and not ko and not gw and safe_dev)
            rows.append({
                "decision_id": did, "mode": mode,
                "baseline_family": base_fam, "baseline_key": base_key,
                "selector_family": sel_fam if ev["changed"] else None, "selector_key": sel_key if ev["changed"] else None,
                "changed": ev["changed"], "transition_class": trans,
                "terminal_override": terminal, "premature_terminal_override": premature_terminal,
                "guaranteed_ko_available": ko, "game_winning_attack_available": gw,
                "safe_development_available": safe_dev, "nonterminal_attack_available": nonterm_atk,
                "prize_diff": tac.get("board.prize_diff"), "my_ready_main_attackers": tac.get("board.my_ready_main_attackers"),
                "my_units": tac.get("board.my_units"), "deckout_pressure": tac.get("value.deckout_pressure"),
                "proposer_prob_of_pick": ev["proposer_prob_of_pick"], "proposer_rank_of_pick": ev["proposer_rank_of_pick"],
                "entropy": ev["entropy"], "top1_margin": ev["top1_margin"], "selector_score_margin": ev["sel_score_margin"],
                "support_status": ev["support"], "source": ev["source"], "hard_veto_on_pick": ev["hard_veto_on_pick"],
                # game_result / matchup are NOT available: the smoke saved aggregate win/loss only, and games may not be re-run.
                "game_result": None, "matchup": None,
            })
    with open(OUT / "changed_decision_classes.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"wrote {OUT / 'changed_decision_classes.jsonl'}  ({len(rows)} rows over {n_base} baseline decisions x 2 modes)",
          flush=True)

    # quick console summary
    for mode in ("top1_gate", "top3_selector"):
        mr = [r for r in rows if r["mode"] == mode]
        ch = [r for r in mr if r["changed"]]
        term = [r for r in ch if r["terminal_override"]]
        prem = [r for r in ch if r["premature_terminal_override"]]
        print(f"  {mode}: {len(ch)}/{len(mr)} overrides ({round(100*len(ch)/max(1,len(mr)),1)}%); "
              f"terminal {len(term)}; premature-terminal {len(prem)} "
              f"({round(100*len(prem)/max(1,len(ch)),1)}% of overrides)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
