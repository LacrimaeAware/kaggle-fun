"""Decision-divergence analysis: where does OUR Starmie agent disagree with top human pilots?

For each winning game of a top Mega-Starmie pilot (from starmie_top_pilots.json), we walk every real
decision the pilot faced, feed the EXACT recorded observation to our agent, and compare our pick to the
pilot's actual move. Pairing is (obs = steps[t][seat].observation, pilot_action = steps[t+1][seat].action)
-- verified 100% legal on the corpus. We categorize each decision, score how pivotal it is, and emit the
ranked disagreements with full board + labeled-option context so the visualizer can render them and the
user can give feedback. This grounds heuristics in what strong pilots actually do, instead of guesses.

  python tools/imitation_gap_v1.py --max-games 60 --budget 0.4 --workers 6

Output: data/imitation_gap.json  { stats, by_category, top_disagreements[], all_disagreements[] }

NOTE on attack damage: attack_stats.json is a flat per-attack number. It does NOT capture conditional /
stacking damage or "ignores weakness/resistance" attacks (e.g. Nebula Beam). Labels show the static
number but it is advisory only; the analysis flags attack decisions for human review rather than trusting it.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPLAYS = r"C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays"
PILOTS = ROOT / "data" / "starmie_top_pilots.json"
OUT = ROOT / "data" / "imitation_gap.json"

TYPE_NAMES = {0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 4: "TOOL_CARD", 5: "ENERGY_CARD", 6: "ENERGY",
              7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY", 11: "DISCARD", 12: "RETREAT",
              13: "ATTACK", 14: "END"}
ATTACK, ATTACH, EVOLVE, ABILITY, PLAY, END, CARD, RETREAT, DISCARD = 13, 8, 9, 10, 7, 14, 3, 12, 11


@contextlib.contextmanager
def _quiet_import():
    old = os.dup(2)
    dn = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(dn, 2)
        yield
    finally:
        os.dup2(old, 2)
        os.close(dn)
        os.close(old)


def _deck_of(d, seat):
    for step in (d.get("steps") or [])[:8]:
        if isinstance(step, list) and len(step) > seat and isinstance(step[seat], dict):
            a = step[seat].get("action")
            if isinstance(a, list) and len(a) == 60:
                return a
    return None


# ---- worker globals (loaded once per process) ----
_G = {}


def _winit(budget, agent_mode="deployed"):
    sys.path.insert(0, str(ROOT / "agent"))
    with _quiet_import(), contextlib.redirect_stdout(io.StringIO()):
        import deck_policy_v3 as DP
        import search_v3 as S
        SH = None
        if agent_mode == "heavy":
            import starmie_heuristics as SH
    S.USE_DYNAMIC_ATTACKS = True
    if budget and budget > 0:
        try:
            S.DEFAULT_BUDGET = budget
        except Exception:
            pass
    CDB = json.load(open(ROOT / "agent" / "card_stats.json", encoding="utf-8"))
    ATK = json.load(open(ROOT / "agent" / "attack_stats.json", encoding="utf-8"))
    CEFF = json.load(open(ROOT / "agent" / "card_effects.json", encoding="utf-8"))
    _G.update(DP=DP, S=S, SH=SH, CDB=CDB, ATK=ATK, CEFF=CEFF, budget=budget, mode=agent_mode)


def _cname(cid):
    if cid is None:
        return None
    return (_G["CDB"].get(str(cid), {}) or {}).get("n", f"#{cid}")


def _our_pick(obs, deck):
    DP, S = _G["DP"], _G["S"]
    if _G.get("mode") == "heavy" and _G.get("SH") is not None:
        SH = _G["SH"]
        h = SH.choose(obs)
        if h is not None:
            return list(h), "sh_heuristic"
        try:
            mv = S.best_option(obs, deck, leaf_mode="deckout", rollout_mode="develop")
            if mv:
                return list(mv), "search"
        except Exception:
            pass
        return DP.default_selection(obs), "default"
    ko = DP.best_ko_attack(obs)
    if ko is not None:
        return [ko[0]], "ko_floor"
    # no-suicide floor
    sel = obs.get("select") or {}
    if (sel.get("maxCount") or 0) == 1:
        opts = sel.get("option") or []
        cur = obs.get("current") or {}
        players = cur.get("players") or []
        yi = cur.get("yourIndex", 0)
        me = players[yi] if yi < len(players) else {}
        in_play = sum(1 for a in (me.get("active") or []) if a) + len(me.get("bench") or [])
        if in_play <= 1:
            end_idx = next((i for i, o in enumerate(opts) if o.get("type") == END), None)
            if end_idx is not None and any(o.get("type") == ABILITY for o in opts):
                return [end_idx], "no_suicide"
    try:
        mv = S.best_option(obs, deck, leaf_mode="deckout", rollout_mode="develop")
        if mv:
            return list(mv), "search"
    except Exception:
        pass
    # legal default
    opts = sel.get("option") or []
    k = sel.get("maxCount") or 0
    mn = sel.get("minCount") or 0
    n = len(opts)
    if n == 0 or k <= 0:
        return [], "default"
    return list(range(max(min(k, n), min(mn, n)))), "default"


def _opt_label(o, obs, sel):
    DP = _G["DP"]
    t = o.get("type")
    tn = TYPE_NAMES.get(t, str(t))
    if t == ATTACK:
        arow = _G["ATK"].get(str(o.get("attackId")), {})
        return f"ATTACK: {arow.get('n','?')} (~{arow.get('d','?')})"
    if t == END:
        return "END turn"
    if t == RETREAT:
        return "RETREAT"
    if t in (1, 2):
        return tn
    cid = None
    try:
        cid = DP.option_card_id(o, obs)
    except Exception:
        cid = None
    nm = _cname(cid) if cid is not None else None
    if t == ATTACH:
        try:
            tgt = DP.option_target_entity(o, obs)
            tnm = _cname(DP._cid(tgt)) if tgt else "?"
        except Exception:
            tnm = "?"
        return f"ATTACH {nm or '?'} -> {tnm}"
    if t == EVOLVE:
        return f"EVOLVE -> {nm or '?'}"
    if t == ABILITY:
        return f"ABILITY {nm or ''}".strip()
    if t == PLAY:
        return f"PLAY {nm or '?'}"
    if t == CARD:
        # selection-local card (search/discard target)
        if nm is None:
            idx = o.get("index")
            for key in ("deck", "discard", "prize", "hand"):
                zone = sel.get(key) or []
                if isinstance(idx, int) and 0 <= idx < len(zone):
                    nm = _cname(DP._cid(zone[idx]))
                    break
        return f"CARD {nm or '?'}"
    if t == DISCARD:
        return f"DISCARD {nm or ''}".strip()
    return tn


def _entity_summary(ent):
    if not ent:
        return None
    return {"name": _cname(ent.get("id")), "hp": ent.get("hp"),
            "energy": len((ent.get("energyCards") or ent.get("energies") or []))}


def _board(obs):
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    yi = cur.get("yourIndex", 0)
    me = players[yi] if yi < len(players) else {}
    opp = players[1 - yi] if len(players) > 1 else {}

    def side(p):
        act = (p.get("active") or [None])[0]
        return {"active": _entity_summary(act),
                "bench": [_entity_summary(b) for b in (p.get("bench") or []) if b],
                "hand": [_cname((c.get("id") if isinstance(c, dict) else c)) for c in (p.get("hand") or [])],
                "handCount": p.get("handCount"),
                "prizeLeft": len(p.get("prize") or []),
                "deckCount": p.get("deckCount")}
    return {"me": side(me), "opp": side(opp), "step": obs.get("step")}


def _category(chosen_types, menu_types, ctx):
    if ctx == 41:
        return "go_first"
    if ATTACK in chosen_types:
        return "attack"
    if ATTACK in menu_types and END in chosen_types:
        return "pass_up_attack"
    for t, name in ((ATTACH, "energy_attach"), (EVOLVE, "evolve"), (ABILITY, "ability"),
                    (PLAY, "play"), (RETREAT, "retreat"), (DISCARD, "discard"), (CARD, "select_card")):
        if t in chosen_types:
            return name
    if END in chosen_types:
        return "end_turn"
    if chosen_types & {1, 2}:
        return "yes_no"
    if 0 in chosen_types:
        return "number"
    return "other"


_BASE_SIG = {"attack": 5.0, "pass_up_attack": 5.0, "play": 4.0, "energy_attach": 3.5, "evolve": 3.0,
             "ability": 2.5, "retreat": 2.5, "select_card": 2.0, "discard": 2.0, "end_turn": 2.0,
             "yes_no": 1.0, "number": 0.7, "go_first": 0.3, "other": 1.0}


def run_game(task):
    epid, seat, deck = task
    fn = os.path.join(REPLAYS, f"{epid}.json")
    try:
        d = json.load(open(fn, encoding="utf-8"))
    except Exception:
        return epid, [], 0, 0
    steps = d.get("steps") or []
    if deck is None:
        deck = _deck_of(d, seat)
    deck = deck or []
    recs = []
    n_dec = n_agree = 0
    cat_dec = Counter()
    cat_agree = Counter()
    for t in range(len(steps) - 1):
        e = steps[t][seat]
        if e.get("status") != "ACTIVE":
            continue
        obs = e.get("observation") or {}
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        if len(opts) < 2 or (sel.get("maxCount") or 0) < 1:
            continue
        pilot = steps[t + 1][seat].get("action")
        if not isinstance(pilot, list) or not pilot:
            continue
        n_dec += 1
        ours, src = _our_pick(obs, deck)
        agree = set(ours) == set(pilot)
        if agree:
            n_agree += 1
        chosen_types = {opts[i].get("type") for i in pilot if 0 <= i < len(opts)}
        menu_types = {o.get("type") for o in opts}
        cat = _category(chosen_types, menu_types, sel.get("context"))
        cat_dec[cat] += 1
        if agree:
            cat_agree[cat] += 1
            continue
        sig = _BASE_SIG.get(cat, 1.0)
        if ATTACK in menu_types:
            sig += 1.5
        cur = obs.get("current") or {}
        players = cur.get("players") or []
        yi = cur.get("yourIndex", 0)
        me = players[yi] if yi < len(players) else {}
        prize_left = len(me.get("prize") or [])
        sig += max(0.0, (6 - prize_left)) * 0.4    # later in the prize race = more pivotal
        labels = [_opt_label(o, obs, sel) for o in opts]
        recs.append({
            "episode": epid, "seat": seat, "step": t, "category": cat,
            "significance": round(sig, 2), "our_source": src,
            "menu_types": sorted(TYPE_NAMES.get(x, x) for x in menu_types),
            "pilot_pick": pilot, "our_pick": ours,
            "pilot_labels": [labels[i] for i in pilot if 0 <= i < len(labels)],
            "our_labels": [labels[i] for i in ours if 0 <= i < len(labels)],
            "options": labels,
            "board": _board(obs),
        })
    return epid, recs, n_dec, n_agree, dict(cat_dec), dict(cat_agree)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-games", type=int, default=60)
    ap.add_argument("--budget", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--agent", choices=["deployed", "heavy"], default="deployed")
    args = ap.parse_args()

    pilots = json.loads(PILOTS.read_text(encoding="utf-8"))
    # round-robin winning games across top pilots so we don't over-sample one pilot
    queues = [list(p.get("winning_episodes") or []) for p in pilots.get("pilots", [])]
    games = []
    i = 0
    while len(games) < args.max_games and any(queues):
        q = queues[i % len(queues)]
        if q:
            epid, seat = q.pop(0)
            games.append((epid, seat, None))
        i += 1
        if i > 100000:
            break
    print(f"imitation-gap: {len(games)} top-pilot winning games, agent={args.agent}, "
          f"budget={args.budget}s, workers={args.workers}", flush=True)

    all_recs = []
    tot_dec = tot_agree = 0
    cat_dec = Counter()
    cat_agree = Counter()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_winit,
                             initargs=(args.budget, args.agent)) as ex:
        futs = [ex.submit(run_game, g) for g in games]
        for f in as_completed(futs):
            done += 1
            try:
                epid, recs, n_dec, n_agree, cd, ca = f.result()
            except Exception as exc:
                print(f"  [{done}/{len(games)}] ERROR {exc!r}", flush=True)
                continue
            tot_dec += n_dec
            tot_agree += n_agree
            all_recs.extend(recs)
            for k, v in cd.items():
                cat_dec[k] += v
            for k, v in ca.items():
                cat_agree[k] += v
            print(f"  [{done}/{len(games)}] ep {epid}: {n_dec} decisions, {n_agree} agree, "
                  f"{len(recs)} disagreements (running agree {tot_agree}/{tot_dec}="
                  f"{(tot_agree/tot_dec*100 if tot_dec else 0):.0f}%)", flush=True)

    all_recs.sort(key=lambda r: -r["significance"])
    top = all_recs[: args.top_n]
    by_cat = {}
    for r in all_recs:
        by_cat.setdefault(r["category"], 0)
        by_cat[r["category"]] += 1

    payload = {
        "n_games": len(games), "budget": args.budget,
        "total_decisions": tot_dec, "agreements": tot_agree,
        "agreement_rate": round(tot_agree / tot_dec, 4) if tot_dec else None,
        "disagreements": len(all_recs),
        "disagreements_by_category": dict(sorted(by_cat.items(), key=lambda kv: -kv[1])),
        "category_agreement": {c: {"decisions": cat_dec[c], "agree": cat_agree[c],
                                   "rate": round(cat_agree[c] / cat_dec[c], 3) if cat_dec[c] else None}
                               for c in sorted(cat_dec, key=lambda c: -cat_dec[c])},
        "top_disagreements": top,
        "all_disagreements": all_recs,
    }
    out_path = OUT if args.agent == "deployed" else (ROOT / "data" / f"imitation_gap_{args.agent}.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n=== agreement {tot_agree}/{tot_dec} = "
          f"{(tot_agree/tot_dec*100 if tot_dec else 0):.1f}% over {len(games)} games (agent={args.agent}) ===", flush=True)
    print("per-category agreement (decisions / agree / rate):", flush=True)
    for cat in sorted(cat_dec, key=lambda c: -cat_dec[c]):
        r = cat_agree[cat] / cat_dec[cat] if cat_dec[cat] else 0
        print(f"  {cat:16s} {cat_dec[cat]:4d} / {cat_agree[cat]:4d} / {r*100:5.1f}%", flush=True)
    print(f"\nwrote {out_path}  (top {len(top)} disagreements carried with full context)", flush=True)


if __name__ == "__main__":
    main()
