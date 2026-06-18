"""H024-v2 Phase 2 (sim-free): action-CONDITIONED imitation dataset for ONE deck, from the winners
who piloted it. For each real decision a WINNER faced while playing the chosen deck, emit one row per
legal option with: root-state features, the action descriptor (type, played card id, decoded effects,
card stats, attack, target), and chosen=1 for the option the winner actually played. Grouped by
decision_id. This is the non-circular target (the winner's move, not the hand eval) and the
action-level inputs (root + action identity) the leaf-only data lacked.

Deck auto-picked as the one with the most winner-decisions in the corpus (override with --deck-rank K).
option_deltas (forward-model one-step consequence) ARE included. With --values, also cache the
frozen-teacher MULTI-TURN search VALUE per option (S.option_evals: finish my turn, opponent reply,
evaluate at the start of my next turn, averaged over N_DETERM worlds). That value is the teacher the
learned ranker is DISTILLED toward: each row gets `val` (search leaf value), `adv` (val centered
within the decision), and `bsrch` (1 if this option is the search argmax). --values is sim-heavy
(~0.6s/decision); the imitation-only build is sim-free for the static features.

    python tools/build_action_dataset.py            # top winner-decision deck
    python tools/build_action_dataset.py --deck-rank 2   # 2nd (e.g. DENPA92)
    python tools/build_action_dataset.py --player KanNinomiya --strategic-only --values  # distill set
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import features as FT  # noqa: E402
import search as S  # noqa: E402  (forward-model option_deltas: the consequence signal)
import main as M  # noqa: E402  (the deployed teacher = _forced_move THEN search.best_option)
import time

DELTA_KEYS = ["prizes_taken", "opp_prizes_taken", "opp_ko", "dmg_dealt", "cards_drawn",
              "energy_attached", "board_dev", "deck_used", "discard_gain", "ends_turn",
              "wins_now", "loses_now"]

CF = json.load(open(ROOT / "agent" / "card_features.json", encoding="utf-8"))
CE = json.load(open(ROOT / "agent" / "card_effects.json", encoding="utf-8"))
ATK = json.load(open(ROOT / "agent" / "attack_stats.json", encoding="utf-8"))
OUT = ROOT / "data" / "replay_db"
A_HAND = 2
EFFECT_KEYS = ["draw", "search", "search_to_bench", "energy_accel", "heal", "switch_gust",
               "recover_discard", "disrupt", "discard_cost", "status", "has_ability"]


def winner_deck(d, win):
    for s in d.get("steps", []):
        if win < len(s) and isinstance(s[win], dict):
            a = s[win].get("action")
            if isinstance(a, list) and len(a) == 60:
                return a
    return None


def opt_card_id(o, me_player):
    if o.get("area") == A_HAND:
        idx = o.get("index")
        hand = me_player.get("hand") or []
        if isinstance(idx, int) and 0 <= idx < len(hand):
            c = hand[idx]
            return (c.get("id") if isinstance(c, dict) else c)
    return None


def option_features(o, cur, me):
    cid = opt_card_id(o, (cur.get("players") or [{}])[me])
    cf = CF.get(str(cid), {}) if cid else {}
    ce = CE.get(str(cid), {}) if cid else {}
    t = o.get("type")
    a = ATK.get(str(o.get("attackId")), {}) if t == 13 else {}
    cost = a.get("c", 0)
    f = {
        "otype": t if isinstance(t, int) else -1,
        "card_id": int(cid) if cid else -1,
        "c_pokemon": 1 if cf.get("ct") == 0 else 0,
        "c_trainer": 1 if cf.get("ct") in (1, 2, 3, 4) else 0,
        "c_energy": 1 if cf.get("ct") in (5, 6) else 0,
        "c_basic": 1 if cf.get("stage") == "basic" else 0,
        "c_evo": 1 if cf.get("stage") in ("stage1", "stage2") else 0,
        "c_ex": 1 if (cf.get("ex") or cf.get("mega")) else 0,
        "c_hp": float(cf.get("hp", 0) or 0),
        "c_bestdmg": float(cf.get("best_dmg", 0) or 0),
        "a_dmg": float(a.get("d", 0) or 0),
        "a_cost": float(len(cost) if isinstance(cost, (list, str)) else (cost or 0)),
        "t_inplay": 1 if o.get("inPlayIndex") is not None else 0,
        "t_isopp": 1 if (o.get("playerIndex") not in (None, me)) else 0,
    }
    for k in EFFECT_KEYS:
        f["e_" + k] = float(ce.get(k, 0) or 0)
    return f


def opt_key(o, cur, me):
    """Canonical key: options with the same key are strategically equivalent (same move)."""
    cid = None
    if o.get("area") == A_HAND and isinstance(o.get("index"), int):
        hand = ((cur.get("players") or [{}])[me]).get("hand") or []
        if 0 <= o["index"] < len(hand):
            c = hand[o["index"]]
            cid = c.get("id") if isinstance(c, dict) else c
    return (o.get("type"), cid, o.get("attackId"), o.get("inPlayArea"), o.get("inPlayIndex"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck-rank", type=int, default=1, help="1=most winner-decisions, 2=next, ...")
    ap.add_argument("--player", default=None, help="filter to ONE winner player's decisions (clean policy)")
    ap.add_argument("--strategic-only", action="store_true", help="only decisions with >1 option type AND >1 distinct move")
    ap.add_argument("--values", action="store_true", help="also cache the frozen-teacher multi-turn search value per option (sim-heavy)")
    ap.add_argument("--label-determ", type=int, default=16, help="determinizations for OFFLINE label gen (no match budget; more = lower label noise)")
    ap.add_argument("--label-budget", type=float, default=4.0, help="per-decision sim budget for OFFLINE label gen (s); larger than the 0.6s match cap so all worlds run")
    args = ap.parse_args()
    if args.values:
        # the cached teacher MUST equal the deployed teacher (agent_search): opp_prior=None (same-deck),
        # opp_k=0 (1-ply), leaf_mode="hand". option_evals hardcodes exactly that, so the label matches
        # the agent. If the deployed teacher ever changes (belief/2-ply promoted), this label gen must too.
        S.N_DETERM = args.label_determ   # offline: average more hidden worlds than the match-time 8

    # pass 1: rank decks by winner-decisions
    stat = defaultdict(lambda: {"wdec": 0, "names": Counter(), "deck": None, "wins": 0})
    files = glob.glob(str(ROOT / "data" / "external" / "replays" / "*.json"))
    for fp in files:
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        rw = d.get("rewards") or []
        if len(rw) != 2 or rw[0] == rw[1] or None in rw:
            continue
        win = 0 if rw[0] > rw[1] else 1
        deck = winner_deck(d, win)
        if not deck:
            continue
        sig = tuple(sorted(deck))
        st = stat[sig]; st["deck"] = deck; st["wins"] += 1
        nm = [a.get("Name") for a in d.get("info", {}).get("Agents", [])]
        if win < len(nm):
            st["names"][nm[win]] += 1
        for s in d.get("steps", []):
            if win >= len(s) or not isinstance(s[win], dict):
                continue
            sel = (s[win].get("observation") or {}).get("select") or {}
            if (sel.get("maxCount") or 0) == 1 and len(sel.get("option") or []) >= 2 and len(s[win].get("action") or []) == 1:
                st["wdec"] += 1
    ranked = sorted(stat.values(), key=lambda x: -x["wdec"])
    chosen = ranked[args.deck_rank - 1]
    target_sig = tuple(sorted(chosen["deck"]))
    owner = chosen["names"].most_common(1)[0][0] if chosen["names"] else "?"
    print(f"deck rank {args.deck_rank}: {owner} | {chosen['wins']} win-games | {chosen['wdec']} winner-decisions")

    # pass 2: emit action-conditioned rows for winners playing the chosen deck
    rows = []
    gid = 0
    t0 = time.time()
    for fp in files:
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        rw = d.get("rewards") or []
        if len(rw) != 2 or rw[0] == rw[1] or None in rw:
            continue
        win = 0 if rw[0] > rw[1] else 1
        deck = winner_deck(d, win)
        if not deck or tuple(sorted(deck)) != target_sig:
            continue
        if args.player:
            nm = [a.get("Name") for a in d.get("info", {}).get("Agents", [])]
            if not (win < len(nm) and nm[win] == args.player):
                continue
        for s in d.get("steps", []):
            if win >= len(s) or not isinstance(s[win], dict):
                continue
            obs = s[win].get("observation") or {}
            sel = obs.get("select") or {}
            opts = sel.get("option") or []
            act = s[win].get("action") or []
            cur = obs.get("current") or {}
            if (sel.get("maxCount") or 0) != 1 or len(opts) < 2 or len(act) != 1 or not cur.get("players"):
                continue
            chosen_idx = act[0]
            if not (isinstance(chosen_idx, int) and 0 <= chosen_idx < len(opts)):
                continue
            me = cur.get("yourIndex", win)
            keys = [opt_key(o, cur, me) if isinstance(o, dict) else None for o in opts]
            uniq = {}
            for k in keys:
                if k not in uniq:
                    uniq[k] = len(uniq)
            types = {o.get("type") for o in opts if isinstance(o, dict)}
            strategic = len(types) > 1 and len(uniq) > 1
            if args.strategic_only and not strategic:
                continue
            try:
                root = FT.vectorize(FT.encode_state(obs))
            except Exception:
                continue
            # forward-model one-step consequence per option (shared determinization within the decision)
            try:
                deltas = S.option_deltas(obs, deck)
            except Exception:
                deltas = None
            # frozen-teacher multi-turn search value per option (distillation target) -- sim-heavy
            vals = None
            dict_idxs = [j for j, o in enumerate(opts) if isinstance(o, dict)]
            if args.values:
                try:
                    ev = S.option_evals(obs, deck, time_budget=args.label_budget)   # list[option] of (mean_val, mean_feats) or None
                except Exception:
                    ev = None
                if ev:
                    vals = [ev[j][0] if (j < len(ev) and ev[j]) else None for j in range(len(opts))]
                    # require FULL coverage: if any sibling failed to simulate, the centering baseline and
                    # the argmax would be over an inconsistent subset (a silent selection effect) -> no label
                    if not all(vals[j] is not None for j in dict_idxs):
                        vals = None
            present = [v for v in (vals or []) if v is not None]
            mean_v = sum(present) / len(present) if present else 0.0
            best_eq = None
            if present:
                bj = max((j for j in range(len(vals)) if vals[j] is not None), key=lambda j: vals[j])
                best_eq = keys[bj]                   # canonical: the eq-class the search prefers
                # the DEPLOYED teacher applies _forced_move BEFORE search (lethal/KO attack, go-first);
                # match it so bsrch == the move agent_search would actually make (esp. high-crit/lethal).
                try:
                    fm = M._forced_move(obs)
                    if fm is not None and 0 <= fm[0] < len(keys):
                        best_eq = keys[fm[0]]
                except Exception:
                    pass
            for j, o in enumerate(opts):
                if not isinstance(o, dict):
                    continue
                af = option_features(o, cur, me)
                dd = deltas[j] if (deltas and j < len(deltas) and deltas[j]) else {}
                for k in DELTA_KEYS:
                    af["d_" + k] = float(dd.get(k, 0.0))
                af["has_delta"] = 1 if dd else 0
                row = {"gid": gid, "chosen": 1 if j == chosen_idx else 0,
                       "dev": 1 if chosen_idx != 0 else 0, "eq": uniq[keys[j]],
                       "strat": 1 if strategic else 0, "root": root, **af}
                if vals is not None:
                    vj = vals[j]
                    row["has_val"] = 1 if vj is not None else 0
                    row["val"] = float(vj) if vj is not None else 0.0
                    row["adv"] = float(vj - mean_v) if vj is not None else 0.0
                    row["bsrch"] = 1 if (best_eq is not None and keys[j] == best_eq) else 0
                rows.append(row)
            gid += 1
            if gid % 1000 == 0:
                print(f"  [build] {gid} decisions, {len(rows)} rows, {time.time()-t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / ("action_adv.jsonl" if args.values else "action_imit.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} option-rows over {gid} decisions -> {out.relative_to(ROOT)}")
    print(f"  avg options/decision: {len(rows)/max(1,gid):.1f}; root dim {len(rows[0]['root']) if rows else 0}; "
          f"action feats: otype, card_id, {len(EFFECT_KEYS)} effects, card stats, attack, target")
    if args.values:
        nv = sum(1 for r in rows if r.get("has_val"))
        gv = len({r["gid"] for r in rows if r.get("has_val")})
        print(f"  search values cached on {nv} rows / {gv} decisions (the distillation target: val/adv/bsrch)")


if __name__ == "__main__":
    main()
