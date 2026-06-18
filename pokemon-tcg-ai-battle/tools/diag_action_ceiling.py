"""Imitation-learnability / feature-ceiling diagnostic.

Question (not "are they heuristic"): can a model trained on the WINNING players' actual moves
reproduce them, and is the ceiling set by our representation? We build a WITHIN-DECISION ranker:
for each real single-select decision the eventual winner faced, score every legal option and check
whether the model's top-ranked option is the one the winner actually chose (top-1 accuracy).

Why this matters: our 47 features are STATE-level (identical across the options of one decision), so
they have ZERO within-decision discriminative power by construction. The only way to rank moves is
ACTION-level features decoded from each option. This script decodes what the engine exposes per
option (type; attack damage/cost via attack_stats; in-play target HP; and, where joinable, the
hand card's class/stage) and ablates feature richness to localize where imitation signal lives.

Baselines top-1 must beat: random (1/n_options), most-frequent-winner-type, and our heuristic
(M.agent re-run on the same obs -- a FLOOR, NOT a claim that opponents use our heuristic).

    python tools/diag_action_ceiling.py [--max-games 200]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import main as M  # our heuristic + ATK/CDB

CF = json.load(open(ROOT / "agent" / "card_features.json", encoding="utf-8"))
ATK = M.ATK
OPT_TYPES = [0, 3, 5, 6, 7, 8, 9, 10, 13, 14, 15]  # OptionType values we one-hot


def _card(cid):
    return CF.get(str(cid), {})


def _inplay(cur, pidx, area, idx):
    """HP of the targeted in-play Pokemon, if recoverable (area 4=active-ish, else bench)."""
    try:
        p = (cur.get("players") or [])[pidx]
        if area == 4 or area == 1:
            slot = (p.get("active") or [None])[0]
        else:
            slot = (p.get("bench") or [])[idx]
        if isinstance(slot, dict):
            return float(slot.get("hp", 0) or 0), float(slot.get("damage", 0) or 0)
    except Exception:
        pass
    return 0.0, 0.0


def option_features(o, cur, me, levels):
    """Decode one option -> dict of features at the requested richness `levels` (set)."""
    f = {}
    t = o.get("type")
    if "type" in levels:
        for ot in OPT_TYPES:
            f[f"t{ot}"] = 1.0 if t == ot else 0.0
    if "attack" in levels:
        a = ATK.get(str(o.get("attackId")), {}) if t == 13 else {}
        f["atk_dmg"] = float(a.get("d", 0) or 0)
        cost = a.get("c", 0)
        f["atk_cost"] = float(len(cost) if isinstance(cost, (list, str)) else (cost or 0))
        f["is_attack"] = 1.0 if t == 13 else 0.0
    if "target" in levels:
        hp, dmg = (0.0, 0.0)
        if o.get("inPlayIndex") is not None:
            hp, dmg = _inplay(cur, o.get("playerIndex", me), o.get("inPlayArea"), o.get("inPlayIndex"))
        f["tgt_hp"] = hp
        f["tgt_dmg_on"] = dmg
        f["tgt_is_opp"] = 1.0 if o.get("playerIndex") not in (None, me) else 0.0
    if "interact" in levels:
        opp = 1 - me  # two-player
        oa = ((cur.get("players") or [{}, {}])[opp]).get("active") or [None]
        oa = oa[0] if oa else None
        opp_hp_left = (float(oa.get("hp", 0) or 0) - float(oa.get("damage", 0) or 0)) if isinstance(oa, dict) else 0.0
        dmg = float(ATK.get(str(o.get("attackId")), {}).get("d", 0) or 0) if t == 13 else 0.0
        f["opp_active_hp_left"] = opp_hp_left
        f["ko_opp"] = 1.0 if (dmg > 0 and opp_hp_left > 0 and dmg >= opp_hp_left) else 0.0
        f["dmg_vs_hp"] = (dmg / opp_hp_left) if opp_hp_left > 0 else 0.0
    if "card" in levels:
        # CORRECT join (was dead code: replay hand entries are dicts {'id':..}, not ints).
        # Use AreaType.HAND==2 + index -> hand[idx]['id'] (the reliable key; option.cardId is null).
        cid = None
        idx = o.get("index")
        if o.get("area") == 2 and isinstance(idx, int):
            hand = ((cur.get("players") or [{}])[me]).get("hand") or []
            if 0 <= idx < len(hand) and isinstance(hand[idx], dict):
                cid = hand[idx].get("id")
        c = _card(cid) if cid else {}
        f["card_pokemon"] = 1.0 if c.get("ct") == 0 else 0.0
        f["card_trainer"] = 1.0 if c.get("ct") in (1, 2, 3, 4) else 0.0
        f["card_energy"] = 1.0 if c.get("ct") in (5, 6) else 0.0
        f["card_basic"] = 1.0 if c.get("stage") == "basic" else 0.0
        f["card_evo"] = 1.0 if c.get("stage") in ("stage1", "stage2") else 0.0
        f["card_ex"] = 1.0 if c.get("ex") or c.get("mega") else 0.0
        f["card_bestdmg"] = float(c.get("best_dmg", 0) or 0)
    return f


def collect(max_games):
    rows = []  # (gid, option, current, me, label, n_options)
    obs_by_gid = {}  # gid -> obs (for heuristic baseline)
    chosen_by_gid = {}  # gid -> chosen option position
    whichcard_by_gid = {}  # gid -> True if all options share one OptionType (a pure "which card" pick)
    gid = 0
    files = sorted(glob.glob(str(ROOT / "data" / "external" / "replays" / "*.json")))
    used = 0
    for fp in files:
        if used >= max_games:
            break
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
        used += 1
        for s in d.get("steps", []):
            if not isinstance(s, list) or win >= len(s) or not isinstance(s[win], dict):
                continue
            ag = s[win]
            obs = ag.get("observation") or {}
            sel = obs.get("select") or {}
            opts = sel.get("option") or []
            act = ag.get("action") or []
            cur = obs.get("current") or {}
            if len(opts) < 2 or len(act) != 1 or sel.get("maxCount") != 1:
                continue  # single-select real decisions only
            chosen = act[0]
            if not (isinstance(chosen, int) and 0 <= chosen < len(opts)):
                continue
            me = cur.get("yourIndex", win)
            obs_by_gid[gid] = obs
            chosen_by_gid[gid] = chosen
            types = {o.get("type") for o in opts if isinstance(o, dict)}
            whichcard_by_gid[gid] = (len(types) == 1)
            for j, o in enumerate(opts):
                if isinstance(o, dict):
                    rows.append((gid, o, cur, me, 1 if j == chosen else 0, len(opts)))
            gid += 1
    return rows, obs_by_gid, chosen_by_gid, whichcard_by_gid


def vectorize(rows, levels):
    keys = sorted({k for (_, o, cur, me, _, _) in rows for k in option_features(o, cur, me, levels)})
    X, y, g, n = [], [], [], []
    for (gid, o, cur, me, lab, no) in rows:
        f = option_features(o, cur, me, levels)
        X.append([f.get(k, 0.0) for k in keys])
        y.append(lab); g.append(gid); n.append(no)
    return np.array(X), np.array(y), np.array(g), np.array(n), keys


def top1(model, X, y, g, subset=None):
    """Fraction of decisions whose argmax-scored option is the winner's chosen one.
    subset: optional set of gids to restrict to (for stratified reporting)."""
    p = model.predict_proba(X)[:, 1]
    hit = tot = 0
    for gid in np.unique(g):
        if subset is not None and gid not in subset:
            continue
        m = g == gid
        if y[m].sum() != 1:
            continue
        tot += 1
        if np.argmax(p[m]) == np.argmax(y[m]):
            hit += 1
    return hit / tot if tot else 0.0, tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-games", type=int, default=200)
    args = ap.parse_args()
    from sklearn.ensemble import GradientBoostingClassifier

    rows, obs_by_gid, chosen_by_gid, whichcard_by_gid = collect(args.max_games)
    gids = sorted(set(r[0] for r in rows))
    n_dec = len(gids)
    wc_gids = {gid for gid in gids if whichcard_by_gid.get(gid)}
    mixed_gids = {gid for gid in gids if not whichcard_by_gid.get(gid)}
    print(f"{n_dec} single-select winner decisions, {len(rows)} options "
          f"(avg {len(rows)/max(1,n_dec):.1f} options/decision)")
    print(f"  which-card (all options same type): {len(wc_gids)}  |  mixed strategic: {len(mixed_gids)}")

    # baselines
    import statistics
    n_by_gid = {gid: 0 for gid in gids}
    for r in rows:
        n_by_gid[r[0]] += 1
    rand = statistics.mean(1.0 / n for n in n_by_gid.values())
    opt0 = statistics.mean(1.0 if chosen_by_gid[gid] == 0 else 0.0 for gid in gids)
    print(f"\nBASELINE top-1:")
    print(f"  random (1/n_options)          : {rand:.3f}")
    print(f"  chose-option-0 (positional)   : {opt0:.3f}   <- engine option-ordering prior")

    # heuristic floor: re-run M.agent on each decision's obs, compare its pick to the winner's
    hk = ht = 0
    for gid, obs in obs_by_gid.items():
        try:
            ours = M.agent(obs)
        except Exception:
            continue
        ht += 1
        if isinstance(ours, list) and ours and ours[0] == chosen_by_gid[gid]:
            hk += 1
    print(f"  our heuristic (floor, n={ht})    : {hk/ht:.3f}" if ht else "  heuristic: n/a")

    # game-wise split
    rng = np.random.default_rng(0)
    gid_arr = np.array(gids)
    rng.shuffle(gid_arr)
    cut = int(0.7 * len(gid_arr))
    train_g, test_g = set(gid_arr[:cut].tolist()), set(gid_arr[cut:].tolist())

    ladders = [("type", {"type"}),
               ("+card", {"type", "card"}),
               ("+attack", {"type", "card", "attack"}),
               ("+target", {"type", "card", "attack", "target"}),
               ("+interact(KO)", {"type", "card", "attack", "target", "interact"})]
    print(f"\nWITHIN-DECISION IMITATION RANKER top-1 (game-wise 70/30, GBM): ALL | which-card | mixed")
    for name, levels in ladders:
        X, y, g, n, keys = vectorize(rows, levels)
        tr = np.isin(g, list(train_g)); te = np.isin(g, list(test_g))
        clf = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0)
        clf.fit(X[tr], y[tr])
        acc_all, n_all = top1(clf, X[te], y[te], g[te])
        acc_wc, n_wc = top1(clf, X[te], y[te], g[te], subset=wc_gids)
        acc_mx, n_mx = top1(clf, X[te], y[te], g[te], subset=mixed_gids)
        print(f"  {name:>14} ({len(keys):>2} feats): {acc_all:.3f} (n={n_all}) | "
              f"wc {acc_wc:.3f} (n={n_wc}) | mixed {acc_mx:.3f} (n={n_mx})")
    print("\nRead: stratified. 'which-card' = all options same type (pure card choice); 'mixed' = the")
    print("hard strategic decisions. Compare mixed top-1 to the chose-option-0 positional baseline,")
    print("not to random. A pointwise GBM understates ranking; listwise loss is the next step.")


if __name__ == "__main__":
    main()
