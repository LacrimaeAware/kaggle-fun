"""Forward-model extension of diag_action_ceiling.py.

Question being tested (the user's hypothesis): is the ~0.5 imitation plateau caused by our STATIC
action representation, such that SIMULATING each option in the real forward model and featuring the
resulting state lifts within-decision top-1 imitation accuracy of the winners' actual moves?

For each single-select decision the EVENTUAL WINNER faced (same selection as diag_action_ceiling),
we run search.option_evals(obs) which, per legal option, plays that option through the competition's
own forward model (cg.api search_begin/search_step), finishes the turn + opponent reply with the
default rollout, and returns (mean_leaf_value, mean_leaf_state_features[47]) over N_DETERM worlds.
The replay carries the true serialized hidden state (search_begin_input), so the simulated board is
the REAL board, not a guessed determinization, for the searching player's own line.

Rankers evaluated (top-1 = model argmax option == winner's actual pick), game-wise 70/30:
  - random, heuristic floor                       : same baselines as diag_action_ceiling
  - STATIC: type+card+attack+target+interact GBM   : reproduces the static ceiling
  - FWD-VALUE argmax (no training)                 : pick the option with the best simulated leaf value
  - FWD-DELTA GBM                                  : GBM on (leaf_state - root_state) 47-dim delta
  - FWD-VALUE GBM                                  : GBM on the scalar simulated leaf value
  - STATIC + FWD (delta+value) GBM                 : everything combined

If any forward-model ranker rises clearly above the static ceiling AND the heuristic floor, the
plateau was a representation problem and simulation-based action features are the fix. If they sit
at the same plateau, the ceiling is not the static features -- it is that the engine's rollout after
the first move does not reproduce the winner's continuation, so the simulated leaf value does not
identify the winner's move.

    python tools/diag_action_fwd.py [--max-games 120]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# sample_submission has its own main.py; insert it FIRST so agent/main.py (inserted last -> index 0)
# wins for `import main`, while cg.api still resolves from the sample dir's cg/ subpackage.
sys.path.insert(0, str(ROOT / "data" / "external" / "official" / "sample_submission"))
sys.path.insert(0, str(ROOT / "agent"))
import main as M  # noqa: E402  (heuristic floor)
import search as S  # noqa: E402  (forward-model option_evals)
import features as FT  # noqa: E402

CF = json.load(open(ROOT / "agent" / "card_features.json", encoding="utf-8"))
ATK = M.ATK
OPT_TYPES = [0, 3, 5, 6, 7, 8, 9, 10, 13, 14, 15]
DECK = [3] * 60  # the serialized hidden state in the replay drives the sim; deck is a count-filler


# ---- static option features (lifted verbatim from diag_action_ceiling.option_features) ----
def _card(cid):
    return CF.get(str(cid), {})


def _inplay(cur, pidx, area, idx):
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


def static_features(o, cur, me):
    f = {}
    t = o.get("type")
    for ot in OPT_TYPES:
        f[f"t{ot}"] = 1.0 if t == ot else 0.0
    a = ATK.get(str(o.get("attackId")), {}) if t == 13 else {}
    f["atk_dmg"] = float(a.get("d", 0) or 0)
    cost = a.get("c", 0)
    f["atk_cost"] = float(len(cost) if isinstance(cost, (list, str)) else (cost or 0))
    f["is_attack"] = 1.0 if t == 13 else 0.0
    hp, dmg = (0.0, 0.0)
    if o.get("inPlayIndex") is not None:
        hp, dmg = _inplay(cur, o.get("playerIndex", me), o.get("inPlayArea"), o.get("inPlayIndex"))
    f["tgt_hp"] = hp
    f["tgt_dmg_on"] = dmg
    f["tgt_is_opp"] = 1.0 if o.get("playerIndex") not in (None, me) else 0.0
    opp = 1 - me
    oa = ((cur.get("players") or [{}, {}])[opp]).get("active") or [None]
    oa = oa[0] if oa else None
    opp_hp_left = (float(oa.get("hp", 0) or 0) - float(oa.get("damage", 0) or 0)) if isinstance(oa, dict) else 0.0
    dmg2 = float(ATK.get(str(o.get("attackId")), {}).get("d", 0) or 0) if t == 13 else 0.0
    f["opp_active_hp_left"] = opp_hp_left
    f["ko_opp"] = 1.0 if (dmg2 > 0 and opp_hp_left > 0 and dmg2 >= opp_hp_left) else 0.0
    f["dmg_vs_hp"] = (dmg2 / opp_hp_left) if opp_hp_left > 0 else 0.0
    cid = None
    idx = o.get("index")
    hand = ((cur.get("players") or [{}])[me]).get("hand") or []
    if t in (3, 5, 6, 7) and isinstance(idx, int) and 0 <= idx < len(hand) and isinstance(hand[idx], int):
        cid = hand[idx]
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
    """Returns per-decision records with static features, the root-state vector, and (lazily) the
    forward-model option evals. One pass; the heavy sim is done here so it is shared across ladders."""
    decisions = []  # each: dict(gid, chosen, n, static[list per opt], root_vec, fwd[list per opt or None])
    obs_by_gid = {}
    chosen_by_gid = {}
    files = sorted(glob.glob(str(ROOT / "data" / "external" / "replays" / "*.json")))
    used = gid = 0
    sim_ok = sim_fail = 0
    t0 = time.time()
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
                continue
            chosen = act[0]
            if not (isinstance(chosen, int) and 0 <= chosen < len(opts)):
                continue
            me = cur.get("yourIndex", win)
            stat = [static_features(o, cur, me) if isinstance(o, dict) else {} for o in opts]
            root_vec = np.array(FT.vectorize(FT.encode_state(obs)), dtype=float)
            fwd = None
            try:
                oe = S.option_evals(obs, DECK, time_budget=0.6)
                if oe is not None and len(oe) == len(opts):
                    fwd = oe  # list per option of (val, [47 feats]) or None
                    sim_ok += 1
                else:
                    sim_fail += 1
            except Exception:
                sim_fail += 1
            decisions.append(dict(gid=gid, chosen=chosen, n=len(opts),
                                  static=stat, root=root_vec, fwd=fwd))
            obs_by_gid[gid] = obs
            chosen_by_gid[gid] = chosen
            gid += 1
    print(f"collected {gid} decisions from {used} games in {time.time()-t0:.0f}s; "
          f"forward-model sim applicable on {sim_ok}/{sim_ok+sim_fail} decisions")
    return decisions, obs_by_gid, chosen_by_gid


def top1_from_scores(scores_by_gid, chosen_by_gid, gids):
    hit = tot = 0
    for gid in gids:
        sc = scores_by_gid.get(gid)
        if sc is None:
            continue
        tot += 1
        if int(np.argmax(sc)) == chosen_by_gid[gid]:
            hit += 1
    return (hit / tot if tot else 0.0), tot


def build_matrix(decisions, mode, fwd_only_subset):
    """Flatten decisions into (X, y, g) for a within-decision GBM. mode selects feature set.
    fwd_only_subset: if True, only include decisions where forward sim succeeded (apples-to-apples)."""
    static_keys = sorted({k for d in decisions for f in d["static"] for k in f})
    X, y, g = [], [], []
    for d in decisions:
        if fwd_only_subset and d["fwd"] is None:
            continue
        for j in range(d["n"]):
            row = []
            if mode in ("static", "all"):
                f = d["static"][j]
                row += [f.get(k, 0.0) for k in static_keys]
            if mode in ("delta", "value", "all"):
                fv = d["fwd"][j] if d["fwd"] is not None else None
                if mode in ("delta", "all"):
                    if fv is not None and fv[1] is not None:
                        row += list(np.array(fv[1], dtype=float) - d["root"])
                    else:
                        row += [0.0] * len(d["root"])
                if mode in ("value", "all"):
                    row.append(float(fv[0]) if fv is not None else 0.0)
            X.append(row)
            y.append(1 if j == d["chosen"] else 0)
            g.append(d["gid"])
    return np.array(X, dtype=float), np.array(y), np.array(g)


def gbm_top1(decisions, mode, train_g, test_g, fwd_only_subset):
    from sklearn.ensemble import GradientBoostingClassifier
    X, y, g = build_matrix(decisions, mode, fwd_only_subset)
    if X.size == 0:
        return 0.0, 0, 0
    tr = np.isin(g, list(train_g))
    te = np.isin(g, list(test_g))
    if tr.sum() == 0 or te.sum() == 0:
        return 0.0, 0, X.shape[1]
    clf = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0)
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    hit = tot = 0
    for gid in np.unique(g[te]):
        m = g[te] == gid
        if y[te][m].sum() != 1:
            continue
        tot += 1
        if int(np.argmax(p[m])) == int(np.argmax(y[te][m])):
            hit += 1
    return (hit / tot if tot else 0.0), tot, X.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-games", type=int, default=120)
    args = ap.parse_args()

    decisions, obs_by_gid, chosen_by_gid = collect(args.max_games)
    gids = [d["gid"] for d in decisions]
    fwd_gids = set(d["gid"] for d in decisions if d["fwd"] is not None)

    # --- baselines on ALL decisions ---
    import statistics
    rand = statistics.mean(1.0 / d["n"] for d in decisions)
    print(f"\nBASELINE top-1 (all {len(decisions)} decisions):")
    print(f"  random (1/n_options)            : {rand:.3f}")
    hk = ht = 0
    for gid, obs in obs_by_gid.items():
        try:
            ours = M.agent(obs)
        except Exception:
            continue
        ht += 1
        if isinstance(ours, list) and ours and ours[0] == chosen_by_gid[gid]:
            hk += 1
    print(f"  our heuristic (floor)           : {hk/ht:.3f} (n={ht})")

    # --- forward-model argmax leaf value (NO training), on the fwd-applicable subset ---
    val_scores = {}
    for d in decisions:
        if d["fwd"] is None:
            continue
        sc = np.array([(fv[0] if fv is not None else -1e18) for fv in d["fwd"]], dtype=float)
        val_scores[d["gid"]] = sc
    acc_val, n_val = top1_from_scores(val_scores, chosen_by_gid, fwd_gids)
    # random + heuristic restricted to the SAME fwd subset for a fair comparison
    rand_sub = statistics.mean(1.0 / d["n"] for d in decisions if d["fwd"] is not None) if fwd_gids else 0.0
    hk2 = ht2 = 0
    for gid in fwd_gids:
        try:
            ours = M.agent(obs_by_gid[gid])
        except Exception:
            continue
        ht2 += 1
        if isinstance(ours, list) and ours and ours[0] == chosen_by_gid[gid]:
            hk2 += 1
    print(f"\nFORWARD-MODEL subset ({len(fwd_gids)} decisions where sim applies):")
    print(f"  random (subset)                 : {rand_sub:.3f}")
    print(f"  heuristic (subset)              : {hk2/ht2:.3f} (n={ht2})" if ht2 else "  heuristic: n/a")
    print(f"  FWD argmax leaf VALUE (no train): {acc_val:.3f} (n={n_val})")

    # --- GBM ladders, game-wise 70/30 ---
    rng = np.random.default_rng(0)
    gid_arr = np.array(sorted(set(gids)))
    rng.shuffle(gid_arr)
    cut = int(0.7 * len(gid_arr))
    train_g, test_g = set(gid_arr[:cut].tolist()), set(gid_arr[cut:].tolist())

    print(f"\nWITHIN-DECISION GBM top-1 (game-wise 70/30):")
    # static ceiling on ALL decisions (matches diag_action_ceiling's richest rung)
    acc, tot, dim = gbm_top1(decisions, "static", train_g, test_g, fwd_only_subset=False)
    print(f"  STATIC (all, {dim:>3} feats)        : {acc:.3f} (n={tot})")
    # on the fwd subset, compare static vs fwd features apples-to-apples
    print(f"  -- on the forward-model subset (same decisions, fair compare) --")
    for name, mode in [("STATIC", "static"), ("FWD-DELTA(47)", "delta"),
                       ("FWD-VALUE(1)", "value"), ("STATIC+FWD", "all")]:
        acc, tot, dim = gbm_top1(decisions, mode, train_g, test_g, fwd_only_subset=True)
        print(f"  {name:>16} ({dim:>3} feats): {acc:.3f} (n={tot})")

    print("\nRead: a forward-model ranker that clears BOTH the heuristic floor and the static GBM on")
    print("the same subset means simulation features break the plateau. If they all cluster together,")
    print("the rollout-after-move (not the static representation) is the binding ceiling.")


if __name__ == "__main__":
    main()
