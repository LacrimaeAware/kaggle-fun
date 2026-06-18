"""Gate 2: do IMMEDIATE forward-model consequence deltas carry within-decision imitation signal that
the static features lacked? For each single-select decision the EVENTUAL WINNER faced, run
search.option_deltas(obs, REAL_DECK) -- one engine step per option, NO rollout -- and rank options by
a grouped listwise model over the delta features. Compare top-1 to the chose-option-0 baseline,
stratified into which-card vs mixed strategic decisions.

This is the clean Gate-2 test from the Codex review (immediate one-step delta), distinct from
diag_action_fwd.py which uses the full-rollout option_evals. Decks come from each replay (the winner's
60-card deck-selection action), NOT the agent's own DECK -- determinizing a replay with the wrong deck
is invalid.

    python tools/diag_action_delta.py [--max-games 80]
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
sys.path.insert(0, str(ROOT / "agent"))
import main as M  # noqa: E402  heuristic floor
import search as S  # noqa: E402  option_deltas

DELTA_KEYS = ["prizes_taken", "opp_prizes_taken", "opp_ko", "dmg_dealt", "cards_drawn",
              "energy_attached", "board_dev", "deck_used", "discard_gain", "ends_turn",
              "wins_now", "loses_now"]


def winner_deck(d, win):
    for s in d.get("steps", []):
        if win < len(s) and isinstance(s[win], dict):
            a = s[win].get("action")
            if isinstance(a, list) and len(a) == 60:
                return a
    return None


def relative_feats(deltas):
    """Within-decision comparative flags (the question is comparative). deltas: list per option (dict|None)."""
    def col(k):
        return [(dd.get(k, 0.0) if dd else -1e9) for dd in deltas]
    dmg = col("dmg_dealt"); draw = col("cards_drawn"); ko = col("opp_ko"); prize = col("prizes_taken")
    out = []
    for j, dd in enumerate(deltas):
        if dd is None:
            out.append(None); continue
        out.append({
            "is_max_dmg": 1.0 if dmg[j] == max(dmg) and dmg[j] > 0 else 0.0,
            "is_only_ko": 1.0 if ko[j] > 0 and sum(1 for x in ko if x > 0) == 1 else 0.0,
            "is_any_ko": 1.0 if ko[j] > 0 else 0.0,
            "is_max_draw": 1.0 if draw[j] == max(draw) and draw[j] > 0 else 0.0,
            "takes_prize": 1.0 if prize[j] > 0 else 0.0,
        })
    return out


def collect(max_games):
    rows = []      # (gid, feat_dict, label, n)
    chosen_by_gid, whichcard_by_gid, obs_by_gid = {}, {}, {}
    gid = 0
    sim_ok = sim_skip = 0
    t0 = time.time()
    for fp in sorted(glob.glob(str(ROOT / "data" / "external" / "replays" / "*.json"))):
        if max_games and _games_used[0] >= max_games:
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
        deck = winner_deck(d, win)
        if not deck:
            continue
        _games_used[0] += 1
        if max_games and _games_used[0] > max_games:
            break
        for s in d.get("steps", []):
            if win >= len(s) or not isinstance(s[win], dict):
                continue
            ag = s[win]; obs = ag.get("observation") or {}
            sel = obs.get("select") or {}; opts = sel.get("option") or []; act = ag.get("action") or []
            if (sel.get("maxCount") or 0) != 1 or len(opts) < 2 or len(act) != 1:
                continue
            chosen = act[0]
            if not (isinstance(chosen, int) and 0 <= chosen < len(opts)):
                continue
            deltas = S.option_deltas(obs, deck)
            if deltas is None or len(deltas) != len(opts):
                sim_skip += 1
                continue
            sim_ok += 1
            rel = relative_feats(deltas)
            for j in range(len(opts)):
                dd = deltas[j]
                if dd is None:
                    continue
                f = {k: float(dd.get(k, 0.0)) for k in DELTA_KEYS}
                f.update(rel[j] or {})
                rows.append((gid, f, 1 if j == chosen else 0, len(opts)))
            chosen_by_gid[gid] = chosen
            whichcard_by_gid[gid] = (len({o.get("type") for o in opts if isinstance(o, dict)}) == 1)
            obs_by_gid[gid] = obs
            gid += 1
    print(f"collected {gid} decisions ({sim_ok} sim-ok, {sim_skip} skipped) from {_games_used[0]} "
          f"games in {time.time()-t0:.0f}s")
    return rows, chosen_by_gid, whichcard_by_gid, obs_by_gid


_games_used = [0]


def _group_sizes(g_sub):
    sizes, cur, c = [], None, 0
    for x in g_sub:
        if x == cur:
            c += 1
        else:
            if cur is not None:
                sizes.append(c)
            cur, c = x, 1
    if cur is not None:
        sizes.append(c)
    return sizes


def topk(p, y, g, subset=None, k=1):
    hit = tot = 0
    for gid in np.unique(g):
        if subset is not None and gid not in subset:
            continue
        m = g == gid
        if y[m].sum() != 1:
            continue
        tot += 1
        if np.argmax(y[m]) in np.argsort(-p[m])[:k]:
            hit += 1
    return (hit / tot if tot else 0.0), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-games", type=int, default=80)
    args = ap.parse_args()
    import statistics
    rows, chosen, whichcard, obs_by = collect(args.max_games)
    if not rows:
        print("no rows"); return
    gids = sorted(set(r[0] for r in rows))
    wc = {g for g in gids if whichcard.get(g)}
    mx = {g for g in gids if not whichcard.get(g)}
    keys = sorted({k for (_, f, _, _) in rows for k in f})
    X = np.array([[f.get(k, 0.0) for k in keys] for (_, f, _, _) in rows])
    y = np.array([r[2] for r in rows]); g = np.array([r[0] for r in rows])
    nbg = {gg: 0 for gg in gids}
    for r in rows:
        nbg[r[0]] += 1

    def opt0(sub):
        s = [gg for gg in gids if gg in sub]
        return statistics.mean(1.0 if chosen[gg] == 0 else 0.0 for gg in s) if s else 0.0
    print(f"\n{len(gids)} decisions | which-card {len(wc)} | mixed {len(mx)} | {len(keys)} delta feats")
    print(f"BASELINE chose-option-0: all {opt0(set(gids)):.3f} | wc {opt0(wc):.3f} | mixed {opt0(mx):.3f}  <- bar to beat")
    hk = ht = 0
    for gg in gids:
        try:
            o = M.agent(obs_by[gg])
        except Exception:
            continue
        ht += 1; hk += 1 if (isinstance(o, list) and o and o[0] == chosen[gg]) else 0
    print(f"BASELINE heuristic: {hk/ht:.3f} (n={ht})")

    rng = np.random.default_rng(0); ga = np.array(gids); rng.shuffle(ga)
    cut = int(0.7 * len(ga)); tr_g, te_g = set(ga[:cut].tolist()), set(ga[cut:].tolist())
    tr = np.isin(g, list(tr_g)); te = np.isin(g, list(te_g))
    try:
        from lightgbm import LGBMRanker
        rk = LGBMRanker(objective="lambdarank", n_estimators=300, num_leaves=31, learning_rate=0.05,
                        min_child_samples=20, random_state=0, verbose=-1)
        rk.fit(X[tr], y[tr], group=_group_sizes(g[tr]))
        p = rk.predict(X[te])
        a_all, n_all = topk(p, y[te], g[te], k=1)
        a_wc, _ = topk(p, y[te], g[te], subset=wc, k=1)
        a_mx, n_mx = topk(p, y[te], g[te], subset=mx, k=1)
        t3, _ = topk(p, y[te], g[te], k=3); t3m, _ = topk(p, y[te], g[te], subset=mx, k=3)
        print(f"\nLISTWISE on DELTA feats: top-1 {a_all:.3f} (n={n_all}) | wc {a_wc:.3f} | mixed {a_mx:.3f} (n={n_mx})"
              f"  || top-3 {t3:.3f} | top-3 mixed {t3m:.3f}")
        imp = sorted(zip(keys, rk.feature_importances_), key=lambda x: -x[1])[:8]
        print("  top delta features by gain:", [(k, int(v)) for k, v in imp])
        print(f"\nGATE 2 verdict: delta-mixed {a_mx:.3f} vs option-0-mixed {opt0(mx):.3f} -> "
              f"{'PASS (deltas beat the prior)' if a_mx > opt0(mx) else 'no lift on mixed (label noisy or needs card-effects/win-rate)'}")
    except Exception as e:
        print(f"listwise skipped: {e!r}")


if __name__ == "__main__":
    main()
