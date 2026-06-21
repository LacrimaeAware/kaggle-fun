"""Branch A / A3.A2 -- search confounder audit.

Instruments the actual determinization loop (mirroring search.option_evals) to measure the confounders
that affect Teacher V2 label quality: fake-Water-Energy hidden-zone padding, determinizations actually
reached under the time budget, per-option coverage, and search time. Runs on self-play states (the deployed
distribution, deck=DENPA92) and replay states (the offline-label distribution, deciding player's own deck),
at the live 0.6s budget and an offline 8s budget. Produces an audit table.

    python tools/audit_search_confounders.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))
import search as S                       # noqa: E402
import main as M                         # noqa: E402
import state_action_schema_v2 as SCH     # noqa: E402
import audit_teacher_stability as A2     # noqa: E402  (reuse gen_selfplay + sampling helpers)

PAD_ID = 3   # Basic Water Energy (the hidden-zone padding card; our deck is Psychic)


def instrument(obs, deck, n_determ, budget):
    A = S._api()
    sel, cur = obs.get("select"), obs.get("current")
    if not sel or not cur:
        return None
    players = cur.get("players") or []
    if len(players) < 2:
        return None
    me = cur.get("yourIndex", 0)
    P, O = players[me], players[1 - me]
    oa = O.get("active") or []
    if oa and oa[0] is None:
        return None
    n_my_deck, n_op_deck = P.get("deckCount", 0) or 0, O.get("deckCount", 0) or 0
    n_my_prize, n_op_prize = len(P.get("prize") or []), len(O.get("prize") or [])
    n_op_hand = O.get("handCount", 0) or 0
    obsd = A.to_observation_class(obs)
    determ, pad_total, opt_counts, nn = 0, 0, None, 0
    t0 = time.time()
    for _ in range(n_determ):
        if time.time() - t0 > budget:
            break
        mp = S._hidden_pool(deck, P, exclude_hand=False)
        pad_my = max(0, (n_my_deck + n_my_prize) - len(mp))
        mp = mp + [PAD_ID] * pad_my
        op = S._hidden_pool(deck, O, exclude_hand=True)
        pad_op = max(0, (n_op_deck + n_op_prize + n_op_hand) - len(op))
        op = op + [PAD_ID] * pad_op
        pad_total += pad_my + pad_op
        try:
            root = A.search_begin(obsd, your_deck=mp[:n_my_deck], your_prize=mp[n_my_deck:n_my_deck + n_my_prize],
                                  opponent_deck=op[n_op_hand + n_op_prize:n_op_hand + n_op_prize + n_op_deck],
                                  opponent_prize=op[n_op_hand:n_op_hand + n_op_prize], opponent_hand=op[:n_op_hand],
                                  opponent_active=[])
        except Exception:
            continue
        determ += 1
        nn = len(root.observation.select.option)
        if opt_counts is None:
            opt_counts = [0] * nn
        try:
            for i in range(nn):
                if time.time() - t0 > budget:
                    break
                try:
                    S._simulate(A, root.searchId, i, me, "hand")
                    opt_counts[i] += 1
                except Exception:
                    continue
        finally:
            try:
                A.search_end()
            except Exception:
                pass
    dt = time.time() - t0
    if not opt_counts:
        return None
    return {"n_options": nn, "determ_reached": determ, "pad_per_determ": pad_total / max(1, determ),
            "any_pad": int(pad_total > 0),
            "min_opt_coverage": min(opt_counts) / max(1, determ),     # 1.0 = every option simulated every world
            "full": int(determ == n_determ and min(opt_counts) == determ), "time_s": dt}


def _agg(rows, key):
    xs = [r[key] for r in rows if r and r.get(key) is not None]
    return round(statistics.fmean(xs), 3) if xs else None


def collect_replay_decisions(n):
    manifest = json.load(open(ROOT / "data" / "manifests" / "replays_20260618.json", encoding="utf-8"))
    split = json.load(open(ROOT / "data" / "splits" / "replays_20260618_split.json", encoding="utf-8"))
    return A2.sample_decisions(manifest, split, n, verify=False)


def main():
    print("[A2 audit] generating self-play states + sampling replay states...", flush=True)
    selfplay = A2.gen_selfplay_decisions(40, 30)
    replay = collect_replay_decisions(40)
    table = {}
    for label, decs, budgets in [("self-play (deployed, DENPA92)", selfplay, [0.6, 8.0]),
                                 ("replay (offline-label, player deck)", replay, [0.6, 8.0])]:
        for b in budgets:
            rows = [instrument(d["obs"], d["deck"], 8, b) for d in decs]
            rows = [r for r in rows if r]
            table[f"{label} @ {b}s"] = {
                "n": len(rows),
                "determ_reached(/8)": _agg(rows, "determ_reached"),
                "pad_cards/determ": _agg(rows, "pad_per_determ"),
                "%decisions_with_padding": round(100 * _agg(rows, "any_pad"), 1) if rows else None,
                "min_option_coverage": _agg(rows, "min_opt_coverage"),
                "%full_coverage": round(100 * _agg(rows, "full"), 1) if rows else None,
                "search_time_s": _agg(rows, "time_s"),
            }
            print(f"  {label} @ {b}s: n={table[f'{label} @ {b}s']['n']}", flush=True)
    out = ROOT / "data" / "manifests" / "search_confounder_audit.json"
    out.write_text(json.dumps(table, indent=1), encoding="utf-8")
    print("\n=== SEARCH CONFOUNDER AUDIT ===")
    for k, v in table.items():
        print(f"\n{k}")
        for mk, mv in v.items():
            print(f"   {mk:26} {mv}")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
