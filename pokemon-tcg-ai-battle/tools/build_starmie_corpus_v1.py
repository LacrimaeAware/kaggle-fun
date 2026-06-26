"""STARMIE_SPECIALIST_CORPUS_V1 (Model B data task).

Produces a clean, replay-grouped, labeled decision corpus for the Cinderace/Mega-Starmie deck that Model A
can train on with the shared C8 feature/extraction infrastructure. Model B produces DATA only -- no learning
pipeline here. Each row is one decision with the observable root state (pointer + a compact summary), all legal
sibling actions, the pilot's chosen action, the action family, and the same-turn sequence. Pilot identity,
game outcome, and future sequence are placed under `meta` and MUST NOT be used as runtime model inputs.

Cohorts:
  C0 exact Starmie deck, all supported pilots
  C1 exact Starmie deck, top high-volume pilots (subset of C0)
  C2 near-exact Starmie deck (deck_distance 1..10), tiered by card distance
  C3 the OPPONENT's decisions in Starmie matchups (for opponent modeling)
Splits are replay-grouped (a whole game is train|val|test) and deterministic by episode hash.

  python tools/build_starmie_corpus_v1.py --max-games 4000
Outputs:
  data/starmie_corpus/starmie_specialist_corpus_v1.jsonl   (rows; large, local for Model A)
  data/starmie_audit/starmie_corpus_manifest_v1.json       (deck hashes, cohorts, support report, splits)
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
REPLAYS = r"C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays"
JSONL = ROOT / "data" / "starmie_corpus" / "starmie_specialist_corpus_v1.jsonl"
MANIFEST = ROOT / "data" / "starmie_audit" / "starmie_corpus_manifest_v1.json"

import deck_policy_v3 as DP            # noqa: E402
import starmie_heuristics as SH        # noqa: E402  (STARMIE_DECK + card ids)
CDB = json.load(open(ROOT / "agent" / "card_stats.json", encoding="utf-8"))
ATK = json.load(open(ROOT / "agent" / "attack_stats.json", encoding="utf-8"))
EXACT = tuple(sorted(SH.STARMIE_DECK))
MEGA = SH.MEGA_STARMIE
TN = {0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY",
      11: "DISCARD", 12: "RETREAT", 13: "ATTACK", 14: "END"}


def _name(cid):
    return (CDB.get(str(cid), {}) or {}).get("n", f"#{cid}") if cid is not None else None


def _deck_distance(deck):
    a, b = Counter(deck), Counter(EXACT)
    return sum(abs(a[c] - b[c]) for c in set(a) | set(b)) // 2


def _deck_of(d, seat):
    for st in (d.get("steps") or [])[:8]:
        if isinstance(st, list) and len(st) > seat and isinstance(st[seat], dict):
            a = st[seat].get("action")
            if isinstance(a, list) and len(a) == 60:
                return a
    return None


def _winner(d):
    r = d.get("rewards") or []
    if len(r) < 2 or r[0] is None or r[1] is None or r[0] == r[1]:
        return None
    return 0 if r[0] > r[1] else 1


def _split(epid):
    h = int(hashlib.sha1(str(epid).encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def _family(opts, action):
    """Action family + detail from the pilot's chosen option(s)."""
    types, dets = set(), []
    for i in (action or []):
        if isinstance(i, int) and 0 <= i < len(opts):
            o = opts[i]
            t = o.get("type")
            types.add(t)
            if t == 13:
                dets.append(ATK.get(str(o.get("attackId")), {}).get("n", "attack"))
    if 13 in types:
        nm = (dets[0] if dets else "").lower()
        if "jetting" in nm:
            return "ATTACK", "Jetting"
        if "nebula" in nm:
            return "ATTACK", "Nebula"
        if "turbo" in nm:
            return "ATTACK", "TurboFlare"
        return "ATTACK", dets[0] if dets else "attack"
    for t, fam in ((8, "ATTACH"), (9, "EVOLVE"), (10, "ABILITY"), (12, "RETREAT"), (11, "DISCARD"),
                   (3, "SELECT_CARD")):
        if t in types:
            return fam, None
    if 7 in types:
        return "PLAY", None
    if types & {1, 2}:
        return "YES_NO", None
    return "OTHER", None


def _legal(opts, obs, sel):
    out = []
    for o in opts:
        t = o.get("type")
        rec = {"type": t, "tname": TN.get(t, str(t)), "index": o.get("index")}
        if t == 13:
            a = ATK.get(str(o.get("attackId")), {})
            rec["attackId"] = o.get("attackId"); rec["attack"] = a.get("n"); rec["dmg"] = a.get("d")
        if o.get("area") is not None:
            rec["area"] = o.get("area")
        if o.get("playerIndex") is not None:
            rec["playerIndex"] = o.get("playerIndex")
        cid = None
        try:
            cid = DP.option_card_id(o, obs)
        except Exception:
            cid = None
        if cid is None and t == 3:
            cid = SH._sel_card(sel, o)
        if cid is not None:
            rec["card_id"] = cid; rec["card"] = _name(cid)
        out.append(rec)
    return out


def _entity(e):
    if not e:
        return None
    return {"id": e.get("id"), "name": _name(e.get("id")), "hp": e.get("hp"),
            "energy": len((e.get("energyCards") or e.get("energies") or []))}


def _root_state(obs):
    cur = obs.get("current") or {}
    pls = cur.get("players") or []
    yi = cur.get("yourIndex", 0)
    me = pls[yi] if yi < len(pls) else {}
    opp = pls[1 - yi] if len(pls) > 1 else {}

    def side(p):
        return {"active": _entity((p.get("active") or [None])[0]),
                "bench": [_entity(b) for b in (p.get("bench") or []) if b],
                "handCount": p.get("handCount"), "prizeLeft": len(p.get("prize") or []),
                "deckCount": p.get("deckCount")}
    return {"me": side(me), "opp": side(opp), "step": obs.get("step")}


def _turn_families(steps, seat, t):
    """Family sequence for the rest of this turn (metadata; NOT a runtime input)."""
    fams = []
    t2 = t
    while t2 < len(steps) - 1 and steps[t2][seat].get("status") == "ACTIVE":
        o2 = (steps[t2][seat].get("observation") or {}).get("select") or {}
        opts2 = o2.get("option") or []
        fam, det = _family(opts2, steps[t2 + 1][seat].get("action"))
        fams.append(det or fam)
        if fam == "ATTACK":
            break
        t2 += 1
    return fams, t, t2


def _seat_rows(d, seat, cohort, dist, pilot, won, opp_seat=False):
    steps = d.get("steps") or []
    epid = (d.get("info") or {}).get("EpisodeId") or "?"
    split = _split(epid)
    rows = []
    turn_id = 0
    prev_active = False
    for t in range(len(steps) - 1):
        e = steps[t][seat]
        active = e.get("status") == "ACTIVE"
        if active and not prev_active:
            turn_id += 1
        prev_active = active
        if not active:
            continue
        obs = e.get("observation") or {}
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        action = steps[t + 1][seat].get("action")
        if len(opts) < 2 or not isinstance(action, list) or not action:
            continue
        fam, det = _family(opts, action)
        fams, ts, te = _turn_families(steps, seat, t)
        rows.append({
            "id": f"{epid}_{t}_{seat}", "episode": epid, "step": t, "seat": seat,
            "cohort": cohort, "deck_distance": dist, "is_opponent": opp_seat,
            "family": fam, "family_detail": det, "context": sel.get("context"),
            "n_legal": len(opts), "legal_actions": _legal(opts, obs, sel),
            "pilot_action": action,
            "root_state": _root_state(obs),
            "root_obs_pointer": {"episode": epid, "step": t, "seat": seat, "field": "steps[step][seat].observation"},
            "split": split,
            # ---- META: must NOT be used as runtime model inputs ----
            "meta": {"pilot": pilot, "won": won, "turn_id": turn_id,
                     "same_turn_family_sequence": fams},
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-games", type=int, default=6000)
    ap.add_argument("--max-distance", type=int, default=10)
    ap.add_argument("--min-pilot-games", type=int, default=10)
    a = ap.parse_args()
    JSONL.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(REPLAYS + "/*.json"))
    print(f"scanning {len(files)} replays for exact/near-exact Starmie (dist<= {a.max_distance})...", flush=True)
    # first pass: count Starmie games per pilot (to define supported pilots + C1 high-volume)
    pilot_games = Counter()
    starmie_seats = []      # (fn, seat, dist, pilot, won)
    for i, fn in enumerate(files[: a.max_games]):
        try:
            d = json.load(open(fn, encoding="utf-8"))
        except Exception:
            continue
        w = _winner(d)
        if w is None:
            continue
        tn = (d.get("info") or {}).get("TeamNames") or []
        for seat in (0, 1):
            dk = _deck_of(d, seat)
            if not dk or MEGA not in dk:
                continue
            dist = _deck_distance(dk)
            if dist > a.max_distance:
                continue
            pilot = str(tn[seat]) if seat < len(tn) and tn[seat] else f"seat{seat}"
            pilot_games[pilot] += 1
            starmie_seats.append((fn, seat, dist, pilot, seat == w))
        if (i + 1) % 1500 == 0:
            print(f"  {i+1} scanned, {len(starmie_seats)} starmie seats", flush=True)

    supported = {p for p, n in pilot_games.items() if n >= a.min_pilot_games}
    top_volume = {p for p, _ in sorted(pilot_games.items(), key=lambda kv: -kv[1])[:10]}
    print(f"supported pilots (>= {a.min_pilot_games} games): {len(supported)} | high-volume(C1): {len(top_volume)}", flush=True)

    fam_counts = Counter()
    cohort_counts = Counter()
    split_counts = Counter()
    n_rows = 0
    fam_detail = Counter()
    with open(JSONL, "w", encoding="utf-8") as out:
        for fn, seat, dist, pilot, won in starmie_seats:
            if pilot not in supported:
                continue
            try:
                d = json.load(open(fn, encoding="utf-8"))
            except Exception:
                continue
            cohort = "C0" if dist == 0 else "C2"
            rows = _seat_rows(d, seat, cohort, dist, pilot, won)
            # C1 is the high-volume subset of exact-deck rows (tag, do not duplicate)
            for r in rows:
                if dist == 0 and pilot in top_volume:
                    r["cohort_c1_highvolume"] = True
                fam_counts[r["family"]] += 1
                if r["family_detail"]:
                    fam_detail[f"{r['family']}:{r['family_detail']}"] += 1
                cohort_counts[r["cohort"]] += 1
                split_counts[r["split"]] += 1
                out.write(json.dumps(r) + "\n")
                n_rows += 1
            # C3: the opponent seat's decisions in this Starmie matchup
            opp_seat = 1 - seat
            opp_tn = (d.get("info") or {}).get("TeamNames") or []
            opp_pilot = str(opp_tn[opp_seat]) if opp_seat < len(opp_tn) and opp_tn[opp_seat] else f"seat{opp_seat}"
            for r in _seat_rows(d, opp_seat, "C3", -1, opp_pilot, (opp_seat == (0 if won else 1)) if False else None, opp_seat=True):
                fam_counts[r["family"]] += 1
                cohort_counts["C3"] += 1
                split_counts[r["split"]] += 1
                out.write(json.dumps(r) + "\n")
                n_rows += 1

    # Ignition / Jetting-vs-Nebula support (subset of families)
    support_report = {
        "ATTACH": fam_counts.get("ATTACH", 0), "SELECT_CARD_tutor": fam_counts.get("SELECT_CARD", 0),
        "ATTACK": fam_counts.get("ATTACK", 0), "RETREAT": fam_counts.get("RETREAT", 0),
        "PLAY_incl_Wally_Hammer_Boss": fam_counts.get("PLAY", 0), "EVOLVE": fam_counts.get("EVOLVE", 0),
        "ABILITY": fam_counts.get("ABILITY", 0),
        "ATTACK_Jetting": fam_detail.get("ATTACK:Jetting", 0), "ATTACK_Nebula": fam_detail.get("ATTACK:Nebula", 0),
        "ATTACK_TurboFlare": fam_detail.get("ATTACK:TurboFlare", 0),
    }
    manifest = {
        "name": "STARMIE_SPECIALIST_CORPUS_V1",
        "producer": "Model B (data only; Model A trains with shared C8 infra)",
        "jsonl": str(JSONL.relative_to(ROOT)), "n_rows": n_rows,
        "exact_deck_sha1": hashlib.sha1(str(EXACT).encode()).hexdigest()[:12],
        "exact_deck_len": len(EXACT),
        "max_deck_distance": a.max_distance, "min_pilot_games": a.min_pilot_games,
        "supported_pilots": sorted(supported), "n_supported_pilots": len(supported),
        "c1_high_volume_pilots": sorted(top_volume),
        "cohorts": dict(cohort_counts),
        "splits_replay_grouped": dict(split_counts),
        "support_report_by_family": support_report,
        "row_schema": {
            "inputs_runtime": ["root_state", "root_obs_pointer", "legal_actions"],
            "label": ["pilot_action", "family", "family_detail"],
            "grouping": ["episode", "split", "cohort", "deck_distance"],
            "META_NOT_runtime_inputs": ["meta.pilot", "meta.won", "meta.same_turn_family_sequence"],
        },
        "notes": "Pilot identity, game outcome, and future (same-turn) sequence are under meta and MUST NOT be runtime inputs. Full observable features can be re-extracted from the replay via root_obs_pointer with the shared C8 observation pipeline; root_state is a compact convenience summary.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nrows {n_rows} | cohorts {dict(cohort_counts)} | splits {dict(split_counts)}", flush=True)
    print(f"support: {support_report}", flush=True)
    print(f"wrote {JSONL} ({round(JSONL.stat().st_size/1e6,1)} MB) + {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
