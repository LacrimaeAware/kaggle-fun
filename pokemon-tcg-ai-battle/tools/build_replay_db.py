"""Aggregate all downloaded cabt replays into a queryable database.

Reads every data/external/replays/*.json and produces three artifacts under data/replay_db/ :

  games.jsonl     -- one row per episode: agent names, both decks (cabt card ids), winner seat,
                     game length, and setup mulligan count per seat (a basic-light deck mulligans
                     a lot, handing the opponent a free card each time).
  decks.json      -- distinct 60-card decks aggregated across games: how many games, win rate,
                     basic-Pokemon count, and the agent names that piloted them. This is how we
                     pick a strong, basic-rich deck to ADOPT instead of our 6-basic mulligan deck.
  decisions.jsonl -- one row per real in-game decision (>1 legal option): the 47 board features,
                     the SelectContext (what kind of choice: attack / play / retreat / mulligan...),
                     #options, and the OUTCOME label (did this seat win the game). This is the
                     non-circular, outcome-labelled signal for the within-decision action ranker
                     and for "should I attack? / should I retreat?"-type decision-quality stats.

Mulligan note: the opening no-basic redraw is engine-forced and shows up in the logs as repeated
SHUFFLE (LogType 0) events before the first TURN_START (LogType 2); it is NOT an agent decision
(SelectContext.MULLIGAN=42 is a separate, rare effect). We count setup shuffles - 1 as mulligans.

Usage: python tools/build_replay_db.py [--min-games 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console: print Japanese agent names
except Exception:
    pass
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import features as FT  # noqa: E402

REPLAYS = ROOT / "data" / "external" / "replays"
OUT = ROOT / "data" / "replay_db"
CF = json.load(open(ROOT / "agent" / "card_features.json", encoding="utf-8"))

# SelectContext names we care to label (subset of cg/api.py SelectContext)
CTX = {0: "main", 1: "setup-active", 2: "setup-bench", 3: "switch", 4: "to-active", 5: "to-bench",
       7: "to-hand", 8: "discard", 18: "evolve-from", 19: "evolve-to", 21: "attach-from",
       22: "attach-to", 35: "attack", 37: "evolve", 41: "go-first", 42: "mulligan", 43: "activate"}
SHUFFLE, HAS_BASIC, TURN_START = 0, 1, 2


def is_basic_pokemon(cid: int) -> bool:
    v = CF.get(str(cid), {})
    return v.get("ct") == 0 and v.get("stage") == "basic"


def deck_basics(deck: list[int]) -> int:
    return sum(1 for c in deck if is_basic_pokemon(c))


def setup_mulligans(logs: list[dict], seat: int) -> int:
    """Setup shuffles (before first TURN_START) for `seat`, minus the initial shuffle."""
    n = 0
    for e in logs:
        if e.get("type") == TURN_START:
            break
        if e.get("type") == SHUFFLE and e.get("playerIndex") == seat:
            n += 1
    return max(0, n - 1)


def longest_logs(steps: list, seat: int) -> list:
    best: list = []
    for s in steps:
        if seat < len(s):
            lg = (s[seat].get("observation") or {}).get("logs") or []
            if len(lg) > len(best):
                best = lg
    return best


def parse_one(path: Path):
    rep = json.load(open(path, encoding="utf-8"))
    steps = rep.get("steps") or []
    if not steps:
        return None
    eid = path.stem
    names = [a.get("Name") for a in rep.get("info", {}).get("Agents", [])][:2]
    rewards = rep.get("rewards") or []
    winner = None
    if len(rewards) == 2 and rewards[0] != rewards[1] and None not in rewards:
        winner = 0 if rewards[0] > rewards[1] else 1

    decks = {0: None, 1: None}
    decisions = []
    for s in steps:
        for seat, ag in enumerate(s):
            if not isinstance(ag, dict):
                continue
            act = ag.get("action")
            if not isinstance(act, list) or not act:
                continue
            if len(act) == 60 and decks.get(seat) is None:
                decks[seat] = act
                continue
            obs = ag.get("observation") or {}
            sel = obs.get("select") or {}
            opts = sel.get("option") or []
            if not obs.get("current") or len(opts) < 2:
                continue  # forced / no real choice -> not a decision
            ctx = sel.get("context")
            feats = FT.encode_state(obs)
            decisions.append({"ep": eid, "seat": seat, "turn": obs["current"].get("turn"),
                              "ctx": ctx, "ctx_name": CTX.get(ctx, str(ctx)),
                              "n_options": len(opts), "won": (None if winner is None else int(winner == seat)),
                              **feats})

    game = {"ep": eid, "names": names, "winner": winner, "n_steps": len(steps),
            "rewards": rewards}
    for seat in (0, 1):
        d = decks.get(seat)
        lg = longest_logs(steps, seat)
        game[f"deck{seat}"] = d
        game[f"basics{seat}"] = deck_basics(d) if d else None
        game[f"mulligans{seat}"] = setup_mulligans(lg, seat) if lg else None
    return game, decisions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-games", type=int, default=5, help="min games for a deck to be ranked by win rate")
    args = ap.parse_args()

    files = sorted(REPLAYS.glob("*.json"))
    if not files:
        ap.error(f"no replays in {REPLAYS}")
    OUT.mkdir(parents=True, exist_ok=True)

    games, all_dec = [], []
    deck_stats = defaultdict(lambda: {"games": 0, "wins": 0, "names": Counter(), "deck": None})
    ctx_stats = defaultdict(lambda: {"n": 0})
    bad = 0
    for f in files:
        try:
            res = parse_one(f)
        except Exception as e:
            bad += 1
            if bad <= 3:
                print(f"  skip {f.name}: {e!r}")
            continue
        if not res:
            continue
        game, decisions = res
        games.append(game)
        all_dec.extend(decisions)
        for seat in (0, 1):
            d = game.get(f"deck{seat}")
            if not d:
                continue
            sig = tuple(sorted(d))
            st = deck_stats[sig]
            st["deck"] = d
            st["games"] += 1
            if game["winner"] == seat:
                st["wins"] += 1
            if seat < len(game["names"]) and game["names"][seat]:
                st["names"][game["names"][seat]] += 1
        for dec in decisions:
            ctx_stats[dec["ctx_name"]]["n"] += 1

    # write artifacts
    with open(OUT / "games.jsonl", "w", encoding="utf-8") as fh:
        for g in games:
            fh.write(json.dumps(g) + "\n")
    with open(OUT / "decisions.jsonl", "w", encoding="utf-8") as fh:
        for d in all_dec:
            fh.write(json.dumps(d) + "\n")

    decks_out = []
    for sig, st in deck_stats.items():
        g = st["games"]
        decks_out.append({
            "n_games": g, "n_wins": st["wins"], "winrate": round(st["wins"] / g, 3) if g else None,
            "basics": deck_basics(st["deck"]), "n_distinct": len(set(st["deck"])),
            "top_names": st["names"].most_common(3),
            "deck": st["deck"],
        })
    decks_out.sort(key=lambda x: (-(x["n_games"]), -(x["winrate"] or 0)))
    json.dump(decks_out, open(OUT / "decks.json", "w", encoding="utf-8"), indent=0)

    # ---- summary ----
    n_dec_lab = sum(1 for d in all_dec if d["won"] is not None)
    print(f"\nparsed {len(games)} games ({bad} skipped) | {len(all_dec)} decisions "
          f"({n_dec_lab} outcome-labelled) | {len(deck_stats)} distinct decks")
    print(f"  -> {OUT.relative_to(ROOT)}/games.jsonl, decisions.jsonl, decks.json")

    print("\nDECK RANKING (>= %d games, by win rate):" % args.min_games)
    ranked = sorted([d for d in decks_out if d["n_games"] >= args.min_games],
                    key=lambda x: -(x["winrate"] or 0))
    print(f"  {'games':>5} {'wr':>5} {'basics':>6}  piloted by")
    for d in ranked[:12]:
        who = ", ".join(f"{n}({c})" for n, c in d["top_names"])
        print(f"  {d['n_games']:>5} {d['winrate']:>5} {d['basics']:>6}  {who[:46]}")

    print("\nBASICS distribution across ALL decks seen (more basics = fewer mulligans):")
    bh = Counter(d["basics"] for d in decks_out)
    for b in sorted(bh):
        print(f"  {b:>2} basics: {bh[b]} decks")
    print("  our current deck has 6 basics (Kyogre x2, Snover x4).")

    print("\nDECISION TYPES (count across all games):")
    for name, st in sorted(ctx_stats.items(), key=lambda x: -x[1]["n"])[:14]:
        print(f"  {name:>14}: {st['n']}")

    # mulligan burden: our basic-light deck vs the field
    mull = [g[f"mulligans{seat}"] for g in games for seat in (0, 1) if g.get(f"mulligans{seat}") is not None]
    if mull:
        import statistics
        print(f"\nMULLIGANS per seat across field: mean {statistics.mean(mull):.2f}, "
              f"max {max(mull)}, share with >=3: {sum(m >= 3 for m in mull) / len(mull):.1%}")


if __name__ == "__main__":
    main()
