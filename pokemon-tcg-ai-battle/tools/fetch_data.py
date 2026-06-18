"""Download reference data into data/external/ (gitignored). Re-runnable; the script is the
canon, the data is regenerable.

Two sources, both free and public:
  1. Card database: the PokemonTCG/pokemon-tcg-data GitHub repo (static JSON per set). We
     pull the Scarlet & Violet era sets (sv*), which are current Standard.
  2. Competitive decklists: the Limitless play API (play.limitlesstcg.com/api). We pull
     recent tournament standings, which include each player's full card-level decklist,
     archetype, placing, and record.

Caveat to remember: these card names/sets are the REAL game's identifiers. The contest's
cabt engine uses its own integer card IDs (e.g. 721, 1092), and the contest card pool is an
organizer-chosen subset that may not match real Standard. So this data is reference and
deck-idea fuel, not something that maps one-to-one onto the cabt pool. Useful for human
strategy and archetype knowledge; the mapping to cabt IDs is a separate, unsolved task.

Stdlib only (urllib). Run: python tools/fetch_data.py
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARDS = ROOT / "data" / "external" / "cards"
DECKS = ROOT / "data" / "external" / "decklists"
GH_RAW = "https://raw.githubusercontent.com/PokemonTCG/pokemon-tcg-data/master/cards/en"
LIMITLESS = "https://play.limitlesstcg.com/api"
SV_SETS = ["sv1", "sv2", "sv3", "sv3pt5", "sv4", "sv4pt5", "sv5", "sv6", "sv6pt5",
           "sv7", "sv8", "sv8pt5", "sv9", "sv10", "sve", "svp"]
MAX_TOURNAMENTS = 40
UA = {"User-Agent": "kaggle-fun-research/1.0"}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def fetch_cards() -> None:
    CARDS.mkdir(parents=True, exist_ok=True)
    have = 0
    for s in SV_SETS:
        dst = CARDS / f"{s}.json"
        if dst.exists() and dst.stat().st_size > 0:
            have += 1
            continue
        try:
            dst.write_bytes(_get(f"{GH_RAW}/{s}.json"))
        except Exception as e:
            print(f"  card set {s} failed: {e}")
    n = sum(len(json.loads((CARDS / f'{s}.json').read_text(encoding='utf-8')))
            for s in SV_SETS if (CARDS / f'{s}.json').exists())
    print(f"cards: {len(SV_SETS)} sets present ({have} already had), {n} cards total in data/external/cards/")


def fetch_decklists() -> None:
    DECKS.mkdir(parents=True, exist_ok=True)
    tours = json.loads(_get(f"{LIMITLESS}/tournaments?game=PTCG&limit=200").decode("utf-8"))
    fmt = {}
    for t in tours:
        fmt[t.get("format", "?")] = fmt.get(t.get("format", "?"), 0) + 1
    print("tournament formats available (recent 200):", fmt)
    standard = [t for t in tours if str(t.get("format", "")).upper() == "STANDARD"]
    chosen = (standard or tours)[:MAX_TOURNAMENTS]
    print(f"downloading standings for {len(chosen)} tournaments "
          f"({'Standard only' if standard else 'all formats, no Standard found'})")

    rows = []
    matches = []
    saved = 0
    for t in chosen:
        tid = t.get("id")
        try:
            standings = json.loads(_get(f"{LIMITLESS}/tournaments/{tid}/standings?game=PTCG").decode("utf-8"))
        except Exception:
            continue
        # pairings give round-by-round who-beat-whom (match results, not moves)
        try:
            pairings = json.loads(_get(f"{LIMITLESS}/tournaments/{tid}/pairings?game=PTCG").decode("utf-8"))
        except Exception:
            pairings = []
        (DECKS / f"{tid}.json").write_text(
            json.dumps({"tournament": t, "standings": standings, "pairings": pairings},
                       ensure_ascii=False), encoding="utf-8")
        saved += 1
        # index player -> archetype so match rows carry deck-vs-deck info
        arch = {p.get("player") or p.get("name"): (p.get("deck") or {}).get("name") for p in standings}
        for p in standings:
            dl = p.get("decklist")
            if not dl:
                continue
            rows.append({
                "tournament": t.get("name"), "date": t.get("date"), "format": t.get("format"),
                "placing": p.get("placing"), "record": p.get("record"),
                "archetype": (p.get("deck") or {}).get("name"), "decklist": dl,
            })
        for g in pairings:
            p1, p2, w = g.get("player1"), g.get("player2"), g.get("winner")
            if not p2 or w in (None, -1):     # skip byes and unreported
                continue
            matches.append({
                "tournament": t.get("name"), "date": t.get("date"), "round": g.get("round"),
                "player1": p1, "player2": p2, "winner": w,
                "p1_archetype": arch.get(p1), "p2_archetype": arch.get(p2),
            })
    (DECKS / "decklists.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (DECKS / "matches.jsonl").write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in matches), encoding="utf-8")
    n_with = sum(1 for r in rows if r["decklist"])
    n_arch = sum(1 for m in matches if m["p1_archetype"] and m["p2_archetype"])
    print(f"decklists: {saved} tournaments, {len(rows)} player rows ({n_with} with full lists)")
    print(f"matches: {len(matches)} reported results ({n_arch} with both archetypes known), "
          f"in data/external/decklists/matches.jsonl  (results only, NOT move-by-move)")


if __name__ == "__main__":
    print("== fetching card reference (GitHub pokemon-tcg-data) ==")
    fetch_cards()
    print("== fetching competitive decklists (Limitless play API) ==")
    fetch_decklists()
    print("done.")
