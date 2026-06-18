"""Card lookup from the official EN_Card_Data.csv (the cabt card database).

The competition ships data/external/official/EN_Card_Data.csv with one row per card (or per
attack for multi-attack cards): Card ID, Card Name, Expansion, Stage/Type, Category, HP,
Type, Weakness, Resistance, Retreat, plus Move Name / Cost / Damage / Effect. This module
loads it into a dict keyed by integer card id, so the rest of the code can turn cabt card
ids (in decks, observations, replays) into real names and stats.

Use:
    from tools.cards import name, card, decode_deck
    name(723)            -> "Mega Abomasnow ex"
    card(723)["hp"]      -> the HP
CLI:
    python tools/cards.py 723                 # show one card
    python tools/cards.py deck path/to/deck.csv   # decode a 60-card deck
    python tools/cards.py ids 721 722 723         # decode a list of ids
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "external" / "official" / "EN_Card_Data.csv"

_CARDS: dict[int, dict] | None = None


def _load() -> dict[int, dict]:
    global _CARDS
    if _CARDS is not None:
        return _CARDS
    cards: dict[int, dict] = {}
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["Card ID"])
            except (KeyError, ValueError):
                continue
            move = {
                "name": (row.get("Move Name") or "").strip(),
                "cost": (row.get("Cost") or "").strip(),
                "damage": (row.get("Damage") or "").strip(),
                "effect": (row.get("Effect Explanation") or "").strip(),
            }
            if cid in cards:                       # multi-attack card: append the move
                if move["name"] and move["name"].lower() != "n/a":
                    cards[cid]["attacks"].append(move)
                continue
            cards[cid] = {
                "id": cid,
                "name": (row.get("Card Name") or "").strip(),
                "expansion": (row.get("Expansion") or "").strip(),
                "category": (row.get("Category") or "").strip(),
                "stage": (row.get("Stage (Pokémon)/Type (Energy and Trainer)") or "").strip(),
                "hp": (row.get("HP") or "").strip(),
                "type": (row.get("Type") or "").strip(),
                "weakness": (row.get("Weakness") or "").strip(),
                "resistance": (row.get("Resistance (Type)") or "").strip(),
                "retreat": (row.get("Retreat") or "").strip(),
                "rule": (row.get("Rule") or "").strip(),
                "prev_stage": (row.get("Previous stage") or "").strip(),
                "attacks": ([move] if move["name"] and move["name"].lower() != "n/a" else []),
            }
    _CARDS = cards
    return cards


def card(cid: int) -> dict:
    return _load().get(int(cid), {"id": cid, "name": f"<unknown id {cid}>", "attacks": []})


def name(cid: int) -> str:
    return card(cid)["name"]


def decode_deck(ids: list[int]) -> list[tuple[int, int, str]]:
    """Return (count, id, name) rows, sorted by count desc."""
    c = Counter(int(i) for i in ids)
    return [(n, cid, name(cid)) for cid, n in sorted(c.items(), key=lambda x: -x[1])]


def _print_card(cid: int) -> None:
    d = card(cid)
    print(f"[{d['id']}] {d['name']}  ({d.get('category','')}, {d.get('stage','')})")
    for k in ("hp", "type", "weakness", "resistance", "retreat", "rule", "prev_stage"):
        if d.get(k) and d[k].lower() != "n/a":
            print(f"  {k}: {d[k]}")
    for a in d.get("attacks", []):
        print(f"  attack: {a['name']}  cost={a['cost']}  dmg={a['damage']}  {a['effect'][:80]}")


def _main() -> None:
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        print(f"\nloaded {len(_load())} cards from {CSV_PATH.name}")
        return
    if a[0] == "deck" and len(a) > 1:
        ids = [int(x) for x in Path(a[1]).read_text().split() if x.strip()]
        for n, cid, nm in decode_deck(ids):
            print(f"  {n:2d}x [{cid}] {nm}")
        print(f"  total {len(ids)} cards")
    elif a[0] == "ids":
        for n, cid, nm in decode_deck([int(x) for x in a[1:]]):
            print(f"  {n:2d}x [{cid}] {nm}")
    else:
        _print_card(int(a[0]))


if __name__ == "__main__":
    _main()
