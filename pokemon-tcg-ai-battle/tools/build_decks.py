"""Build registry/decks.json: the deck store the HTML viewer reads.

Collects decks from (a) the official sample_submission/deck.csv (Mega Abomasnow ex) and
(b) decks embedded inline in the downloaded notebooks (a 60-int list). Dedupes identical
decks by card multiset, auto-names each by its strongest Pokemon, records the source.

Decks are also where the user edits/clones/versions. decks.json is the canon (hand-editable);
re-run this to regenerate the auto-collected ones. User-authored decks (source "user") are
preserved across re-runs.

Run: python tools/build_decks.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import cards  # tools/cards.py

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "research" / "notebooks"
OFFICIAL_DECK = ROOT / "data" / "external" / "official" / "sample_submission" / "deck.csv"
OUT = ROOT / "registry" / "decks.json"

# notebook file -> a human source label. Only distinct archetypes; clones dedupe out.
SOURCES = [
    ("crustle-wall-mirror-ok", "crustle notebook"),
    ("validated-rule-based-agent-matchup-tests(1)", "validated-matchup notebook"),
    ("pokemon-ai-battle-agent-mega-lucario", "lucario-search notebook"),
    ("reinforcement-learning-and-mcts-sample-code", "RL+MCTS sample"),
    ("pokemon", "beginner notebook"),
    ("strong-start-safe-agent-turn-search-lb-860", "strong-start notebook"),
]


def deck_from_notebook(stem: str) -> list[int] | None:
    nb = json.loads((NB / f"{stem}.ipynb").read_text(encoding="utf-8"))
    src = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code")
    for m in re.finditer(r"\[([\s\d,]+)\]", src):
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        if 58 <= len(nums) <= 62:
            return nums
    return None


def auto_name(ids: list[int]) -> str:
    """Name a deck by its strongest Pokemon (prefer megaEx/ex, then highest HP)."""
    best = None
    for cid in set(ids):
        c = cards.card(cid)
        if c.get("category", "").lower() != "pokemon" and "Pok" not in c.get("stage", ""):
            # category field is sometimes blank; fall back to HP presence
            if not (c.get("hp") or "").isdigit():
                continue
        hp = int(c["hp"]) if (c.get("hp") or "").isdigit() else 0
        is_ex = "ex" in c.get("name", "").lower()
        key = (is_ex, hp)
        if best is None or key > best[0]:
            best = (key, c["name"])
    return f"{best[1]} deck" if best else "Unnamed deck"


def main() -> None:
    existing = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
    user_decks = [d for d in existing if d.get("source") == "user"]

    collected = []
    seen = set()

    def add(ids, source, name=None):
        key = tuple(sorted(ids))
        if key in seen or len(ids) != 60:
            return
        seen.add(key)
        collected.append({
            "id": f"D{len(collected) + 1:03d}",
            "name": name or auto_name(ids),
            "source": source,
            "cards": ids,
            "notes": "",
        })

    # official Abomasnow sample
    if OFFICIAL_DECK.exists():
        ids = [int(x) for x in OFFICIAL_DECK.read_text().split() if x.strip()]
        add(ids, "official sample", "Mega Abomasnow ex (official sample)")

    for stem, label in SOURCES:
        if not (NB / f"{stem}.ipynb").exists():
            continue
        ids = deck_from_notebook(stem)
        if ids:
            add(ids, label)

    out = collected + user_decks
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {len(collected)} auto decks + {len(user_decks)} user decks to {OUT.relative_to(ROOT)}")
    for d in collected:
        print(f"  {d['id']}  {d['name']:42s}  ({d['source']})")


if __name__ == "__main__":
    main()
