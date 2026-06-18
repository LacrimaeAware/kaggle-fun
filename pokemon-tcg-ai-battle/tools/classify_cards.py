"""Classify every card by FUNCTION (not energy type): attacker, energy, energy-acceleration
(your "pseudo-energy"), draw, search, gust, switch, disruption, heal, plus structural tags
(basic / evolution / ability / item / supporter / tool / stadium).

This is the keystone the rest builds on: it turns 1267 opaque card ids into interpretable
functional features that (a) make heuristics meaningful (e.g. "play cards whose energy you can
support, counting energy-acceleration as pseudo-energy"), (b) become the per-card vectors for
the embedding work, and (c) are far better RL inputs than raw ids.

This first pass is a transparent rule parse of the card type + ability/attack effect text
(from cards_full.json, which comes from the engine's all_card_data()). It is meant to be
inspected and corrected; an LLM pass can refine the ambiguous cards later. Output:
registry/card_roles.json (id -> {cat, tags}).

Run: python tools/classify_cards.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "external" / "official" / "cards_full.json"
OUT = ROOT / "registry" / "card_roles.json"

CAT = {0: "Pokemon", 1: "Item", 2: "Tool", 3: "Supporter", 4: "Stadium",
       5: "Basic Energy", 6: "Special Energy"}


def classify(c: dict) -> tuple[str, list[str]]:
    ct = c.get("ct", 0)
    name = (c.get("n") or "").lower()
    text = (" ".join(s.get("t", "") for s in c.get("skills", []))
            + " " + " ".join(a.get("n", "") for a in c.get("atk", []))).lower()
    tags: set[str] = set()

    if ct == 5:
        tags.add("energy")
    if ct == 6:
        tags.add("special_energy")
    if ct == 0:
        if any((a.get("d") or 0) > 0 for a in c.get("atk", [])):
            tags.add("attacker")
        if c.get("skills"):
            tags.add("ability")
        if c.get("basic"):
            tags.add("basic_mon")
        if c.get("s1") or c.get("s2"):
            tags.add("evolution")
        if c.get("ex"):
            tags.add("ex")
        if c.get("mega"):
            tags.add("mega")

    # functional roles from effect text (Trainers and Pokemon abilities)
    if "draw" in text and "card" in text:
        tags.add("draw")
    if "search your deck" in text:
        tags.add("search")
    if "attach" in text and "energy" in text and ("from your" in text or re.search(r"search your deck for[^.]*energy", text)):
        tags.add("energy_accel")          # pseudo-energy
    if ("switch in 1 of your opponent" in text or ("switch" in text and "opponent" in text and "bench" in text)
            or "boss" in name):
        tags.add("gust")
    if "switch your active" in text or "switch this pok" in text:
        tags.add("switch")
    if "opponent" in text and ("shuffle" in text or "discard" in text) and "hand" in text:
        tags.add("disruption")
    if "heal" in text or "remove" in text and "damage" in text:
        tags.add("heal")
    return CAT.get(ct, "?"), sorted(tags)


def main() -> None:
    full = json.loads(FULL.read_text(encoding="utf-8"))
    roles = {}
    tagcount: Counter = Counter()
    catcount: Counter = Counter()
    examples: dict[str, list[str]] = {}
    for cid, c in full.items():
        cat, tags = classify(c)
        roles[cid] = {"n": c.get("n"), "cat": cat, "tags": tags}
        catcount[cat] += 1
        for t in tags:
            tagcount[t] += 1
            examples.setdefault(t, [])
            if len(examples[t]) < 4:
                examples[t].append(c.get("n"))
    OUT.write_text(json.dumps(roles, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(roles)} cards)\n")
    print("by card type:")
    for k, v in catcount.most_common():
        print(f"  {v:4d}  {k}")
    print("\nby function tag (with examples):")
    for t, v in tagcount.most_common():
        print(f"  {v:4d}  {t:14s} e.g. {', '.join(examples[t])}")


if __name__ == "__main__":
    main()
