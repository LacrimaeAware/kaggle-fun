"""Decode per-card EFFECT features from cards_full.json effect text -> agent/card_effects.json.

The static replay option exposes only type/index/target/attackId, never what a card DOES. The card
text is highly stereotyped ("draw 6 cards", "Search your deck for up to 2 Basic Pokemon", "attach a
Basic {L} Energy"), so a small regex layer turns it into quantified effect features. These become
per-OPTION action features once joined to the card an option plays (Track B / Codex Tier 3): the
representation a learned move-ranker needs and static features lack.

Two-layer by design: regex over the stereotyped text now; a hand-built override table can be added
later for the few cards regex gets wrong. Quantities, not just presence flags (draw HOW MANY).

    python tools/build_card_effects.py        # writes agent/card_effects.json, prints coverage
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FULL = json.load(open(ROOT / "data" / "external" / "official" / "cards_full.json", encoding="utf-8"))
OUT = ROOT / "agent" / "card_effects.json"


def _maxint(pat, text, default=0):
    vals = [int(x) for x in re.findall(pat, text, re.I)]
    return max(vals) if vals else default


def decode(text: str) -> dict:
    """Effect features from one card's combined skill/attack text (lowercased matching, quantified)."""
    t = " ".join((text or "").split())
    low = t.lower()
    f = {}
    # draw / card advantage
    f["draw"] = _maxint(r"draw (\d+) cards?", low)
    if "draw a card" in low and not f["draw"]:
        f["draw"] = 1
    # search / tutor (deck -> hand/bench/field)
    if re.search(r"search your deck", low):
        f["search"] = _maxint(r"up to (\d+)", low) or 1
        f["search_to_bench"] = 1 if "onto your bench" in low else 0
    else:
        f["search"] = 0; f["search_to_bench"] = 0
    # recover from discard
    f["recover_discard"] = 1 if "from your discard pile" in low else 0
    # energy acceleration (attach energy outside the normal one-per-turn, in a skill/ability/trainer)
    f["energy_accel"] = 1 if re.search(r"attach .*?energy", low) else 0
    # heal
    f["heal"] = _maxint(r"heal (\d+) damage", low)
    # switch / gust (move a Pokemon to the Active Spot)
    f["switch_gust"] = 1 if ("to the active spot" in low or "switch" in low) else 0
    # disruption (opponent shuffles/discards hand)
    f["disrupt"] = 1 if re.search(r"opponent.*(shuffle|discard).*hand", low) else 0
    # costs / downside
    f["discard_cost"] = _maxint(r"discard (\d+) (?:other )?cards? from your hand", low)
    f["shuffle_hand"] = 1 if "shuffle your hand into your deck" in low else 0
    # special conditions inflicted
    f["status"] = 1 if re.search(r"\b(asleep|paralyzed|confused|burned|poisoned)\b", low) else 0
    return f


def main():
    out = {}
    for cid, c in FULL.items():
        texts = [sk.get("t", "") for sk in (c.get("skills") or [])]
        texts += [a.get("t", "") for a in (c.get("atk") or [])]
        eff = decode(" ".join(texts))
        # ability flag: a Pokemon skill whose text is an ongoing/at-will effect (not an attack)
        eff["has_ability"] = 1 if (c.get("ct") == 0 and (c.get("skills") or [])) else 0
        if any(eff.values()):
            out[cid] = {k: v for k, v in eff.items() if v}
    json.dump(out, open(OUT, "w", encoding="utf-8"), separators=(",", ":"))

    # coverage report
    from collections import Counter
    cov = Counter()
    for e in out.values():
        for k in e:
            cov[k] += 1
    print(f"decoded effects for {len(out)}/{len(FULL)} cards -> {OUT.relative_to(ROOT)}")
    print("coverage (cards with each effect):")
    for k, n in cov.most_common():
        print(f"  {k:>16}: {n}")
    # sanity-check the known DENPA92/Iono cards
    print("\nsanity (known cards):")
    for cid in ["1086", "1227", "1121", "1097", "1233", "269"]:
        print(f"  {cid} {FULL[cid]['n'][:24]:<24}: {out.get(cid, {})}")


if __name__ == "__main__":
    main()
