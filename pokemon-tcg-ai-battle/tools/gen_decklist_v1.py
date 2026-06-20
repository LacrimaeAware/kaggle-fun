"""Generate a human+model-readable card reference for the best pilot's deck (episode 80723114, player 0):
every card with counts, type/weakness/resistance, HP, stage, attacks (cost/damage), and full effect text from
EN_Card_Data.csv, plus our hard-won card-mechanics gotchas. For brainstorming tactical heuristics.

    python tools/gen_decklist_v1.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import main as M   # noqa: E402

OUT = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/kaggle-fun/pokemon-tcg-ai-battle/research/BEST_PILOT_DECKLIST.md")
EPISODE = ROOT / "data" / "external" / "replays" / "80723114.json"
CSVP = ROOT / "data" / "external" / "official" / "EN_Card_Data.csv"

# our hard-won card-mechanics gotchas (from prior research; prevent repeating mistakes)
GOTCHAS = {
    "Dudunsparce": "Run Away Draw shuffles ITSELF (the 140-HP body) back into the deck when used -- it is a draw engine that removes its own attacker. Do not treat it as a stable wall.",
    "Dunsparce": "Basic that evolves into Dudunsparce. The deck runs it mainly to access the Dudunsparce draw engine.",
    "Alakazam": "Psychic Draw (the draw ability) only triggers on EVOLVE-from-hand, not every turn. Sequencing matters: evolve to draw, do not sit on it.",
    "Kadabra": "Same as Alakazam: draw only on the evolve step, not passively.",
}


def load_pilot_deck():
    d = json.load(open(EPISODE, encoding="utf-8"))
    for step in d["steps"][:6]:
        if isinstance(step, list) and len(step) > 0:
            ag = step[0]
            if isinstance(ag, dict) and isinstance(ag.get("action"), list) and len(ag["action"]) == 60:
                return Counter(ag["action"])
    raise SystemExit("pilot deck not found")


def load_cards():
    rows = defaultdict(lambda: {"moves": [], "effects": []})
    with open(CSVP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cid = r.get("Card ID")
            if not cid:
                continue
            try:
                cid = int(cid)
            except ValueError:
                continue
            c = rows[cid]
            c["name"] = r.get("Card Name", "")
            c["category"] = r.get("Category", "")
            c["stage"] = r.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
            c["rule"] = r.get("Rule", "")
            c["hp"] = r.get("HP", "")
            c["type"] = r.get("Type", "")
            c["weak"] = r.get("Weakness", "")
            c["resist"] = r.get("Resistance (Type)", "")
            c["retreat"] = r.get("Retreat", "")
            mv, cost, dmg = r.get("Move Name", ""), r.get("Cost", ""), r.get("Damage", "")
            eff = (r.get("Effect Explanation", "") or "").strip()
            if mv and mv != "n/a":
                c["moves"].append((mv, cost, dmg, eff))
            elif eff and eff not in c["effects"]:
                c["effects"].append(eff)
    return rows


def main():
    deck = load_pilot_deck()
    cards = load_cards()
    by_cat = defaultdict(list)
    for cid, n in deck.items():
        c = cards.get(cid, {"name": f"id {cid}", "category": "?"})
        stage = c.get("stage") or ""
        if "Tool" in stage:
            bucket = "Trainer"
        elif "Pokémon" in stage or "Pokemon" in stage:
            bucket = "Pokemon"
        elif "Energy" in stage:
            bucket = "Energy"
        else:
            bucket = "Trainer"
        by_cat[bucket].append((n, cid, c))

    L = []
    L.append("# Best-Pilot Deck Reference (for tactical-heuristic brainstorming)")
    L.append("")
    L.append("Source: Kaggle Pokemon TCG AI Battle, submission 53802029, episode 80723114, **player 0** "
             "(a strong ladder pilot). This is THEIR exact 60-card list, not ours. Card IDs are the cabt "
             "engine IDs (what the agent sees). Effect text is from the official EN_Card_Data.csv.")
    L.append("")
    L.append(f"**Deck:** {sum(deck.values())} cards, {len(deck)} distinct. "
             "Archetype: Dudunsparce/Alakazam draw-engine toolbox (near-identical to our DENPA92 list; "
             "theirs leans more on the evolution/draw engine + recovery, ours leans more on tools).")
    L.append("")
    L.append("## How to read this for heuristics")
    L.append("- **Weakness** = takes 2x damage from that type (huge: a weakness-hit often = a KO). "
             "**Resistance** = takes less. These are the 'effectiveness types'.")
    L.append("- **Prize liability:** ex = opponent takes 2 prizes on KO, Mega ex = 3, basic = 1. Avoid feeding multi-prize KOs.")
    L.append("- The engine is authoritative; some paper-rule edge cases differ. Heuristics should be conditional "
             "on board state (bench space, energy, prizes, KO availability), not raw card value.")
    L.append("")

    order = ["Pokemon", "Trainer", "Energy"]
    for cat in order:
        items = sorted(by_cat.get(cat, []), key=lambda x: (-x[0], cards.get(x[1], {}).get("name", "")))
        if not items:
            continue
        L.append(f"## {cat} ({sum(n for n, _, _ in items)} cards)")
        L.append("")
        for n, cid, c in items:
            nm = c.get("name", f"id {cid}")
            head = f"### {n}x {nm}  (id {cid})"
            L.append(head)
            meta = []
            if c.get("stage") and c["stage"] not in ("n/a", ""):
                meta.append(c["stage"])
            if c.get("rule") and c["rule"] not in ("n/a", ""):
                meta.append(c["rule"])
            if c.get("hp") and c["hp"] not in ("n/a", ""):
                meta.append(f"HP {c['hp']}")
            if c.get("type") and c["type"] not in ("n/a", ""):
                meta.append(f"Type {c['type']}")
            if c.get("weak"):
                meta.append(f"Weak {c['weak']}")
            if c.get("resist"):
                meta.append(f"Resist {c['resist']}")
            if c.get("retreat") and c["retreat"] not in ("n/a", ""):
                meta.append(f"Retreat {c['retreat']}")
            if meta:
                L.append("- " + " | ".join(meta))
            for mv, cost, dmg, eff in c.get("moves", []):
                seg = f"- **{mv}** (cost {cost or '0'}{', dmg ' + dmg if dmg else ''}): {eff or '-'}"
                L.append(seg)
            for eff in c.get("effects", []):
                L.append(f"- Effect: {eff}")
            for key, note in GOTCHAS.items():
                if key.lower() in nm.lower():
                    L.append(f"- **GOTCHA (from our testing):** {note}")
            L.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT} ({len(L)} lines, {sum(deck.values())} cards / {len(deck)} distinct)")
    print("category counts:", {k: sum(n for n, _, _ in v) for k, v in by_cat.items()})


if __name__ == "__main__":
    main()
