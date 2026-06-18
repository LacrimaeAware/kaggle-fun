"""Emit agent/card_stats.json: a compact card-stats table the battler reads at runtime.

Parsed from the official EN_Card_Data.csv (via cards.py): per card id, HP, ex/mega flags,
weakness/resistance energy letters, and each attack's energy cost (as a count and letters)
and damage (int). Small enough to bundle next to main.py in a submission. The battler uses it
for lethal/best-attack decisions and prize-liability (ex=2, mega=3) valuation.

Run: python tools/build_card_stats.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cards

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "agent" / "card_stats.json"


def energy_letters(s: str) -> list[str]:
    return re.findall(r"\{(.)\}", s or "")


def to_int(s: str) -> int:
    m = re.search(r"\d+", s or "")
    return int(m.group()) if m else 0


def main() -> None:
    out = {}
    for cid, c in cards._load().items():
        name = c["name"]
        is_ex = bool(re.search(r"\bex\b", name, re.I))
        is_mega = bool(re.search(r"mega", name, re.I)) and is_ex
        atks = []
        for a in c.get("attacks", []):
            cost = energy_letters(a["cost"])
            atks.append({"cost": len(cost), "cE": cost, "dmg": to_int(a["damage"]), "name": a["name"]})
        out[cid] = {
            "n": name,
            "hp": to_int(c.get("hp", "")),
            "ex": is_ex, "mega": is_mega,
            "wk": "".join(energy_letters(c.get("weakness", ""))),
            "rs": "".join(energy_letters(c.get("resistance", ""))),
            "ty": "".join(energy_letters(c.get("type", ""))),
            "atk": atks,
        }
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(out)} cards, {OUT.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
