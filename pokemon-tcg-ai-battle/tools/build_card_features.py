"""L0: build agent/card_features.json, the per-card feature vector the state encoder uses.

Combines the functional classification (registry/card_review.json: confirmed human_tags win
over proposed llm_tags) with the RULE attributes from cards_full.json (the complete engine
data). The rule attributes are deterministic features we must encode because they ARE the
rules of the game:
  - prize: prizes the opponent takes when this Pokemon is KO'd (basic/evolved 1, ex 2, Mega 3).
  - cardType: Pokemon / Item / Supporter / Stadium / Tool / Basic|Special Energy (drives the
    per-turn play limits: Items unlimited, Supporter 1/turn, Stadium 1/turn, Energy 1 attach).
  - ex / mega / tera (no bench damage) / aceSpec (max 1 per deck) / stage / retreat /
    weakness / resistance / energy type.
plus the numeric attack data for energy-affordance reasoning.

Re-run after the classification or card data changes: python tools/build_card_features.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "external" / "official" / "cards_full.json"
REVIEW = ROOT / "registry" / "card_review.json"
OUT = ROOT / "agent" / "card_features.json"
CT = {0: "pokemon", 1: "item", 2: "tool", 3: "supporter", 4: "stadium", 5: "basic_energy", 6: "special_energy"}


def main() -> None:
    full = json.loads(FULL.read_text(encoding="utf-8"))
    review = json.loads(REVIEW.read_text(encoding="utf-8")).get("cards", {})
    out = {}
    for cid, c in full.items():
        r = review.get(cid, {})
        tags = r.get("human_tags") if r.get("status") == "confirmed" and r.get("human_tags") is not None else (r.get("llm_tags") or [])
        atks = c.get("atk", [])
        ct = c.get("ct", 0)
        prize = 3 if c.get("mega") else 2 if c.get("ex") else (1 if ct == 0 else 0)
        stage = "stage2" if c.get("s2") else "stage1" if c.get("s1") else "basic" if c.get("basic") else None
        out[cid] = {
            "n": c.get("n"),
            "tags": tags,
            "ct": ct, "type": CT.get(ct, "?"),
            "hp": c.get("hp", 0),
            "prize": prize,                       # prizes given to the opponent on KO
            "ex": bool(c.get("ex")), "mega": bool(c.get("mega")),
            "tera": bool(c.get("tera")), "ace_spec": bool(c.get("aceSpec")),
            "stage": stage, "retreat": c.get("retreat", 0),
            "ty": c.get("ty") or "", "wk": c.get("wk") or "", "rs": c.get("rs") or "",
            "best_dmg": max([a.get("d", 0) for a in atks], default=0),
            "min_cost": min([a.get("c", 0) for a in atks], default=0),
            "max_cost": max([a.get("c", 0) for a in atks], default=0),
            "n_atk": len(atks),
            "atks": [{"dmg": a.get("d", 0), "cost": a.get("c", 0), "cE": a.get("cE", [])} for a in atks],
        }
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tagged = sum(1 for v in out.values() if v["tags"])
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} cards, {tagged} with functional tags, "
          f"with rule attributes (prize/type/ex/mega/tera/ace_spec/stage/retreat). {OUT.stat().st_size//1024} KB")


if __name__ == "__main__":
    main()
