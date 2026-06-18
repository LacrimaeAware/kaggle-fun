"""Ingest an external classification (e.g. from a stronger model) into the review tool store.

Reads registry/card_functional_classification.json ({id: {tags, why}}) plus optional
card_functional_manual_overrides.json and card_functional_reviewed_220_ids.json, and writes
registry/card_review.json in the tool's format so the human can verify the work in the
browser. Confidence is tiered so the smart queue surfaces the least-vetted cards first:
manual override 0.97, reviewed 0.90, otherwise 0.75. Any card the human already confirmed in
the tool is preserved. Re-runnable.

Run: python tools/ingest_labels.py   (stop the server first; restart after)
"""
from __future__ import annotations

import json
from pathlib import Path

import review_server as R   # for the canonical taxonomy keys

ROOT = Path(__file__).resolve().parent.parent
CLS = ROOT / "registry" / "card_functional_classification.json"
OVR = ROOT / "registry" / "card_functional_manual_overrides.json"
REV = ROOT / "registry" / "card_functional_reviewed_220_ids.json"
STORE = ROOT / "registry" / "card_review.json"


def load(p, default):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def main() -> None:
    cls = load(CLS, {})
    ovr = load(OVR, {})
    reviewed = set(load(REV, []))
    prev = load(STORE, {}).get("cards", {})
    tax = set(R.TAXONOMY)

    cards = {}
    kept_confirmed = 0
    for cid, v in cls.items():
        src = ovr.get(cid) or v
        tags = [t for t in (src.get("tags") or []) if t in tax]
        why = src.get("why", "")
        conf = 0.97 if cid in ovr else (0.90 if cid in reviewed else 0.75)
        p = prev.get(cid)
        if p and p.get("status") == "confirmed" and p.get("human_tags") is not None:
            cards[cid] = {"llm_tags": tags, "conf": conf, "why": why,
                          "human_tags": p["human_tags"], "status": "confirmed"}
            kept_confirmed += 1
        else:
            cards[cid] = {"llm_tags": tags, "conf": conf, "why": why,
                          "human_tags": None, "status": "proposed"}

    STORE.write_text(json.dumps({"taxonomy": {}, "cards": cards}, ensure_ascii=False, indent=0),
                     encoding="utf-8")
    n_unreviewed = sum(1 for cid in cls if cid not in reviewed and cid not in ovr)
    print(f"ingested {len(cards)} cards: {kept_confirmed} kept as your confirmations, "
          f"{len(reviewed)} reviewed (conf .90), {len(ovr)} overrides (conf .97), "
          f"{n_unreviewed} un-vetted (conf .75, surface first).")


if __name__ == "__main__":
    main()
