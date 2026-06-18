"""Extract the official card art from Card_ID_List_EN.pdf into per-card image files.

The PDF is a card table (pages 0..~38) whose "View Image" cells are internal links to image
pages (each later page holds one card's art). We pair each table link to the card id on its
row, then save that linked page's image as card_images/<card_id>.jpg. The art is the
organizer-provided official card image (jpeg ~451x630, ~28KB), so ~35MB total for 1267 cards.

Run: python tools/scrape_card_images.py
Requires PyMuPDF (pip install pymupdf). Output is gitignored (under data/).
"""
from __future__ import annotations

import json
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "data" / "external" / "official" / "Card_ID_List_EN.pdf"
OUT = ROOT / "data" / "external" / "official" / "card_images"


def card_id_for_link(words: list, link_rect) -> int | None:
    """The card id is the leftmost numeric word on the same table row as the link."""
    ly = (link_rect.y0 + link_rect.y1) / 2
    row = [w for w in words if abs((w[1] + w[3]) / 2 - ly) < 6 and w[4].isdigit()]
    if not row:
        return None
    row.sort(key=lambda w: w[0])      # leftmost column = Card ID
    return int(row[0][4])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(PDF))
    dest_to_id: dict[int, int] = {}
    for pno in range(doc.page_count):
        page = doc[pno]
        links = [l for l in page.get_links() if l.get("kind") == fitz.LINK_GOTO and l.get("page") is not None]
        if not links:
            continue
        words = page.get_text("words")
        for l in links:
            cid = card_id_for_link(words, l["from"])
            if cid is not None:
                dest_to_id[l["page"]] = cid

    manifest: dict[int, str] = {}
    saved = 0
    for dest_page, cid in dest_to_id.items():
        imgs = doc[dest_page].get_images(full=True)
        if not imgs:
            continue
        d = doc.extract_image(imgs[0][0])
        ext = "jpg" if d["ext"] == "jpeg" else d["ext"]
        fn = f"{cid}.{ext}"
        (OUT / fn).write_bytes(d["image"])
        manifest[cid] = fn
        saved += 1
    (OUT / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    print(f"saved {saved} card images to {OUT.relative_to(ROOT)} (manifest.json has {len(manifest)} ids)")


if __name__ == "__main__":
    main()
