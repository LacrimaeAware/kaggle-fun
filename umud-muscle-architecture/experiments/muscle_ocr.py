"""Read the printed muscle label off each test image via OCR (reuses easyocr, like scale_ocr.py).

Many test images print the muscle in German device UI text, e.g. "Quadriceps", "links/rechts VL",
"rechts RF", "GM rechts", "Tri surae". The muscle tells us the expected pennation range, which lets
us correct PA per muscle instead of with a blunt global shift:
  RF  = Rectus Femoris      -> LOW pennation  (~5-15deg)   model usually fine; DO NOT lift
  VL  = Vastus Lateralis    -> MODERATE       (~15-20deg)
  GM  = Gastrocnemius Med.  -> HIGH           (~17-25deg)  model tends to UNDER-read; lift if low
  SOL = Soleus              -> HIGH-ish
  TA  = Tibialis Anterior   -> LOW-MODERATE   (~9-12deg)
  QUAD= Quadriceps group    -> ambiguous VL/RF unless a VL/RF token is also present

Run in the environment that has easyocr installed (same one that runs scale_ocr.py):
    python experiments/muscle_ocr.py
Outputs results/muscle_ocr.csv (image_id, muscle, side, ocr_text, has_muscle) and prints coverage.
Tune CROP_REGIONS / patterns after the first run if coverage is low.
"""
from __future__ import annotations
import re
import glob
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "muscle_ocr.csv"

# specific muscles first so e.g. "rechts RF" maps to RF, not the QUAD fallback
MUSCLE_PATTERNS = [
    ("RF",  r"rectus|\brf\b"),
    ("GM",  r"gastro\w*|\bgm\b|tri.?surae|trizeps|\bsurae\b"),
    ("SOL", r"soleus|\bsol\b"),
    ("VL",  r"vastus|\bvl\b"),
    ("TA",  r"tibialis|\bta\b"),
    ("QUAD", r"quadri\w*|\bquad\b"),
]
SIDE_PATTERNS = [("R", r"rechts|\bright\b|\br\b"), ("L", r"links|\bleft\b|\bl\b")]


def find_test_dir() -> Path:
    hits = list(ROOT.glob("data/test_images*/**/IMG_*.tif")) + list(ROOT.glob("data/test_images*/IMG_*.tif"))
    if not hits:
        raise SystemExit("could not find test images under data/test_images*/")
    return hits[0].parent


def classify(text: str):
    t = text.lower()
    muscle = next((m for m, pat in MUSCLE_PATTERNS if re.search(pat, t)), "")
    side = next((s for s, pat in SIDE_PATTERNS if re.search(pat, t)), "")
    return muscle, side


def make_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=False, verbose=False)  # add "de" if German words are missed


def main():
    test_dir = find_test_dir()
    files = sorted(glob.glob(str(test_dir / "*.tif")) + glob.glob(str(test_dir / "*.png")))
    print(f"test dir: {test_dir}  ({len(files)} images)")
    rd = make_reader()
    rows = []
    for i, f in enumerate(files):
        im = cv2.imread(f)
        if im is None:
            rows.append((Path(f).name, "", "", "", 0)); continue
        H, W = im.shape[:2]
        # UI muscle text usually sits in the top strip and/or the top-left panel
        crops = [im[0:int(H * 0.18), 0:W], im[0:int(H * 0.55), 0:int(W * 0.35)]]
        texts = []
        for c in crops:
            try:
                texts += [r[1] for r in rd.readtext(c, detail=1, paragraph=False)]
            except Exception:
                pass
        text = " | ".join(texts)
        muscle, side = classify(text)
        rows.append((Path(f).name, muscle, side, text[:160], int(bool(muscle))))
        if i % 25 == 0:
            print(f"  {i}/{len(files)} ...", flush=True)
    df = pd.DataFrame(rows, columns=["image_id", "muscle", "side", "ocr_text", "has_muscle"])
    df.to_csv(OUT, index=False)
    print(f"\ncoverage (muscle read): {int(df.has_muscle.sum())}/{len(df)}")
    print(df["muscle"].value_counts(dropna=False).to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
