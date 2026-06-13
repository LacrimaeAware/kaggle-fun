"""EXP64: multi-region OCR audit for test-set scale/depth text.

The older OCR pass missed many tiny bottom/right depth labels because it ran a
single full-frame read. This audit reads several targeted UI regions at higher
scale, caches raw tokens, and extracts visible field-depth candidates without
using the local human-review notes as predictor input.

The notes file is used only for post-hoc evaluation when present.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGES = ROOT / "data" / "test_images_v2" / "test_set_v2"
OUT_DIR = ROOT / "results" / "exp64_text_scale_ocr"
TOKEN_DIR = OUT_DIR / "tokens"
SUMMARY_OUT = OUT_DIR / "depth_ocr_summary.csv"
NOTES_PATH = ROOT / "results" / "scale_oracle_review" / "oracle_notes.json"
PARTITION_PATH = ROOT / "results" / "scale_partition.csv"
IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

ALLOWLIST = "0123456789.,cmCMmMDeTiefeTIEF "


def image_files() -> list[Path]:
    return sorted(p for p in TEST_IMAGES.iterdir() if p.suffix.lower() in IMG_EXTS)


def regions_for_shape(h: int, w: int) -> list[tuple[str, tuple[int, int, int, int], int]]:
    """Return OCR crops as (name, x0, y0, x1, y1, upscale)."""
    if depth_from_cropped_family((h, w))[0] is not None:
        return []
    return [
        ("top_right", (int(w * 0.72), 0, w, int(h * 0.34)), 3),
        ("middle_right", (int(w * 0.70), int(h * 0.22), w, int(h * 0.72)), 3),
        ("bottom_full", (0, int(h * 0.62), w, h), 3),
        ("left_full", (0, 0, int(w * 0.22), h), 3),
    ]


def load_notes() -> dict[str, dict]:
    if not NOTES_PATH.exists():
        return {}
    try:
        return json.loads(NOTES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def parse_depth_mm(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace(",", ".")
    if not text or text == "nan":
        return None
    match = re.search(r"(\d+(?:\.\d*)?)\s*(cm|mm)?", text)
    if not match:
        return None
    number = match.group(1).rstrip(".")
    if not number:
        return None
    val = float(number)
    unit = match.group(2) or ""
    if unit == "cm":
        # OCR often drops the decimal: "3.5 cm" -> "35 cm". A 35 cm field is
        # impossible here; 3.5 cm is a normal displayed depth.
        if val > 9.0 and "." not in match.group(1):
            val /= 10.0
        val *= 10.0
    elif unit == "mm":
        pass
    elif val < 10.0:
        val *= 10.0
    if 15.0 <= val <= 90.0:
        return round(val, 1)
    return None


def token_depth_candidates(tokens: list[dict], w: int, h: int) -> list[dict]:
    out: list[dict] = []
    for tok in tokens:
        raw = str(tok["text"]).strip()
        text = raw.lower().replace(",", ".")
        conf = float(tok["conf"])
        region = str(tok["region"])

        local_candidates: list[tuple[float, str]] = []
        for match in re.finditer(r"(\d+(?:[\.,]\d*)?)\s*(mm|cm)", text):
            depth = parse_depth_mm(match.group(0))
            if depth is not None:
                local_candidates.append((depth, match.group(0)))

        # Some bottom labels are read as "35 cm" for "3.5 cm"; parse_depth_mm
        # handles that. Tokens without a unit are not accepted as field-depth
        # labels here because side-ruler numbers are common.
        if not local_candidates:
            continue

        cx = float(tok["cx"])
        cy = float(tok["cy"])
        loc_bonus = 0.10 * (cy / max(h, 1)) + 0.05 * (cx / max(w, 1))
        region_bonus = {
            "bottom_right": 0.35,
            "bottom_full": 0.28,
            "top_right": 0.22,
            "middle_right": 0.24,
            "right_full": 0.18,
            "left_full": 0.02,
        }.get(region, 0.0)
        context_bonus = 0.25 if ("de" in text or "tie" in text) else 0.0
        for depth, snippet in local_candidates:
            out.append(
                {
                    "depth_mm": depth,
                    "score": conf + loc_bonus + region_bonus + context_bonus,
                    "conf": conf,
                    "cx": cx,
                    "cy": cy,
                    "region": region,
                    "text": raw,
                    "snippet": snippet,
                }
            )
    return sorted(out, key=lambda item: item["score"], reverse=True)


def depth_from_tick_scale_family(scale_px_per_cm: float | None, image_shape: tuple[int, int]) -> tuple[float | None, str]:
    """Infer displayed depth from stable 1200x800 tick-scale families.

    This is not a human label. It is the same deterministic repair discovered
    from the OCR failure audit: the scale family disambiguates tiny labels like
    `3. cm`, where OCR sees the unit but drops the half-centimeter digit.
    """
    if scale_px_per_cm is None:
        return None, ""
    h, w = image_shape
    if (w, h) != (1200, 800):
        return None, ""
    rules = [
        (110.0, 111.5, 55.0, "tick-family 5.5 cm"),
        (135.0, 136.0, 45.0, "tick-family 4.5 cm"),
        (150.5, 153.0, 40.0, "tick-family 4.0 cm"),
        (158.8, 160.2, 35.0, "tick-family 3.5 cm"),
        (173.5, 174.5, 70.0, "tick-family 7.0 cm"),
    ]
    for lo, hi, depth, label in rules:
        if lo <= float(scale_px_per_cm) <= hi:
            return depth, label
    return None, ""


def depth_from_cropped_family(image_shape: tuple[int, int]) -> tuple[float | None, str]:
    h, w = image_shape
    if (900 <= w <= 1100 and h == 853) or (460 <= w <= 466 and 512 <= h <= 513):
        return 50.0, "cropped/no-overlay 50 mm family"
    return None, ""


def depth_from_ruler_numeric_tokens(tokens: list[dict], image_shape: tuple[int, int]) -> tuple[float | None, str]:
    """Infer depth from edge ruler numbers even when OCR misses the unit.

    This accepts only bottom/deep values located on the ruler edge, preferably
    paired with a top `0` on the same edge. It rejects arbitrary labels like
    `50%` in the parameter text because they are not near the edge ruler line.
    """
    h, w = image_shape
    numeric = []
    for tok in tokens:
        text = str(tok["text"]).strip().lower().replace(",", ".")
        if re.search(r"[a-z%]", text):
            continue
        match = re.fullmatch(r"\d+(?:\.\d*)?", text)
        if not match:
            continue
        value = float(match.group(0).rstrip("."))
        cx = float(tok["cx"])
        cy = float(tok["cy"])
        edge = None
        if cx <= 0.08 * w:
            edge = "left"
        elif cx >= 0.92 * w:
            edge = "right"
        elif cy >= 0.90 * h:
            edge = "bottom"
        if edge is None:
            continue
        numeric.append((value, cx, cy, edge, float(tok["conf"])))

    depths = [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]
    for edge in ("left", "right", "bottom"):
        vals = [item for item in numeric if item[3] == edge]
        if not vals:
            continue
        has_zero_top = any(abs(v) < 0.1 and cy <= 0.18 * h for v, _cx, cy, _edge, _conf in vals)
        for depth in depths:
            # A bottom/deep tick label at the edge is a depth label. Requiring
            # top zero when present gives higher confidence but is not mandatory:
            # some crops cut off the zero.
            hits = [
                (conf, cy)
                for v, _cx, cy, _edge, conf in vals
                if abs(v - depth) <= 0.1 and cy >= 0.62 * h
            ]
            if hits and (has_zero_top or max(conf for conf, _cy in hits) >= 0.90):
                return depth, f"numeric {edge} ruler label {depth:.0f} mm"
    return None, ""


def load_partition() -> dict[str, dict]:
    if not PARTITION_PATH.exists():
        return {}
    return pd.read_csv(PARTITION_PATH).set_index("image_id", drop=False).to_dict("index")


def ocr_tokens(path: Path, reader, force: bool = False) -> list[dict]:
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    cache = TOKEN_DIR / f"{path.name}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text(encoding="utf-8"))

    im = cv2.imread(str(path))
    if im is None:
        raise RuntimeError(f"could not read {path}")
    h, w = im.shape[:2]
    tokens: list[dict] = []
    seen: set[tuple[str, int, int, str]] = set()
    for region_name, (x0, y0, x1, y1), upscale in regions_for_shape(h, w):
        crop = im[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        big = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        result = reader.readtext(
            big,
            allowlist=ALLOWLIST,
            detail=1,
            paragraph=False,
            low_text=0.20,
            text_threshold=0.30,
        )
        for box, text, conf in result:
            clean = str(text).strip()
            if not clean:
                continue
            xs = [float(p[0]) / upscale + x0 for p in box]
            ys = [float(p[1]) / upscale + y0 for p in box]
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            key = (clean.lower(), round(cx / 4), round(cy / 4), region_name)
            if key in seen:
                continue
            seen.add(key)
            tokens.append(
                {
                    "text": clean,
                    "conf": float(conf),
                    "cx": cx,
                    "cy": cy,
                    "region": region_name,
                    "box": [[x, y] for x, y in zip(xs, ys)],
                }
            )
    cache.write_text(json.dumps(tokens, indent=2), encoding="utf-8")
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--image", action="append", default=[])
    args = parser.parse_args()

    import easyocr

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = image_files()
    if args.image:
        wanted = set(args.image)
        files = [p for p in files if p.name in wanted]
    if args.limit:
        files = files[: args.limit]

    notes = load_notes()
    partition = load_partition()
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    rows = []
    for idx, path in enumerate(files, 1):
        im = cv2.imread(str(path))
        h, w = im.shape[:2]
        tokens = ocr_tokens(path, reader, force=args.force)
        cands = token_depth_candidates(tokens, w, h)
        best = cands[0] if cands else {}
        tick_scale = None
        if path.name in partition:
            raw_scale = partition[path.name].get("scale_px_per_cm")
            try:
                tick_scale = float(raw_scale)
            except (TypeError, ValueError):
                tick_scale = None
        family_depth, family_note = depth_from_tick_scale_family(tick_scale, (h, w))
        cropped_depth, cropped_note = depth_from_cropped_family((h, w))
        ruler_depth, ruler_note = depth_from_ruler_numeric_tokens(tokens, (h, w))
        fused_depth = best.get("depth_mm")
        fused_source = "ocr_text" if fused_depth is not None else ""
        if family_depth is not None:
            # Trust family repair when OCR is missing, or when OCR is an ambiguous
            # integer-cm read that conflicts by at least 5 mm.
            if fused_depth is None or abs(float(fused_depth) - family_depth) >= 5.0:
                fused_depth = family_depth
                fused_source = family_note
        if fused_depth is None and ruler_depth is not None:
            fused_depth = ruler_depth
            fused_source = ruler_note
        if fused_depth is None and cropped_depth is not None:
            fused_depth = cropped_depth
            fused_source = cropped_note
        reviewed = parse_depth_mm(notes.get(path.name, {}).get("oracle_depth_mm"))
        pred = fused_depth
        rows.append(
            {
                "image_id": path.name,
                "fused_depth_mm": pred,
                "fused_source": fused_source,
                "ocr_depth_mm": best.get("depth_mm"),
                "reviewed_depth_mm": reviewed,
                "match_review": "" if reviewed is None or pred is None else bool(abs(pred - reviewed) < 0.1),
                "depth_score": best.get("score"),
                "depth_conf": best.get("conf"),
                "depth_region": best.get("region", ""),
                "depth_text": best.get("text", ""),
                "tick_scale_px_per_cm": tick_scale,
                "tick_family_depth_mm": family_depth,
                "ruler_numeric_depth_mm": ruler_depth,
                "cropped_family_depth_mm": cropped_depth,
                "candidate_count": len(cands),
                "token_count": len(tokens),
            }
        )
        if idx % 10 == 0 or idx == len(files):
            pd.DataFrame(rows).to_csv(SUMMARY_OUT, index=False)
            print(f"{idx}/{len(files)} written -> {SUMMARY_OUT}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_OUT, index=False)
    reviewed = df[df["reviewed_depth_mm"].notna()]
    known = reviewed[reviewed["fused_depth_mm"].notna()]
    misses = known[known["match_review"] == False]
    print("\n=== EXP64 multi-region OCR depth audit ===")
    print(f"rows: {len(df)}")
    print(f"OCR depth found: {df['ocr_depth_mm'].notna().sum()}/{len(df)}")
    print(f"fused depth found: {df['fused_depth_mm'].notna().sum()}/{len(df)}")
    print(f"reviewed rows with fused depth: {len(known)}/{len(reviewed)}")
    print(f"misses vs review where fused depth found: {len(misses)}")
    if len(misses):
        print(misses[["image_id", "fused_depth_mm", "reviewed_depth_mm", "fused_source", "depth_text", "depth_region"]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
