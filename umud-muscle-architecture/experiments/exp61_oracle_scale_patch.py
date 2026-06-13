"""Apply human-oracle scale notes to the current scale partition.

This does not assume the human note is a final px/cm label. If the note only
states visible field depth (for example 4 cm), the script estimates px/cm from
the detected ultrasound field height and reports disagreement with the existing
tick/router scale. The output is an audit table, not an automatic submission.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TEST_IMAGES = ROOT / "data" / "test_images_v2" / "test_set_v2"
NOTES = RESULTS / "scale_oracle_review" / "oracle_notes.json"
PARTITION = RESULTS / "scale_partition.csv"
OUT = RESULTS / "scale_oracle_review" / "oracle_scale_patch_audit.csv"
OVERRIDES = RESULTS / "scale_oracle_review" / "oracle_scale_overrides.csv"


def longest_true_run(mask: np.ndarray, min_len: int) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    start: int | None = None
    for idx, ok in enumerate(mask.tolist() + [False]):
        if ok and start is None:
            start = idx
        elif not ok and start is not None:
            end = idx
            if end - start >= min_len and (best is None or end - start > best[1] - best[0]):
                best = (start, end)
            start = None
    return best


def detect_field_rect(gray: np.ndarray) -> dict[str, float] | None:
    """Detect the visible ultrasound field, not the whole UI panel.

    The field is the largest textured non-uniform rectangle. This deliberately
    rejects black UI margins, flat parameter panels, and thin ruler tick strips.
    """
    g = gray.astype(np.float32)
    # Local vertical/horizontal texture. The scan field has sustained texture;
    # UI margins and labels are mostly flat with sparse strokes.
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    texture = cv2.GaussianBlur((gx + gy), (7, 7), 0)
    thresh = max(5.0, float(np.percentile(texture, 75)) * 0.55)
    active = texture > thresh
    active &= g > 6
    active = cv2.morphologyEx(active.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(active, 8)
    if n <= 1:
        return None
    candidates = []
    h_img, w_img = gray.shape
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if w < 0.12 * w_img or h < 0.25 * h_img:
            continue
        fill = area / max(w * h, 1)
        # Ruler strips are tall but too narrow/sparse; parameter panels are not textured.
        score = area * min(fill, 0.8)
        candidates.append((score, x, y, w, h, area, fill))
    if not candidates:
        rows = active.mean(axis=1)
        cols = active.mean(axis=0)
        yr = longest_true_run(rows > 0.05, max(20, gray.shape[0] // 8))
        xr = longest_true_run(cols > 0.05, max(40, gray.shape[1] // 8))
        if yr is None or xr is None:
            return None
        x0, x1 = xr
        y0, y1 = yr
        return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "method": "density_fallback"}
    _score, x, y, w, h, area, fill = max(candidates, key=lambda item: item[0])
    return {"x": x, "y": y, "w": w, "h": h, "area": area, "fill": round(fill, 4), "method": "textured_component"}


def as_float(value: object) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        out = float(s)
        if not np.isfinite(out):
            return None
        return out
    except ValueError:
        return None


def main() -> None:
    if not NOTES.exists():
        raise SystemExit(f"missing {NOTES}; review some scale rows first")
    if not PARTITION.exists():
        raise SystemExit(f"missing {PARTITION}; run scale_ocr.py first")

    notes = json.loads(NOTES.read_text(encoding="utf-8"))
    partition = pd.read_csv(PARTITION).set_index("image_id", drop=False)
    rows = []
    for image_id, note in sorted(notes.items()):
        p = TEST_IMAGES / image_id
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        rect = detect_field_rect(gray) if gray is not None else None
        oracle_depth = as_float(note.get("oracle_depth_mm"))
        oracle_scale = as_float(note.get("oracle_scale_px_per_cm"))
        field_scale = None
        if rect and oracle_depth and oracle_depth > 0:
            field_scale = rect["h"] / oracle_depth * 10.0
        old = partition.loc[image_id].to_dict() if image_id in partition.index else {}
        old_scale = as_float(old.get("scale_px_per_cm"))
        tick_scale = as_float(old.get("tick_px_cm"))
        chosen = oracle_scale or field_scale or old_scale
        old_vs_field_pct = None
        if old_scale and field_scale:
            old_vs_field_pct = 100.0 * abs(old_scale - field_scale) / ((old_scale + field_scale) / 2.0)
        action = "no_scale_change"
        if oracle_scale:
            action = "oracle_pxcm_override"
        elif field_scale and (old_scale is None or (old_vs_field_pct is not None and old_vs_field_pct > 8.0)):
            action = "field_depth_candidate"
        elif field_scale:
            action = "field_depth_confirms_existing"
        rows.append(
            {
                "image_id": image_id,
                "status": note.get("status", ""),
                "old_tier": old.get("tier", ""),
                "old_scale_px_per_cm": old_scale,
                "tick_px_cm": tick_scale,
                "old_text_depth_mm": old.get("text_depth_mm", ""),
                "oracle_depth_mm": oracle_depth,
                "oracle_scale_px_per_cm": oracle_scale,
                "field_h_px": rect.get("h") if rect else None,
                "field_method": rect.get("method") if rect else None,
                "field_scale_px_per_cm": field_scale,
                "old_vs_field_pct": old_vs_field_pct,
                "chosen_scale_px_per_cm": chosen,
                "action": action,
                "comment": note.get("comment", ""),
            }
        )
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    overrides = out[out["action"].isin(["oracle_pxcm_override", "field_depth_candidate"])].copy()
    overrides[["image_id", "chosen_scale_px_per_cm", "action", "comment"]].to_csv(OVERRIDES, index=False)

    print("\n=== EXP61 oracle scale patch audit ===")
    print(out["action"].value_counts().to_string())
    cols = [
        "image_id",
        "old_tier",
        "old_scale_px_per_cm",
        "oracle_depth_mm",
        "field_h_px",
        "field_scale_px_per_cm",
        "old_vs_field_pct",
        "action",
    ]
    print("\nrows:")
    print(out[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nwrote:\n  {OUT}\n  {OVERRIDES}")


if __name__ == "__main__":
    main()
