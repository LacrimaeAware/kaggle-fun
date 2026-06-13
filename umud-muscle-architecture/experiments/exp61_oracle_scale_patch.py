"""Apply human-oracle scale notes to the current scale partition.

This does not assume the human note is a final px/cm label. If the note only
states visible field depth (for example 4 cm), the script estimates px/cm from
the detected ultrasound field height and reports disagreement with the existing
tick/router scale. The output is an audit table, not an automatic submission.
"""

from __future__ import annotations

import json
import re
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


def _texture_anchor(gray: np.ndarray) -> tuple[int, int, tuple[int, int, int, int]] | None:
    g = gray.astype(np.float32)
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    texture = cv2.GaussianBlur((gx + gy), (7, 7), 0)
    thresh = max(5.0, float(np.percentile(texture, 75)) * 0.55)
    active = texture > thresh
    active &= g > 6
    active = cv2.morphologyEx(active.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(active, 8)
    h_img, w_img = gray.shape
    candidates = []
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if w < 0.12 * w_img or h < 0.25 * h_img:
            continue
        fill = area / max(w * h, 1)
        score = area * min(fill, 0.8)
        candidates.append((score, x, y, w, h))
    if not candidates:
        ys, xs = np.where(active)
        if len(xs) == 0:
            return None
        return int(np.median(xs)), int(np.median(ys)), (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    _score, x, y, w, h = max(candidates, key=lambda item: item[0])
    component = active[y : y + h, x : x + w].astype(bool)
    ys, xs = np.where(component)
    if len(xs) == 0:
        return int(x + w / 2), int(y + h / 2), (x, y, x + w - 1, y + h - 1)
    return int(x + np.median(xs)), int(y + np.median(ys)), (x, y, x + w - 1, y + h - 1)


def _has_uniform_vertical_run(gray: np.ndarray, x: int, y: int, run: int, tol: int) -> bool:
    y0 = max(0, y - run)
    y1 = min(gray.shape[0], y + run + 1)
    vals = gray[y0:y1, x].astype(int)
    return len(vals) >= run and int(vals.max()) - int(vals.min()) <= tol


def _has_uniform_horizontal_run(gray: np.ndarray, x: int, y: int, run: int, tol: int) -> bool:
    x0 = max(0, x - run)
    x1 = min(gray.shape[1], x + run + 1)
    vals = gray[y, x0:x1].astype(int)
    return len(vals) >= run and int(vals.max()) - int(vals.min()) <= tol


def detect_field_rect(gray: np.ndarray) -> dict[str, float] | None:
    """Detect the visible ultrasound field by scanning out from image texture.

    The earlier detector selected the largest textured component. That often
    overcounted UI/canvas height. This follows the user's simpler heuristic:
    start from a point inside the scan texture, move outward, and stop at
    sustained constant-color runs, which are overwhelmingly UI/background rather
    than speckled ultrasound.
    """
    anchor = _texture_anchor(gray)
    if anchor is None:
        return None
    cx, cy, (px0, py0, px1, py1) = anchor
    h_img, w_img = gray.shape
    run = max(10, min(h_img, w_img) // 45)
    tol = 1
    y_samples = [int(v) for v in np.linspace(max(0, py0 + 5), min(h_img - 1, py1 - 5), 17)]
    x_samples = [int(v) for v in np.linspace(max(0, px0 + 5), min(w_img - 1, px1 - 5), 17)]

    def vertical_background_vote(x: int) -> float:
        votes = [_has_uniform_vertical_run(gray, x, y, run, tol) for y in y_samples]
        return float(sum(votes) / max(len(votes), 1))

    def horizontal_background_vote(y: int) -> float:
        votes = [_has_uniform_horizontal_run(gray, x, y, run, tol) for x in x_samples]
        return float(sum(votes) / max(len(votes), 1))

    def left_edge() -> int:
        streak = 0
        for x in range(cx, -1, -1):
            streak = streak + 1 if vertical_background_vote(x) >= 0.65 else 0
            if streak >= 4:
                return min(w_img - 1, x + streak)
        return 0

    def right_edge() -> int:
        streak = 0
        for x in range(cx, w_img):
            streak = streak + 1 if vertical_background_vote(x) >= 0.65 else 0
            if streak >= 4:
                return max(0, x - streak)
        return w_img - 1

    def top_edge() -> int:
        streak = 0
        for y in range(cy, -1, -1):
            streak = streak + 1 if horizontal_background_vote(y) >= 0.65 else 0
            if streak >= 4:
                return min(h_img - 1, y + streak)
        return 0

    def bottom_edge() -> int:
        streak = 0
        for y in range(cy, h_img):
            streak = streak + 1 if horizontal_background_vote(y) >= 0.65 else 0
            if streak >= 4:
                return max(0, y - streak)
        return h_img - 1

    x0, x1 = left_edge(), right_edge()
    y0, y1 = top_edge(), bottom_edge()
    if x1 <= x0 or y1 <= y0:
        return None
    return {
        "x": int(x0),
        "y": int(y0),
        "w": int(x1 - x0 + 1),
        "h": int(y1 - y0 + 1),
        "anchor_x": int(cx),
        "anchor_y": int(cy),
        "texture_box_x": int(px0),
        "texture_box_y": int(py0),
        "texture_box_w": int(px1 - px0 + 1),
        "texture_box_h": int(py1 - py0 + 1),
        "uniform_run": int(run),
        "uniform_tol": int(tol),
        "method": "texture_anchor_uniform_run_scan",
    }


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


def parse_depth_mm(value: object) -> float | None:
    """Parse reviewed/guessed visible depth into millimeters.

    Bare values under 10 are treated as centimeters (`3.5` -> 35 mm), matching
    the scale-review UI convention and the visible depth labels in these files.
    """
    if value is None:
        return None
    s = str(value).strip().lower().replace(",", ".")
    if not s or s == "nan":
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(cm|mm)?", s)
    if not match:
        return None
    out = float(match.group(1))
    unit = match.group(2) or ""
    if unit == "cm" or (not unit and out < 10.0):
        out *= 10.0
    if not np.isfinite(out) or out < 15.0 or out > 90.0:
        return None
    return out


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
        oracle_depth = parse_depth_mm(note.get("oracle_depth_mm"))
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
