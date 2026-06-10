"""Experiment 26: machine-learning-ready scale-cue pseudo-labels.

This does not train a model and does not generate a submission. It exports
reproducible weak labels for visual scale cues so the current deterministic
detectors can become teachers for a learned cue detector.

Important rule posture: labels on target images are generated only by code. Do
not hand-correct these masks/boxes and then use them for a competition
submission; that would cross into human labeling of test records.

Outputs:
    results/scale_cue_pseudolabels/manifest.csv
    results/scale_cue_pseudolabels/summary.csv
    results/scale_cue_pseudolabels/masks/*.png
    results/scale_cue_pseudolabels/overlays/*.jpg
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import scale_ticks as ST  # noqa: E402
import tick_calibration as TC  # noqa: E402
from exp21_scale_tail_recovery import recover_bottom_scale_bar_3cm  # noqa: E402

IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
TEST = ROOT / "data" / "test_images_v2" / "test_set_v2"
OUT = ROOT / "results" / "scale_cue_pseudolabels"
MASK_DIR = OUT / "masks"
OVERLAY_DIR = OUT / "overlays"
MANIFEST = OUT / "manifest.csv"
SUMMARY = OUT / "summary.csv"

STRICT_CONF = {
    "png_left_ruler": 0.5,
    "left_ruler_1cm": 0.5,
    "bottom_ticks": 0.9,
    "right_ruler_5mm": 0.5,
    "family_b_signature": 1.0,
    "bottom_scale_bar_3cm": 0.5,
}

COLORS = {
    "bottom_ticks": (0, 255, 0),
    "left_ruler_1cm": (0, 220, 255),
    "png_left_ruler": (0, 160, 255),
    "right_ruler_5mm": (255, 80, 80),
    "family_b_signature": (255, 180, 0),
    "bottom_scale_bar_3cm": (255, 0, 255),
}


def display_rgb(gray: np.ndarray) -> Image.Image:
    lo, hi = np.percentile(gray, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    disp = np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB))


def clamp_box(box: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (
        max(0, min(w - 1, int(x0))),
        max(0, min(h - 1, int(y0))),
        max(1, min(w, int(x1))),
        max(1, min(h, int(y1))),
    )


def rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: int | tuple[int, int, int], width: int = 1):
    x0, y0, x1, y1 = box
    if isinstance(fill, int):
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=fill)
    else:
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=fill, width=width)


def base_record(image_id: str, gray: np.ndarray, method: str, det: dict, cue_class: str, box, points):
    h, w = gray.shape
    conf = float(det.get("conf", det.get("confidence", 0.0)))
    return {
        "image_id": image_id,
        "height": h,
        "width": w,
        "cue_method": method,
        "cue_class": cue_class,
        "scale_px_per_cm": det.get("scale_px_per_cm"),
        "confidence": conf,
        "strict": conf >= STRICT_CONF.get(method, 1.0),
        "box_x0": box[0],
        "box_y0": box[1],
        "box_x1": box[2],
        "box_y1": box[3],
        "n_marks": len(points),
        "points_json": json.dumps(points, separators=(",", ":")),
    }


def add_bottom_ticks(rows, image_id: str, gray: np.ndarray):
    det = ST.recover_scale(gray, tick_cm=1.0)
    if not det or det.get("conf", 0.0) < STRICT_CONF["bottom_ticks"]:
        return
    h, w = gray.shape
    yb = int(det["baseline_y"])
    top = max(0, yb - max(24, int(0.03 * h)))
    box = clamp_box((0, top, w, min(h, yb + 3)), w, h)
    points = [{"x": int(x), "y0": top, "y1": min(h - 1, yb + 2)} for x in det.get("peaks", [])]
    rows.append(("bottom_ticks", det, base_record(image_id, gray, "bottom_ticks", det, "bottom_tick_axis", box, points)))


def add_left_ruler(rows, image_id: str, gray: np.ndarray):
    det = ST.recover_scale_left_ruler(gray, x_max=30, tick_cm=1.0)
    if not det or det.get("conf", 0.0) < STRICT_CONF["left_ruler_1cm"]:
        return
    h, w = gray.shape
    ys = [int(y) for y in det.get("peaks", [])]
    if not ys:
        return
    box = clamp_box((0, max(0, min(ys) - 5), min(48, w), min(h, max(ys) + 6)), w, h)
    points = [{"x0": 0, "x1": min(47, w - 1), "y": y} for y in ys]
    rows.append(("left_ruler_1cm", det, base_record(image_id, gray, "left_ruler_1cm", det, "left_ruler_ticks", box, points)))


def add_png_left_ruler(rows, image_id: str, gray: np.ndarray):
    if not image_id.lower().endswith(".png"):
        return
    cand = TC.png_left_ruler_candidate(gray, 5.0)
    if cand is None or cand.confidence < STRICT_CONF["png_left_ruler"]:
        return
    h, w = gray.shape
    x0, _y0, x1, _y1 = cand.strip_box
    ys = [int(y) for y in cand.peaks]
    if not ys:
        return
    det = {
        "scale_px_per_cm": cand.px_per_mm * 10.0,
        "conf": cand.confidence,
        "spacing_px": cand.spacing_px,
    }
    box = clamp_box((x0, max(0, min(ys) - 5), x1, min(h, max(ys) + 6)), w, h)
    points = [{"x0": int(x0), "x1": int(x1), "y": y} for y in ys]
    rows.append(("png_left_ruler", det, base_record(image_id, gray, "png_left_ruler", det, "left_ruler_ticks", box, points)))


def add_right_ruler(rows, image_id: str, gray: np.ndarray):
    det = ST.recover_scale_right_ruler(gray, tick_cm=0.5)
    if not det or det.get("conf", 0.0) < STRICT_CONF["right_ruler_5mm"]:
        return
    h, w = gray.shape
    x = int(det.get("x", w - 55))
    ys = [int(y) for y in det.get("peaks", [])]
    if not ys:
        return
    box = clamp_box((x - 28, max(0, min(ys) - 5), x + 40, min(h, max(ys) + 6)), w, h)
    points = [{"x0": max(0, x - 28), "x1": min(w - 1, x + 40), "y": y} for y in ys]
    rows.append(("right_ruler_5mm", det, base_record(image_id, gray, "right_ruler_5mm", det, "right_ruler_ticks", box, points)))


def add_family_b_signature(rows, image_id: str, gray: np.ndarray):
    det = ST.recover_scale_family_b_signature(gray)
    if not det:
        return
    h, w = gray.shape
    sig = [73, 82, 293, 302]
    box = clamp_box((0, min(sig) - 8, min(32, w), max(sig) + 9), w, h)
    points = [{"x0": 0, "x1": min(31, w - 1), "y": y} for y in sig]
    rows.append(("family_b_signature", det, base_record(image_id, gray, "family_b_signature", det, "ui_signature_marks", box, points)))


def add_bottom_scale_bar(rows, image_id: str, gray: np.ndarray):
    det = recover_bottom_scale_bar_3cm(gray)
    if not det or det.get("conf", 0.0) < STRICT_CONF["bottom_scale_bar_3cm"]:
        return
    h, w = gray.shape
    y = int(det["bar_y"])
    x0 = int(det["bar_x0"])
    x1 = int(det["bar_x1"])
    box = clamp_box((x0, y - 4, x1 + 1, y + 5), w, h)
    points = [{"x0": x0, "x1": x1, "y": y}]
    rows.append(("bottom_scale_bar_3cm", det, base_record(image_id, gray, "bottom_scale_bar_3cm", det, "bottom_scale_bar", box, points)))


def draw_label(mask: Image.Image, overlay: Image.Image, rec: dict, color):
    mask_draw = ImageDraw.Draw(mask)
    overlay_draw = ImageDraw.Draw(overlay)
    box = (int(rec["box_x0"]), int(rec["box_y0"]), int(rec["box_x1"]), int(rec["box_y1"]))
    points = json.loads(rec["points_json"])

    if rec["cue_class"] == "bottom_tick_axis":
        for p in points:
            rect(mask_draw, (p["x"] - 3, p["y0"], p["x"] + 4, p["y1"] + 1), 255)
            rect(overlay_draw, (p["x"] - 3, p["y0"], p["x"] + 4, p["y1"] + 1), color, width=1)
        rect(mask_draw, (box[0], box[3] - 3, box[2], box[3]), 255)
        rect(overlay_draw, box, color, width=2)
    elif rec["cue_class"] in {"left_ruler_ticks", "right_ruler_ticks", "ui_signature_marks"}:
        for p in points:
            y = int(p["y"])
            rect(mask_draw, (p["x0"], y - 3, p["x1"] + 1, y + 4), 255)
            rect(overlay_draw, (p["x0"], y - 3, p["x1"] + 1, y + 4), color, width=1)
        rect(overlay_draw, box, color, width=2)
    elif rec["cue_class"] == "bottom_scale_bar":
        for p in points:
            rect(mask_draw, (p["x0"], p["y"] - 3, p["x1"] + 1, p["y"] + 4), 255)
            rect(overlay_draw, (p["x0"], p["y"] - 3, p["x1"] + 1, p["y"] + 4), color, width=2)
        rect(overlay_draw, box, color, width=2)


def write_artifacts(image_id: str, gray: np.ndarray, labels: list[tuple[str, dict, dict]]):
    stem = Path(image_id).stem
    for method, _det, rec in labels:
        mask = Image.new("L", (gray.shape[1], gray.shape[0]), 0)
        overlay = display_rgb(gray)
        color = COLORS.get(method, (255, 255, 0))
        draw_label(mask, overlay, rec, color)
        safe_method = method.replace("/", "_")
        mask_path = MASK_DIR / f"{stem}__{safe_method}.png"
        overlay_path = OVERLAY_DIR / f"{stem}__{safe_method}.jpg"
        mask.save(mask_path)
        overlay.save(overlay_path, quality=92)
        rec["mask_path"] = str(mask_path.relative_to(ROOT))
        rec["overlay_path"] = str(overlay_path.relative_to(ROOT))


def label_image(path: Path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"could not read {path}")
    labels: list[tuple[str, dict, dict]] = []
    route = ST.recover_for_image_detail(gray, path.name)
    method = route.get("method", "none")

    # Export high-precision teacher labels only from the production-accepted
    # route. Raw cue detectors can over-fire when run out of family context.
    if method == "bottom_ticks":
        add_bottom_ticks(labels, path.name, gray)
    elif method == "left_ruler_1cm":
        add_left_ruler(labels, path.name, gray)
    elif method == "png_left_ruler":
        add_png_left_ruler(labels, path.name, gray)
    elif method == "right_ruler_5mm":
        add_right_ruler(labels, path.name, gray)
    elif method == "family_b_signature":
        add_family_b_signature(labels, path.name, gray)
    elif method == "none":
        add_bottom_scale_bar(labels, path.name, gray)
    return gray, labels


def clear_artifact_dir(path: Path, suffixes: set[str]):
    path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()
    expected = OUT.resolve()
    if expected not in resolved.parents and resolved != expected:
        raise RuntimeError(f"refusing to clear unexpected path: {path}")
    for item in path.iterdir():
        if item.is_file() and item.suffix.lower() in suffixes:
            item.unlink()


def main():
    if not TEST.exists():
        raise SystemExit(f"missing test directory: {TEST}")
    OUT.mkdir(parents=True, exist_ok=True)
    clear_artifact_dir(MASK_DIR, {".png"})
    clear_artifact_dir(OVERLAY_DIR, {".jpg", ".jpeg"})

    rows = []
    image_counts = Counter()
    files = sorted(p for p in TEST.iterdir() if p.suffix.lower() in IMG_EXTS)
    for path in files:
        gray, labels = label_image(path)
        if labels:
            write_artifacts(path.name, gray, labels)
        for method, _det, rec in labels:
            rows.append(rec)
            image_counts[path.name] += 1

    df = pd.DataFrame(rows)
    df.to_csv(MANIFEST, index=False)

    summary_rows = [
        {"view": "all", "key": "images_total", "n": len(files)},
        {"view": "all", "key": "images_with_any_cue", "n": len(image_counts)},
        {"view": "all", "key": "label_rows", "n": len(df)},
    ]
    if len(df):
        for method, sub in df.groupby("cue_method"):
            summary_rows.append({
                "view": "cue_method",
                "key": method,
                "n": int(len(sub)),
                "images": int(sub["image_id"].nunique()),
                "strict_rows": int(sub["strict"].sum()),
                "median_confidence": float(sub["confidence"].median()),
            })
        for cls, sub in df.groupby("cue_class"):
            summary_rows.append({
                "view": "cue_class",
                "key": cls,
                "n": int(len(sub)),
                "images": int(sub["image_id"].nunique()),
                "strict_rows": int(sub["strict"].sum()),
                "median_confidence": float(sub["confidence"].median()),
            })
    pd.DataFrame(summary_rows).to_csv(SUMMARY, index=False)

    print(f"images: {len(files)}")
    print(f"images with any cue label: {len(image_counts)}")
    print(f"label rows: {len(df)}")
    if len(df):
        print("\nby cue method:")
        print(df["cue_method"].value_counts().sort_index().to_string())
        print("\nby cue class:")
        print(df["cue_class"].value_counts().sort_index().to_string())
    print(f"\nwrote {MANIFEST}")
    print(f"wrote {SUMMARY}")
    print(f"wrote masks -> {MASK_DIR}")
    print(f"wrote overlays -> {OVERLAY_DIR}")
    print("\nread: these are reproducible weak labels for training/QA, not hand annotations or a submission.")


if __name__ == "__main__":
    main()
