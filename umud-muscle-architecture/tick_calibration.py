"""Estimate UMUD pixels-per-millimetre from visible ultrasound ruler ticks.

Diagnostic prototype only: it scans test-image edges for repeated ruler ticks, writes a
per-image calibration table, and saves overlays for visual QA.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
HERE = Path(__file__).resolve().parent
LOCAL_DATA = HERE / "data"


@dataclass
class Candidate:
    method: str
    edge: str
    score: float
    spacing_px: float
    tick_mm: float
    crop_box: tuple[int, int, int, int]
    strip_box: tuple[int, int, int, int]
    peaks: list[int]
    n_regular: int
    n_peaks: int

    @property
    def px_per_mm(self) -> float:
        return self.spacing_px / self.tick_mm

    @property
    def confidence(self) -> float:
        return float(max(0.0, min(1.0, self.score / 8.0)))


def list_images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def index_root(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for dirpath, dirnames, _files in os.walk(root):
        for dn in dirnames:
            index.setdefault(dn, []).append(Path(dirpath) / dn)
        dirnames[:] = [d for d in dirnames if d != "test_set_v2"]
    return index


def resolve_test_dir() -> Path:
    roots: list[Path] = []
    if Path("/kaggle/input").exists():
        roots += sorted(p for p in Path("/kaggle/input").iterdir() if p.is_dir())
    if LOCAL_DATA.exists():
        roots.append(LOCAL_DATA)

    best: tuple[int, Path] | None = None
    for root in roots:
        index = index_root(root)
        for leaf in ("test_set_v2", "test_images_v2"):
            for cand in index.get(leaf, []):
                n = len(list_images(cand))
                if n and (best is None or n > best[0]):
                    best = (n, cand)
    if best is None:
        raise SystemExit("Could not find UMUD test images.")
    return best[1]


def read_gray(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = np.asarray(Image.fromarray(arr[..., :3]).convert("L"))
    return arr.astype(np.uint8, copy=False)


def display_rgb(gray: np.ndarray) -> Image.Image:
    lo, hi = np.percentile(gray, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    disp = np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return Image.fromarray(disp).convert("RGB")


def smooth(x: np.ndarray, width: int = 5) -> np.ndarray:
    return np.convolve(x.astype(float), np.ones(width) / width, mode="same")


def find_peaks(signal: np.ndarray, min_dist: int) -> list[int]:
    sm = smooth(signal)
    median = float(np.median(sm))
    mad = float(np.median(np.abs(sm - median))) + 1e-6
    level = max(median + 5.0 * mad, float(np.percentile(sm, 97.5)))
    raw: list[tuple[float, int]] = []
    for i in range(2, len(sm) - 2):
        if sm[i] >= level and sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1]:
            raw.append((float(sm[i]), i))

    selected: list[int] = []
    for _value, idx in sorted(raw, reverse=True):
        if all(abs(idx - prev) >= min_dist for prev in selected):
            selected.append(idx)
    return sorted(selected)


def find_png_ruler_peaks(signal: np.ndarray, min_dist: int = 20) -> list[int]:
    """Less aggressive peak picker for faint ticks on the PNG left depth ruler."""
    sm = np.convolve(signal.astype(float), np.ones(3) / 3, mode="same")
    median = float(np.median(sm))
    mad = float(np.median(np.abs(sm - median))) + 1e-6
    level = max(median + 3.0 * mad, float(np.percentile(sm, 95.0)))
    raw: list[tuple[float, int]] = []
    for i in range(1, len(sm) - 1):
        if sm[i] >= level and sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1]:
            raw.append((float(sm[i]), i))

    selected: list[int] = []
    for _value, idx in sorted(raw, reverse=True):
        if all(abs(idx - prev) >= min_dist for prev in selected):
            selected.append(idx)
    return sorted(selected)


def spacing_from_peaks(peaks: list[int], min_spacing: int, max_spacing: int) -> tuple[float | None, int]:
    diffs: list[int] = []
    for i, a in enumerate(peaks):
        for b in peaks[i + 1 :]:
            d = b - a
            if min_spacing <= d <= max_spacing:
                diffs.append(d)
    if not diffs:
        return None, 0

    bins: dict[int, list[int]] = {}
    for d in diffs:
        bins.setdefault(int(round(d / 4.0) * 4), []).append(d)
    _key, values = max(bins.items(), key=lambda item: len(item[1]))
    spacing = float(np.median(values))
    tol = max(5.0, spacing * 0.08)
    regular = sum(
        1
        for i, a in enumerate(peaks)
        for b in peaks[i + 1 :]
        if abs((b - a) - spacing) <= tol
    )
    return spacing, regular


def spacing_from_peak_diffs(peaks: list[int], min_spacing: int, max_spacing: int) -> tuple[float | None, int]:
    diffs: list[int] = []
    for i, a in enumerate(peaks):
        for b in peaks[i + 1 :]:
            d = b - a
            if min_spacing <= d <= max_spacing:
                diffs.append(d)
    if not diffs:
        return None, 0

    bins: dict[int, list[int]] = {}
    for d in diffs:
        bins.setdefault(int(round(d / 4.0) * 4), []).append(d)
    _key, values = max(bins.items(), key=lambda item: len(item[1]))
    spacing = float(np.median(values))
    tol = max(5.0, spacing * 0.08)
    regular = sum(1 for d in diffs if abs(d - spacing) <= tol)
    return spacing, regular


def candidate_from_projection(
    *,
    method: str,
    edge: str,
    crop_box: tuple[int, int, int, int],
    strip_box: tuple[int, int, int, int],
    projection: np.ndarray,
    tick_mm: float,
    min_spacing: int,
    max_spacing: int,
) -> Candidate | None:
    peaks = find_peaks(projection, min_dist=max(22, min_spacing // 2))
    if len(peaks) < 4:
        return None
    spacing, regular = spacing_from_peaks(peaks, min_spacing, max_spacing)
    if spacing is None or regular < 2:
        return None
    score = regular + 0.08 * len(peaks) - 0.004 * abs(spacing - 70.0)
    return Candidate(method, edge, float(score), float(spacing), float(tick_mm), crop_box, strip_box, peaks, regular, len(peaks))


def png_left_ruler_candidate(gray: np.ndarray, tick_mm: float) -> Candidate | None:
    """Read the PNG family's left numbered ruler and ignore right-side UI text rows."""
    h, w = gray.shape
    crop_box = (0, 0, min(70, w), h)
    candidates: list[Candidate] = []
    for x1, x2 in ((0, 8), (0, 16), (0, 32), (0, 60), (8, 50)):
        x2 = min(x2, w)
        if x2 <= x1:
            continue
        strip_gray = gray[:, x1:x2]
        bright = strip_gray > 150
        if int(bright.sum()) < 40:
            continue
        peaks = find_png_ruler_peaks(bright.sum(axis=1), min_dist=20)
        if len(peaks) < 4:
            continue
        spacing, regular = spacing_from_peak_diffs(peaks, min_spacing=50, max_spacing=120)
        if spacing is None or regular < 3:
            continue
        score = regular + 0.12 * len(peaks) - 0.002 * abs(spacing - 85.0)
        candidates.append(Candidate(
            method="png_left_ruler",
            edge="left",
            score=float(score),
            spacing_px=float(spacing),
            tick_mm=float(tick_mm),
            crop_box=crop_box,
            strip_box=(x1, 0, x2, h),
            peaks=peaks,
            n_regular=int(regular),
            n_peaks=len(peaks),
        ))
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.score)


def side_candidates(gray: np.ndarray, edge: str, tick_mm: float) -> list[Candidate]:
    h, w = gray.shape
    crop_box = (0, 0, int(w * 0.22), h) if edge == "left" else (int(w * 0.78), 0, w, h)
    x1, y1, x2, y2 = crop_box
    crop = gray[y1:y2, x1:x2]
    threshold = max(165.0, float(np.percentile(crop, 99.5)) * 0.65)
    bright = crop > threshold

    out: list[Candidate] = []
    for strip_w in (8, 12, 18, 24, 32, 44, 60):
        step = max(3, strip_w // 3)
        for sx in range(0, max(1, crop.shape[1] - strip_w), step):
            strip_gray = crop[:, sx : sx + strip_w]
            strip = bright[:, sx : sx + strip_w]
            dark_fraction = float((strip_gray < 45).mean())
            bright_fraction = float(strip.mean())
            if dark_fraction < 0.45 or bright_fraction > 0.20 or int(strip.sum()) < 20:
                continue
            cand = candidate_from_projection(
                method="side_ticks",
                edge=edge,
                crop_box=crop_box,
                strip_box=(x1 + sx, y1, x1 + sx + strip_w, y2),
                projection=strip.sum(axis=1),
                tick_mm=tick_mm,
                min_spacing=35,
                max_spacing=130,
            )
            if cand is not None:
                out.append(cand)
    return out


def bottom_candidates(gray: np.ndarray, tick_mm: float) -> list[Candidate]:
    h, w = gray.shape
    crop_box = (0, int(h * 0.80), w, h)
    x1, y1, x2, y2 = crop_box
    crop = gray[y1:y2, x1:x2]
    threshold = max(165.0, float(np.percentile(crop, 99.5)) * 0.65)
    bright = crop > threshold

    out: list[Candidate] = []
    for strip_h in (8, 12, 18, 24, 32, 44):
        step = max(3, strip_h // 3)
        for sy in range(0, max(1, crop.shape[0] - strip_h), step):
            strip_gray = crop[sy : sy + strip_h, :]
            strip = bright[sy : sy + strip_h, :]
            dark_fraction = float((strip_gray < 45).mean())
            bright_fraction = float(strip.mean())
            if dark_fraction < 0.70 or bright_fraction > 0.08 or int(strip.sum()) < 20:
                continue
            cand = candidate_from_projection(
                method="bottom_ticks",
                edge="bottom",
                crop_box=crop_box,
                strip_box=(x1, y1 + sy, x2, y1 + sy + strip_h),
                projection=strip.sum(axis=0),
                tick_mm=tick_mm,
                min_spacing=45,
                max_spacing=240,
            )
            if cand is not None:
                out.append(cand)
    return out


def choose_candidate(
    gray: np.ndarray,
    side_tick_mm: float,
    bottom_tick_mm: float,
    image_name: str = "",
) -> Candidate | None:
    if image_name.lower().endswith(".png"):
        return png_left_ruler_candidate(gray, side_tick_mm)
    candidates = (
        side_candidates(gray, "left", side_tick_mm)
        + side_candidates(gray, "right", side_tick_mm)
        + bottom_candidates(gray, bottom_tick_mm)
    )
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.score)


def draw_overlay(gray: np.ndarray, cand: Candidate | None, out_path: Path, title: str) -> None:
    im = display_rgb(gray)
    draw = ImageDraw.Draw(im)
    draw.text((8, 8), title, fill=(255, 255, 0))
    if cand is None:
        draw.text((8, 28), "no calibration candidate", fill=(255, 80, 80))
    else:
        draw.rectangle(cand.crop_box, outline=(255, 180, 0), width=2)
        draw.rectangle(cand.strip_box, outline=(0, 255, 255), width=2)
        if cand.edge in {"left", "right"}:
            x1, _y1, x2, _y2 = cand.strip_box
            for p in cand.peaks:
                draw.line((x1, p, x2, p), fill=(255, 0, 0), width=1)
        else:
            _x1, y1, _x2, y2 = cand.strip_box
            for p in cand.peaks:
                draw.line((p, y1, p, y2), fill=(255, 0, 0), width=1)
        draw.text(
            (8, 28),
            (
                f"{cand.method}/{cand.edge} spacing={cand.spacing_px:.1f}px "
                f"tick={cand.tick_mm:g}mm px/mm={cand.px_per_mm:.3f} "
                f"conf={cand.confidence:.2f}"
            ),
            fill=(255, 255, 0),
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, quality=92)


def run(args: argparse.Namespace) -> None:
    test_dir = Path(args.test_dir) if args.test_dir else resolve_test_dir()
    files = list_images(test_dir)
    out_csv = Path(args.out_csv)
    overlay_dir = Path(args.overlay_dir)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for i, path in enumerate(files):
        gray = read_gray(path)
        cand = choose_candidate(gray, args.side_tick_mm, args.bottom_tick_mm, image_name=path.name)
        if cand is None:
            row = {
                "image_id": path.name,
                "px_per_mm": math.nan,
                "confidence": 0.0,
                "method": "none",
                "edge": "",
                "spacing_px": math.nan,
                "tick_mm": math.nan,
                "n_peaks": 0,
                "n_regular": 0,
                "height": gray.shape[0],
                "width": gray.shape[1],
            }
        else:
            row = {
                "image_id": path.name,
                "px_per_mm": round(cand.px_per_mm, 6),
                "confidence": round(cand.confidence, 6),
                "method": cand.method,
                "edge": cand.edge,
                "spacing_px": round(cand.spacing_px, 3),
                "tick_mm": cand.tick_mm,
                "n_peaks": cand.n_peaks,
                "n_regular": cand.n_regular,
                "height": gray.shape[0],
                "width": gray.shape[1],
            }
        rows.append(row)
        if i < args.overlay_limit or (cand is None and args.overlay_failures):
            draw_overlay(gray, cand, overlay_dir / f"{path.stem}_calibration.jpg", path.name)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    px = np.array([float(r["px_per_mm"]) for r in rows if r["method"] != "none"], dtype=float)
    conf = np.array([float(r["confidence"]) for r in rows], dtype=float)
    print(f"test_dir: {test_dir}")
    print(f"wrote: {out_csv} ({len(rows)} rows)")
    print(f"overlays: {overlay_dir}")
    print(f"detected: {len(px)}/{len(rows)}")
    if len(px):
        print(
            "px_per_mm:",
            f"median={np.nanmedian(px):.3f}",
            f"p10={np.nanpercentile(px, 10):.3f}",
            f"p90={np.nanpercentile(px, 90):.3f}",
        )
    print(
        "confidence:",
        f"median={np.nanmedian(conf):.3f}",
        f">=0.5={(conf >= 0.5).sum()}",
        f">=0.7={(conf >= 0.7).sum()}",
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="")
    ap.add_argument("--out-csv", default=str(HERE / "results" / "calibration_debug" / "tick_calibration.csv"))
    ap.add_argument("--overlay-dir", default=str(HERE / "results" / "calibration_debug" / "overlays"))
    ap.add_argument("--overlay-limit", type=int, default=40)
    ap.add_argument("--overlay-failures", action="store_true")
    ap.add_argument("--side-tick-mm", type=float, default=5.0)
    ap.add_argument("--bottom-tick-mm", type=float, default=10.0)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
