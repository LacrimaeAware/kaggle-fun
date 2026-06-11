"""Score saved human benchmark labels.

This script intentionally has a low-dependency first layer: mask overlap metrics work with only
OpenCV + NumPy. Geometry metrics are attempted if the project pipeline imports cleanly in the current
environment.

Examples:
    python benchmark_lab/score_labels.py --manifest results/human_benchmark/manifest.csv
    python benchmark_lab/score_labels.py --manifest results/human_benchmark/manifest.csv --pred-dir results/pred_masks
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
LAYERS = ("apo", "fasc")
FASC_MIN_AREA = 40
FASC_MIN_ANG = 6.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def path_or_none(value: str | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return Path(str(value))


def read_meta(label_dir: Path) -> dict[str, str]:
    path = label_dir / "meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_mask(path: Path) -> np.ndarray | None:
    if not path or not path.exists():
        return None
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    if arr.ndim == 3 and arr.shape[2] == 4:
        return (arr[:, :, 3] > 0).astype(np.uint8)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return (arr > 0).astype(np.uint8)


def overlap(a: np.ndarray | None, b: np.ndarray | None) -> dict[str, str]:
    if a is None or b is None:
        return {"dice": "", "iou": "", "shape": ""}
    if a.shape != b.shape:
        return {"dice": "", "iou": "", "shape": f"{a.shape}!={b.shape}"}
    aa = a.astype(bool)
    bb = b.astype(bool)
    inter = int(np.logical_and(aa, bb).sum())
    sa = int(aa.sum())
    sb = int(bb.sum())
    union = int(np.logical_or(aa, bb).sum())
    dice = 1.0 if sa + sb == 0 else (2.0 * inter / (sa + sb))
    iou = 1.0 if union == 0 else (inter / union)
    return {"dice": f"{dice:.6f}", "iou": f"{iou:.6f}", "shape": "ok"}


def parse_float(value: str | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def import_measure():
    sys.path.insert(0, str(ROOT))
    try:
        import segment_then_measure as M  # noqa: WPS433
        return M, ""
    except Exception as exc:  # dependency-light environments can still score mask overlap.
        return None, str(exc)


def fit_line(ys: np.ndarray, xs: np.ndarray) -> tuple[float, float]:
    m = np.polyfit(xs.astype(float), ys.astype(float), 1)
    return float(m[0]), float(m[1])


def pca_line(ys: np.ndarray, xs: np.ndarray) -> tuple[float, float]:
    pts = np.column_stack([xs.astype(float), ys.astype(float)])
    cen = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - cen, full_matrices=False)
    vx, vy = vh[0]
    if abs(vx) < 1e-6:
        vx = 1e-6
    slope = float(vy / vx)
    return slope, float(cen[1] - slope * cen[0])


def weighted_median(vals: list[float], wts: list[float]) -> float | None:
    if not vals:
        return None
    order = np.argsort(vals)
    v = np.asarray(vals, dtype=float)[order]
    w = np.asarray(wts, dtype=float)[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, c[-1] / 2.0)])


def line_y(line: tuple[float, float], x: float) -> float:
    return line[0] * x + line[1]


def line_intersection(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float] | None:
    denom = a[0] - b[0]
    if abs(denom) < 1e-6:
        return None
    x = (b[1] - a[1]) / denom
    return float(x), float(line_y(a, x))


def visible_length(xs: np.ndarray, ys: np.ndarray, slope: float) -> float:
    if len(xs) < 2:
        return 0.0
    # Projection length along the fitted line direction.
    ux = 1.0 / np.sqrt(1.0 + slope * slope)
    uy = slope * ux
    proj = xs.astype(float) * ux + ys.astype(float) * uy
    return float(proj.max() - proj.min())


def apo_boundary_groups(apo_mask: np.ndarray) -> list[tuple[float, np.ndarray, np.ndarray]] | None:
    """Group hand-drawn apo fragments into upper/lower boundaries.

    Human labels are often made of several strokes. Treating connected components literally would
    mistake two pieces of the same boundary for two separate boundaries, so split components at the
    largest vertical gap between component centroids.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(apo_mask, connectivity=8)
    comps = []
    for i in range(1, n):
        area = int(stats[i, 4])
        if area < 5:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 2:
            continue
        comps.append({"mean_y": float(np.mean(ys)), "xs": xs, "ys": ys, "area": area})
    if len(comps) < 2:
        return None
    comps.sort(key=lambda c: c["mean_y"])
    if len(comps) == 2:
        split = 1
    else:
        gaps = [comps[i + 1]["mean_y"] - comps[i]["mean_y"] for i in range(len(comps) - 1)]
        split = int(np.argmax(gaps)) + 1
    groups = [comps[:split], comps[split:]]
    if not groups[0] or not groups[1]:
        return None
    out = []
    for group in groups:
        xs = np.concatenate([g["xs"] for g in group])
        ys = np.concatenate([g["ys"] for g in group])
        out.append((float(np.average([g["mean_y"] for g in group], weights=[g["area"] for g in group])), xs, ys))
    return out


def measure_light(apo_mask: np.ndarray, fasc_mask: np.ndarray) -> dict[str, float] | None:
    """Production-shaped mask geometry using only cv2/numpy.

    This intentionally mirrors the safe fragment-FL baseline: top/bottom aponeurosis components,
    inner-edge MT, area-weighted PA, and median fragment extrapolated FL.
    """
    apo_mask = np.ascontiguousarray(apo_mask, np.uint8)
    fasc_mask = np.ascontiguousarray(fasc_mask, np.uint8)
    band_info = apo_boundary_groups(apo_mask)
    if band_info is None:
        return None
    band_info.sort(key=lambda item: item[0])
    fit = []
    for role, (_, xs, ys) in zip(("sup", "deep"), band_info):
        if len(xs) < 10:
            return None
        ux, inv = np.unique(xs, return_inverse=True)
        if role == "sup":
            ey = np.full(len(ux), -1.0)
            np.maximum.at(ey, inv, ys.astype(float))
        else:
            ey = np.full(len(ux), 1e18)
            np.minimum.at(ey, inv, ys.astype(float))
        fit.append(fit_line(ey, ux.astype(float)))
    superficial, deep = fit
    deep_s = deep[0]
    x_center = apo_mask.shape[1] / 2.0
    mt_px = abs(line_y(deep, x_center) - line_y(superficial, x_center)) / np.sqrt(1 + deep_s**2)

    nf, labf, statsf, _ = cv2.connectedComponentsWithStats(fasc_mask, connectivity=8)
    angs: list[float] = []
    wts: list[float] = []
    fls: list[float] = []
    for i in range(1, nf):
        area = int(statsf[i, 4])
        if area < FASC_MIN_AREA:
            continue
        ys, xs = np.where(labf == i)
        if len(xs) < 8:
            continue
        fs, fb = pca_line(ys, xs)
        a = abs(np.degrees(np.arctan(fs) - np.arctan(deep_s)))
        if a > 90:
            a = 180 - a
        upper = line_intersection((fs, fb), superficial)
        lower = line_intersection((fs, fb), deep)
        if FASC_MIN_ANG <= a <= 75:
            angs.append(float(a))
            wts.append(float(max(area, 1)))
            if upper is not None and lower is not None:
                fl = float(np.hypot(upper[0] - lower[0], upper[1] - lower[1]))
                if 10.0 <= fl <= 4000.0:
                    fls.append(fl)
    pa = weighted_median(angs, wts)
    if pa is None:
        return None
    return {
        "pa_deg": pa,
        "fl_px": None if not fls else float(np.median(fls)),
        "mt_px": float(mt_px),
        "n_fascicles": float(len(angs)),
    }


def find_pred_mask(pred_dir: Path | None, label_id: str, image_id: str, layer: str) -> Path | None:
    if pred_dir is None:
        return None
    candidates = [
        pred_dir / label_id / f"{layer}.png",
        pred_dir / f"{label_id}_{layer}.png",
        pred_dir / image_id / f"{layer}.png",
        pred_dir / f"{image_id}_{layer}.png",
    ]
    return next((p for p in candidates if p.exists()), None)


def metric_delta(row: dict[str, str], meta: dict[str, str], measured: dict[str, float]) -> dict[str, str]:
    out = {}
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        manual = parse_float(meta.get(col, ""))
        pred = parse_float(measured.get(col))
        out[f"{col}_manual"] = "" if manual is None else f"{manual:.6f}"
        out[f"{col}_measured"] = "" if pred is None else f"{pred:.6f}"
        out[f"{col}_delta"] = "" if manual is None or pred is None else f"{pred - manual:.6f}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "results" / "human_benchmark" / "manifest.csv"))
    ap.add_argument("--labels-dir", default=str(ROOT / "results" / "human_benchmark" / "labels"))
    ap.add_argument("--pred-dir", default="", help="optional predicted masks to compare against human labels")
    ap.add_argument("--out", default=str(ROOT / "results" / "human_benchmark" / "scores.csv"))
    ap.add_argument("--production-measure", action="store_true",
                    help="use segment_then_measure.measure instead of the split-stroke-tolerant label scorer")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    labels_dir = Path(args.labels_dir)
    pred_dir = Path(args.pred_dir) if args.pred_dir else None
    out_path = Path(args.out)
    rows = read_csv(manifest)
    M, import_error = import_measure() if args.production_measure else (None, "")
    scored = []

    for src in rows:
        label_id = src["label_id"]
        image_id = src.get("image_id", "")
        label_dir = labels_dir / label_id
        meta = read_meta(label_dir)
        apo = load_mask(label_dir / "apo.png")
        fasc = load_mask(label_dir / "fasc.png")
        ignore = load_mask(label_dir / "ignore.png")
        apo_pixels = 0 if apo is None else int(apo.sum())
        fasc_pixels = 0 if fasc is None else int(fasc.sum())

        out = {
            "label_id": label_id,
            "image_id": image_id,
            "source": src.get("source", ""),
            "label_mode": src.get("label_mode", ""),
            "quality": meta.get("quality", ""),
            "has_apo": str(apo_pixels > 0).lower(),
            "has_fasc": str(fasc_pixels > 0).lower(),
            "apo_pixels": str(apo_pixels),
            "fasc_pixels": str(fasc_pixels),
            "ignore_pixels": "" if ignore is None else str(int(ignore.sum())),
            "notes": meta.get("notes", ""),
        }

        for layer, human in (("apo", apo), ("fasc", fasc)):
            ref_path = path_or_none(src.get(f"reference_{layer}_mask_path", ""))
            ref = load_mask(ref_path)
            ref_ov = overlap(human, ref)
            out[f"{layer}_dice_vs_reference"] = ref_ov["dice"]
            out[f"{layer}_iou_vs_reference"] = ref_ov["iou"]
            out[f"{layer}_reference_shape"] = ref_ov["shape"]

            pred = load_mask(find_pred_mask(pred_dir, label_id, image_id, layer))
            pred_ov = overlap(pred, human)
            out[f"{layer}_dice_pred_vs_human"] = pred_ov["dice"]
            out[f"{layer}_iou_pred_vs_human"] = pred_ov["iou"]
            out[f"{layer}_pred_shape"] = pred_ov["shape"]

        measured = {}
        if apo_pixels == 0 or fasc_pixels == 0:
            out["measure_error"] = "missing human apo/fasc mask"
        else:
            try:
                geom = M.measure(apo, fasc) if args.production_measure and M is not None else measure_light(apo, fasc)
                if geom is None:
                    out["measure_error"] = "measure returned None"
                else:
                    out["measure_engine"] = "production" if args.production_measure and M is not None else "light_cv2_numpy"
                    scale = parse_float(meta.get("scale_px_per_mm")) or parse_float(src.get("scale_px_per_mm"))
                    measured["pa_deg"] = geom.get("pa_deg")
                    if scale:
                        measured["fl_mm"] = None if geom.get("fl_px") is None else geom["fl_px"] / scale
                        measured["mt_mm"] = geom["mt_px"] / scale
                    out["measure_error"] = ""
                    out["fl_px_measured"] = "" if geom.get("fl_px") is None else f"{geom['fl_px']:.6f}"
                    out["mt_px_measured"] = f"{geom['mt_px']:.6f}"
                    out["n_fascicles"] = str(geom.get("n_fascicles", ""))
            except Exception as exc:
                out["measure_error"] = str(exc)
        out.update(metric_delta(src, meta, measured))
        scored.append(out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in scored for k in row})
    preferred = [
        "label_id", "image_id", "source", "label_mode", "quality", "has_apo", "has_fasc",
        "apo_pixels", "fasc_pixels",
        "measure_error", "pa_deg_manual", "pa_deg_measured", "pa_deg_delta",
        "fl_mm_manual", "fl_mm_measured", "fl_mm_delta", "mt_mm_manual", "mt_mm_measured",
        "mt_mm_delta", "fl_px_measured", "mt_px_measured", "n_fascicles",
        "apo_dice_vs_reference", "fasc_dice_vs_reference",
        "apo_dice_pred_vs_human", "fasc_dice_pred_vs_human",
        "ignore_pixels", "notes",
    ]
    ordered = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for row in scored:
            w.writerow(row)

    labeled = sum(1 for row in scored if row["has_apo"] == "true" or row["has_fasc"] == "true")
    print(f"wrote {len(scored)} rows -> {out_path}")
    print(f"labeled rows with any mask: {labeled}/{len(scored)}")
    if args.production_measure and import_error:
        print(f"production geometry import unavailable; used light cv2/numpy fallback where masks exist: {import_error}")


if __name__ == "__main__":
    main()
