"""Derive muscle-architecture pseudo-labels from UMUD segmentation masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


@dataclass(frozen=True)
class Line:
    slope: float
    intercept: float

    def y_at(self, x: float) -> float:
        return self.slope * x + self.intercept


def read_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"OpenCV failed to read {path}")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr


def image_shape(path: Path) -> tuple[int, int]:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"OpenCV failed to read {path}")
    return int(arr.shape[0]), int(arr.shape[1])


def _component_points(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    binary = mask > 0
    lab = label(binary, connectivity=2)
    points = []
    for region in regionprops(lab):
        if region.area < min_area:
            continue
        coords = region.coords.astype(float)
        points.append(coords)
    return points


def _scale_points(
    coords_yx: np.ndarray,
    mask_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    y = coords_yx[:, 0] * (target_shape[0] / mask_shape[0])
    x = coords_yx[:, 1] * (target_shape[1] / mask_shape[1])
    return x, y


def fit_line_xy(x: np.ndarray, y: np.ndarray) -> Line:
    if len(x) < 2:
        raise ValueError("Need at least two points to fit a line")
    slope, intercept = np.polyfit(x, y, deg=1)
    return Line(float(slope), float(intercept))


def angle_between_degrees(a: Line, b: Line) -> float:
    angle = abs(np.degrees(np.arctan(a.slope) - np.arctan(b.slope)))
    if angle > 90.0:
        angle = 180.0 - angle
    return float(angle)


def line_intersection(a: Line, b: Line) -> tuple[float, float] | None:
    denom = a.slope - b.slope
    if abs(denom) < 1e-6:
        return None
    x = (b.intercept - a.intercept) / denom
    return float(x), float(a.y_at(x))


def aponeurosis_lines(mask_path: Path, target_shape: tuple[int, int]) -> tuple[Line, Line]:
    mask = read_gray(mask_path)
    components = _component_points(mask, min_area=200)
    if len(components) < 2:
        raise ValueError(f"Expected two aponeurosis components in {mask_path}")

    fitted = []
    for coords in components:
        x, y = _scale_points(coords, mask.shape[:2], target_shape)
        line = fit_line_xy(x, y)
        fitted.append((float(np.mean(y)), line))

    fitted.sort(key=lambda item: item[0])
    superficial = fitted[0][1]
    deep = fitted[-1][1]
    return superficial, deep


def fascicle_lines(mask_path: Path, target_shape: tuple[int, int]) -> list[Line]:
    mask = read_gray(mask_path)
    components = _component_points(mask, min_area=20)
    lines: list[Line] = []
    for coords in components:
        if len(coords) < 8:
            continue
        x, y = _scale_points(coords, mask.shape[:2], target_shape)
        try:
            lines.append(fit_line_xy(x, y))
        except ValueError:
            continue
    return lines


def measure_from_masks(
    apo_mask_path: Path,
    fasc_mask_path: Path,
    image_path: Path,
) -> dict[str, float]:
    target_shape = image_shape(image_path)
    superficial, deep = aponeurosis_lines(apo_mask_path, target_shape)
    fascicles = fascicle_lines(fasc_mask_path, target_shape)
    if not fascicles:
        raise ValueError(f"No fascicle lines found in {fasc_mask_path}")

    x_center = target_shape[1] / 2.0
    mt_px = abs(deep.y_at(x_center) - superficial.y_at(x_center)) / np.sqrt(1 + deep.slope**2)

    pa_values = []
    fl_values = []
    for fascicle in fascicles:
        pa = angle_between_degrees(fascicle, deep)
        upper = line_intersection(fascicle, superficial)
        lower = line_intersection(fascicle, deep)
        if upper is None or lower is None:
            continue
        fl = float(np.hypot(upper[0] - lower[0], upper[1] - lower[1]))
        if 2.0 <= pa <= 75.0 and 10.0 <= fl <= 4000.0:
            pa_values.append(pa)
            fl_values.append(fl)

    if not pa_values or not fl_values:
        raise ValueError(f"No valid fascicle geometry found for {image_path.name}")

    return {
        "pa_deg": float(np.median(pa_values)),
        "fl_px": float(np.median(fl_values)),
        "mt_px": float(mt_px),
        "n_fascicles": float(len(pa_values)),
    }


def paired_mask_rows(data_dir: Path = DATA) -> list[tuple[str, Path, Path, Path]]:
    apo_mask_dir = data_dir / "apo_masks_v1" / "apo_masks_new_model_v1"
    fasc_mask_dir = data_dir / "fasc_masks_v1" / "fasc_masks_new_model_v1"
    image_dir = data_dir / "fasc_imgs_v1" / "fasc_images_new_model_v1"
    names = sorted(
        {p.name for p in apo_mask_dir.glob("*.tif")}
        & {p.name for p in fasc_mask_dir.glob("*.tif")}
        & {p.name for p in image_dir.glob("*.tif")}
    )
    return [
        (name, image_dir / name, apo_mask_dir / name, fasc_mask_dir / name)
        for name in names
    ]


def build_geometry_table(data_dir: Path = DATA) -> pd.DataFrame:
    rows = []
    failures = []
    for name, image_path, apo_path, fasc_path in paired_mask_rows(data_dir):
        try:
            row = measure_from_masks(apo_path, fasc_path, image_path)
            row["image_id"] = name
            rows.append(row)
        except Exception as exc:
            failures.append((name, str(exc)))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[["image_id", "pa_deg", "fl_px", "mt_px", "n_fascicles"]]
    if failures:
        fail_path = HERE / "results" / "mask_geometry_failures.csv"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures, columns=["image_id", "error"]).to_csv(fail_path, index=False)
    return df


def main() -> None:
    out = HERE / "results" / "mask_geometry_pseudolabels.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = build_geometry_table()
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    print(f"rows: {len(df)}")
    if not df.empty:
        print(df[["pa_deg", "fl_px", "mt_px", "n_fascicles"]].describe().to_string())


if __name__ == "__main__":
    main()
