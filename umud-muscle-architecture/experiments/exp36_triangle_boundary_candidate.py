"""Generate expert-benchmark candidates for triangle upper-boundary modes.

This uses the production `segment_then_measure.measure()` function against the
saved 35-image expert masks, so the local benchmark candidate and submission
flag exercise the same geometry path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import benchmark_validate as BV  # noqa: E402
import segment_then_measure as M  # noqa: E402

MASK_DIR = ROOT / "results" / "visual_review"
OUT_ROBUST = ROOT / "results" / "benchmark_pred_robust_triangle.csv"
OUT_EXACT = ROOT / "results" / "benchmark_pred_exact_triangle.csv"
SUMMARY = ROOT / "results" / "benchmark_triangle_boundary_summary.csv"


def load_mask(path: Path):
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(path)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return (arr[:, :, 3] > 0).astype("uint8")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return (arr > 0).astype("uint8")


def predict_for_mode(truth: pd.DataFrame, mode: str) -> pd.DataFrame:
    old_mode = M.TOP_BOUNDARY_MODE
    M.TOP_BOUNDARY_MODE = mode
    rows = []
    try:
        for r in truth.itertuples():
            image_id = str(r.ImageID)
            ppm = float(r.scale_px_per_cm) / 10.0
            geom = M.measure(load_mask(MASK_DIR / f"{image_id}_apo.png"), load_mask(MASK_DIR / f"{image_id}_fasc.png"))
            if geom is None:
                raise RuntimeError(f"measurement failed for {image_id}")
            rows.append({
                "image_id": image_id,
                "pa_deg": float(geom["pa_deg"]),
                "fl_mm": float(geom["fl_px"]) / ppm,
                "mt_mm": float(geom["mt_px"]) / ppm,
            })
    finally:
        M.TOP_BOUNDARY_MODE = old_mode
    return pd.DataFrame(rows)


def main() -> None:
    truth, _ = BV.load_truth()
    outputs = {
        "exact_triangle": (OUT_EXACT, predict_for_mode(truth, "triangle")),
        "robust_triangle": (OUT_ROBUST, predict_for_mode(truth, "robust_triangle")),
    }
    summary_rows = []
    for name, (path, df) in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        df.round({"pa_deg": 6, "fl_mm": 6, "mt_mm": 6}).to_csv(path, index=False)
        score = BV.score(df, truth)
        summary_rows.append({
            "candidate": name,
            "path": str(path),
            "overall": score["overall"],
            "pa": score["pa_deg"],
            "fl": score["fl_mm"],
            "mt": score["mt_mm"],
            "n": score["n"],
        })
    summary = pd.DataFrame(summary_rows).sort_values("overall")
    summary.to_csv(SUMMARY, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote:\n  {OUT_ROBUST}\n  {OUT_EXACT}\n  {SUMMARY}")


if __name__ == "__main__":
    main()
