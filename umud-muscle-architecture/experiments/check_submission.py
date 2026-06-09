"""Sanity-check the router-scaled submission: distributions and clipping (the tell for a bad scale)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import segment_then_measure as M  # noqa: E402

sub = pd.read_csv(ROOT / "results" / "submission_local.csv")
dbg = pd.read_csv(ROOT / "results" / "calibration_measurement_debug.csv")
print(f"rows {len(sub)}")
for c, lo, hi in [("pa_deg", M.PA_MIN, M.PA_MAX), ("mt_mm", M.MT_MIN, M.MT_MAX), ("fl_mm", M.FL_MIN, M.FL_MAX)]:
    v = sub[c]
    print(f"{c:7s} mean {v.mean():6.2f} std {v.std():5.2f} min {v.min():6.2f} max {v.max():6.2f} "
          f"| at_min {int((v <= lo + 1e-6).sum()):3d} at_max {int((v >= hi - 1e-6).sum()):3d}")

# scaled vs constant rows
scaled = dbg[dbg["calibration_method"] != "none"]
const = dbg[dbg["calibration_method"] == "none"]
print(f"\nscaled images: {len(scaled)}  | constant images: {len(const)}")
if "calibration_method" in dbg:
    print("methods:", dbg["calibration_method"].value_counts().to_dict())
print(f"\nscaled MT: mean {scaled['mt_mm'].mean():.2f} std {scaled['mt_mm'].std():.2f} "
      f"min {scaled['mt_mm'].min():.2f} max {scaled['mt_mm'].max():.2f} "
      f"| at MT_MAX(50): {int((scaled['mt_mm'] >= M.MT_MAX - 1e-6).sum())}")
print(f"scaled px/mm: mean {scaled['px_per_mm'].mean():.2f} "
      f"(px/cm mean {scaled['px_per_mm'].mean()*10:.0f}, range {scaled['px_per_mm'].min()*10:.0f}-{scaled['px_per_mm'].max()*10:.0f})")
