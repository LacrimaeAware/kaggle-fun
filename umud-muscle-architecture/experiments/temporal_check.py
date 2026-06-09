"""Sanity-check temporal smoothing on the test set WITHOUT overwriting submission_local.csv.

Loads the current submission, computes fingerprints for the 309 images, detects sequence clips, and
reports how many clips exist and how much median-smoothing would move PA/FL/MT. Validates the clip
detection (can't validate the SCORE without labels).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import segment_then_measure as M  # noqa: E402

sub = pd.read_csv(ROOT / "results" / "submission_local.csv")
files = sorted(p for p in M.DIRS["test"].iterdir() if p.is_file() and p.suffix.lower() in M.IMG_EXTS)
assert [p.name for p in files] == list(sub["image_id"]), "submission order != file order"

fps = np.asarray([M.fingerprint(M.read_rgb(p)) for p in files], np.float32)
sim = (fps[:-1] * fps[1:]).sum(axis=1)
for thr in (0.88, 0.90, 0.92, 0.95):
    clip = np.zeros(len(fps), int)
    for i in range(1, len(fps)):
        clip[i] = clip[i - 1] + (1 if sim[i - 1] < thr else 0)
    sizes = np.bincount(clip)
    multi = sizes[(sizes >= 2) & (sizes <= 12)]
    print(f"thr {thr}: {len(multi)} clips of 2-12 frames covering {int(multi.sum())} images "
          f"| longest clip {sizes.max()} | pairs>thr {int((sim >= thr).sum())}/{len(sim)}")

# how much would smoothing (thr 0.92) move each column?
sm = M.temporal_smooth(sub.copy(), fps, thresh=0.92)
for col in ("pa_deg", "fl_mm", "mt_mm"):
    d = (sm[col] - sub[col]).abs()
    print(f"{col}: {int((d > 1e-6).sum())} rows changed, mean |change| {d[d > 1e-6].mean():.2f} "
          f"(tol {M.PRIOR and {'pa_deg':6,'fl_mm':12,'mt_mm':3}[col]})")
