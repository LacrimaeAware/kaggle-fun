"""Experiment 02: (a) which of our inputs limits FL=MT/sin(PA), and (b) is the Kaggle test
made of time-series clips (so temporal smoothing is a lever)?

(a) Swap in the EXPERTS' MT or PA one at a time to see whether our PA error or our MT error
    bounds the derived FL. Tells us where to spend effort.
(b) Compute consecutive-frame image similarity over the 309 Kaggle test images to detect clips.

CPU only. Needs results/benchmark_pred_truescale.csv and the Kaggle test images under data/.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import benchmark_validate as BV  # noqa: E402

TOL = BV.TOL


def part_a():
    truth, _ = BV.load_truth()
    pred = pd.read_csv(ROOT / "results" / "benchmark_pred_truescale.csv")
    pred["ImageID"] = pred["image_id"]
    m = truth.merge(pred, on="ImageID")
    print("(a) FL = MT/sin(PA): which input limits us? (FL-term, lower better)")
    combos = {
        "our MT, our PA (real)": (m["mt_mm"], m["pa_deg"]),
        "EXPERT MT, our PA": (m["mt_mm_true"], m["pa_deg"]),
        "our MT, EXPERT PA": (m["mt_mm"], m["pa_deg_true"]),
        "EXPERT MT, EXPERT PA (floor)": (m["mt_mm_true"], m["pa_deg_true"]),
    }
    for name, (mt, pa) in combos.items():
        fl = np.clip(mt / np.sin(np.radians(pa)), 30, 200)
        term = (fl - m["fl_mm_true"]).abs().mean() / TOL["fl_mm"]
        print(f"    {name:32s} FL-term {term:.3f}")
    print("    -> whichever swap helps most is the input worth improving.")


def part_b():
    test_dir = next((p.parent for p in ROOT.glob("data/**/IMG_00001.tif")), None)
    if test_dir is None:
        print("\n(b) Kaggle test images not found under data/, skipping sequence detection.")
        return
    files = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in (".tif", ".png")], key=lambda p: p.name)

    def feat(p):
        a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        a = cv2.resize(a, (64, 64)).astype(np.float32)
        return ((a - a.mean()) / (a.std() + 1e-6)).ravel()

    fs = [feat(p) for p in files]
    sims = np.array([float(np.dot(fs[i], fs[i + 1]) / fs[i].size) for i in range(len(fs) - 1)])
    # a clip boundary = consecutive similarity drops low
    thr = 0.6
    boundaries = [i + 1 for i, s in enumerate(sims) if s < thr]
    starts = [0] + boundaries
    ends = boundaries + [len(files)]
    runs = [e - s for s, e in zip(starts, ends)]
    print("\n(b) Test sequence detection (consecutive-frame similarity, threshold %.2f):" % thr)
    print(f"    {len(files)} images -> {len(runs)} clips; clip lengths: median {int(np.median(runs))}, "
          f"mode-ish {pd.Series(runs).mode().tolist()[:3]}, max {max(runs)}")
    print(f"    consecutive similarity: median {np.median(sims):.2f}, "
          f"{int((sims>0.9).sum())}/{len(sims)} pairs >0.9 (clearly same clip)")
    # show first few clip groupings
    show = []
    for s, e in list(zip(starts, ends))[:6]:
        show.append(f"{files[s].name}..{files[e-1].name} (n={e-s})")
    print("    first clips: " + " | ".join(show))


if __name__ == "__main__":
    part_a()
    part_b()
